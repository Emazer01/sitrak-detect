import base64
import math
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import (
    VisionServiceException,
    TargetNotFoundError,
    ImageQualityError,
    InvalidImageFormatError
)
from app.schemas.detection import (
    DetectionResponse,
    ShotDetail,
    TargetAnalytics
)
from app.services.scoring_engine import scoring_engine
from app.storage.file_storage import storage_service


class VisionService:
    """
    Core Computer Vision pipeline for shooting target analysis:
    1. Pre-processing & image quality check.
    2. Perspective correction & homography warping.
    3. Target center & concentric rings detection.
    4. Bullet hole segmentation & contour filtering.
    5. Scoring calculation & dispersion analytics.
    6. Visual overlay annotation & persistence.
    """

    def __init__(self):
        self.canonical_size = settings.TARGET_CANONICAL_SIZE
        self.default_scale = settings.CALIBRATION_MM_PER_PIXEL

    def decode_image_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Decodes raw binary byte stream into OpenCV BGR numpy array."""
        if not image_bytes:
            raise InvalidImageFormatError("Empty image payload received.")
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise InvalidImageFormatError("Failed to decode image. Unsupported or corrupted file format.")
        return img

    def decode_base64_image(self, b64_string: str) -> np.ndarray:
        """Decodes base64 string (including data URI prefixes) into OpenCV BGR image."""
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(b64_string)
            return self.decode_image_bytes(image_bytes)
        except Exception as e:
            raise InvalidImageFormatError(f"Invalid Base64 image payload: {str(e)}")

    def validate_image_quality(self, img_bgr: np.ndarray) -> None:
        """
        Validates basic image quality: dimensions, blurriness, and brightness.
        """
        h, w = img_bgr.shape[:2]
        if h < 200 or w < 200:
            raise ImageQualityError(f"Image resolution too low ({w}x{h}). Minimum required: 200x200.")

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Check blur using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 15.0:
            raise ImageQualityError(
                f"Image is excessively blurry (sharpness score: {laplacian_var:.1f}). "
                "Please capture a clearer, stabilized photo of the target."
            )

        # Check illumination
        mean_brightness = float(np.mean(gray))
        if mean_brightness < 25.0:
            raise ImageQualityError("Image is too dark to reliably distinguish target markings and bullet holes.")
        elif mean_brightness > 245.0:
            raise ImageQualityError("Image is overexposed / washed out.")

    def correct_perspective(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Detects quadrilateral target paper border and applies perspective warp
        to normalize coordinates into canonical square space (e.g. 800x800).
        Returns (warped_image, was_corrected).
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 30, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return cv2.resize(img_bgr, (self.canonical_size, self.canonical_size)), False

        # Sort contours by area descending
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        target_contour = None
        img_area = float(w * h)

        for c in contours[:5]:
            area = cv2.contourArea(c)
            if area < (img_area * 0.18):
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(approx) == 4:
                target_contour = approx
                break

        if target_contour is not None:
            # Order corners: top-left, top-right, bottom-right, bottom-left
            pts = target_contour.reshape(4, 2).astype(np.float32)
            rect = self._order_points(pts)

            dst = np.array([
                [0, 0],
                [self.canonical_size - 1, 0],
                [self.canonical_size - 1, self.canonical_size - 1],
                [0, self.canonical_size - 1]
            ], dtype=np.float32)

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img_bgr, M, (self.canonical_size, self.canonical_size))
            return warped, True

        # Fallback: square resize to canonical dimensions
        resized = cv2.resize(img_bgr, (self.canonical_size, self.canonical_size))
        return resized, False

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Orders 4 coordinates clockwise: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def detect_target_and_rings(
        self,
        img_bgr: np.ndarray
    ) -> Tuple[float, float, Dict[int, float], float]:
        """
        Detects target center (bullseye) and the radii of scoring rings 1 to 10.
        Returns (center_x, center_y, ring_radii_dict, outer_radius).
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 7)

        # Hough circle transform for concentric rings or bullseye
        min_radius = int(w * 0.03)
        max_radius = int(w * 0.48)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=w * 0.15,
            param1=100,
            param2=35,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        center_x = float(w / 2.0)
        center_y = float(h / 2.0)
        outer_radius = float(w * 0.42)

        if circles is not None and len(circles[0]) > 0:
            circles = np.uint16(np.around(circles[0]))
            # Find the circle closest to the geometric center
            best_circle = None
            min_dist = float("inf")
            for c in circles:
                cx, cy, r = c
                dist = math.hypot(cx - w / 2.0, cy - h / 2.0)
                if dist < min_dist:
                    min_dist = dist
                    best_circle = (float(cx), float(cy), float(r))

            if best_circle and min_dist < (w * 0.25):
                center_x, center_y, detected_r = best_circle
                # If detected radius is the bullseye (rings 7-10) or outer target:
                if detected_r < (w * 0.22):
                    # Detected central black bullseye (~40% of target radius in standard targets)
                    outer_radius = detected_r / 0.4
                else:
                    outer_radius = detected_r

        # Calculate standard 10 scoring ring radii (10 = innermost bullseye, 1 = outermost)
        # Ring k outer radius is proportional: r_k = outer_radius * ((11 - k) / 10.0)
        ring_radii: Dict[int, float] = {}
        for k in range(1, 11):
            ring_radii[k] = outer_radius * ((11.0 - k) / 10.0)

        # Ring 10 (bullseye) minimum boundary
        ring_radii[10] = max(ring_radii[10], outer_radius * 0.08)

        return center_x, center_y, ring_radii, outer_radius

    def detect_bullet_holes(
        self,
        img_bgr: np.ndarray,
        center_x: float,
        center_y: float,
        ring_radii: Dict[int, float],
        max_radius: float
    ) -> List[Tuple[float, float, float]]:
        """
        Segments bullet hole punctures on paper target.
        Returns list of (x, y, radius) tuples.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # Multi-technique detection:
        # Technique 1: Adaptive thresholding for punctured dark/shadow edges
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 5
        )

        # Technique 2: Gradient / Edge magnitude for circular hole borders
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_binary = np.uint8(grad_mag > 50) * 255

        # Morphological opening to clean single-pixel noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.bitwise_or(adaptive_thresh, grad_binary)
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        min_hole_area = 15.0      # min area for a bullet hole
        max_hole_area = 1200.0    # max area for a single bullet hole
        detected_candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_hole_area or area > max_hole_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * math.pi * (area / (perimeter * perimeter))
            if circularity < 0.35:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)

            # Ensure candidate is within target boundary (+ 10% leeway)
            dist_to_center = math.hypot(cx - center_x, cy - center_y)
            if dist_to_center > (max_radius * 1.12):
                continue

            # Ignore central bullseye crosshair intersection if detected as small spot
            if dist_to_center < 3.0 and area < 25.0:
                continue

            detected_candidates.append((float(cx), float(cy), float(radius)))

        # Non-Maximum Suppression (NMS) to merge overlapping or multi-edge detections
        merged_holes = self._merge_overlapping_holes(detected_candidates, merge_threshold=14.0)
        return merged_holes

    def _merge_overlapping_holes(
        self,
        candidates: List[Tuple[float, float, float]],
        merge_threshold: float = 14.0
    ) -> List[Tuple[float, float, float]]:
        """Deduplicates closely packed detections belonging to the same bullet hole."""
        if not candidates:
            return []

        merged = []
        used = [False] * len(candidates)

        for i in range(len(candidates)):
            if used[i]:
                continue
            group = [candidates[i]]
            used[i] = True

            for j in range(i + 1, len(candidates)):
                if used[j]:
                    continue
                d = math.hypot(candidates[i][0] - candidates[j][0], candidates[i][1] - candidates[j][1])
                if d <= merge_threshold:
                    group.append(candidates[j])
                    used[j] = True

            # Calculate weighted center of cluster
            avg_x = float(np.mean([p[0] for p in group]))
            avg_y = float(np.mean([p[1] for p in group]))
            avg_r = float(np.mean([p[2] for p in group]))
            merged.append((avg_x, avg_y, avg_r))

        return merged

    def annotate_target(
        self,
        img_bgr: np.ndarray,
        center_x: float,
        center_y: float,
        ring_radii: Dict[int, float],
        shots: List[ShotDetail],
        analytics: TargetAnalytics
    ) -> np.ndarray:
        """
        Draws visual annotations:
        - Target center crosshair
        - Rings overlay
        - Bullet hole markers with shot index and scores
        - Minimum enclosing grouping circle / dispersion ellipse
        - Directional bias arrow
        - Top stats HUD
        """
        overlay = img_bgr.copy()
        cx_int, cy_int = int(round(center_x)), int(round(center_y))

        # 1. Subtle ring guidelines
        for ring_val in range(1, 11):
            r_px = int(round(ring_radii[ring_val]))
            cv2.circle(overlay, (cx_int, cy_int), r_px, (100, 100, 100), 1, cv2.LINE_AA)
            if ring_val in [3, 5, 7, 9]:
                # Add tiny ring label
                cv2.putText(
                    overlay,
                    str(ring_val),
                    (cx_int, cy_int - r_px + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (140, 140, 140),
                    1,
                    cv2.LINE_AA
                )

        # 2. Bullseye crosshair
        cv2.line(overlay, (cx_int - 15, cy_int), (cx_int + 15, cy_int), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(overlay, (cx_int, cy_int - 15), (cx_int, cy_int + 15), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(overlay, (cx_int, cy_int), 3, (0, 0, 255), -1, cv2.LINE_AA)

        # 3. Grouping dispersion & Mean Point of Impact (MPI)
        if len(shots) >= 2:
            pts = np.array([[int(s.x), int(s.y)] for s in shots], dtype=np.int32)
            (grp_cx, grp_cy), grp_r = cv2.minEnclosingCircle(pts)
            # Translucent grouping circle
            disp_overlay = overlay.copy()
            cv2.circle(disp_overlay, (int(grp_cx), int(grp_cy)), int(grp_r), (255, 200, 0), 2, cv2.LINE_AA)
            cv2.addWeighted(disp_overlay, 0.7, overlay, 0.3, 0, overlay)

            # Directional bias arrow from bullseye to MPI
            if analytics.directional_bias != "centered":
                mpi_x = int(round(center_x + (analytics.mean_point_of_impact_x or 0.0) / self.default_scale))
                mpi_y = int(round(center_y + (analytics.mean_point_of_impact_y or 0.0) / self.default_scale))
                cv2.arrowedLine(overlay, (cx_int, cy_int), (mpi_x, mpi_y), (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.25)

        # 4. Color-coded Shot annotations
        score_colors = {
            10: (0, 215, 255),   # Gold / Yellow
            9:  (50, 205, 50),   # Lime Green
            8:  (0, 200, 0),     # Green
            7:  (0, 165, 255),   # Orange
            6:  (0, 100, 255),   # Orange-Red
        }

        for shot in shots:
            sx, sy = int(round(shot.x)), int(round(shot.y))
            color = score_colors.get(shot.score, (0, 0, 230))  # Default red for <6

            # Inner hole circle
            cv2.circle(overlay, (sx, sy), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, (sx, sy), 8, color, 2, cv2.LINE_AA)

            # Badge pill / text
            badge_text = f"#{shot.id}:{shot.score}"
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            text_size, _ = cv2.getTextSize(badge_text, font_face, font_scale, thickness)

            badge_x = sx + 10
            badge_y = sy - 10
            # Ensure badge stays within image boundary
            h, w = overlay.shape[:2]
            badge_x = min(badge_x, w - text_size[0] - 8)
            badge_y = max(badge_y, text_size[1] + 8)

            # Badge background
            cv2.rectangle(
                overlay,
                (badge_x - 3, badge_y - text_size[1] - 3),
                (badge_x + text_size[0] + 3, badge_y + 3),
                (30, 30, 30),
                -1
            )
            cv2.putText(overlay, badge_text, (badge_x, badge_y), font_face, font_scale, color, thickness, cv2.LINE_AA)

        # 5. Top HUD banner
        hud_h = 42
        hud_bg = overlay.copy()
        cv2.rectangle(hud_bg, (0, 0), (overlay.shape[1], hud_h), (20, 20, 20), -1)
        cv2.addWeighted(hud_bg, 0.85, overlay, 0.15, 0, overlay)

        total_pts = sum(s.score for s in shots)
        max_pts = max(len(shots) * 10, 10)
        acc_pct = (total_pts / max_pts) * 100.0 if shots else 0.0

        hud_text = (
            f"SCORE: {total_pts}/{max_pts} ({acc_pct:.1f}%) | "
            f"SHOTS: {len(shots)} | "
            f"GRP: {analytics.grouping_diameter_mm}mm | "
            f"BIAS: {analytics.directional_bias.upper()}"
        )
        cv2.putText(
            overlay,
            hud_text,
            (14, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        return overlay

    def process_image(
        self,
        image_bytes: bytes,
        original_filename: str = "",
        caliber_mm: float = 4.5,
        target_type: str = "issf_standard_10_ring",
        scale_override: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DetectionResponse:
        """
        Executes end-to-end pipeline:
        decode -> validate -> warp -> detect rings -> detect shots -> score -> annotate -> persist.
        """
        # 1. Decode & Validate
        img_bgr = self.decode_image_bytes(image_bytes)
        self.validate_image_quality(img_bgr)

        # 2. Perspective warp to canonical square
        canonical_bgr, was_warped = self.correct_perspective(img_bgr)

        # 3. Detect Target & Rings
        center_x, center_y, ring_radii, outer_r = self.detect_target_and_rings(canonical_bgr)

        # Scale calibration (mm per pixel)
        # Standard ISSF 25m Pistol target ring 1 is 500mm diameter -> radius 250mm
        # If calibrated scale provided, use it; otherwise compute or default
        scale_mm_px = scale_override or self.default_scale

        # 4. Bullet Hole Detection
        detected_holes = self.detect_bullet_holes(canonical_bgr, center_x, center_y, ring_radii, outer_r)

        # Sort holes chronologically/spatially (top-to-bottom, left-to-right)
        detected_holes = sorted(detected_holes, key=lambda h: (h[1], h[0]))

        # Calculate bullet radius in pixels at canonical scale
        bullet_radius_px = max(4.0, (caliber_mm / 2.0) / scale_mm_px)

        # 5. Evaluate individual shot scores
        shots: List[ShotDetail] = []
        for idx, (hx, hy, hr) in enumerate(detected_holes, start=1):
            shot_detail = scoring_engine.evaluate_shot(
                shot_id=idx,
                x=hx,
                y=hy,
                center_x=center_x,
                center_y=center_y,
                ring_radii=ring_radii,
                bullet_radius_px=bullet_radius_px,
                mm_per_pixel=scale_mm_px
            )
            shots.append(shot_detail)

        # 6. Analytics (dispersion & directional bias)
        analytics = scoring_engine.compute_analytics(
            shots=shots,
            center_x=center_x,
            center_y=center_y,
            mm_per_pixel=scale_mm_px
        )

        # 7. Draw Visual Annotations
        annotated_bgr = self.annotate_target(
            img_bgr=canonical_bgr,
            center_x=center_x,
            center_y=center_y,
            ring_radii=ring_radii,
            shots=shots,
            analytics=analytics
        )

        # 8. Persist annotated result
        _, annotated_url = storage_service.save_annotated_image(annotated_bgr)

        # Calculate aggregate stats
        total_score = sum(s.score for s in shots)
        shots_count = len(shots)
        max_possible_score = max(shots_count * 10, 10 if shots_count == 0 else shots_count * 10)
        accuracy_percentage = round((total_score / max_possible_score) * 100.0, 1) if shots_count > 0 else 0.0

        meta_info = {
            "perspective_corrected": was_warped,
            "target_type": target_type,
            "caliber_mm": caliber_mm,
            "scale_mm_per_pixel": scale_mm_px,
            "center": {"x": round(center_x, 1), "y": round(center_y, 1)}
        }
        if metadata:
            meta_info.update(metadata)

        return DetectionResponse(
            status="success",
            target_detected=True,
            total_score=total_score,
            max_possible_score=max_possible_score,
            accuracy_percentage=accuracy_percentage,
            shots_count=shots_count,
            shots=shots,
            analytics=analytics,
            annotated_image_url=annotated_url,
            metadata=meta_info
        )


vision_service = VisionService()

