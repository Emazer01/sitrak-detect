import base64
import math
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
import cv2
import numpy as np

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
from app.services.gemini_service import GeminiVisionService
from app.storage.file_storage import storage_service


class VisionService:
    """
    State-of-the-art Multi-Algorithm Computer Vision & Deep Learning Engine
    for Shooting Target Hole Detection and Automated Scoring:

    Multi-Model / Multi-Algorithm Ensemble:
    1. Deep Learning (YOLOv8) - Learned feature extraction for paper tears & punctures.
    2. Difference of Gaussians (DoG) - Multi-scale circular blob filter immune to printed lines/numbers.
    3. MSER (Maximally Stable Extremal Regions) - Affine & illumination invariant puncture locator.
    4. Distance Transform & Watershed - Automatic decomposition of overlapping double-taps / keyholes.
    5. Strict False-Positive Filtering - Rejection of printed numbers, target ring arcs, and crosshairs.
    6. Consensus Ensemble Fusion - Multi-engine consensus voting for ultra-high precision.
    7. Gemini Multimodal Cloud AI - High-precision cloud multimodal target reasoning.
    """

    def __init__(self):
        self.canonical_size = settings.TARGET_CANONICAL_SIZE
        self.default_scale = settings.CALIBRATION_MM_PER_PIXEL
        self.ai_model = None
        self.ai_model_name = ""
        self.models_cache: Dict[str, Any] = {}
        self.gemini_service = GeminiVisionService()
        self._load_ai_model()

    def _load_ai_model(self) -> None:
        """Attempts to load trained YOLO bullet hole detector weights (YOLO26 & YOLO11)."""
        self.models_cache = {}
        try:
            from ultralytics import YOLO

            # 1. Attempt loading YOLO26
            yolo26_candidates = [
                settings.AI_YOLO26_PATH,
                settings.BASE_DIR / "models" / "bullet_hole_yolo26.pt",
                settings.BASE_DIR / "training" / "runs" / "yolo26_kruger_exp" / "weights" / "best.pt",
                settings.BASE_DIR / "runs" / "detect" / "training" / "runs" / "yolo26_kruger_exp" / "weights" / "best.pt"
            ]
            for p in yolo26_candidates:
                if p and p.exists():
                    self.models_cache["yolo26"] = YOLO(str(p))
                    print(f"[VisionService] Loaded YOLO26: {p.resolve()}")
                    break

            # 2. Attempt loading YOLO11
            yolo11_candidates = [
                settings.AI_YOLO11_PATH,
                settings.BASE_DIR / "models" / "bullet_hole_yolo11.pt",
                settings.BASE_DIR / "training" / "runs" / "yolo11_kruger_exp" / "weights" / "best.pt",
                settings.BASE_DIR / "runs" / "detect" / "training" / "runs" / "yolo11_kruger_exp" / "weights" / "best.pt"
            ]
            for p in yolo11_candidates:
                if p and p.exists():
                    self.models_cache["yolo11"] = YOLO(str(p))
                    print(f"[VisionService] Loaded YOLO11: {p.resolve()}")
                    break

            # 3. Default AI model
            default_candidates = [
                settings.AI_MODEL_PATH,
                settings.BASE_DIR / "models" / "bullet_hole_yolo.pt",
                settings.BASE_DIR / "models" / "bullet_hole_yolo26.pt",
                settings.BASE_DIR / "models" / "bullet_hole_yolo11.pt"
            ]
            for p in default_candidates:
                if p and p.exists():
                    self.ai_model = YOLO(str(p))
                    self.ai_model_name = p.name
                    print(f"[VisionService] Loaded default AI detector ({p.name}): {p.resolve()}")
                    return
        except ImportError:
            print("[VisionService] ultralytics not installed or still loading. Classical CV active.")
        except Exception as e:
            print(f"[VisionService] Warning: Failed to load YOLO weights ({str(e)}). Classical CV active.")

        self.ai_model = None
        self.ai_model_name = ""

    def reload_model(self) -> bool:
        """Reloads model weights (useful after training a new model checkpoint)."""
        self._load_ai_model()
        return self.ai_model is not None

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
        """Validates basic image quality: dimensions, blurriness, and brightness."""
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
        if mean_brightness < 20.0:
            raise ImageQualityError("Image is too dark to reliably distinguish target markings and bullet holes.")
        elif mean_brightness > 248.0:
            raise ImageQualityError("Image is overexposed / washed out.")

    def _find_target_center_for_crop(self, img_bgr: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """
        Attempts to detect the center (cx, cy) of the shooting target on the full image
        prior to 1:1 cropping, ensuring the square crop window accurately centers on the target.
        """
        h, w = img_bgr.shape[:2]
        max_dim = max(h, w)
        if max_dim <= 0:
            return None, None

        scale = 400.0 / float(max_dim)
        sw, sh = int(w * scale), int(h * scale)
        if sw < 20 or sh < 20:
            return None, None

        small = cv2.resize(img_bgr, (sw, sh))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 7)

        side_small = min(sw, sh)
        min_r = int(side_small * 0.05)
        max_r = int(side_small * 0.48)

        # 1. Hough Circles to find target concentric rings / bullseye
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=int(side_small * 0.15),
            param1=100,
            param2=35,
            minRadius=min_r,
            maxRadius=max_r
        )
        if circles is not None and len(circles[0]) > 0:
            center_dist = []
            img_mid_x, img_mid_y = sw / 2.0, sh / 2.0
            for c in circles[0]:
                cx_s, cy_s = float(c[0]), float(c[1])
                dist = math.hypot(cx_s - img_mid_x, cy_s - img_mid_y)
                center_dist.append((dist, cx_s, cy_s))
            center_dist.sort(key=lambda item: item[0])
            best_cx = center_dist[0][1] / scale
            best_cy = center_dist[0][2] / scale
            return best_cx, best_cy

        # 2. Fallback: Contrast/dark contour centroid (black target center)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            valid_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > (side_small * side_small * 0.02):
                    valid_contours.append((area, cnt))
            if valid_contours:
                valid_contours.sort(key=lambda x: x[0], reverse=True)
                M = cv2.moments(valid_contours[0][1])
                if M["m00"] > 0:
                    cx_s = M["m10"] / M["m00"]
                    cy_s = M["m01"] / M["m00"]
                    return cx_s / scale, cy_s / scale

        return None, None

    def crop_to_square(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, bool, Dict[str, int]]:
        """
        Crops any non-1:1 aspect ratio image into an exact 1:1 square (min(w, h) x min(w, h)).
        Preserves circular geometry of the shooting target without stretching or distortion.
        If target/bullseye center is detected, centers the crop on the target; otherwise uses center crop.
        Returns:
            cropped_img: 1:1 square numpy array
            was_cropped: bool indicating whether cropping was performed
            crop_box: dict with {'x': start_x, 'y': start_y, 'width': side, 'height': side, 'orig_w': w, 'orig_h': h}
        """
        h, w = img_bgr.shape[:2]
        if w == h:
            return img_bgr, False, {
                "x": 0, "y": 0, "width": w, "height": h,
                "original_width": w, "original_height": h
            }

        side = min(w, h)
        target_center_x, target_center_y = self._find_target_center_for_crop(img_bgr)

        if w > h:
            # Landscape orientation: full height used, crop horizontally
            start_y = 0
            max_start_x = w - side
            if target_center_x is not None:
                ideal_start_x = int(target_center_x - side / 2.0)
                start_x = max(0, min(ideal_start_x, max_start_x))
            else:
                start_x = (w - side) // 2
        else:
            # Portrait orientation: full width used, crop vertically
            start_x = 0
            max_start_y = h - side
            if target_center_y is not None:
                ideal_start_y = int(target_center_y - side / 2.0)
                start_y = max(0, min(ideal_start_y, max_start_y))
            else:
                start_y = (h - side) // 2

        cropped = img_bgr[start_y:start_y + side, start_x:start_x + side]
        crop_box = {
            "x": int(start_x),
            "y": int(start_y),
            "width": int(side),
            "height": int(side),
            "original_width": int(w),
            "original_height": int(h)
        }
        return cropped, True, crop_box

    def correct_perspective(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Detects quadrilateral target paper border and applies perspective warp
        to normalize coordinates into canonical square space (e.g. 800x800).
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
                max_reachable = float(min(center_x, w - center_x, center_y, h - center_y))

                # 1. Estimate based on standard ISSF/NRA bullseye ratio (rings 7-10 is 4/10 = 0.4 of target radius, so outer_r = bullseye_r * 2.5)
                cand_bullseye = min(max_reachable * 0.98, detected_r * 2.5) if detected_r < max_reachable * 0.55 else detected_r

                # 2. Refine outer_radius by profiling radial contrast gradients outward
                grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                mag = cv2.magnitude(grad_x, grad_y)
                angles = np.linspace(0, 2 * math.pi, 90, endpoint=False)
                cos_a = np.cos(angles)
                sin_a = np.sin(angles)

                search_start = int(max(detected_r * 0.9, max_reachable * 0.35))
                search_end = int(max_reachable * 0.98)
                best_r = detected_r
                best_score = 0.0

                for r in range(search_start, search_end, 2):
                    xs = (center_x + r * cos_a).astype(int)
                    ys = (center_y + r * sin_a).astype(int)
                    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
                    if np.sum(valid) > 40:
                        energy = float(np.mean(mag[ys[valid], xs[valid]]))
                        score = energy * (1.0 + (r / float(w)))
                        if score > best_score:
                            best_score = score
                            best_r = float(r)

                outer_radius = max(best_r, cand_bullseye)

        # Calculate standard 10 scoring ring radii (10 = bullseye, 1 = outermost)
        ring_radii: Dict[int, float] = {}
        for k in range(1, 11):
            ring_radii[k] = outer_radius * ((11.0 - k) / 10.0)

        ring_radii[10] = max(ring_radii[10], outer_radius * 0.08)
        return center_x, center_y, ring_radii, outer_radius

    # =========================================================================
    # ALGORITHM 1: Deep Learning (YOLO26 & YOLO11) Detector
    # =========================================================================
    def detect_bullet_holes_ai(
        self,
        img_bgr: np.ndarray,
        center_x: float,
        center_y: float,
        max_radius: float,
        conf_threshold: Optional[float] = None,
        model_name: Optional[str] = None
    ) -> List[Tuple[float, float, float, float, str]]:
        """Deep Learning (YOLO26/YOLO11) inference for detecting bullet holes."""
        target_model = None
        if model_name and model_name.lower() in self.models_cache:
            target_model = self.models_cache[model_name.lower()]
        elif self.ai_model is not None:
            target_model = self.ai_model

        if target_model is None:
            return []

        conf = conf_threshold if conf_threshold is not None else settings.AI_CONFIDENCE_THRESHOLD
        iou = settings.AI_IOU_THRESHOLD

        try:
            results = target_model.predict(
                img_bgr,
                conf=conf,
                iou=iou,
                verbose=False
            )
            candidates = []
            for box in results[0].boxes:
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bw = x2 - x1
                bh = y2 - y1
                r = max(bw, bh) / 2.0
                c = float(box.conf[0])

                # Discard candidates outside target perimeter
                dist = math.hypot(cx - center_x, cy - center_y)
                if dist > (max_radius * 1.05):
                    continue

                # Suppress false detection on the printed central crosshair/inner-10 circle when confidence is low
                if dist < 5.0 and c < 0.35:
                    continue

                # Check if candidate is an elongated double-tap (two bullets touching/keyhole)
                aspect_ratio = max(bw, bh) / max(1.0, min(bw, bh))
                tag = f"{model_name}_ai" if model_name else "yolo_ai"
                if aspect_ratio >= 1.35 and c >= 0.30:
                    if bw > bh:
                        # Horizontal double-tap
                        offset = bw * 0.22
                        candidates.append((cx - offset, cy, bh / 2.0, c, f"{tag}_double_tap"))
                        candidates.append((cx + offset, cy, bh / 2.0, c, f"{tag}_double_tap"))
                    else:
                        # Vertical double-tap
                        offset = bh * 0.22
                        candidates.append((cx, cy - offset, bw / 2.0, c, f"{tag}_double_tap"))
                        candidates.append((cx, cy + offset, bw / 2.0, c, f"{tag}_double_tap"))
                else:
                    candidates.append((cx, cy, r, c, tag))

            return candidates
        except Exception as e:
            print(f"[VisionService] AI inference warning: {e}. Falling back to classical CV.")
            return []

    # =========================================================================
    # ALGORITHM 2: Multi-Scale Difference of Gaussians (DoG) Blob Detector
    # =========================================================================
    def detect_bullet_holes_dog(
        self,
        gray: np.ndarray,
        center_x: float,
        center_y: float,
        max_radius: float
    ) -> List[Tuple[float, float, float, float, str]]:
        """
        Difference of Gaussians filter tuned for small circular bullet hole punctures.
        Immune to printed text numbers and linear target boundary rings.
        """
        candidates = []
        # Bullet hole diameter scales in pixels (e.g. 5px, 8px, 12px)
        scales = [(3, 7), (5, 11), (7, 15)]

        for (s1, s2) in scales:
            g1 = cv2.GaussianBlur(gray, (s1, s1), 0)
            g2 = cv2.GaussianBlur(gray, (s2, s2), 0)
            dog = cv2.subtract(g2, g1)  # Highlights dark depressions on bright background

            _, thresh = cv2.threshold(dog, 18, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 16.0 or area > 1000.0:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circ = 4 * math.pi * (area / (perimeter * perimeter))
                if circ < 0.45:
                    continue

                (cx, cy), r = cv2.minEnclosingCircle(cnt)
                if math.hypot(cx - center_x, cy - center_y) > (max_radius * 1.10):
                    continue

                candidates.append((float(cx), float(cy), float(r), min(0.92, 0.5 + 0.4 * circ), "dog_blob"))

        return candidates

    # =========================================================================
    # ALGORITHM 3: MSER (Maximally Stable Extremal Regions) Detector
    # =========================================================================
    def detect_bullet_holes_mser(
        self,
        gray: np.ndarray,
        center_x: float,
        center_y: float,
        ring_radii: Dict[int, float],
        max_radius: float
    ) -> List[Tuple[float, float, float, float, str]]:
        """
        MSER identifies puncture regions that remain stable over a wide range
        of intensity thresholds. Detects both dark and bright tears in the bullseye.
        """
        candidates = []
        mser = cv2.MSER_create(
            delta=5,
            min_area=15,
            max_area=1200,
            max_variation=0.35
        )

        # Detect on normal gray (dark holes on light paper) and inverted gray (light tears in bullseye)
        for is_inverted, img in [(False, gray), (True, 255 - gray)]:
            regions, _ = mser.detectRegions(img)
            for reg in regions:
                hull = cv2.convexHull(reg)
                area = cv2.contourArea(hull)
                if area < 15.0 or area > 1200.0:
                    continue

                perimeter = cv2.arcLength(hull, True)
                if perimeter == 0:
                    continue
                circ = 4 * math.pi * (area / (perimeter * perimeter))
                if circ < 0.38:
                    continue

                # Solidity check: rejects text numbers like '1', '2', '8'
                solidity = float(cv2.contourArea(reg)) / area if area > 0 else 0.0
                if solidity < 0.70:
                    continue

                (cx, cy), r = cv2.minEnclosingCircle(hull)
                dist = math.hypot(cx - center_x, cy - center_y)
                if dist > (max_radius * 1.10):
                    continue
                if dist < 3.0 and area < 25.0:
                    continue  # center crosshair suppression

                candidates.append((float(cx), float(cy), float(r), min(0.90, 0.45 + 0.45 * circ), "mser_puncture"))

        return candidates

    # =========================================================================
    # ALGORITHM 4: Distance Transform & Watershed Double-Tap Decomposition
    # =========================================================================
    def split_overlapping_double_taps(
        self,
        binary_mask: np.ndarray,
        expected_bullet_radius: float = 6.0
    ) -> List[Tuple[float, float, float]]:
        """
        Decomposes overlapping double-tap / keyhole bullet holes into separate
        impact centers using Distance Transform and local peak detection.
        """
        split_shots = []
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        single_hole_area = math.pi * (expected_bullet_radius ** 2)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 15.0 or area > 2500.0:
                continue

            # If contour is elongated or larger than 1.65x single bullet area, check for double-tap
            if area > (single_hole_area * 1.65):
                c_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
                cv2.drawContours(c_mask, [cnt], -1, 255, -1)

                dist_map = cv2.distanceTransform(c_mask, cv2.DIST_L2, 5)
                max_val = np.max(dist_map)
                if max_val > 0:
                    peaks_mask = (dist_map > (max_val * 0.60)).astype(np.uint8) * 255
                    p_cnts, _ = cv2.findContours(peaks_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    if len(p_cnts) >= 2:
                        for p in p_cnts[:2]:
                            M = cv2.moments(p)
                            if M["m00"] > 0:
                                px = float(M["m10"] / M["m00"])
                                py = float(M["m01"] / M["m00"])
                                split_shots.append((px, py, expected_bullet_radius))
                        continue

            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            split_shots.append((float(cx), float(cy), float(r)))

        return split_shots

    # =========================================================================
    # ALGORITHM 5: Classical CV Pipeline (Combined MSER + DoG + Morphology)
    # =========================================================================
    def detect_bullet_holes_cv(
        self,
        img_bgr: np.ndarray,
        center_x: float,
        center_y: float,
        ring_radii: Dict[int, float],
        max_radius: float
    ) -> List[Tuple[float, float, float, float, str]]:
        """
        Combines MSER, Difference of Gaussians (DoG), Sobel Gradient, and Adaptive Thresholding
        with strict text/number artifact rejection.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        all_candidates: List[Tuple[float, float, float, float, str]] = []

        # 1. DoG Blobs
        dog_candidates = self.detect_bullet_holes_dog(gray, center_x, center_y, max_radius)
        all_candidates.extend(dog_candidates)

        # 2. MSER Extremal Regions
        mser_candidates = self.detect_bullet_holes_mser(gray, center_x, center_y, ring_radii, max_radius)
        all_candidates.extend(mser_candidates)

        # 3. Adaptive Thresholding + Sobel Gradient Magnitude (for contrast & paper edge tears)
        adaptive_dark = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 5
        )
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_binary = np.uint8(grad_mag > 48) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.bitwise_or(adaptive_dark, grad_binary)

        # Bullseye light punctures
        inv_gray = 255 - gray
        adaptive_light = cv2.adaptiveThreshold(
            inv_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 5
        )
        bullseye_r = ring_radii.get(7, max_radius * 0.4)
        mask_bullseye = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_bullseye, (int(center_x), int(center_y)), int(bullseye_r), 255, -1)
        bullseye_punctures = cv2.bitwise_and(adaptive_light, mask_bullseye)
        combined = cv2.bitwise_or(combined, bullseye_punctures)

        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        # Multi-scale contour analysis using hierarchical tree
        contours, _ = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 15.0 or area > 1200.0:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * math.pi * (area / (perimeter * perimeter))
            if circularity < 0.35:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            dist_to_center = math.hypot(cx - center_x, cy - center_y)
            if dist_to_center > (max_radius * 1.12):
                continue
            if dist_to_center < 3.0 and area < 25.0:
                continue

            conf = min(0.95, float(0.5 + 0.45 * circularity))
            all_candidates.append((float(cx), float(cy), float(radius), conf, "cv_contour"))

        # Decompose potential double taps via Watershed
        split_results = self.split_overlapping_double_taps(cleaned, expected_bullet_radius=6.0)
        for (sx, sy, sr) in split_results:
            dist = math.hypot(sx - center_x, sy - center_y)
            if dist <= (max_radius * 1.10) and dist > 4.0:
                all_candidates.append((sx, sy, sr, 0.85, "watershed_split"))

        # Merge overlapping detections with spatial NMS
        merged = self._merge_overlapping_holes(all_candidates, merge_threshold=14.0)
        return merged

    def _merge_overlapping_holes(
        self,
        candidates: List[Tuple[float, float, float, float, str]],
        merge_threshold: float = 14.0
    ) -> List[Tuple[float, float, float, float, str]]:
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

            avg_x = float(np.mean([p[0] for p in group]))
            avg_y = float(np.mean([p[1] for p in group]))
            avg_r = float(np.mean([p[2] for p in group]))
            max_c = float(np.max([p[3] for p in group]))
            # Determine consensus source
            sources = [p[4] for p in group]
            if "yolo_ai" in sources and any(s != "yolo_ai" for s in sources):
                consensus_source = "ensemble"
                max_c = min(0.98, max_c + 0.10)
            elif "yolo_ai" in sources:
                consensus_source = "yolo_ai"
            else:
                consensus_source = sources[0]

            merged.append((avg_x, avg_y, avg_r, max_c, consensus_source))

        return merged

    # =========================================================================
    # ALGORITHM 6: Consensus Ensemble Fusion (YOLO26/YOLO11 + Classical CV)
    # =========================================================================
    def detect_bullet_holes_hybrid(
        self,
        img_bgr: np.ndarray,
        center_x: float,
        center_y: float,
        ring_radii: Dict[int, float],
        max_radius: float,
        model_name: Optional[str] = None
    ) -> Tuple[List[Tuple[float, float, float, float, str]], str]:
        """
        Consensus Ensemble Engine:
        When AI model is active, AI serves as the primary detector (trained on real and synthetic targets).
        Classical CV is used as fallback when AI is unavailable or when specifically configured.
        """
        target_model = self.models_cache.get(model_name.lower()) if (model_name and model_name.lower() in self.models_cache) else self.ai_model
        model_tag = f"{model_name}_ai" if model_name else "yolo_ai"

        if target_model is not None and settings.DETECTION_ENGINE != "cv":
            ai_holes = self.detect_bullet_holes_ai(img_bgr, center_x, center_y, max_radius, model_name=model_name)
            if settings.DETECTION_ENGINE == "ai":
                return ai_holes, model_tag

            # In hybrid mode: if AI found shots, use AI detections merged for double-taps
            # If AI found 0 shots (e.g. edge case or uncharacteristic target), fall back to CV
            if len(ai_holes) > 0:
                merged_ai = self._merge_overlapping_holes(ai_holes, merge_threshold=10.0)
                return merged_ai, model_tag

        cv_holes = self.detect_bullet_holes_cv(img_bgr, center_x, center_y, ring_radii, max_radius)
        return cv_holes, "classical_cv_ensemble"

    def annotate_target(
        self,
        img_bgr: np.ndarray,
        center_x: float,
        center_y: float,
        ring_radii: Dict[int, float],
        shots: List[ShotDetail],
        analytics: TargetAnalytics,
        engine_name: str = "hybrid_ensemble"
    ) -> np.ndarray:
        """Draws visual annotations including shot markers, dispersion, and HUD banner."""
        overlay = img_bgr.copy()
        cx_int, cy_int = int(round(center_x)), int(round(center_y))

        # 1. Subtle ring guidelines
        for ring_val in range(1, 11):
            r_px = int(round(ring_radii[ring_val]))
            cv2.circle(overlay, (cx_int, cy_int), r_px, (100, 100, 100), 1, cv2.LINE_AA)
            if ring_val in [3, 5, 7, 9]:
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
            10: (0, 215, 255),   # Gold
            9:  (50, 205, 50),   # Lime Green
            8:  (0, 200, 0),     # Green
            7:  (0, 165, 255),   # Orange
            6:  (0, 100, 255),   # Orange-Red
        }

        for shot in shots:
            sx, sy = int(round(shot.x)), int(round(shot.y))
            color = score_colors.get(shot.score, (0, 0, 230))

            # Inner & outer rings
            cv2.circle(overlay, (sx, sy), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, (sx, sy), 8, color, 2, cv2.LINE_AA)

            # Text badge
            conf_str = f" {int(shot.confidence * 100)}%" if shot.confidence else ""
            badge_text = f"#{shot.id}:{shot.score}{conf_str}"
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.42
            thickness = 1
            text_size, _ = cv2.getTextSize(badge_text, font_face, font_scale, thickness)

            badge_x = min(sx + 10, overlay.shape[1] - text_size[0] - 8)
            badge_y = max(sy - 10, text_size[1] + 8)

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
            f"ENGINE: {engine_name.upper()}"
        )
        cv2.putText(
            overlay,
            hud_text,
            (14, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        return overlay

    def get_available_engines(self) -> List[Dict[str, Any]]:
        """Returns list of detection engines supported by this service."""
        engines = [
            {
                "id": "model_sendiri",
                "name": "Model Sendiri (Local YOLO & Classical CV)",
                "type": "local_vision",
                "model": self.ai_model_name or "Classical CV Ensemble",
                "is_available": True,
                "description": "Deteksi lokal otomatis menggunakan model YOLO terlatih dan algoritma Computer Vision."
            }
        ]

        # YOLO26 option
        yolo26_available = "yolo26" in self.models_cache or settings.AI_YOLO26_PATH.exists()
        engines.append({
            "id": "yolo26",
            "name": "YOLO26 (Small-Target Puncture Detector)",
            "type": "local_deep_learning",
            "model": "bullet_hole_yolo26.pt",
            "is_available": yolo26_available,
            "description": "Arsitektur YOLO26 dengan STAL (Small-Target-Aware Label Assignment) khusus target Krüger."
        })

        # YOLO11 option
        yolo11_available = "yolo11" in self.models_cache or settings.AI_YOLO11_PATH.exists()
        engines.append({
            "id": "yolo11",
            "name": "YOLO11 (Ultralytics High-Precision Detector)",
            "type": "local_deep_learning",
            "model": "bullet_hole_yolo11.pt",
            "is_available": yolo11_available,
            "description": "Arsitektur Ultralytics YOLO11 terlatih untuk deteksi sasaran tembak Krüger."
        })

        # Gemini API option
        engines.append({
            "id": "gemini",
            "name": f"Google Gemini API ({settings.GEMINI_MODEL})",
            "type": "cloud_multimodal_ai",
            "model": settings.get_effective_gemini_model(),
            "is_available": self.gemini_service.is_configured(),
            "description": "Model multimodal vision Google Gemini untuk deteksi sasaran tembak berbasis cloud."
        })

        return engines

    def process_image(
        self,
        image_bytes: bytes,
        original_filename: str = "",
        caliber_mm: float = 4.5,
        target_type: str = "issf_standard_10_ring",
        scale_override: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        engine: Optional[str] = None
    ) -> DetectionResponse:
        """
        Executes end-to-end pipeline:
        decode -> validate -> warp -> detect rings -> detect shots -> score -> annotate -> persist.
        Supports choosing between 'model_sendiri', 'yolo26', 'yolo11', or 'gemini'.
        """
        # 1. Decode & Validate
        img_bgr = self.decode_image_bytes(image_bytes)
        self.validate_image_quality(img_bgr)

        # 1b. Auto-Crop to 1:1 Aspect Ratio (Bujur Sangkar) to eliminate elongation/distortion
        img_bgr, was_cropped, crop_box = self.crop_to_square(img_bgr)

        # 2. Perspective warp to canonical square
        canonical_bgr, was_warped = self.correct_perspective(img_bgr)

        # 3. Detect Target & Rings (Baseline Geometric Pipeline)
        center_x, center_y, ring_radii, outer_r = self.detect_target_and_rings(canonical_bgr)
        scale_mm_px = scale_override or self.default_scale

        # 4. Bullet Hole Detection Engine Selection
        chosen_engine = (engine or settings.DETECTION_ENGINE).lower().strip()
        use_gemini = chosen_engine in ["gemini", "api_gemini", "gemini_api", "cloud"]
        model_name = "yolo26" if "26" in chosen_engine else ("yolo11" if "11" in chosen_engine else None)

        detected_holes: List[Tuple[float, float, float, float, str]] = []
        engine_used: str = chosen_engine if chosen_engine in ["yolo26", "yolo11"] else "model_sendiri"

        if use_gemini:
            try:
                g_holes, g_geom, g_engine = self.gemini_service.analyze_target(
                    canonical_bgr, target_type=target_type
                )
                detected_holes = g_holes
                engine_used = g_engine
                if g_geom is not None:
                    center_x, center_y, outer_r = g_geom
                    ring_radii = {r: outer_r * (11 - r) / 10.0 for r in range(1, 11)}
            except Exception as e:
                print(f"[VisionService] Gemini detection failed: {e}. Falling back to model_sendiri.")
                detected_holes, engine_used = self.detect_bullet_holes_hybrid(
                    canonical_bgr, center_x, center_y, ring_radii, outer_r, model_name=model_name
                )
                if metadata is None:
                    metadata = {}
                metadata["gemini_error"] = str(e)
                metadata["fallback_triggered"] = True
        else:
            detected_holes, engine_used = self.detect_bullet_holes_hybrid(
                canonical_bgr, center_x, center_y, ring_radii, outer_r, model_name=model_name
            )
            if engine_used in ["yolo_ai", "classical_cv_ensemble"] and chosen_engine == "model_sendiri":
                engine_used = "model_sendiri"

        # Sort holes spatially (top-to-bottom, left-to-right)
        detected_holes = sorted(detected_holes, key=lambda h: (h[1], h[0]))
        bullet_radius_px = max(4.0, (caliber_mm / 2.0) / scale_mm_px)

        # 5. Evaluate individual shot scores
        shots: List[ShotDetail] = []
        for idx, (hx, hy, hr, hconf, hsource) in enumerate(detected_holes, start=1):
            shot_detail = scoring_engine.evaluate_shot(
                shot_id=idx,
                x=hx,
                y=hy,
                center_x=center_x,
                center_y=center_y,
                ring_radii=ring_radii,
                bullet_radius_px=bullet_radius_px,
                mm_per_pixel=scale_mm_px,
                confidence=hconf,
                detection_source=hsource
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
            analytics=analytics,
            engine_name=engine_used
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
            "cropped_to_square": was_cropped,
            "crop_box": crop_box,
            "target_type": target_type,
            "caliber_mm": caliber_mm,
            "scale_mm_per_pixel": scale_mm_px,
            "center": {"x": round(center_x, 1), "y": round(center_y, 1)},
            "ai_model": self.ai_model_name if self.ai_model else "none",
            "selected_engine": chosen_engine
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
            detection_engine_used=engine_used,
            metadata=meta_info
        )


vision_service = VisionService()
