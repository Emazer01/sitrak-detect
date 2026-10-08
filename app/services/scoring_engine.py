import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from app.schemas.detection import ShotDetail, TargetAnalytics


class ScoringEngine:
    """
    Engine to compute shot scores, Euclidean distances, grouping dispersion,
    and shooter directional bias (MPI / compass / clock position).
    """

    def __init__(self, default_mm_per_pixel: float = 0.35):
        self.default_mm_per_pixel = default_mm_per_pixel

    def evaluate_shot(
        self,
        shot_id: int,
        x: float,
        y: float,
        center_x: float,
        center_y: float,
        ring_radii: Dict[int, float],
        bullet_radius_px: float = 6.0,
        mm_per_pixel: Optional[float] = None
    ) -> ShotDetail:
        """
        Calculates distance from bullseye and ring score (1-10) for a given shot.
        Supports standard shooting rule where touching a higher ring line grants the higher score.
        """
        scale = mm_per_pixel or self.default_mm_per_pixel

        # Euclidean distance in pixels
        dx = x - center_x
        dy = y - center_y
        dist_px = math.hypot(dx, dy)
        dist_mm = dist_px * scale

        # Determine score: ring_radii maps score (10 to 1) -> outer radius in px
        # If the bullet touches the ring line, inward-gauging grants the higher point
        score = 0
        effective_dist = max(0.0, dist_px - (bullet_radius_px * 0.75))

        # Check rings from 10 down to 1
        sorted_rings = sorted(ring_radii.items(), key=lambda item: item[0], reverse=True)
        for ring_val, radius in sorted_rings:
            if effective_dist <= radius:
                score = ring_val
                break

        # Decimal score (ISSF 10th ring decimal score approximation)
        r10 = ring_radii.get(10, 20.0)
        r9 = ring_radii.get(9, 40.0)
        ring_step = max(1.0, r9 - r10)
        if dist_px <= r10:
            dec_score = round(10.9 - (dist_px / r10) * 0.9, 1)
        else:
            dec_score = max(0.0, round(10.0 - (dist_px - r10) / ring_step, 1))

        return ShotDetail(
            id=shot_id,
            x=round(x, 1),
            y=round(y, 1),
            score=score,
            distance_from_center=round(dist_mm, 2),
            decimal_score=dec_score
        )

    def compute_analytics(
        self,
        shots: List[ShotDetail],
        center_x: float,
        center_y: float,
        mm_per_pixel: Optional[float] = None
    ) -> TargetAnalytics:
        """
        Calculates dispersion (grouping diameter) and directional bias based on
        shot distribution and Mean Point of Impact (MPI).
        """
        scale = mm_per_pixel or self.default_mm_per_pixel

        if not shots:
            return TargetAnalytics(
                grouping_diameter_mm=0.0,
                directional_bias="centered",
                mean_point_of_impact_x=0.0,
                mean_point_of_impact_y=0.0,
                dispersion_radius_px=0.0
            )

        coords = np.array([[s.x, s.y] for s in shots], dtype=np.float32)

        # 1. Grouping diameter (Extreme spread / Minimum Enclosing Circle)
        if len(shots) == 1:
            grouping_diam_px = 0.0
        elif len(shots) == 2:
            grouping_diam_px = float(np.linalg.norm(coords[0] - coords[1]))
        else:
            # Pairwise max distance (Extreme Spread standard)
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
            grouping_diam_px = float(np.max(dist_matrix))

        grouping_diam_mm = round(grouping_diam_px * scale, 1)

        # 2. Mean Point of Impact (MPI)
        mean_x = float(np.mean(coords[:, 0]))
        mean_y = float(np.mean(coords[:, 1]))
        offset_x = mean_x - center_x
        offset_y = mean_y - center_y  # In image coordinates, positive Y is down

        # 3. Directional Bias Calculation
        mpi_dist_px = math.hypot(offset_x, offset_y)
        bias_str = self._resolve_directional_bias(offset_x, offset_y, mpi_dist_px, scale)

        return TargetAnalytics(
            grouping_diameter_mm=grouping_diam_mm,
            directional_bias=bias_str,
            mean_point_of_impact_x=round(offset_x * scale, 2),
            mean_point_of_impact_y=round(offset_y * scale, 2),
            dispersion_radius_px=round(grouping_diam_px / 2.0, 1)
        )

    def _resolve_directional_bias(
        self,
        offset_x: float,
        offset_y: float,
        dist_px: float,
        scale: float
    ) -> str:
        """
        Maps vector (offset_x, offset_y) to compass & clock direction.
        offset_y is positive DOWN in image coordinate system.
        """
        # Threshold: if average offset is less than 3 mm, consider it centered
        if (dist_px * scale) < 4.0:
            return "centered"

        # Mathematical angle in degrees from center (0 = right, 90 = top, 180 = left, 270 = bottom)
        # Note: -offset_y because image Y is downwards
        angle_rad = math.atan2(-offset_y, offset_x)
        angle_deg = math.degrees(angle_rad) % 360

        # Compass & clock direction mapping
        if 22.5 <= angle_deg < 67.5:
            return "top-right"
        elif 67.5 <= angle_deg < 112.5:
            return "top"
        elif 112.5 <= angle_deg < 157.5:
            return "top-left"
        elif 157.5 <= angle_deg < 202.5:
            return "left"
        elif 202.5 <= angle_deg < 247.5:
            return "bottom-left"
        elif 247.5 <= angle_deg < 292.5:
            return "bottom"
        elif 292.5 <= angle_deg < 337.5:
            return "bottom-right"
        else:
            return "right"


scoring_engine = ScoringEngine()

