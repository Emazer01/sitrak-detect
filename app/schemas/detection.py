from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ShotDetail(BaseModel):
    """Information for an individual detected shot/impact."""
    id: int = Field(..., description="Chronological or sequential shot index")
    x: float = Field(..., description="X coordinate of bullet hole on canonical/target plane")
    y: float = Field(..., description="Y coordinate of bullet hole on canonical/target plane")
    score: int = Field(..., description="Ring score earned by this shot (0-10)")
    distance_from_center: float = Field(..., description="Euclidean distance from target center (mm or normalized px)")
    decimal_score: Optional[float] = Field(None, description="Tenth-ring decimal score (e.g., 10.4)")
    confidence: Optional[float] = Field(None, description="Detection confidence score (0.0 to 1.0)")
    detection_source: Optional[str] = Field("cv", description="Detection method: 'yolo_ai', 'cv', or 'ensemble'")


class TargetAnalytics(BaseModel):
    """Shooting performance metrics and dispersion analysis."""
    grouping_diameter_mm: float = Field(..., description="Minimum enclosing circle / extreme spread diameter in mm")
    directional_bias: str = Field(..., description="Primary cluster offset direction (e.g. top-left, bottom, centered)")
    mean_point_of_impact_x: Optional[float] = Field(None, description="Average X offset from bullseye")
    mean_point_of_impact_y: Optional[float] = Field(None, description="Average Y offset from bullseye")
    dispersion_radius_px: Optional[float] = Field(None, description="Dispersion radius in pixels")


class DetectionResponse(BaseModel):
    """Full detection response payload."""
    status: str = Field(default="success", description="Status code or message")
    target_detected: bool = Field(..., description="Whether the target bullseye and rings were found")
    total_score: int = Field(..., description="Sum of all detected shot scores")
    max_possible_score: int = Field(..., description="Maximum possible score (shots_count * 10)")
    accuracy_percentage: float = Field(..., description="Percentage ratio of total_score to max_possible_score")
    shots_count: int = Field(..., description="Total count of detected bullet impacts")
    shots: List[ShotDetail] = Field(default_factory=list, description="List of detected shots")
    analytics: TargetAnalytics = Field(..., description="Dispersion and directional bias analysis")
    annotated_image_url: Optional[str] = Field(None, description="Relative or absolute URL to annotated image result")
    detection_engine_used: Optional[str] = Field(default="cv", description="Engine used: 'yolo_ai', 'cv', or 'hybrid'")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata echoed back")


class FrameDetectionRequest(BaseModel):
    """Payload for incremental video/webcam frame detection."""
    image_base64: str = Field(..., description="Base64-encoded image string (with or without data URI header)")
    target_type: Optional[str] = Field(default="issf_standard_10_ring", description="Target model standard")
    caliber_mm: Optional[float] = Field(default=4.5, description="Bullet caliber in mm (e.g. 4.5mm airgun, 9mm pistol)")
    calibration_mm_per_pixel: Optional[float] = Field(default=None, description="Custom scale override (mm per pixel)")
    known_shots_count: Optional[int] = Field(default=0, description="Known shots count from previous frames to detect deltas")
    engine: Optional[str] = Field(default="model_sendiri", description="Detection engine: 'model_sendiri', 'yolo26', or 'yolo11'")

