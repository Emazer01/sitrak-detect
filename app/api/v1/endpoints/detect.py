from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
import base64

from app.core.config import settings
from app.core.exceptions import InvalidImageFormatError
from app.schemas.detection import (
    DetectionResponse,
    FrameDetectionRequest
)
from app.services.vision_service import vision_service
from app.storage.file_storage import storage_service

router = APIRouter()


@router.post(
    "/upload",
    response_model=DetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect shots from uploaded target photo",
    description="Accepts multipart/form-data image file, performs computer vision analysis, scoring, and returns visual overlay."
)
async def detect_shots_upload(
    file: UploadFile = File(..., description="Target image file (JPEG, PNG, etc.)"),
    target_type: Optional[str] = Form("issf_standard_10_ring", description="Target discipline model"),
    caliber_mm: Optional[float] = Form(4.5, description="Bullet caliber in mm (e.g. 4.5, 5.56, 9.0)"),
    calibration_mm_per_pixel: Optional[float] = Form(None, description="Custom scale override (mm/pixel)"),
    shooter_name: Optional[str] = Form(None, description="Shooter identifier or name"),
    session_id: Optional[str] = Form(None, description="Shooting session tracking ID")
) -> DetectionResponse:
    # 1. Validate file content and size
    file_bytes = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded image exceeds maximum allowable size of {settings.MAX_UPLOAD_SIZE_MB}MB."
        )

    # 2. Persist raw input
    raw_path, raw_url = storage_service.save_raw_upload(file_bytes, original_filename=file.filename or "")

    # 3. Assemble metadata
    metadata = {
        "original_filename": file.filename,
        "raw_image_url": raw_url,
        "shooter_name": shooter_name,
        "session_id": session_id
    }

    # 4. Execute Vision Pipeline
    result = vision_service.process_image(
        image_bytes=file_bytes,
        original_filename=file.filename or "",
        caliber_mm=caliber_mm or 4.5,
        target_type=target_type or "issf_standard_10_ring",
        scale_override=calibration_mm_per_pixel,
        metadata=metadata
    )

    return result


@router.post(
    "/frame",
    response_model=DetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Process video/webcam frame incrementally",
    description="Accepts base64 frame stream from front-end camera, computes shot detection and delta metrics."
)
async def detect_shots_frame(
    payload: FrameDetectionRequest
) -> DetectionResponse:
    # 1. Decode base64 to binary bytes
    b64_str = payload.image_base64
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(b64_str)
    except Exception as e:
        raise InvalidImageFormatError(f"Failed to decode base64 frame: {str(e)}")

    # 2. Execute Vision Pipeline
    metadata = {
        "frame_mode": "incremental_stream",
        "known_shots_count": payload.known_shots_count
    }

    result = vision_service.process_image(
        image_bytes=image_bytes,
        caliber_mm=payload.caliber_mm or 4.5,
        target_type=payload.target_type or "issf_standard_10_ring",
        scale_override=payload.calibration_mm_per_pixel,
        metadata=metadata
    )

    # Add delta shots calculation in metadata
    if payload.known_shots_count is not None:
        new_shots = max(0, result.shots_count - payload.known_shots_count)
        if result.metadata is not None:
            result.metadata["new_shots_detected"] = new_shots

    return result

