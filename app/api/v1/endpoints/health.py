import cv2
from fastapi import APIRouter, status
from app.schemas.health import HealthCheckResponse
from app.core.config import settings

from app.services.vision_service import vision_service

router = APIRouter()


@router.get(
    "/healthcheck",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Health Check",
    description="Returns current operational status, version, and hardware acceleration status."
)
async def get_healthcheck() -> HealthCheckResponse:
    gpu_available = False
    try:
        gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
        gpu_available = gpu_count > 0
    except Exception:
        gpu_available = False

    return HealthCheckResponse(
        status="healthy",
        service=settings.APP_NAME,
        version="1.0.0",
        gpu_available=gpu_available,
        ai_model_loaded=vision_service.ai_model is not None,
        ai_model_path=vision_service.ai_model_name or "none"
    )

