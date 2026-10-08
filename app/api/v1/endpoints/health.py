import cv2
from fastapi import APIRouter, status
from app.schemas.health import HealthCheckResponse
from app.core.config import settings

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
        gpu_available=gpu_available
    )

