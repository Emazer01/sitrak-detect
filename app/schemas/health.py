from pydantic import BaseModel, Field
from datetime import datetime, timezone


class HealthCheckResponse(BaseModel):
    status: str = Field(default="healthy", description="Service health state")
    service: str = Field(default="sitrak-detect", description="Service name")
    version: str = Field(default="1.0.0", description="Application version")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp")
    gpu_available: bool = Field(default=False, description="OpenCV/PyTorch GPU acceleration indicator")
