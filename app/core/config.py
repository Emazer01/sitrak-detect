import os
from pathlib import Path
from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings configuration via environment variables or .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # General
    APP_NAME: str = "Sitrak Shooting Target Detection Service"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*"
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Storage paths (Relative to project root or absolute)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "static" / "uploads"
    ANNOTATED_DIR: Path = BASE_DIR / "static" / "annotated"
    MAX_UPLOAD_SIZE_MB: int = 15

    # Vision defaults
    TARGET_CANONICAL_SIZE: int = 800  # Normalized square dimension for homography warp
    CALIBRATION_MM_PER_PIXEL: float = 0.35  # Approx mm per pixel at canonical size
    DEFAULT_TARGET_TYPE: str = "issf_standard_10_ring"

    # AI Model Detection settings
    MODELS_DIR: Path = BASE_DIR / "models"
    AI_MODEL_PATH: Path = BASE_DIR / "models" / "bullet_hole_yolo.pt"
    AI_YOLO26_PATH: Path = BASE_DIR / "models" / "bullet_hole_yolo26.pt"
    AI_YOLO11_PATH: Path = BASE_DIR / "models" / "bullet_hole_yolo11.pt"
    AI_CONFIDENCE_THRESHOLD: float = 0.15
    AI_IOU_THRESHOLD: float = 0.58
    DETECTION_ENGINE: str = "hybrid"  # "hybrid", "ai", "cv", "yolo26", "yolo11", "model_sendiri"


settings = Settings()

# Ensure required storage directories exist at startup
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)


