from fastapi import Request, status
from fastapi.responses import JSONResponse


class VisionServiceException(Exception):
    """Base exception for computer vision processing errors."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class TargetNotFoundError(VisionServiceException):
    """Raised when circular target / bullseye could not be detected."""
    pass


class ImageQualityError(VisionServiceException):
    """Raised when image is too dark, overexposed, or blurred to be processed."""
    pass


class InvalidImageFormatError(VisionServiceException):
    """Raised when uploaded file is corrupted or not a valid image."""
    pass


async def vision_exception_handler(request: Request, exc: VisionServiceException) -> JSONResponse:
    """Standardized handler for domain vision errors."""
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    if isinstance(exc, InvalidImageFormatError):
        status_code = status.HTTP_400_BAD_REQUEST

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "error_type": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details,
            "target_detected": False,
        }
    )

