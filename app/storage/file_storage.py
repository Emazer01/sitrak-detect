import uuid
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np

from app.core.config import settings
from app.core.exceptions import InvalidImageFormatError


class FileStorageService:
    """Handles persistence of raw input photos and annotated computer vision outputs."""

    def __init__(self):
        self.upload_dir = settings.UPLOAD_DIR
        self.annotated_dir = settings.ANNOTATED_DIR
        self.allowed_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def save_raw_upload(self, content: bytes, original_filename: str = "") -> Tuple[Path, str]:
        """Saves raw uploaded image bytes to disk and returns (file_path, public_url)."""
        suffix = Path(original_filename).suffix.lower() if original_filename else ".jpg"
        if suffix not in self.allowed_extensions:
            suffix = ".jpg"

        file_id = f"raw_{uuid.uuid4().hex[:12]}{suffix}"
        target_path = self.upload_dir / file_id

        try:
            with open(target_path, "wb") as f:
                f.write(content)
        except Exception as e:
            raise InvalidImageFormatError(f"Failed to persist uploaded file: {str(e)}")

        public_url = f"/static/uploads/{file_id}"
        return target_path, public_url

    def save_annotated_image(self, image_bgr: np.ndarray, prefix: str = "shot_result") -> Tuple[Path, str]:
        """Encodes and saves an OpenCV BGR image to static/annotated and returns (file_path, public_url)."""
        file_id = f"{prefix}_{uuid.uuid4().hex[:12]}.jpg"
        target_path = self.annotated_dir / file_id

        success = cv2.imwrite(str(target_path), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not success:
            raise InvalidImageFormatError("Failed to encode and save annotated image output.")

        public_url = f"/static/annotated/{file_id}"
        return target_path, public_url


storage_service = FileStorageService()

