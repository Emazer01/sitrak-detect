"""
gemini_service.py
Multimodal Computer Vision Service using Google Gemini API
Detects target bullseye, scoring rings, and bullet hole punctures.
"""

import os
import cv2
import json
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger("sitrak.gemini_service")


class GeminiHoleDetection(BaseModel):
    x: float = Field(..., description="Normalized X coordinate (0 to 1000, left to right)")
    y: float = Field(..., description="Normalized Y coordinate (0 to 1000, top to bottom)")
    description: Optional[str] = Field(None, description="Brief description of the impact location")
    confidence: Optional[float] = Field(0.92, description="Confidence score")


class GeminiTargetAnalysis(BaseModel):
    target_detected: bool = Field(True, description="Whether target rings or bullseye was found")
    bullseye_x: float = Field(500.0, description="Normalized bullseye center X (0 to 1000)")
    bullseye_y: float = Field(500.0, description="Normalized bullseye center Y (0 to 1000)")
    outer_ring_radius: float = Field(450.0, description="Normalized outer scoring ring radius (0 to 1000)")
    bullet_holes: List[GeminiHoleDetection] = Field(default_factory=list, description="List of detected bullet holes")


class GeminiVisionService:
    """
    Multimodal Gemini Vision Service for shooting target detection and scoring.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not configured in .env or settings. "
                    "Please provide a valid Gemini API key to use the 'gemini' detection engine."
                )
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def is_configured(self) -> bool:
        """Returns True if Gemini API key is configured."""
        return bool(self.api_key and len(self.api_key.strip()) > 10)

    def analyze_target(
        self,
        image_bgr: np.ndarray,
        target_type: str = "issf_standard_10_ring"
    ) -> Tuple[List[Tuple[float, float, float, float, str]], Optional[Tuple[float, float, float]], str]:
        """
        Calls Gemini Multimodal API to detect target bullseye and bullet holes.

        Returns:
            - holes: List of (x, y, radius, confidence, source) in pixel coordinates
            - target_geom: Optional tuple of (center_x, center_y, outer_radius) in pixel coordinates
            - model_used: Name of the Gemini model used
        """
        client = self._get_client()
        from google.genai import types

        h, w = image_bgr.shape[:2]

        # Encode image to high-quality JPEG bytes
        success, encoded_jpg = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not success:
            raise ValueError("Failed to encode target image to JPEG format for Gemini API")
        jpg_bytes = encoded_jpg.tobytes()

        prompt = (
            "You are an expert shooting target analysis Computer Vision AI. "
            f"Analyze this shooting target image carefully (target type hint: {target_type}). "
            "The image coordinate space is normalized from 0 to 1000 for both X (horizontal, left=0 to right=1000) "
            "and Y (vertical, top=0 to bottom=1000).\n\n"
            "Identify:\n"
            "1. The exact center (bullseye) of the target scoring rings (bullseye_x, bullseye_y on 0-1000 scale).\n"
            "2. The radius of the outermost scoring ring (outer_ring_radius on 0-1000 scale).\n"
            "3. Every distinct bullet impact hole, puncture, tear, or cluster on the target paper.\n"
            "Do not miss subtle punctures, edge tears, or overlapping double taps. "
            "Ignore printed text, mounting staples, or paper target clips."
        )

        model_candidates = [
            settings.get_effective_gemini_model(),
            "gemini-3.8-flash",
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash-lite"
        ]
        # Remove duplicates while preserving order
        unique_models = list(dict.fromkeys([m for m in model_candidates if m]))

        last_error = None
        analysis_data: Optional[GeminiTargetAnalysis] = None
        model_used = unique_models[0]

        import time
        for model_name in unique_models:
            for attempt in range(2):
                try:
                    logger.info(f"[GeminiVisionService] Querying Gemini model: {model_name} (attempt {attempt+1})...")
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[
                            types.Part.from_bytes(data=jpg_bytes, mime_type="image/jpeg"),
                            prompt
                        ],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=GeminiTargetAnalysis,
                            temperature=0.1
                        )
                    )

                    if response and response.text:
                        raw_text = response.text.strip()
                        analysis_data = GeminiTargetAnalysis.model_validate_json(raw_text)
                        model_used = model_name
                        logger.info(f"[GeminiVisionService] Success with {model_name}: found {len(analysis_data.bullet_holes)} holes.")
                        break
                except Exception as e:
                    err_msg = str(e)
                    logger.warning(f"[GeminiVisionService] Model {model_name} attempt {attempt+1} error: {err_msg}")
                    last_error = e
                    # If deprecated or not found (404), do not retry the same model
                    if "404" in err_msg or "NOT_FOUND" in err_msg:
                        break
                    # Transient error (503 / 429), back off slightly
                    if attempt == 0:
                        time.sleep(1.0)
            if analysis_data is not None:
                break

        if analysis_data is None:
            raise RuntimeError(f"Gemini API detection failed on all model candidates: {last_error}")

        # Map normalized (0..1000) coordinates to image pixel space (w, h)
        scale_x = w / 1000.0
        scale_y = h / 1000.0
        avg_scale = (scale_x + scale_y) / 2.0

        holes: List[Tuple[float, float, float, float, str]] = []
        default_hole_radius_px = max(4.0, min(w, h) * 0.015)

        for hole in analysis_data.bullet_holes:
            px = float(hole.x * scale_x)
            py = float(hole.y * scale_y)
            conf = float(hole.confidence or 0.90)
            holes.append((px, py, default_hole_radius_px, conf, f"gemini_{model_used}"))

        target_geom: Optional[Tuple[float, float, float]] = None
        if analysis_data.target_detected:
            bx = float(analysis_data.bullseye_x * scale_x)
            by = float(analysis_data.bullseye_y * scale_y)
            br = float(analysis_data.outer_ring_radius * avg_scale)
            target_geom = (bx, by, br)

        return holes, target_geom, f"gemini_{model_used}"

