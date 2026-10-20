import io
import base64
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.scoring_engine import scoring_engine
from app.services.vision_service import vision_service
from app.schemas.detection import ShotDetail

client = TestClient(app)


def generate_synthetic_target_image() -> bytes:
    """
    Generates a realistic synthetic shooting target image with concentric rings
    and 3 distinct bullet holes at known coordinates.
    """
    size = 800
    img = np.full((size, size, 3), 245, dtype=np.uint8)  # Off-white paper background
    center = (size // 2, size // 2)
    outer_radius = int(size * 0.42)  # ~336 px

    # Draw rings 1 to 10
    for k in range(1, 11):
        r = int(outer_radius * ((11 - k) / 10.0))
        # Draw black central bullseye for rings 7 to 10
        if k >= 7:
            cv2.circle(img, center, r, (30, 30, 30), -1)
        # Ring border lines
        line_color = (200, 200, 200) if k >= 7 else (50, 50, 50)
        cv2.circle(img, center, r, line_color, 2)

    # Crosshair at center
    cv2.line(img, (center[0] - 8, center[1]), (center[0] + 8, center[1]), (255, 255, 255), 1)
    cv2.line(img, (center[0], center[1] - 8), (center[0], center[1] + 8), (255, 255, 255), 1)

    # Bullet holes (dark high-contrast circular punctures with ragged edge)
    # Hole 1: Near bullseye center (10-ring) -> offset (-10, -15) -> top-left
    # Hole 2: Ring 9 -> offset (-35, -40) -> top-left
    # Hole 3: Ring 8 -> offset (-60, -75) -> top-left
    bullet_offsets = [(-10, -15), (-35, -40), (-60, -75)]
    for ox, oy in bullet_offsets:
        bx = center[0] + ox
        by = center[1] + oy
        # Draw dark puncture hole
        cv2.circle(img, (bx, by), 7, (15, 15, 15), -1)
        # Inner contrast rim
        cv2.circle(img, (bx, by), 8, (70, 70, 70), 1)

    # Add minor Gaussian noise to simulate real paper texture
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Encode to JPEG bytes
    success, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert success
    return encoded.tobytes()


def test_healthcheck():
    response = client.get("/api/v1/healthcheck")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_scoring_engine_math():
    ring_radii = {k: 336.0 * ((11.0 - k) / 10.0) for k in range(1, 11)}
    center_x = 400.0
    center_y = 400.0

    # Test center shot -> should be score 10
    shot_center = scoring_engine.evaluate_shot(
        shot_id=1,
        x=405.0,
        y=405.0,
        center_x=center_x,
        center_y=center_y,
        ring_radii=ring_radii
    )
    assert shot_center.score == 10
    assert shot_center.distance_from_center < 10.0

    # Test outer shot
    shot_outer = scoring_engine.evaluate_shot(
        shot_id=2,
        x=400.0,
        y=400.0 + 280.0,
        center_x=center_x,
        center_y=center_y,
        ring_radii=ring_radii
    )
    assert shot_outer.score <= 3

    # Test analytics with top-left offset shots
    shots = [
        ShotDetail(id=1, x=390.0, y=385.0, score=10, distance_from_center=6.3),
        ShotDetail(id=2, x=365.0, y=360.0, score=9, distance_from_center=18.6),
        ShotDetail(id=3, x=340.0, y=325.0, score=8, distance_from_center=33.1)
    ]
    analytics = scoring_engine.compute_analytics(shots, center_x, center_y)
    assert analytics.grouping_diameter_mm > 0.0
    assert analytics.directional_bias == "top-left"


def test_detect_upload_endpoint():
    image_bytes = generate_synthetic_target_image()

    response = client.post(
        "/api/v1/detect/upload",
        files={"file": ("target_test.jpg", image_bytes, "image/jpeg")},
        data={"target_type": "issf_standard_10_ring", "caliber_mm": "4.5"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["target_detected"] is True
    assert data["shots_count"] >= 1
    assert data["total_score"] > 0
    assert data["max_possible_score"] == data["shots_count"] * 10
    assert data["accuracy_percentage"] > 0.0
    assert "analytics" in data
    assert "grouping_diameter_mm" in data["analytics"]
    assert "directional_bias" in data["analytics"]
    assert data["annotated_image_url"] is not None
    assert data["annotated_image_url"].startswith("/static/annotated/")


def test_detect_frame_endpoint():
    image_bytes = generate_synthetic_target_image()
    b64_image = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("utf-8")

    response = client.post(
        "/api/v1/detect/frame",
        json={
            "image_base64": b64_image,
            "target_type": "issf_standard_10_ring",
            "caliber_mm": 4.5,
            "known_shots_count": 0
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["shots_count"] >= 1
    assert data["metadata"]["new_shots_detected"] == data["shots_count"]


def test_invalid_image_upload():
    # Test uploading corrupted/random binary payload
    bad_bytes = b"This is not a real image header"
    response = client.post(
        "/api/v1/detect/upload",
        files={"file": ("corrupt.jpg", bad_bytes, "image/jpeg")}
    )
    assert response.status_code in [400, 422]
    data = response.json()
    assert data["target_detected"] is False


def test_model_status_endpoint():
    response = client.get("/api/v1/detect/model-status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "detection_engine" in data
    assert "confidence_threshold" in data


def test_model_reload_endpoint():
    response = client.post("/api/v1/detect/model-reload")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["success", "warning"]
    assert "reloaded" in data


def test_detection_engines_endpoint():
    response = client.get("/api/v1/detect/engines")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "engines" in data
    engine_ids = [e["id"] for e in data["engines"]]
    assert "model_sendiri" in engine_ids
    assert "gemini" in engine_ids


def test_detect_upload_with_engine_model_sendiri():
    image_bytes = generate_synthetic_target_image()
    response = client.post(
        "/api/v1/detect/upload",
        files={"file": ("target_test.jpg", image_bytes, "image/jpeg")},
        data={
            "target_type": "issf_standard_10_ring",
            "caliber_mm": "4.5",
            "engine": "model_sendiri"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["target_detected"] is True
    assert data["detection_engine_used"] == "model_sendiri"


def test_gemini_configuration_status():
    response = client.get("/api/v1/detect/model-status")
    assert response.status_code == 200
    data = response.json()
    assert "gemini_api_configured" in data
    assert data["gemini_api_configured"] is True
    assert "gemini_model" in data
    assert "available_engines" in data
    assert "gemini" in data["available_engines"]
    assert "model_sendiri" in data["available_engines"]


def test_detect_upload_with_engine_yolo26():
    image_bytes = generate_synthetic_target_image()
    response = client.post(
        "/api/v1/detect/upload",
        files={"file": ("target_test.jpg", image_bytes, "image/jpeg")},
        data={
            "target_type": "issf_standard_10_ring",
            "caliber_mm": "4.5",
            "engine": "yolo26"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["target_detected"] is True
    assert data["detection_engine_used"] in ["yolo26", "yolo26_ai", "model_sendiri", "classical_cv_ensemble"]


def test_detect_upload_with_engine_yolo11():
    image_bytes = generate_synthetic_target_image()
    response = client.post(
        "/api/v1/detect/upload",
        files={"file": ("target_test.jpg", image_bytes, "image/jpeg")},
        data={
            "target_type": "issf_standard_10_ring",
            "caliber_mm": "4.5",
            "engine": "yolo11"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["target_detected"] is True
    assert data["detection_engine_used"] in ["yolo11", "yolo11_ai", "model_sendiri", "classical_cv_ensemble"]


def test_auto_crop_square_on_landscape_upload():
    """Verify that a non-1:1 landscape image (1200x800) is automatically cropped to 1:1 square."""
    img_square = generate_synthetic_target_image()
    nparr = np.frombuffer(img_square, np.uint8)
    base_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Pad horizontally to make 1200x800 (landscape 3:2)
    landscape = cv2.copyMakeBorder(base_img, 0, 0, 200, 200, cv2.BORDER_CONSTANT, value=[220, 220, 220])
    _, buf = cv2.imencode(".jpg", landscape)

    response = client.post(
        "/api/v1/detect/upload",
        files={"file": ("target_landscape.jpg", buf.tobytes(), "image/jpeg")},
        data={"target_type": "issf_standard_10_ring"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["target_detected"] is True
    assert data["metadata"]["cropped_to_square"] is True
    assert data["metadata"]["crop_box"]["width"] == data["metadata"]["crop_box"]["height"]
    assert data["metadata"]["crop_box"]["width"] == 800


def test_auto_crop_square_on_portrait_upload():
    """Verify that a non-1:1 portrait image (800x1200) is automatically cropped to 1:1 square."""
    img_square = generate_synthetic_target_image()
    nparr = np.frombuffer(img_square, np.uint8)
    base_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Pad vertically to make 800x1200 (portrait 2:3)
    portrait = cv2.copyMakeBorder(base_img, 200, 200, 0, 0, cv2.BORDER_CONSTANT, value=[220, 220, 220])
    _, buf = cv2.imencode(".jpg", portrait)

    response = client.post(
        "/api/v1/detect/upload",
        files={"file": ("target_portrait.jpg", buf.tobytes(), "image/jpeg")},
        data={"target_type": "issf_standard_10_ring"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["target_detected"] is True
    assert data["metadata"]["cropped_to_square"] is True
    assert data["metadata"]["crop_box"]["width"] == data["metadata"]["crop_box"]["height"]
    assert data["metadata"]["crop_box"]["height"] == 800




