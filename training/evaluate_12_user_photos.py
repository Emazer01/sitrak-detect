import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.vision_service import vision_service

PHOTOS_DIR = Path("training/user_target_photos")

# Expected shots ground-truth for each photo in sorted WhatsApp order
GROUND_TRUTH_COUNTS = [10, 10, 10, 10, 10, 3, 10, 3, 10, 3, 10, 10]


def evaluate():
    vision_service.reload_model()
    wa_files = sorted([f for f in os.listdir(PHOTOS_DIR) if f.startswith("WhatsApp") and f.endswith(".jpeg")])
    assert len(wa_files) == 12, f"Expected 12 WhatsApp photos, found {len(wa_files)}"

    print("\n" + "=" * 80)
    print(" END-TO-END EVALUATION: 12 RAW USER SMARTPHONE TARGET PHOTOS")
    print(" Through vision_service.process_image() (Auto-Crop 1:1 + Perspective + Scoring)")
    print("=" * 80)

    for engine_name in ["yolo26", "yolo11", "model_sendiri"]:
        print(f"\n--- Testing Engine: {engine_name.upper()} ---")
        exact_matches = 0
        close_matches = 0  # within +/- 1 shot

        print(f"{'Photo':<10} | {'Expected':<8} | {'Detected':<8} | {'Score':<6} | {'Status':<12} | Filename")
        print("-" * 80)

        for idx, (filename, expected) in enumerate(zip(wa_files, GROUND_TRUTH_COUNTS), 1):
            photo_path = PHOTOS_DIR / filename
            with open(photo_path, "rb") as f:
                img_bytes = f.read()

            try:
                res = vision_service.process_image(
                    image_bytes=img_bytes,
                    original_filename=filename,
                    engine=engine_name
                )
                detected = res.shots_count
                score = res.total_score
                diff = detected - expected

                if diff == 0:
                    status = "EXACT"
                    exact_matches += 1
                    close_matches += 1
                elif abs(diff) <= 1:
                    status = f"CLOSE ({diff:+d})"
                    close_matches += 1
                else:
                    status = f"DIFF ({diff:+d})"

                short_fn = filename if len(filename) <= 30 else filename[:27] + "..."
                print(f"Photo {idx:<4d} | {expected:<8d} | {detected:<8d} | {score:<6d} | {status:<12} | {short_fn}")
            except Exception as e:
                print(f"Photo {idx:<4d} | {expected:<8d} | ERROR: {e}")

        total = len(wa_files)
        print("-" * 80)
        print(f"Summary for {engine_name.upper()}: Exact = {exact_matches}/{total} ({exact_matches/total*100:.1f}%), "
              f"Exact/Close (+/-1) = {close_matches}/{total} ({close_matches/total*100:.1f}%)\n")


if __name__ == "__main__":
    evaluate()
