import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.vision_service import vision_service

vision_service.reload_model()

imgs = [
    r'C:/Users/Advan/.gemini/antigravity/brain/3130756e-b1e5-42fd-88b6-c01cf73c6b69/.user_uploaded/media_1792458102887_034f03cf.jpg',
    r'C:/Users/Advan/.gemini/antigravity/brain/3130756e-b1e5-42fd-88b6-c01cf73c6b69/.user_uploaded/media_1792458102886_2d42a6d2.jpg',
    r'C:/Users/Advan/.gemini/antigravity/brain/3130756e-b1e5-42fd-88b6-c01cf73c6b69/.user_uploaded/media_1792458102886_253d812c.jpg',
    r'C:/Users/Advan/.gemini/antigravity/brain/3130756e-b1e5-42fd-88b6-c01cf73c6b69/.user_uploaded/media_1792458102886_4bf9e183.jpg',
    r'C:/Users/Advan/.gemini/antigravity/brain/3130756e-b1e5-42fd-88b6-c01cf73c6b69/.user_uploaded/media_1792458102886_3b3cbd74.jpg',
]

expected_counts = [3, 10, 10, 10, 3]

for i, p in enumerate(imgs, 1):
    b = Path(p).read_bytes()
    res = vision_service.process_image(b, original_filename=f'img_{i}.jpg')
    match_str = "PASSED" if res.shots_count == expected_counts[i-1] else "FAIL"
    print(f"=== Image {i} ({match_str}) ===")
    print(f"Expected: {expected_counts[i-1]} shots | Detected: {res.shots_count} shots")
    print(f"Total Score: {res.total_score}/{res.max_possible_score} ({res.accuracy_percentage}%)")
    scores = [s.score for s in res.shots]
    print(f"Scores per shot: {scores}")
    for s in res.shots:
        print(f"  Shot {s.id}: ({s.x}, {s.y}) -> Score {s.score} (dec: {s.decimal_score}, dist: {s.distance_from_center}mm, conf: {s.confidence})")
