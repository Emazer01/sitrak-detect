import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def evaluate_model(weights_path: str = "models/bullet_hole_yolo.pt", dataset_yaml: str = "training/dataset_combined/data.yaml", imgsz: int = 416):
    """Evaluates YOLO model on validation set."""
    from ultralytics import YOLO

    model_path = Path(weights_path)
    if not model_path.exists():
        print(f"Error: Model weights not found at {model_path}")
        return None

    yaml_path = Path(dataset_yaml)
    if not yaml_path.exists():
        print(f"Error: Dataset yaml not found at {yaml_path}")
        return None

    print(f"\n{'='*60}")
    print(f"Evaluating {model_path.name} on {yaml_path} (imgsz={imgsz})...")
    print(f"{'='*60}")

    model = YOLO(str(model_path))
    metrics = model.val(data=str(yaml_path.resolve()), imgsz=imgsz, device="cpu", verbose=False)

    p = metrics.results_dict.get("metrics/precision(B)", 0)
    r = metrics.results_dict.get("metrics/recall(B)", 0)
    map50 = metrics.results_dict.get("metrics/mAP50(B)", 0)
    map95 = metrics.results_dict.get("metrics/mAP50-95(B)", 0)

    print("\n--- Validation Performance Metrics ---")
    print(f"Precision: {p:.4f} ({p*100:.1f}%)")
    print(f"Recall:    {r:.4f} ({r*100:.1f}%)")
    print(f"mAP50:     {map50:.4f} ({map50*100:.1f}%)")
    print(f"mAP50-95:  {map95:.4f} ({map95*100:.1f}%)")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate bullet hole detector")
    parser.add_argument("--weights", type=str, default="models/bullet_hole_yolo.pt", help="Path to weights")
    parser.add_argument("--data", type=str, default="training/dataset_combined/data.yaml", help="Path to dataset yaml")
    parser.add_argument("--imgsz", type=int, default=416, help="Image size")
    args = parser.parse_args()
    evaluate_model(weights_path=args.weights, dataset_yaml=args.data, imgsz=args.imgsz)
