import os
import sys
import shutil
from pathlib import Path
import argparse
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def train_model(
    model_name: str,
    base_model_file: str,
    output_target_file: str,
    dataset_yaml: Path,
    epochs: int = 15,
    img_size: int = 640,
    batch_size: int = 8,
    workers: int = 2,
    device: str = "cpu"
) -> Path:
    print(f"\n{'='*75}")
    print(f" TRAINING {model_name.upper()} ON DATASET FINAL (Electronic + Puncture Targets)")
    print(f" Base model weights: {base_model_file}")
    print(f" Epochs: {epochs} | Imgsz: {img_size} | Batch: {batch_size} | Device: {device}")
    print(f"{'='*75}\n")

    model = YOLO(base_model_file)

    exp_name = f"{model_name.lower()}_final_exp"
    results = model.train(
        data=str(dataset_yaml.resolve()),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        workers=workers,
        project="training/runs",
        name=exp_name,
        exist_ok=True,
        # Fine-tuned for small impact dots (both white and grey)
        degrees=10.0,
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.015,
        hsv_s=0.2,
        hsv_v=0.3,
        mosaic=0.3,
        close_mosaic=3,
        save=True,
        plots=True,
        verbose=True
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path("training/runs") / exp_name
    candidates = [
        save_dir / "weights" / "best.pt",
        Path("training/runs") / exp_name / "weights" / "best.pt",
        Path(f"runs/detect/training/runs/{exp_name}/weights/best.pt"),
        Path(f"runs/detect/{exp_name}/weights/best.pt"),
        save_dir / "weights" / "last.pt",
        Path("training/runs") / exp_name / "weights" / "last.pt",
        Path(f"runs/detect/training/runs/{exp_name}/weights/last.pt"),
        Path(f"runs/detect/{exp_name}/weights/last.pt"),
    ]
    best_weights = None
    for c in candidates:
        if c.exists():
            best_weights = c
            break

    target_path = Path(output_target_file)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if best_weights and best_weights.exists():
        shutil.copy(best_weights, target_path)
        print(f"\n[SUCCESS] {model_name} trained and deployed to: {target_path.resolve()} (Size: {target_path.stat().st_size} bytes)")
    else:
        print(f"\n[WARNING] Could not find weights in any candidate locations: {candidates}")

    return target_path


def main():
    parser = argparse.ArgumentParser(description="Train models on dataset_final")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image dimension")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--workers", type=int, default=2, help="Worker count")
    args = parser.parse_args()

    dataset_yaml = Path("training/dataset_final/data.yaml")
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {dataset_yaml}")

    # 1. Train YOLO26
    base_26 = "models/bullet_hole_yolo26.pt" if Path("models/bullet_hole_yolo26.pt").exists() else "yolo26n.pt"
    t26 = train_model(
        model_name="yolo26",
        base_model_file=base_26,
        output_target_file="models/bullet_hole_yolo26.pt",
        dataset_yaml=dataset_yaml,
        epochs=args.epochs,
        img_size=args.imgsz,
        batch_size=args.batch,
        workers=args.workers,
        device=args.device
    )
    shutil.copy(t26, "models/bullet_hole_yolo.pt")
    print(f"[DEPLOY] Copied YOLO26 to default: models/bullet_hole_yolo.pt")

    # 2. Train YOLO11
    base_11 = "models/bullet_hole_yolo11.pt" if Path("models/bullet_hole_yolo11.pt").exists() else "yolo11n.pt"
    train_model(
        model_name="yolo11",
        base_model_file=base_11,
        output_target_file="models/bullet_hole_yolo11.pt",
        dataset_yaml=dataset_yaml,
        epochs=args.epochs,
        img_size=args.imgsz,
        batch_size=args.batch,
        workers=args.workers,
        device=args.device
    )


if __name__ == "__main__":
    main()

