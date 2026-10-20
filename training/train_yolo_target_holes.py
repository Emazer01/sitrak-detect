import os
import sys
import shutil
from pathlib import Path
import argparse
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def train_on_target_holes_dataset(
    model_name: str,
    base_model_file: str,
    output_target_file: str,
    dataset_yaml: Path,
    epochs: int = 30,
    img_size: int = 640,
    batch_size: int = 4,
    workers: int = 2,
    device: str = "cpu"
) -> Path:
    print(f"\n{'='*70}")
    print(f" TRAINING {model_name.upper()} ON yolo_target_holes_dataset")
    print(f" Base model: {base_model_file}")
    print(f" Epochs: {epochs} | Imgsz: {img_size} | Batch: {batch_size} | Device: {device}")
    print(f"{'='*70}\n")

    model = YOLO(base_model_file)

    exp_name = f"{model_name.lower()}_target_holes_exp"
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
        # Fine-tuning parameters for small dataset
        degrees=10.0,
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.015,
        hsv_s=0.2,
        hsv_v=0.3,
        mosaic=0.2,
        close_mosaic=5,
        save=True,
        plots=True,
        verbose=True
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path("training/runs") / exp_name
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = Path(f"runs/detect/training/runs/{exp_name}/weights/best.pt")

    if not best_weights.exists():
        best_weights = save_dir / "weights" / "last.pt"
        if not best_weights.exists():
            best_weights = Path(f"runs/detect/training/runs/{exp_name}/weights/last.pt")

    target_path = Path(output_target_file)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if best_weights.exists():
        shutil.copy(best_weights, target_path)
        print(f"\n[SUCCESS] {model_name} trained and deployed to: {target_path.resolve()} (Size: {target_path.stat().st_size} bytes)")
    else:
        print(f"\n[WARNING] Could not find weights at {best_weights}")

    return target_path


def main():
    parser = argparse.ArgumentParser(description="Train models on yolo_target_holes_dataset")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--workers", type=int, default=2, help="Workers")
    args = parser.parse_args()

    dataset_yaml = Path("training/yolo_target_holes_dataset/data.yaml")
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {dataset_yaml}")

    # Prefer starting from existing trained weights if present, otherwise base models
    yolo26_init = "models/bullet_hole_yolo26.pt" if Path("models/bullet_hole_yolo26.pt").exists() else "yolo26n.pt"
    yolo11_init = "models/bullet_hole_yolo11.pt" if Path("models/bullet_hole_yolo11.pt").exists() else "yolo11n.pt"

    # 1. Train YOLO26
    t26 = train_on_target_holes_dataset(
        model_name="yolo26",
        base_model_file=yolo26_init,
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
    train_on_target_holes_dataset(
        model_name="yolo11",
        base_model_file=yolo11_init,
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

