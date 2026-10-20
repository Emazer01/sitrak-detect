import os
import sys
import shutil
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.dataset_generator import SyntheticTargetDatasetGenerator


def train_bullet_detector(
    epochs: int = 15,
    img_size: int = 640,
    batch_size: int = 8,
    train_samples: int = 120,
    val_samples: int = 25,
    base_model: str = "yolov8n.pt",
    device: str = "cpu",
    workers: int = 4
):
    """
    Trains / fine-tunes a YOLO model to detect bullet holes on shooting targets.
    """
    from ultralytics import YOLO

    dataset_yaml = Path("training/dataset/data.yaml")
    if not dataset_yaml.exists():
        print("Dataset not found. Generating photorealistic synthetic target dataset...")
        generator = SyntheticTargetDatasetGenerator(output_dir="training/dataset", img_size=img_size)
        generator.generate(train_samples=train_samples, val_samples=val_samples)

    print(f"Loading base YOLO model: {base_model}")
    model = YOLO(base_model)

    print(f"Starting training on device={device} for {epochs} epochs...")
    results = model.train(
        data=str(dataset_yaml.resolve()),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        workers=workers,
        project="training/runs",
        name="bullet_hole_exp",
        exist_ok=True,
        # Augmentations optimized for small circular bullet hole punctures
        degrees=10.0,
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.015,
        hsv_s=0.2,
        hsv_v=0.3,
        mosaic=0.5,
        close_mosaic=3,
        save=True,
        plots=True
    )

    # Copy best weights to models/bullet_hole_yolo.pt for production deployment
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path("training/runs/bullet_hole_exp")
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = Path("runs/detect/training/runs/bullet_hole_exp/weights/best.pt")

    target_models_dir = Path("models")
    target_models_dir.mkdir(parents=True, exist_ok=True)
    deployed_weights = target_models_dir / "bullet_hole_yolo.pt"

    if best_weights.exists():
        shutil.copy(best_weights, deployed_weights)
        print(f"\n[SUCCESS] Model successfully trained and deployed to: {deployed_weights.resolve()}")
    else:
        last_weights = save_dir / "weights" / "last.pt"
        if not last_weights.exists():
            last_weights = Path("runs/detect/training/runs/bullet_hole_exp/weights/last.pt")
        if last_weights.exists():
            shutil.copy(last_weights, deployed_weights)
            print(f"\n[SUCCESS] Model deployed (last.pt) to: {deployed_weights.resolve()}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train bullet hole detector")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, 0, cuda)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--base_model", type=str, default="models/bullet_hole_yolo.pt", help="Base/pretrained YOLO model weights")
    args = parser.parse_args()

    # Fall back to yolov8n.pt if specified base model does not exist
    base_m = args.base_model
    if not Path(base_m).exists():
        base_m = "yolov8n.pt"

    train_bullet_detector(
        epochs=args.epochs,
        img_size=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        workers=args.workers,
        base_model=base_m
    )
