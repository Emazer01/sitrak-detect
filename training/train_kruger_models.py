import os
import sys
import shutil
from pathlib import Path
import argparse
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def train_single_model(
    model_name: str,
    base_model_file: str,
    output_target_file: str,
    dataset_yaml: Path,
    epochs: int = 10,
    img_size: int = 416,
    batch_size: int = 16,
    workers: int = 2,
    device: str = "cpu"
) -> Path:
    print(f"\n{'='*70}")
    print(f" TRAINING {model_name.upper()} ({base_model_file})")
    print(f" Epochs: {epochs} | Imgsz: {img_size} | Batch: {batch_size} | Device: {device}")
    print(f"{'='*70}\n")

    model = YOLO(base_model_file)

    exp_name = f"{model_name.lower()}_kruger_exp"
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
        # Augmentations suitable for bullet hole punctures on paper target
        degrees=10.0,
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.015,
        hsv_s=0.2,
        hsv_v=0.3,
        mosaic=0.5,
        close_mosaic=2,
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
    parser = argparse.ArgumentParser(description="Train YOLO26 and YOLO11 on Kruger Target Dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=416, help="Image dimension for training")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--model", type=str, default="all", choices=["all", "yolo26", "yolo11"], help="Which model to train")
    args = parser.parse_args()

    dataset_yaml = Path("training/dataset_kruger/data.yaml")
    if not dataset_yaml.exists():
        print(f"Dataset yaml not found at {dataset_yaml}. Generating dataset...")
        from training.generate_kruger_dataset import KrugerTargetDatasetGenerator
        gen = KrugerTargetDatasetGenerator()
        gen.generate(train_count=400, val_count=80)

    # 1. Train YOLO26
    if args.model in ["all", "yolo26"]:
        trained_yolo26 = train_single_model(
            model_name="yolo26",
            base_model_file="yolo26n.pt",
            output_target_file="models/bullet_hole_yolo26.pt",
            dataset_yaml=dataset_yaml,
            epochs=args.epochs,
            img_size=args.imgsz,
            batch_size=args.batch,
            workers=args.workers,
            device=args.device
        )
        # Also deploy as default bullet_hole_yolo.pt
        shutil.copy(trained_yolo26, "models/bullet_hole_yolo.pt")
        print(f"[DEPLOY] Copied YOLO26 to default: models/bullet_hole_yolo.pt")

    # 2. Train YOLO11
    if args.model in ["all", "yolo11"]:
        train_single_model(
            model_name="yolo11",
            base_model_file="yolo11n.pt",
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

