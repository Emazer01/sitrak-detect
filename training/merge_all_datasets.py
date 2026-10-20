import shutil
from pathlib import Path


def merge_datasets():
    base_dir = Path(__file__).resolve().parent
    combined_dir = base_dir / "dataset_combined"

    # Reset destination
    if combined_dir.exists():
        shutil.rmtree(combined_dir)

    train_img_dir = combined_dir / "images" / "train"
    val_img_dir = combined_dir / "images" / "val"
    train_lbl_dir = combined_dir / "labels" / "train"
    val_lbl_dir = combined_dir / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    sources = [
        {
            "name": "kruger",
            "dir": base_dir / "dataset_kruger",
            "train_split": "train",
            "val_split": "val"
        },
        {
            "name": "target_holes",
            "dir": base_dir / "yolo_target_holes_dataset",
            "train_split": "train",
            "val_split": "val",
            "also_val_splits": ["test"]
        }
    ]

    total_train = 0
    total_val = 0

    for src in sources:
        s_dir = src["dir"]
        s_name = src["name"]
        if not s_dir.exists():
            print(f"Warning: source {s_dir} does not exist, skipping.")
            continue

        print(f"Processing source: {s_name} from {s_dir.resolve()}...")

        # 1. Train split
        t_imgs = list((s_dir / "images" / src["train_split"]).glob("*.*"))
        for img_path in t_imgs:
            lbl_path = s_dir / "labels" / src["train_split"] / f"{img_path.stem}.txt"
            new_stem = f"{s_name}_{img_path.stem}"
            dst_img = train_img_dir / f"{new_stem}{img_path.suffix}"
            dst_lbl = train_lbl_dir / f"{new_stem}.txt"

            shutil.copy(img_path, dst_img)
            if lbl_path.exists():
                shutil.copy(lbl_path, dst_lbl)
            else:
                dst_lbl.touch()
            total_train += 1

        # 2. Val split
        val_splits = [src["val_split"]] + src.get("also_val_splits", [])
        for v_split in val_splits:
            v_dir = s_dir / "images" / v_split
            if not v_dir.exists():
                continue
            v_imgs = list(v_dir.glob("*.*"))
            for img_path in v_imgs:
                lbl_path = s_dir / "labels" / v_split / f"{img_path.stem}.txt"
                new_stem = f"{s_name}_{v_split}_{img_path.stem}"
                dst_img = val_img_dir / f"{new_stem}{img_path.suffix}"
                dst_lbl = val_lbl_dir / f"{new_stem}.txt"

                shutil.copy(img_path, dst_img)
                if lbl_path.exists():
                    shutil.copy(lbl_path, dst_lbl)
                else:
                    dst_lbl.touch()
                total_val += 1

    # Write data.yaml
    yaml_content = f"""path: {combined_dir.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: bullet_hole
"""
    yaml_path = combined_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("\n" + "=" * 60)
    print(f"MERGED DATASET CREATED: {combined_dir.resolve()}")
    print(f"Total Train Images: {total_train}")
    print(f"Total Val Images:   {total_val}")
    print(f"Config YAML:        {yaml_path.resolve()}")
    print("=" * 60)
    return yaml_path


if __name__ == "__main__":
    merge_datasets()

