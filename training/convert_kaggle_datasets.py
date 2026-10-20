"""
convert_kaggle_datasets.py
Converts all 3 user-provided Kaggle datasets into unified YOLOv8 format:
1. pfrmusic/yolo-shot-datasets-with-annots (already YOLO format)
2. karimhossam02/computer-vision-shooting-target-score-dataset (converts 'hit' class to normalized YOLO)
3. bryanb/standup-targets (converts targets 5-10 and 'out' bullet impacts to normalized YOLO)
"""

import os
import shutil
import pandas as pd
import cv2
from pathlib import Path


OUTPUT_DIR = Path("training/dataset")
KAGGLE_DIR = Path("training/kaggle_datasets")


def process_pfrmusic():
    print("[Importer] Processing pfrmusic (yolo-shot-datasets-with-annots)...")
    base = KAGGLE_DIR / "pfrmusic" / "yolo_shot_dataset"
    count = 0
    for split in ["train", "val"]:
        img_src = base / "images" / split
        lbl_src = base / "labels" / split
        
        img_dest = OUTPUT_DIR / "images" / split
        lbl_dest = OUTPUT_DIR / "labels" / split
        
        for img_p in img_src.glob("*.jpg"):
            lbl_p = lbl_src / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                continue
            
            # Format: '0 cx cy w h'
            dest_img_p = img_dest / f"pfrmusic_{img_p.name}"
            dest_lbl_p = lbl_dest / f"pfrmusic_{img_p.stem}.txt"
            
            shutil.copy(img_p, dest_img_p)
            shutil.copy(lbl_p, dest_lbl_p)
            count += 1
            
    print(f"  [pfrmusic] Imported {count} images successfully!")
    return count


def process_karimhossam02():
    print("[Importer] Processing karimhossam02 (computer-vision-shooting-target-score-dataset)...")
    base = KAGGLE_DIR / "karimhossam02"
    img_dir = base / "images"
    lbl_dir = base / "labels"
    
    count = 0
    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    
    for idx, img_p in enumerate(img_files):
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue
            
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            continue
            
        yolo_lines = []
        with open(lbl_p, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_name = parts[0].lower()
                # We specifically extract bullet hole impact punctures ('hit')
                if cls_name == "hit":
                    x1, y1, bw, bh = [float(val) for val in parts[1:5]]
                    cx = (x1 + bw / 2.0) / w
                    cy = (y1 + bh / 2.0) / h
                    nw = bw / w
                    nh = bh / h
                    
                    # Clamp
                    cx = max(0.001, min(0.999, cx))
                    cy = max(0.001, min(0.999, cy))
                    nw = max(0.002, min(0.5, nw))
                    nh = max(0.002, min(0.5, nh))
                    yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                    
        # Allocate 1 of 6 to val, rest to train
        split = "val" if (idx % 6 == 0) else "train"
        dest_img_p = OUTPUT_DIR / "images" / split / f"karim_{img_p.stem}.jpg"
        dest_lbl_p = OUTPUT_DIR / "labels" / split / f"karim_{img_p.stem}.txt"
        
        cv2.imwrite(str(dest_img_p), img)
        with open(dest_lbl_p, "w") as f:
            for l in yolo_lines:
                f.write(l + "\n")
        count += 1
        
    print(f"  [karimhossam02] Imported {count} images successfully!")
    return count


def process_bryanb(max_samples: int = 500):
    print(f"[Importer] Processing bryanb (standup-targets, taking {max_samples} diverse samples)...")
    base = KAGGLE_DIR / "bryanb"
    csv_p = base / "train.csv"
    train_img_dir = base / "train" / "train"
    
    if not csv_p.exists():
        print("  train.csv not found, skipping bryanb.")
        return 0
        
    df = pd.read_csv(csv_p)
    # Group bounding boxes by image
    grouped = df.groupby("image_name")
    img_names = list(grouped.groups.keys())
    
    # Subsample if more than max_samples to keep train balanced
    step = max(1, len(img_names) // max_samples)
    selected_imgs = img_names[::step][:max_samples]
    
    count = 0
    for idx, img_name in enumerate(selected_imgs):
        group = grouped.get_group(img_name)
        img_src = train_img_dir / img_name
        if not img_src.exists():
            continue
            
        img = cv2.imread(str(img_src))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        yolo_lines = []
        for _, row in group.iterrows():
            bx = float(row["bbox_x"])
            by = float(row["bbox_y"])
            bw = float(row["bbox_width"])
            bh = float(row["bbox_height"])
            
            cx = (bx + bw / 2.0) / w
            cy = (by + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            
            cx = max(0.001, min(0.999, cx))
            cy = max(0.001, min(0.999, cy))
            nw = max(0.002, min(0.5, nw))
            nh = max(0.002, min(0.5, nh))
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            
        split = "val" if (idx % 6 == 0) else "train"
        stem = Path(img_name).stem
        dest_img_p = OUTPUT_DIR / "images" / split / f"bryanb_{stem}.jpg"
        dest_lbl_p = OUTPUT_DIR / "labels" / split / f"bryanb_{stem}.txt"
        
        cv2.imwrite(str(dest_img_p), img)
        with open(dest_lbl_p, "w") as f:
            for l in yolo_lines:
                f.write(l + "\n")
        count += 1
        
    print(f"  [bryanb] Imported {count} images successfully!")
    return count


def process_bobsuna():
    print("[Importer] Processing bobsuna (datasets-of-bullet-hole-on-the-target-paper)...")
    base = KAGGLE_DIR / "bobsuna" / "dataset"
    count = 0
    for split in ["train", "val"]:
        img_src = base / "images" / split
        lbl_src = base / "labels" / split
        
        img_dest = OUTPUT_DIR / "images" / split
        lbl_dest = OUTPUT_DIR / "labels" / split
        
        if not img_src.exists():
            continue
            
        for img_p in list(img_src.glob("*.jpg")) + list(img_src.glob("*.png")):
            lbl_p = lbl_src / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                continue
                
            dest_img_p = img_dest / f"bobsuna_{img_p.name}"
            dest_lbl_p = lbl_dest / f"bobsuna_{img_p.stem}.txt"
            
            shutil.copy(img_p, dest_img_p)
            shutil.copy(lbl_p, dest_lbl_p)
            count += 1
            
    print(f"  [bobsuna] Imported {count} images successfully!")
    return count


def process_qiyan527(max_train: int = 5000, max_val: int = 1000):
    print(f"[Importer] Processing qiyan527 (target-paper-images-with-bullet-holes)...")
    base = KAGGLE_DIR / "qiyan527" / "datasets"
    count = 0
    
    splits_map = {
        "train": (base / "train", OUTPUT_DIR / "images" / "train", OUTPUT_DIR / "labels" / "train", max_train),
        "val": (base / "val", OUTPUT_DIR / "images" / "val", OUTPUT_DIR / "labels" / "val", max_val)
    }
    
    for split_name, (src_base, img_dest, lbl_dest, max_limit) in splits_map.items():
        img_dir = src_base / "images"
        lbl_dir = src_base / "labels"
        
        if not img_dir.exists():
            continue
            
        img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        if max_limit and len(img_files) > max_limit:
            step = max(1, len(img_files) // max_limit)
            selected = img_files[::step][:max_limit]
        else:
            selected = img_files
            
        for img_p in selected:
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                continue
                
            dest_img_p = img_dest / f"qiyan_{img_p.name}"
            dest_lbl_p = lbl_dest / f"qiyan_{img_p.stem}.txt"
            
            shutil.copy(img_p, dest_img_p)
            shutil.copy(lbl_p, dest_lbl_p)
            count += 1
            
    print(f"  [qiyan527] Imported {count} images successfully!")
    return count


def main():
    p_cnt = process_pfrmusic()
    k_cnt = process_karimhossam02()
    b_cnt = process_bryanb(max_samples=1000)
    bob_cnt = process_bobsuna()
    qiy_cnt = process_qiyan527(max_train=5000, max_val=1000)
    
    print("\n==========================================")
    print(f"Total Kaggle images imported: {p_cnt + k_cnt + b_cnt + bob_cnt + qiy_cnt}")
    
    train_total = len(list((OUTPUT_DIR / "images" / "train").glob("*.jpg")))
    val_total = len(list((OUTPUT_DIR / "images" / "val").glob("*.jpg")))
    print(f"Grand Total Dataset: {train_total} train, {val_total} val ({train_total + val_total} total images)")
    print("==========================================")


if __name__ == "__main__":
    main()

