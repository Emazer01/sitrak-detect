import os
import random
import math
import shutil
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np


def build_augmented_user_samples(output_train_img: Path, output_train_lbl: Path, output_val_img: Path, output_val_lbl: Path):
    """
    Augments the 5 user-uploaded Krüger target images with exact ground-truth labels
    via rotations (0, 90, 180, 270, slight angle), flips, and photometric variations.
    """
    user_samples_dir = Path("training/assets/user_samples")
    sample_files = sorted(list(user_samples_dir.glob("user_kruger_*.jpg")))

    print(f"[Augmenter] Found {len(sample_files)} user ground-truth images.")

    count = 0
    val_set = {2, 4}  # Reserve sample 2 and 4 as parts of validation

    for s_idx, img_file in enumerate(sample_files, 1):
        lbl_file = img_file.with_suffix(".txt")
        img = cv2.imread(str(img_file))
        h, w = img.shape[:2]

        # Read base bboxes
        base_bboxes = []
        if lbl_file.exists():
            for line in lbl_file.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    _, ncx, ncy, nbw, nbh = map(float, parts)
                    xmin = (ncx - nbw / 2.0) * w
                    ymin = (ncy - nbh / 2.0) * h
                    xmax = (ncx + nbw / 2.0) * w
                    ymax = (ncy + nbh / 2.0) * h
                    base_bboxes.append((xmin, ymin, xmax, ymax))

        # Generate 45 augmented versions per user image
        for aug_i in range(45):
            is_val = (s_idx in val_set and aug_i < 8)
            dst_img_dir = output_val_img if is_val else output_train_img
            dst_lbl_dir = output_val_lbl if is_val else output_train_lbl

            aug_img = img.copy()
            bboxes = list(base_bboxes)

            # 1. Rotation mode
            rot = random.choice(["none", "90", "180", "270", "slight"])
            if rot == "90":
                aug_img = cv2.rotate(aug_img, cv2.ROTATE_90_CLOCKWISE)
                bboxes = [(h - ymax, xmin, h - ymin, xmax) for (xmin, ymin, xmax, ymax) in bboxes]
            elif rot == "180":
                aug_img = cv2.rotate(aug_img, cv2.ROTATE_180)
                bboxes = [(w - xmax, h - ymax, w - xmin, h - ymin) for (xmin, ymin, xmax, ymax) in bboxes]
            elif rot == "270":
                aug_img = cv2.rotate(aug_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                bboxes = [(ymin, w - xmax, ymax, w - xmin) for (xmin, ymin, xmax, ymax) in bboxes]
            elif rot == "slight":
                angle = random.uniform(-6.0, 6.0)
                M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
                aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                rad = math.radians(angle)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                cx_f, cy_f = w / 2.0, h / 2.0
                new_boxes = []
                for (xmin, ymin, xmax, ymax) in bboxes:
                    bcx, bcy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
                    bw_b, bh_b = xmax - xmin, ymax - ymin
                    rx = cos_a * (bcx - cx_f) - sin_a * (bcy - cy_f) + cx_f
                    ry = sin_a * (bcx - cx_f) + cos_a * (bcy - cy_f) + cy_f
                    new_boxes.append((rx - bw_b / 2.0, ry - bh_b / 2.0, rx + bw_b / 2.0, ry + bh_b / 2.0))
                bboxes = new_boxes

            # 2. Lighting & contrast jitter
            if random.random() < 0.50:
                bright = random.uniform(0.90, 1.10)
                cont = random.uniform(0.92, 1.08)
                aug_img = np.clip(aug_img.astype(np.float32) * cont * bright, 0, 255).astype(np.uint8)

            # Resize to 640x640 standard YOLO input
            target_sz = 640
            img_resized = cv2.resize(aug_img, (target_sz, target_sz), interpolation=cv2.INTER_AREA)
            scale_x = target_sz / float(w)
            scale_y = target_sz / float(h)

            yolo_lines = []
            for (xmin, ymin, xmax, ymax) in bboxes:
                r_xmin = max(0.0, xmin * scale_x)
                r_ymin = max(0.0, ymin * scale_y)
                r_xmax = min(float(target_sz), xmax * scale_x)
                r_ymax = min(float(target_sz), ymax * scale_y)

                bw_box = r_xmax - r_xmin
                bh_box = r_ymax - r_ymin
                if bw_box < 3 or bh_box < 3:
                    continue
                bcx = (r_xmin + r_xmax) / 2.0
                bcy = (r_ymin + r_ymax) / 2.0
                yolo_lines.append(f"0 {bcx/target_sz:.6f} {bcy/target_sz:.6f} {bw_box/target_sz:.6f} {bh_box/target_sz:.6f}")

            prefix = "val" if is_val else "train"
            fn = f"user_aug_{s_idx}_{aug_i:03d}"
            cv2.imwrite(str(dst_img_dir / f"{fn}.jpg"), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(dst_lbl_dir / f"{fn}.txt", "w") as f:
                f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
            count += 1

    print(f"[Augmenter] Created {count} augmented user sample images.")


def generate_synthetic_digital_dots(
    output_train_img: Path,
    output_train_lbl: Path,
    output_val_img: Path,
    output_val_lbl: Path,
    num_samples: int = 400
):
    """
    Synthesizes Krüger targets containing both white digital dots (in bullseye)
    and grey digital dots (on cream paper), matching the user's electronic simulator format.
    Also includes 16% HARD NEGATIVES (clean target with inner-10 white circle but NO shots).
    """
    master_path = Path("training/assets/kruger_master.webp")
    master_img = cv2.imread(str(master_path))
    h, w = master_img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    bullseye_r = 148.6
    outer_r = 376.0

    target_sz = 640
    scale_x = target_sz / float(w)
    scale_y = target_sz / float(h)

    print(f"[Synthesizer] Generating {num_samples} digital dots target images...")

    for i in range(num_samples):
        is_val = (i < int(num_samples * 0.18))
        dst_img_dir = output_val_img if is_val else output_train_img
        dst_lbl_dir = output_val_lbl if is_val else output_train_lbl

        img = master_img.copy()

        # 16% Hard Negative (Clean target with 0 shots, to suppress false positives on inner-10 and numbers)
        is_hard_negative = (random.random() < 0.16)
        bboxes = []

        if not is_hard_negative:
            num_shots = random.choices([1, 2, 3, 5, 8, 10], weights=[0.1, 0.15, 0.25, 0.25, 0.15, 0.10])[0]

            for _ in range(num_shots):
                # Sample position across target
                dist = outer_r * (random.uniform(0.05, 0.98) ** 0.85)
                ang = random.uniform(0, 2.0 * math.pi)
                sx = int(round(cx + dist * math.cos(ang)))
                sy = int(round(cy + dist * math.sin(ang)))
                sx = max(25, min(w - 25, sx))
                sy = max(25, min(h - 25, sy))

                dist_c = math.hypot(sx - cx, sy - cy)
                is_bullseye = (dist_c <= bullseye_r)

                dot_r = random.uniform(7.0, 8.2)

                if is_bullseye:
                    # Pure white / light grey disc inside black bullseye
                    dot_color = (random.randint(245, 255), random.randint(245, 255), random.randint(245, 255))
                    cv2.circle(img, (sx, sy), int(round(dot_r)), dot_color, -1, cv2.LINE_AA)
                else:
                    # Medium grey disc on cream paper
                    g_val = random.randint(180, 210)
                    dot_color = (g_val, g_val, g_val)
                    cv2.circle(img, (sx, sy), int(round(dot_r)), dot_color, -1, cv2.LINE_AA)

                # Bounding box
                bw_box = dot_r * 2.3
                bh_box = dot_r * 2.3
                xmin = sx - bw_box / 2.0
                ymin = sy - bh_box / 2.0
                xmax = sx + bw_box / 2.0
                ymax = sy + bh_box / 2.0
                bboxes.append((xmin, ymin, xmax, ymax))

        # Rotate / augment
        rot = random.choice(["none", "90", "180", "270", "slight"])
        if rot == "90":
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            bboxes = [(h - ymax, xmin, h - ymin, xmax) for (xmin, ymin, xmax, ymax) in bboxes]
        elif rot == "180":
            img = cv2.rotate(img, cv2.ROTATE_180)
            bboxes = [(w - xmax, h - ymax, w - xmin, h - ymin) for (xmin, ymin, xmax, ymax) in bboxes]
        elif rot == "270":
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            bboxes = [(ymin, w - xmax, ymax, w - xmin) for (xmin, ymin, xmax, ymax) in bboxes]
        elif rot == "slight":
            angle = random.uniform(-6.0, 6.0)
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            cx_f, cy_f = w / 2.0, h / 2.0
            new_boxes = []
            for (xmin, ymin, xmax, ymax) in bboxes:
                bcx, bcy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
                bw_b, bh_b = xmax - xmin, ymax - ymin
                rx = cos_a * (bcx - cx_f) - sin_a * (bcy - cy_f) + cx_f
                ry = sin_a * (bcx - cx_f) + cos_a * (bcy - cy_f) + cy_f
                new_boxes.append((rx - bw_b / 2.0, ry - bh_b / 2.0, rx + bw_b / 2.0, ry + bh_b / 2.0))
            bboxes = new_boxes

        # Resize to 640x640
        img_resized = cv2.resize(img, (target_sz, target_sz), interpolation=cv2.INTER_AREA)

        yolo_lines = []
        for (xmin, ymin, xmax, ymax) in bboxes:
            r_xmin = max(0.0, xmin * scale_x)
            r_ymin = max(0.0, ymin * scale_y)
            r_xmax = min(float(target_sz), xmax * scale_x)
            r_ymax = min(float(target_sz), ymax * scale_y)
            bw_box = r_xmax - r_xmin
            bh_box = r_ymax - r_ymin
            if bw_box < 3 or bh_box < 3:
                continue
            bcx = (r_xmin + r_xmax) / 2.0
            bcy = (r_ymin + r_ymax) / 2.0
            yolo_lines.append(f"0 {bcx/target_sz:.6f} {bcy/target_sz:.6f} {bw_box/target_sz:.6f} {bh_box/target_sz:.6f}")

        fn = f"syn_dot_{i:04d}"
        cv2.imwrite(str(dst_img_dir / f"{fn}.jpg"), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(dst_lbl_dir / f"{fn}.txt", "w") as f:
            f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

    print(f"[Synthesizer] Generated {num_samples} digital dots target images.")


def build_final_dataset():
    out_dir = Path("training/dataset_final")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    train_img = out_dir / "images" / "train"
    val_img = out_dir / "images" / "val"
    train_lbl = out_dir / "labels" / "train"
    val_lbl = out_dir / "labels" / "val"

    for d in [train_img, val_img, train_lbl, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Augment user samples
    build_augmented_user_samples(train_img, train_lbl, val_img, val_lbl)

    # 2. Generate synthetic digital dots targets + hard negatives
    generate_synthetic_digital_dots(train_img, train_lbl, val_img, val_lbl, num_samples=400)

    # 3. Incorporate selected physical puncture targets from dataset_kruger for generalization
    kruger_dir = Path("training/dataset_kruger")
    if kruger_dir.exists():
        for split in ["train", "val"]:
            dst_i = train_img if split == "train" else val_img
            dst_l = train_lbl if split == "train" else val_lbl
            imgs = list((kruger_dir / "images" / split).glob("*.jpg"))[:120]
            for im_path in imgs:
                shutil.copy(im_path, dst_i / f"phys_{im_path.name}")
                lp = kruger_dir / "labels" / split / f"{im_path.stem}.txt"
                if lp.exists():
                    shutil.copy(lp, dst_l / f"phys_{im_path.stem}.txt")

    # 4. Generate data.yaml
    yaml_content = f"""path: {out_dir.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: bullet_hole
"""
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    t_count = len(list(train_img.glob("*.jpg")))
    v_count = len(list(val_img.glob("*.jpg")))
    print(f"\n[DONE] Final Dataset ready at {out_dir.resolve()}")
    print(f"  Train images: {t_count}")
    print(f"  Val images:   {v_count}")
    return yaml_path


if __name__ == "__main__":
    build_final_dataset()

