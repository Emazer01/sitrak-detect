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
    Augments all user target images (5 Krüger master scans + 12 smartphone photos of targets)
    with exact ground-truth labels via rotations (0, 90, 180, 270, slight angle), flips,
    and photometric / lighting variations.
    """
    user_samples_dir = Path("training/assets/user_samples")
    sample_files = sorted(list(user_samples_dir.glob("user_*.jpg")))

    print(f"[Augmenter] Found {len(sample_files)} user ground-truth images (master scans + real smartphone photos).")

    count = 0
    # Reserve selected real user photos and master samples for validation
    val_set = {"user_kruger_2", "user_kruger_4", "user_photo_3", "user_photo_8", "user_photo_11"}

    for s_idx, img_file in enumerate(sample_files, 1):
        lbl_file = img_file.with_suffix(".txt")
        img = cv2.imread(str(img_file))
        if img is None:
            continue
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

        num_augs = 28
        for aug_i in range(num_augs):
            is_val = (img_file.stem in val_set and aug_i < 5)
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
            if random.random() < 0.60:
                bright = random.uniform(0.88, 1.12)
                cont = random.uniform(0.90, 1.10)
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
            fn = f"usr_aug_{img_file.stem}_{aug_i:03d}"
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
    num_samples: int = 500
):
    """
    Synthesizes Krüger / Pistol targets with:
    - White / cyan digital dots (in bullseye)
    - Blue digital dots (in scoring rings)
    - Dark grey dots (in scoring rings)
    - Faint low-contrast light grey dots (Foto 1 & 4)
    - ADJACENT, TOUCHING & OVERLAPPING pairs and triplets (double-taps & clusters)
    - LCD moiré scanline simulation
    - 16% HARD NEGATIVES (clean target with 0 shots to suppress false positives on ring numbers).
    """
    templates = []
    
    # 1. Clean lesan template from user
    t_lesan = Path("training/user_target_photos/lesan_extracted_image.png")
    if t_lesan.exists():
        im = cv2.imread(str(t_lesan))
        if im is not None:
            templates.append((im, 357.0, 443.0, 443.0))

    # 2. Master webp
    t_master = Path("training/assets/kruger_master.webp")
    if t_master.exists():
        im = cv2.imread(str(t_master))
        if im is not None:
            h, w = im.shape[:2]
            templates.append((im, 148.6, w / 2.0, h / 2.0))

    if not templates:
        print("[Synthesizer] Warning: No base templates found.")
        return

    target_sz = 640
    print(f"[Synthesizer] Generating {num_samples} digital dots target images with adjacent & overlapping shots...")

    for i in range(num_samples):
        is_val = (i < int(num_samples * 0.16))
        dst_img_dir = output_val_img if is_val else output_train_img
        dst_lbl_dir = output_val_lbl if is_val else output_train_lbl

        base_tpl, bullseye_r, cx, cy = random.choice(templates)
        img = base_tpl.copy()
        h, w = img.shape[:2]
        outer_r = min(cx, cy, w - cx, h - cy) * 0.94
        scale_x = target_sz / float(w)
        scale_y = target_sz / float(h)

        # 16% Hard Negative (clean target with 0 shots)
        is_hard_negative = (random.random() < 0.16)
        bboxes = []

        if not is_hard_negative:
            target_total_shots = random.choices([1, 2, 3, 5, 8, 10], weights=[0.05, 0.10, 0.25, 0.25, 0.20, 0.15])[0]
            current_shots = 0

            while current_shots < target_total_shots:
                remaining = target_total_shots - current_shots
                if remaining >= 3 and random.random() < 0.25:
                    event_type = "triplet"
                elif remaining >= 2 and random.random() < 0.45:
                    event_type = "pair"
                else:
                    event_type = "single"

                dist = outer_r * (random.uniform(0.04, 0.98) ** 0.85)
                ang = random.uniform(0, 2.0 * math.pi)
                sx = int(round(cx + dist * math.cos(ang)))
                sy = int(round(cy + dist * math.sin(ang)))
                sx = max(25, min(w - 25, sx))
                sy = max(25, min(h - 25, sy))

                dist_c = math.hypot(sx - cx, sy - cy)
                is_bullseye = (dist_c <= bullseye_r)

                dot_r = random.uniform(6.5, 9.5)

                if is_bullseye:
                    # White / light cyan disc inside black bullseye (screen format)
                    dot_color = random.choice([
                        (random.randint(235, 255), random.randint(235, 255), random.randint(235, 255)), # pure white
                        (random.randint(240, 255), random.randint(230, 250), random.randint(200, 230)), # cyan/light-blue
                        (random.randint(210, 230), random.randint(210, 230), random.randint(210, 230))  # light grey
                    ])
                else:
                    color_choice = random.random()
                    if color_choice < 0.50:
                        # Blue dot (BGR: B in 180-255, G in 60-140, R in 10-50)
                        dot_color = (random.randint(180, 255), random.randint(60, 140), random.randint(10, 50))
                    elif color_choice < 0.75:
                        # Dark grey / black dot
                        g_val = random.randint(25, 75)
                        dot_color = (g_val, g_val, g_val)
                    else:
                        # Faint low-contrast light grey dot (like Foto 1 & Foto 4)
                        g_val = random.randint(170, 195)
                        dot_color = (g_val, min(255, g_val + random.randint(0, 5)), max(0, g_val - random.randint(0, 5)))

                event_pts = [(sx, sy)]
                if event_type == "pair":
                    pair_ang = random.uniform(0, 2.0 * math.pi)
                    pair_sep = dot_r * random.uniform(1.2, 2.1)
                    p2x = int(round(sx + pair_sep * math.cos(pair_ang)))
                    p2y = int(round(sy + pair_sep * math.sin(pair_ang)))
                    event_pts.append((p2x, p2y))
                elif event_type == "triplet":
                    pair_ang = random.uniform(0, 2.0 * math.pi)
                    pair_sep = dot_r * random.uniform(1.2, 2.1)
                    p2x = int(round(sx + pair_sep * math.cos(pair_ang)))
                    p2y = int(round(sy + pair_sep * math.sin(pair_ang)))
                    p3x = int(round(sx + pair_sep * math.cos(pair_ang + 1.1)))
                    p3y = int(round(sy + pair_sep * math.sin(pair_ang + 1.1)))
                    event_pts.extend([(p2x, p2y), (p3x, p3y)])

                for px, py in event_pts:
                    px = max(15, min(w - 15, px))
                    py = max(15, min(h - 15, py))
                    cv2.circle(img, (px, py), int(round(dot_r)), dot_color, -1, cv2.LINE_AA)
                    bw_box = dot_r * 2.2
                    bh_box = dot_r * 2.2
                    xmin = px - bw_box / 2.0
                    ymin = py - bh_box / 2.0
                    xmax = px + bw_box / 2.0
                    ymax = py + bh_box / 2.0
                    bboxes.append((xmin, ymin, xmax, ymax))

                current_shots += len(event_pts)

        # Add optional LCD scanline effect on 30% of samples
        if random.random() < 0.30:
            lines = np.zeros_like(img, dtype=np.float32)
            for y_line in range(0, h, 3):
                lines[y_line, :] = random.randint(10, 25)
            img = np.clip(img.astype(np.float32) - lines, 0, 255).astype(np.uint8)

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

    # 1. Augment user samples (both master Kruger scans and 12 real smartphone photos)
    build_augmented_user_samples(train_img, train_lbl, val_img, val_lbl)

    # 2. Generate synthetic digital dots targets + hard negatives
    generate_synthetic_digital_dots(train_img, train_lbl, val_img, val_lbl, num_samples=350)

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
