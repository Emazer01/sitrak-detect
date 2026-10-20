import os
import random
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np


class SyntheticTargetDatasetGenerator:
    """
    State-of-the-art Multi-Discipline Shooting Target Dataset Generator.
    Produces highly realistic targets covering:
    - Style 1: Monochrome Target with White Center (ISSF / NRA Pistol 10-ring: rings 1-5 white, rings 6-8 black band, rings 9-10 white center)
    - Style 2: 5-Color Archery / Commercial Target (White 1-2, Black 3-4, Blue 5-6, Red 7-8, Gold/Yellow 9-10)
    - Style 3: Classic Solid Black Bullseye (rings 1-6 white, rings 7-10 solid black)
    - Realistic Backgrounds: Wood grain tables, desk surfaces, corner clips/clamps
    - Realistic Text & Logos: "10PCS", watermarks, brand badges, and numbers (1-10)
    - Photorealistic Bullet Punctures: Dark cores, lead-wipe rings, torn paper fibers, double-taps
    - Hard Negative Samples: Clean targets with text, numbers, clamps, and wood grains with 0 bullet holes.
    """

    def __init__(self, output_dir: str = "training/dataset", img_size: int = 640):
        self.output_dir = Path(output_dir)
        self.img_size = img_size

    def _setup_directories(self):
        for split in ["train", "val"]:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        yaml_content = f"""path: {self.output_dir.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: bullet_hole
"""
        with open(self.output_dir / "data.yaml", "w") as f:
            f.write(yaml_content)

    def _generate_wood_background(self, size: int) -> np.ndarray:
        """Generates realistic wood plank table background texture."""
        bg = np.zeros((size, size, 3), dtype=np.uint8)
        base_wood = np.array([45, 85, 145], dtype=np.float32)  # Brown BGR

        # Horizontal wood grain lines
        y_coords = np.arange(size)
        grain_freq = np.sin(y_coords * 0.08) * 15.0 + np.sin(y_coords * 0.02) * 25.0
        for y in range(size):
            color = np.clip(base_wood + grain_freq[y] + random.uniform(-10, 10), 0, 255)
            bg[y, :] = color.astype(np.uint8)

        # Wood plank vertical seams
        num_planks = random.randint(3, 5)
        step = size // num_planks
        for p in range(1, num_planks):
            px = p * step + random.randint(-10, 10)
            cv2.line(bg, (px, 0), (px, size), (25, 45, 80), 2)

        # Noise
        noise = np.random.normal(0, 4.0, bg.shape).astype(np.int16)
        return np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def _draw_target_canvas(self, size: int, style: str) -> Tuple[np.ndarray, Tuple[int, int], float, Dict[str, Any]]:
        """
        Draws the target paper onto canvas (either plain or on a table background).
        Returns (image, center, outer_radius, target_info).
        """
        has_table_bg = (random.random() < 0.40)
        if has_table_bg:
            canvas = self._generate_wood_background(size)
            # Paper target inside table
            margin = random.randint(25, 60)
            target_w = size - 2 * margin
            paper_rect = [margin, margin, size - margin, size - margin]
            # Draw paper shadow
            cv2.rectangle(canvas, (paper_rect[0] + 4, paper_rect[1] + 4), (paper_rect[2] + 4, paper_rect[3] + 4), (20, 20, 20), -1)
            # Draw white target paper
            paper_val = random.randint(238, 252)
            cv2.rectangle(canvas, (paper_rect[0], paper_rect[1]), (paper_rect[2], paper_rect[3]), (paper_val, paper_val, paper_val), -1)
            # Target center & radius
            cx = size // 2 + random.randint(-5, 5)
            cy = size // 2 + random.randint(-5, 5)
            outer_r = float(target_w * random.uniform(0.42, 0.47))
        else:
            paper_val = random.randint(240, 252)
            canvas = np.full((size, size, 3), (paper_val - 2, paper_val, paper_val - 2), dtype=np.uint8)
            cx = size // 2 + random.randint(-4, 4)
            cy = size // 2 + random.randint(-4, 4)
            outer_r = float(size * random.uniform(0.41, 0.46))

        center = (cx, cy)
        ring_radii = {k: outer_r * ((11.0 - k) / 10.0) for k in range(1, 11)}

        # =====================================================================
        # STYLE 1: White Center Monochrome (Image 1 Style)
        # Rings 1-5: White with black lines
        # Rings 6-8: Black band
        # Rings 9-10: White center with black lines and numbers
        # =====================================================================
        if style == "white_center_mono":
            # 1. Fill black band (from ring 6 down to ring 8 outer edge)
            r6 = int(ring_radii[6])
            cv2.circle(canvas, center, r6, (30, 30, 30), -1, cv2.LINE_AA)
            # 2. Fill white center (rings 9-10)
            r9 = int(ring_radii[9])
            cv2.circle(canvas, center, r9, (paper_val, paper_val, paper_val), -1, cv2.LINE_AA)

            # Draw concentric circle borders
            for k in range(1, 11):
                r = int(ring_radii[k])
                line_color = (200, 200, 200) if (6 <= k <= 8) else (50, 50, 50)
                th = 2 if k in [6, 9, 10] else 1
                cv2.circle(canvas, center, r, line_color, th, cv2.LINE_AA)

                # Numbers
                num_str = str(k)
                for ang in [0, 90, 180, 270]:
                    rad = math.radians(ang)
                    tx = int(cx + (r - 10) * math.cos(rad))
                    ty = int(cy + (r - 10) * math.sin(rad))
                    c_num = (220, 220, 220) if (6 <= k <= 8) else (40, 40, 40)
                    cv2.putText(canvas, num_str, (tx - 5, ty + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c_num, 1, cv2.LINE_AA)

        # =====================================================================
        # STYLE 2: 5-Color Archery/Commercial Target (Image 2 Style)
        # Rings 1-2: White
        # Rings 3-4: Black
        # Rings 5-6: Blue
        # Rings 7-8: Red
        # Rings 9-10: Yellow
        # =====================================================================
        elif style == "multicolor":
            # Color palette (BGR)
            c_white = (245, 245, 245)
            c_black = (28, 28, 28)
            c_blue = (210, 155, 45)     # Cyan/Blue in BGR
            c_red = (55, 65, 215)       # Red in BGR
            c_yellow = (45, 215, 240)   # Gold/Yellow in BGR

            # Draw colored concentric bands from outer to inner
            cv2.circle(canvas, center, int(ring_radii[1]), c_white, -1, cv2.LINE_AA)
            cv2.circle(canvas, center, int(ring_radii[3]), c_black, -1, cv2.LINE_AA)
            cv2.circle(canvas, center, int(ring_radii[5]), c_blue, -1, cv2.LINE_AA)
            cv2.circle(canvas, center, int(ring_radii[7]), c_red, -1, cv2.LINE_AA)
            cv2.circle(canvas, center, int(ring_radii[9]), c_yellow, -1, cv2.LINE_AA)

            # Draw thin ring lines and numbers
            for k in range(1, 11):
                r = int(ring_radii[k])
                cv2.circle(canvas, center, r, (35, 35, 35), 1, cv2.LINE_AA)
                num_str = str(k)
                for ang in [0, 90, 180, 270]:
                    rad = math.radians(ang)
                    tx = int(cx + (r - 10) * math.cos(rad))
                    ty = int(cy + (r - 10) * math.sin(rad))
                    c_num = (230, 230, 230) if k in [3, 4] else (25, 25, 25)
                    cv2.putText(canvas, num_str, (tx - 4, ty + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, c_num, 1, cv2.LINE_AA)

        # =====================================================================
        # STYLE 3: Solid Black Bullseye (Classic ISSF Target)
        # Rings 1-6: White
        # Rings 7-10: Black
        # =====================================================================
        else:
            cv2.circle(canvas, center, int(ring_radii[7]), (30, 30, 30), -1, cv2.LINE_AA)
            for k in range(1, 11):
                r = int(ring_radii[k])
                line_c = (190, 190, 190) if k >= 7 else (50, 50, 50)
                cv2.circle(canvas, center, r, line_c, 1, cv2.LINE_AA)
                if k <= 8:
                    for ang in [0, 90, 180, 270]:
                        rad = math.radians(ang)
                        tx = int(cx + (r - 10) * math.cos(rad))
                        ty = int(cy + (r - 10) * math.sin(rad))
                        c_num = (190, 190, 190) if k >= 7 else (40, 40, 40)
                        cv2.putText(canvas, str(k), (tx - 4, ty + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, c_num, 1, cv2.LINE_AA)

        # Center X crosshair
        cv2.line(canvas, (cx - 7, cy), (cx + 7, cy), (40, 40, 40), 1, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy - 7), (cx, cy + 7), (40, 40, 40), 1, cv2.LINE_AA)

        # Corner clamps or text badges (like Image 2 "10PCS", clips at top corners)
        if has_table_bg or random.random() < 0.35:
            # Top-left and top-right black clamps
            clamp_sz = random.randint(18, 30)
            cv2.rectangle(canvas, (15, 15), (15 + clamp_sz, 15 + clamp_sz // 2), (25, 25, 25), -1)
            cv2.rectangle(canvas, (size - 15 - clamp_sz, 15), (size - 15, 15 + clamp_sz // 2), (25, 25, 25), -1)

            # Random promo text (e.g. "10PCS", "TARGET")
            if random.random() < 0.40:
                cv2.putText(canvas, "10PCS", (25, size // 3), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.putText(canvas, "10PCS", (25, size // 3), cv2.FONT_HERSHEY_DUPLEX, 1.1, (30, 30, 240), 2, cv2.LINE_AA)

        info = {"center": center, "outer_radius": outer_r, "ring_radii": ring_radii, "style": style}
        return canvas, center, outer_r, info

    def _render_realistic_bullet_hole(
        self,
        img: np.ndarray,
        center_x: int,
        center_y: int,
        radius: float,
        is_on_dark_bg: bool
    ) -> Tuple[int, int, int, int]:
        """
        Renders photorealistic torn bullet puncture:
        - Outer lead-wipe ring (grey smudge)
        - Torn bevel ring
        - Dark void / inner shadow
        - Fiber tearing spikes
        """
        r = int(radius)
        h, w = img.shape[:2]

        # 1. Lead wipe / graphite burn ring (always present around bullet impact)
        wipe_r = int(r * random.uniform(1.25, 1.45))
        wipe_color = (random.randint(55, 80), random.randint(55, 80), random.randint(55, 80))
        cv2.circle(img, (center_x, center_y), wipe_r, wipe_color, 2, cv2.LINE_AA)

        # 2. Torn paper bevel:
        # If background is dark/black, the tear exposes grey/light fibers
        # If background is light/colored, the bevel is dark grey
        if is_on_dark_bg:
            bevel_color = (random.randint(190, 235), random.randint(190, 235), random.randint(190, 235))
            inner_color = (random.randint(15, 30), random.randint(15, 30), random.randint(15, 30))
        else:
            bevel_color = (random.randint(40, 75), random.randint(40, 75), random.randint(40, 75))
            inner_color = (random.randint(8, 20), random.randint(8, 20), random.randint(8, 20))

        # Irregular torn polygon
        n_pts = random.randint(12, 18)
        pts = []
        for i in range(n_pts):
            ang = 2 * math.pi * i / n_pts
            dist = r * random.uniform(0.85, 1.18)
            pts.append([int(center_x + dist * math.cos(ang)), int(center_y + dist * math.sin(ang))])

        cv2.fillPoly(img, np.array([pts], dtype=np.int32), bevel_color, cv2.LINE_AA)

        # Dark void core
        core_r = max(2, int(r * 0.70))
        cv2.circle(img, (center_x, center_y), core_r, inner_color, -1, cv2.LINE_AA)

        # Radial fiber fraying spikes
        for _ in range(random.randint(4, 9)):
            fang = random.uniform(0, 2 * math.pi)
            fr = r * random.uniform(0.95, 1.35)
            fx = int(center_x + fr * math.cos(fang))
            fy = int(center_y + fr * math.sin(fang))
            cv2.line(img, (center_x, center_y), (fx, fy), wipe_color, 1, cv2.LINE_AA)

        margin = int(r * 0.3)
        return (
            max(0, center_x - r - margin),
            max(0, center_y - r - margin),
            min(w - 1, center_x + r + margin),
            min(h - 1, center_y + r + margin)
        )

    def generate_sample(self, split: str, idx: int) -> None:
        """Generates a diverse synthetic sample with accurate bounding boxes."""
        size = self.img_size
        styles = ["white_center_mono", "multicolor", "classic_black_bullseye"]
        style = random.choice(styles)

        img, (cx, cy), outer_r, info = self._draw_target_canvas(size, style)
        bboxes: List[Tuple[float, float, float, float]] = []

        # 12% HARD NEGATIVE samples (clean target with NO holes, but with numbers, logos, clamps)
        is_hard_negative = (random.random() < 0.12)

        if not is_hard_negative:
            num_shots = random.randint(2, 14)
            shot_coords = []

            for _ in range(num_shots):
                # Distribute shots across the target (some center, some outer)
                ang = random.uniform(0, 2 * math.pi)
                dist = outer_r * (random.uniform(0.05, 0.95) ** 0.8)
                hx = int(cx + dist * math.cos(ang))
                hy = int(cy + dist * math.sin(ang))
                shot_coords.append((hx, hy))

            # Simulate occasional double-taps (closely overlapping pairs like in Image 1 & 2)
            if len(shot_coords) >= 2 and random.random() < 0.50:
                base_s = shot_coords[0]
                shot_coords[1] = (base_s[0] + random.randint(-8, 8), base_s[1] + random.randint(-8, 8))

            for (hx, hy) in shot_coords:
                # Check background tone under hole
                h_dist = math.hypot(hx - cx, hy - cy)
                is_dark = False
                if style == "white_center_mono":
                    # Rings 6-8 are dark (between ring 8 and ring 6)
                    r6 = info["ring_radii"][6]
                    r8 = info["ring_radii"][8]
                    is_dark = (r8 <= h_dist <= r6)
                elif style == "multicolor":
                    r3 = info["ring_radii"][3]
                    r4 = info["ring_radii"][4]
                    is_dark = (r4 <= h_dist <= r3)
                else:
                    is_dark = (h_dist <= info["ring_radii"][7])

                bullet_r = random.uniform(4.5, 9.5)
                xmin, ymin, xmax, ymax = self._render_realistic_bullet_hole(img, hx, hy, bullet_r, is_dark)

                norm_cx = (xmin + xmax) / (2.0 * size)
                norm_cy = (ymin + ymax) / (2.0 * size)
                norm_w = (xmax - xmin) / float(size)
                norm_h = (ymax - ymin) / float(size)
                bboxes.append((norm_cx, norm_cy, norm_w, norm_h))

        # Photometric augmentations
        if random.random() < 0.40:
            illum = random.uniform(0.85, 1.15)
            img = np.clip(img.astype(np.float32) * illum, 0, 255).astype(np.uint8)

        # Save image and YOLO labels
        img_filename = f"syn_{split}_{idx:04d}.jpg"
        lbl_filename = f"syn_{split}_{idx:04d}.txt"

        cv2.imwrite(str(self.output_dir / "images" / split / img_filename), img, [cv2.IMWRITE_JPEG_QUALITY, 93])

        with open(self.output_dir / "labels" / split / lbl_filename, "w") as f:
            for bx, by, bw, bh in bboxes:
                f.write(f"0 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")

    def incorporate_user_reference_targets(self):
        """
        Embeds the user's uploaded real-world target images (with ground-truth bullet holes)
        augmented via rotations, crops, and flips directly into the training & val sets.
        """
        ref1_path = Path("C:/Users/Advan/.gemini/antigravity/brain/3130756e-b1e5-42fd-88b6-c01cf73c6b69/.user_uploaded/media_1791509690040_bdc91957.jpg")
        ref2_path = Path("C:/Users/Advan/.gemini/antigravity/brain/3130756e-b1e5-42fd-88b6-c01cf73c6b69/.user_uploaded/media_1791509690263_d59002ad.png")
        ref3_path = Path("C:/Users/Advan/.gemini/antigravity/brain/3130756e-b1e5-42fd-88b6-c01cf73c6b69/.user_uploaded/media_1791513965088_0efb5b1c.jpg")
        ref4_path = Path("C:/Users/Advan/.gemini/antigravity/brain/3130756e-b1e5-42fd-88b6-c01cf73c6b69/.user_uploaded/media_1791513965095_50730071.jpg")

        # Ground truth hole centroids and radii for Image 1 (360x360)
        holes_img1 = [
            (191, 56, 7), (198, 62, 7),     # top double tap
            (107, 107, 7), (119, 114, 7),   # top-left double tap
            (112, 252, 7), (124, 256, 7),   # bot-left double tap
            (52, 224, 7),                   # mid-left
            (83, 310, 7),                   # far bot-left
            (238, 263, 7),                  # bot-right
            (277, 266, 7),                  # far bot-right
            (271, 112, 7),                  # top-right
            (326, 128, 7),                  # far right
            (161, 181, 7),                  # ring 9 left
            (180, 180, 7),                  # center 10
            (207, 173, 7),                  # ring 9 right
            (136, 198, 7),                  # black ring tear (left)
            (217, 211, 7),                  # black ring tear (right)
        ]

        # Ground truth hole centroids and radii for Image 2 (800x800)
        holes_img2 = [
            (213, 203, 14), # black ring 4
            (353, 309, 13), # red ring 7
            (450, 366, 12), # yellow ring 9
            (420, 380, 11), # center 10
            (310, 512, 13), # blue ring 6 (left)
            (506, 501, 14), # blue ring 6 (right)
            (688, 304, 13), # white ring 2
        ]

        # Ground truth hole centroids and radii for Image 3 (297x235)
        holes_img3 = [
            (105, 45, 6),   # ring 4 top-left
            (117, 50, 6),   # ring 5 top-left
            (181, 80, 6),   # ring 7 top-right
            (130, 83, 6),   # ring 8 top
            (157, 107, 6),  # ring 9 top
            (178, 114, 6),  # ring 9 right
            (122, 140, 6),  # ring 8 bottom-left
            (143, 135, 6),  # ring 9 bottom
            (105, 153, 6),  # ring 7 bottom-left
            (163, 185, 6),  # ring 7 bottom
        ]

        # Ground truth centroids for Image 4 (1024x768 - yellow target with pasters)
        # Includes both covered paster centers and open bullet holes so all impact markers are captured
        holes_img4 = [
            (197, 141, 18), (318, 97, 18), (204, 326, 18),
            (385, 386, 18), (439, 433, 18), (473, 427, 18), (445, 507, 18),
            (334, 420, 18), (403, 508, 18), (480, 584, 18), (496, 568, 18),
            (509, 583, 18), (530, 550, 18), (381, 577, 18), (398, 590, 18),
            (389, 614, 18), (568, 588, 18), (601, 552, 18), (628, 483, 18),
            (663, 485, 18), (768, 505, 18), (489, 684, 18), (645, 623, 18),
            (706, 810, 18), (733, 826, 18), (785, 715, 18), (892, 729, 18),
            (144, 626, 18), (186, 588, 18), (204, 626, 18), (266, 608, 18),
            (350, 693, 18), (273, 508, 12), (304, 504, 12), (370, 485, 12),
            (407, 532, 12), (439, 537, 12), (506, 529, 12),
        ]

        ref_list = []
        if ref1_path.exists():
            ref_list.append((ref1_path, holes_img1, 360, 360))
        if ref2_path.exists():
            ref_list.append((ref2_path, holes_img2, 800, 800))
        if ref3_path.exists():
            ref_list.append((ref3_path, holes_img3, 297, 235))
        if ref4_path.exists():
            ref_list.append((ref4_path, holes_img4, 768, 1024))

        # Generate augmented variations (0°, 90°, 180°, 270°, horizontal flip)
        for ref_idx, (path, holes, orig_w, orig_h) in enumerate(ref_list, start=1):
            img = cv2.imread(str(path))
            for aug_i, rot in enumerate([0, 90, 180, 270]):
                for flip in [False, True]:
                    aug_img = img.copy()
                    if rot == 90:
                        aug_img = cv2.rotate(aug_img, cv2.ROTATE_90_CLOCKWISE)
                    elif rot == 180:
                        aug_img = cv2.rotate(aug_img, cv2.ROTATE_180)
                    elif rot == 270:
                        aug_img = cv2.rotate(aug_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    if flip:
                        aug_img = cv2.flip(aug_img, 1)

                    aug_h, aug_w = aug_img.shape[:2]
                    bboxes = []

                    for (hx, hy, hr) in holes:
                        # Transform coordinates
                        x, y = hx, hy
                        if rot == 90:
                            x, y = orig_h - 1 - hy, hx
                        elif rot == 180:
                            x, y = orig_w - 1 - hx, orig_h - 1 - hy
                        elif rot == 270:
                            x, y = hy, orig_w - 1 - hx

                        if flip:
                            x = aug_w - 1 - x

                        margin = hr * 0.35
                        bw = (hr * 2 + margin) / aug_w
                        bh = (hr * 2 + margin) / aug_h
                        cx_norm = x / aug_w
                        cy_norm = y / aug_h
                        bboxes.append((cx_norm, cy_norm, bw, bh))

                    split = "val" if (aug_i == 3 and not flip) else "train"
                    fname = f"ref_{ref_idx}_rot{rot}_flip{int(flip)}"
                    cv2.imwrite(str(self.output_dir / "images" / split / f"{fname}.jpg"), aug_img)
                    with open(self.output_dir / "labels" / split / f"{fname}.txt", "w") as f:
                        for bx, by, bw, bh in bboxes:
                            f.write(f"0 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")

    def incorporate_real_downloaded_dataset(self, real_raw_dir: str = "training/real_raw"):
        """
        Integrates real-world target images downloaded from the internet into the YOLO dataset splits.
        Splits ~85% into train and ~15% into val.
        """
        raw_path = Path(real_raw_dir)
        img_dir = raw_path / "images"
        lbl_dir = raw_path / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"[DatasetGenerator] Real dataset directory {real_raw_dir} not found. Skipping.")
            return

        image_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        if not image_files:
            return

        import shutil
        train_count = 0
        val_count = 0

        for idx, img_file in enumerate(image_files):
            stem = img_file.stem
            lbl_file = lbl_dir / f"{stem}.txt"
            if not lbl_file.exists():
                continue

            # Assign 1 out of 7 (~14%) to val, rest to train
            split = "val" if (idx % 7 == 0) else "train"

            dest_img = self.output_dir / "images" / split / f"real_{stem}.jpg"
            dest_lbl = self.output_dir / "labels" / split / f"real_{stem}.txt"

            shutil.copy(img_file, dest_img)
            shutil.copy(lbl_file, dest_lbl)

            if split == "train":
                train_count += 1
            else:
                val_count += 1

        print(f"[DatasetGenerator] Successfully integrated {train_count + val_count} real internet targets ({train_count} train, {val_count} val)!")

    def generate(self, train_samples: int = 250, val_samples: int = 40) -> Path:
        """Full dataset compilation: synthetic multi-style + real internet targets + augmented user targets."""
        self._setup_directories()
        print(f"Generating {train_samples} multi-style synthetic training samples...")
        for i in range(train_samples):
            self.generate_sample("train", i)

        print(f"Generating {val_samples} validation samples...")
        for i in range(val_samples):
            self.generate_sample("val", i)

        # User reference incorporation removed to prevent test set data leakage
        # self.incorporate_user_reference_targets()
        self.incorporate_real_downloaded_dataset()
        print(f"Dataset successfully compiled at: {self.output_dir}")
        return self.output_dir


if __name__ == "__main__":
    gen = SyntheticTargetDatasetGenerator()
    gen.generate(train_samples=250, val_samples=40)
