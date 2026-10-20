import os
import random
import math
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np


class KrugerTargetDatasetGenerator:
    """
    Advanced Dataset Generator specifically engineered for Krüger 50-Foot Pistol Targets.
    Features:
    - Master Krüger Target (886x886)
    - Punctures across Cream Paper (rings 1-6) with graphite/lead wipe rings & torn bevels
    - Punctures inside Black Bullseye (rings 7-10) with torn white fiber halos
    - Border-straddling punctures (half cream, half black bullseye)
    - Realistic clusters (double-taps and 3-shot cloverleaf groupings)
    - Variable bullet calibers (.177/4.5mm, .22LR, 9mm, .38, .45 ACP)
    - Photometric gradient illumination (simulating indoor shooting range lighting)
    - Hard negative samples (clean target with numbers 1-10, text, and logos with 0 shots)
    """

    def __init__(
        self,
        master_image_path: str = "training/assets/kruger_master.webp",
        output_dir: str = "training/dataset_kruger",
        target_size: int = 416
    ):
        self.master_path = Path(master_image_path)
        self.output_dir = Path(output_dir)
        self.target_size = target_size

        if not self.master_path.exists():
            raise FileNotFoundError(f"Master Krüger target image not found at {self.master_path}")

        self.master_img = cv2.imread(str(self.master_path))
        if self.master_img is None:
            raise ValueError(f"Could not decode image at {self.master_path}")

        self.orig_h, self.orig_w = self.master_img.shape[:2]
        self.center_x = self.orig_w / 2.0
        self.center_y = self.orig_h / 2.0
        self.bullseye_radius = 148.6
        self.outer_ring_radius = 370.0

    def _setup_directories(self):
        """Prepares train/val directories and data.yaml."""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        for split in ["train", "val"]:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        yaml_content = f"""path: {self.output_dir.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: bullet_hole
"""
        with open(self.output_dir / "data.yaml", "w", encoding="utf-8") as f:
            f.write(yaml_content)

    def _render_bullet_hole(
        self,
        img: np.ndarray,
        hx: int,
        hy: int,
        radius: float,
        dist_to_center: float
    ) -> Tuple[int, int, int, int]:
        """
        Renders a photorealistic bullet hole puncture considering background context.
        """
        h, w = img.shape[:2]
        r = max(4, int(round(radius)))
        is_wadcutter = (random.random() < 0.45)

        # Context: Deep bullseye vs Border-straddling vs Cream paper
        is_deep_bullseye = dist_to_center <= (self.bullseye_radius - r * 0.5)
        is_straddling = abs(dist_to_center - self.bullseye_radius) < (r * 1.1)

        if is_deep_bullseye:
            # Puncture entirely within Black Bullseye:
            tear_ring_r = int(r * random.uniform(1.15, 1.35))
            tear_color = (
                random.randint(180, 230),
                random.randint(200, 240),
                random.randint(210, 245)
            )

            # Irregular white/cream paper fraying halo
            n_pts = random.randint(12, 18)
            pts = []
            for i in range(n_pts):
                ang = 2.0 * math.pi * i / n_pts
                dist = tear_ring_r * random.uniform(0.85, 1.20)
                px = int(hx + dist * math.cos(ang))
                py = int(hy + dist * math.sin(ang))
                pts.append([px, py])
            cv2.fillPoly(img, np.array([pts], dtype=np.int32), tear_color, cv2.LINE_AA)

            # Radial fiber spikes
            for _ in range(random.randint(4, 8)):
                ang = random.uniform(0, 2.0 * math.pi)
                fr = tear_ring_r * random.uniform(1.1, 1.45)
                fx = int(hx + fr * math.cos(ang))
                fy = int(hy + fr * math.sin(ang))
                cv2.line(img, (hx, hy), (fx, fy), tear_color, 1, cv2.LINE_AA)

            # Dark inner cavity void
            core_r = max(2, int(r * random.uniform(0.75, 0.90)))
            core_color = (random.randint(10, 28), random.randint(10, 28), random.randint(10, 28))
            cv2.circle(img, (hx, hy), core_r, core_color, -1, cv2.LINE_AA)

            box_r = tear_ring_r

        elif is_straddling:
            # Bullet hole straddles the boundary between black 7-ring and cream 6-ring
            wipe_r = int(r * random.uniform(1.20, 1.40))
            wipe_color = (random.randint(45, 75), random.randint(45, 75), random.randint(45, 75))
            fiber_color = (random.randint(190, 235), random.randint(205, 245), random.randint(215, 250))

            # Outer ring: mixture of wipe and fibers
            cv2.circle(img, (hx, hy), wipe_r, wipe_color, 2, cv2.LINE_AA)
            cv2.circle(img, (hx, hy), int(r * 1.15), fiber_color, 1, cv2.LINE_AA)

            # Dark core
            core_r = max(2, int(r * 0.80))
            core_color = (random.randint(8, 22), random.randint(8, 22), random.randint(8, 22))
            cv2.circle(img, (hx, hy), core_r, core_color, -1, cv2.LINE_AA)
            box_r = wipe_r

        else:
            # Puncture in Cream/Tan paper (rings 1-6):
            wipe_r = int(r * random.uniform(1.25, 1.50))
            wipe_color = (random.randint(35, 68), random.randint(35, 68), random.randint(35, 68))
            cv2.circle(img, (hx, hy), wipe_r, wipe_color, random.randint(1, 2), cv2.LINE_AA)

            bevel_r = int(r * random.uniform(0.95, 1.15))
            bevel_color = (random.randint(22, 42), random.randint(22, 42), random.randint(22, 42))
            if is_wadcutter:
                cv2.circle(img, (hx, hy), bevel_r, bevel_color, -1, cv2.LINE_AA)
            else:
                n_pts = random.randint(10, 16)
                pts = []
                for i in range(n_pts):
                    ang = 2.0 * math.pi * i / n_pts
                    dist = bevel_r * random.uniform(0.88, 1.15)
                    pts.append([int(hx + dist * math.cos(ang)), int(hy + dist * math.sin(ang))])
                cv2.fillPoly(img, np.array([pts], dtype=np.int32), bevel_color, cv2.LINE_AA)

            # Core void
            core_r = max(2, int(r * 0.75))
            core_color = (random.randint(5, 16), random.randint(5, 16), random.randint(5, 16))
            cv2.circle(img, (hx, hy), core_r, core_color, -1, cv2.LINE_AA)

            for _ in range(random.randint(3, 7)):
                ang = random.uniform(0, 2.0 * math.pi)
                fr = r * random.uniform(1.1, 1.45)
                fx = int(hx + fr * math.cos(ang))
                fy = int(hy + fr * math.sin(ang))
                cv2.line(img, (hx, hy), (fx, fy), wipe_color, 1, cv2.LINE_AA)

            box_r = wipe_r

        margin = int(box_r * 0.28) + 2
        xmin = max(0, hx - box_r - margin)
        ymin = max(0, hy - box_r - margin)
        xmax = min(w - 1, hx + box_r + margin)
        ymax = min(h - 1, hy + box_r + margin)
        return (xmin, ymin, xmax, ymax)

    def generate_sample(self, split: str, idx: int) -> None:
        """Generates one training or validation sample with augmentations and YOLO labels."""
        img = self.master_img.copy()
        h, w = img.shape[:2]

        # 1. Decide if Hard Negative (Target with NO holes, clean paper with numbers & text)
        is_hard_negative = (random.random() < 0.14)
        bboxes = []

        if not is_hard_negative:
            # Number of shots (1 to 14)
            num_shots = random.choices([1, 2, 3, 5, 8, 10, 12, 14], weights=[0.08, 0.12, 0.22, 0.26, 0.16, 0.10, 0.04, 0.02])[0]

            shot_centers = []
            group_offset_x = random.gauss(0, 28.0)
            group_offset_y = random.gauss(0, 28.0)

            for _ in range(num_shots):
                dispersion = abs(random.gauss(0, 105.0))
                dispersion = min(dispersion, self.outer_ring_radius * 1.05)
                ang = random.uniform(0, 2.0 * math.pi)

                hx = int(round(self.center_x + group_offset_x + dispersion * math.cos(ang)))
                hy = int(round(self.center_y + group_offset_y + dispersion * math.sin(ang)))

                hx = max(25, min(w - 25, hx))
                hy = max(25, min(h - 25, hy))
                shot_centers.append((hx, hy))

            # Simulate clusters:
            # A. Double-tap (2 touching/overlapping holes) in 30% of multi-shot targets
            if len(shot_centers) >= 2 and random.random() < 0.30:
                bx, by = shot_centers[0]
                dt_dist = random.uniform(7.0, 16.0)
                dt_ang = random.uniform(0, 2.0 * math.pi)
                shot_centers[1] = (
                    int(round(bx + dt_dist * math.cos(dt_ang))),
                    int(round(by + dt_dist * math.sin(dt_ang)))
                )

            # B. 3-shot Cloverleaf Group in 18% of targets with >= 3 shots
            if len(shot_centers) >= 3 and random.random() < 0.18:
                cx_clover, cy_clover = shot_centers[0]
                clover_rad = random.uniform(10.0, 18.0)
                for c_idx, angle_deg in enumerate([120, 240], start=1):
                    rad = math.radians(angle_deg + random.uniform(-15, 15))
                    shot_centers[c_idx] = (
                        int(round(cx_clover + clover_rad * math.cos(rad))),
                        int(round(cy_clover + clover_rad * math.sin(rad)))
                    )

            # Render each bullet hole
            for (hx, hy) in shot_centers:
                dist_to_center = math.hypot(hx - self.center_x, hy - self.center_y)
                bullet_r = random.uniform(6.0, 12.5)  # covers .177 up to .45
                xmin, ymin, xmax, ymax = self._render_bullet_hole(img, hx, hy, bullet_r, dist_to_center)
                bboxes.append((xmin, ymin, xmax, ymax))

        # 2. Geometric & Photometric Augmentations
        aug_type = random.choice(["none", "rotate_90", "rotate_180", "rotate_270", "slight_rotate", "perspective"])

        if aug_type == "rotate_90":
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            bboxes = [(h - ymax, xmin, h - ymin, xmax) for (xmin, ymin, xmax, ymax) in bboxes]
        elif aug_type == "rotate_180":
            img = cv2.rotate(img, cv2.ROTATE_180)
            bboxes = [(w - xmax, h - ymax, w - xmin, h - ymin) for (xmin, ymin, xmax, ymax) in bboxes]
        elif aug_type == "rotate_270":
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            bboxes = [(ymin, w - xmax, ymax, w - xmin) for (xmin, ymin, xmax, ymax) in bboxes]
        elif aug_type == "slight_rotate":
            angle = random.uniform(-7.0, 7.0)
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            cx_f, cy_f = w / 2.0, h / 2.0
            new_bboxes = []
            for (xmin, ymin, xmax, ymax) in bboxes:
                bcx, bcy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
                bw, bh = xmax - xmin, ymax - ymin
                rx = cos_a * (bcx - cx_f) - sin_a * (bcy - cy_f) + cx_f
                ry = sin_a * (bcx - cx_f) + cos_a * (bcy - cy_f) + cy_f
                new_bboxes.append((rx - bw / 2.0, ry - bh / 2.0, rx + bw / 2.0, ry + bh / 2.0))
            bboxes = new_bboxes

        # Directional lighting gradient (simulates overhead indoor range lighting)
        if random.random() < 0.55:
            grad_y = np.linspace(random.uniform(0.85, 1.0), random.uniform(1.0, 1.15), h)[:, None]
            img = np.clip(img.astype(np.float32) * grad_y, 0, 255).astype(np.uint8)

        # Contrast and brightness jitter
        if random.random() < 0.60:
            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.88, 1.12)
            img = np.clip(img.astype(np.float32) * contrast * brightness, 0, 255).astype(np.uint8)

        # Resize to target training size (416x416)
        curr_h, curr_w = img.shape[:2]
        img_resized = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)

        scale_x = self.target_size / float(curr_w)
        scale_y = self.target_size / float(curr_h)

        # Normalized YOLO labels
        yolo_labels = []
        for (xmin, ymin, xmax, ymax) in bboxes:
            r_xmin = max(0.0, xmin * scale_x)
            r_ymin = max(0.0, ymin * scale_y)
            r_xmax = min(float(self.target_size), xmax * scale_x)
            r_ymax = min(float(self.target_size), ymax * scale_y)

            bw = r_xmax - r_xmin
            bh = r_ymax - r_ymin
            if bw <= 3.0 or bh <= 3.0:
                continue

            bcx = (r_xmin + r_xmax) / 2.0
            bcy = (r_ymin + r_ymax) / 2.0

            norm_cx = bcx / float(self.target_size)
            norm_cy = bcy / float(self.target_size)
            norm_w = bw / float(self.target_size)
            norm_h = bh / float(self.target_size)
            yolo_labels.append((norm_cx, norm_cy, norm_w, norm_h))

        img_name = f"kruger_{split}_{idx:04d}.jpg"
        lbl_name = f"kruger_{split}_{idx:04d}.txt"

        cv2.imwrite(str(self.output_dir / "images" / split / img_name), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 94])
        with open(self.output_dir / "labels" / split / lbl_name, "w", encoding="utf-8") as f:
            for (cx, cy, bw, bh) in yolo_labels:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    def generate(self, train_count: int = 600, val_count: int = 120):
        """Generates full train and val dataset."""
        print(f"[KrugerDatasetGenerator] Initializing directories at {self.output_dir.resolve()}...")
        self._setup_directories()

        print(f"[KrugerDatasetGenerator] Generating {train_count} training samples...")
        for i in range(train_count):
            self.generate_sample("train", i)
            if (i + 1) % 150 == 0 or (i + 1) == train_count:
                print(f"  -> Generated {i + 1}/{train_count} train samples")

        print(f"[KrugerDatasetGenerator] Generating {val_count} validation samples...")
        for i in range(val_count):
            self.generate_sample("val", i)
            if (i + 1) % 60 == 0 or (i + 1) == val_count:
                print(f"  -> Generated {i + 1}/{val_count} val samples")

        print(f"[SUCCESS] Expanded Kruger Target Dataset ready at {self.output_dir.resolve()}")


if __name__ == "__main__":
    generator = KrugerTargetDatasetGenerator()
    generator.generate(train_count=600, val_count=120)
