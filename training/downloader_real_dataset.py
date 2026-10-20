"""
downloader_real_dataset.py
Downloads real annotated shooting target images from the public internet (cdmstrong/bullet_yolov5 VOC2007)
and converts Pascal VOC bounding boxes into standardized YOLO annotation format.
"""

import os
import sys
import json
import time
import random
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np


def download_and_convert_voc_samples(
    output_dir: str = "training/real_raw",
    max_samples: int = 120,
    max_workers: int = 8
) -> int:
    """
    Downloads real shooting target images + annotations from GitHub repository
    and converts them to YOLO format.
    """
    dest_path = Path(output_dir)
    images_dir = dest_path / "images"
    labels_dir = dest_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print("[Downloader] Fetching GitHub repository tree for cdmstrong/bullet_yolov5...")
    req = urllib.request.Request(
        "https://api.github.com/repos/cdmstrong/bullet_yolov5/git/trees/main?recursive=1",
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        tree = json.loads(resp.read().decode("utf-8"))

    ann_dict = {}
    img_dict = {}
    for item in tree.get("tree", []):
        p = item["path"]
        if p.startswith("VOC2007/Annotations/") and p.endswith(".xml"):
            name = Path(p).stem
            ann_dict[name] = p
        elif p.startswith("VOC2007/JPEGImages/") and p.lower().endswith((".jpg", ".jpeg", ".png")):
            name = Path(p).stem
            img_dict[name] = p

    common_names = sorted(list(set(ann_dict.keys()) & set(img_dict.keys())))
    print(f"[Downloader] Found {len(common_names)} real verified target pairs in repository.")

    # Evenly sample across the full repository to get diverse targets
    step = max(1, len(common_names) // max_samples)
    sampled_names = common_names[::step][:max_samples]
    print(f"[Downloader] Selected {len(sampled_names)} diverse samples for download...")

    def download_pair(sample_name: str) -> bool:
        xml_rel = ann_dict[sample_name]
        img_rel = img_dict[sample_name]

        xml_url = f"https://raw.githubusercontent.com/cdmstrong/bullet_yolov5/main/{xml_rel}"
        img_url = f"https://raw.githubusercontent.com/cdmstrong/bullet_yolov5/main/{img_rel}"

        try:
            # 1. Download XML
            with urllib.request.urlopen(xml_url, timeout=15) as r:
                xml_data = r.read().decode("utf-8")

            root = ET.fromstring(xml_data)
            size_elem = root.find("size")
            if size_elem is None:
                return False
            img_w = float(size_elem.find("width").text)
            img_h = float(size_elem.find("height").text)
            if img_w <= 0 or img_h <= 0:
                return False

            objects = root.findall("object")
            yolo_boxes = []
            for obj in objects:
                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)

                # Clamp
                xmin = max(0.0, min(xmin, img_w))
                xmax = max(0.0, min(xmax, img_w))
                ymin = max(0.0, min(ymin, img_h))
                ymax = max(0.0, min(ymax, img_h))

                bw = xmax - xmin
                bh = ymax - ymin
                if bw < 2.0 or bh < 2.0:
                    continue

                cx = xmin + bw / 2.0
                cy = ymin + bh / 2.0

                # Normalize to [0, 1]
                x_norm = cx / img_w
                y_norm = cy / img_h
                w_norm = bw / img_w
                h_norm = bh / img_h

                yolo_boxes.append((x_norm, y_norm, w_norm, h_norm))

            # 2. Download Image
            with urllib.request.urlopen(img_url, timeout=20) as r:
                img_data = r.read()

            # Save files
            img_dest = images_dir / f"{sample_name}.jpg"
            lbl_dest = labels_dir / f"{sample_name}.txt"

            with open(img_dest, "wb") as f:
                f.write(img_data)

            with open(lbl_dest, "w") as f:
                for (x, y, w, h) in yolo_boxes:
                    f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

            return True
        except Exception as e:
            # print(f"Error on {sample_name}: {e}")
            return False

    downloaded = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_pair, name): name for name in sampled_names}
        for future in as_completed(futures):
            if future.result():
                downloaded += 1
                if downloaded % 20 == 0:
                    print(f"  [Progress] Downloaded {downloaded}/{len(sampled_names)} real target images...")

    print(f"[Downloader] Completed! Successfully downloaded {downloaded} real target images with YOLO annotations.")
    return downloaded


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    download_and_convert_voc_samples(max_samples=count)

