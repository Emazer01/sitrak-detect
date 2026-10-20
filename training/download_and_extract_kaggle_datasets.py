"""
download_and_extract_kaggle_datasets.py
Downloads and inspects the 3 target shooting datasets requested by the user:
1. pfrmusic/yolo-shot-datasets-with-annots (~7.8 MB)
2. karimhossam02/computer-vision-shooting-target-score-dataset (~103 MB)
3. bryanb/standup-targets (~176 MB)
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path


DATASETS = [
    {
        "id": "bobsuna",
        "name": "datasets-of-bullet-hole-on-the-target-paper",
        "url": "https://www.kaggle.com/api/v1/datasets/download/bobsuna/datasets-of-bullet-hole-on-the-target-paper",
        "dest": Path("training/kaggle_datasets/bobsuna")
    },
    {
        "id": "qiyan527",
        "name": "target-paper-images-with-bullet-holes",
        "url": "https://www.kaggle.com/api/v1/datasets/download/qiyan527/target-paper-images-with-bullet-holes",
        "dest": Path("training/kaggle_datasets/qiyan527")
    },
    {
        "id": "pfrmusic",
        "name": "yolo-shot-datasets-with-annots",
        "url": "https://www.kaggle.com/api/v1/datasets/download/pfrmusic/yolo-shot-datasets-with-annots",
        "dest": Path("training/kaggle_datasets/pfrmusic")
    },
    {
        "id": "karimhossam02",
        "name": "computer-vision-shooting-target-score-dataset",
        "url": "https://www.kaggle.com/api/v1/datasets/download/karimhossam02/computer-vision-shooting-target-score-dataset",
        "dest": Path("training/kaggle_datasets/karimhossam02")
    },
    {
        "id": "bryanb",
        "name": "standup-targets",
        "url": "https://www.kaggle.com/api/v1/datasets/download/bryanb/standup-targets",
        "dest": Path("training/kaggle_datasets/bryanb")
    }
]


def download_file(url: str, dest_file: Path) -> bool:
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Downloader] Downloading {url} -> {dest_file}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    with urllib.request.urlopen(req, timeout=120) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB
        
        with open(dest_file, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100.0
                    mb = downloaded / (1024 * 1024)
                    tot_mb = total_size / (1024 * 1024)
                    print(f"\r  Progress: {mb:.1f}/{tot_mb:.1f} MB ({pct:.1f}%)", end="", flush=True)
    print("\n[Downloader] Download completed successfully!")
    return True


def extract_zip(zip_file: Path, extract_to: Path):
    print(f"[Extractor] Extracting {zip_file} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_file, "r") as zf:
        zf.extractall(extract_to)
    print(f"[Extractor] Extracted {len(zf.namelist())} files.")


def main():
    base_zip_dir = Path("training/kaggle_zips")
    base_zip_dir.mkdir(parents=True, exist_ok=True)
    
    for item in DATASETS:
        print(f"\n==========================================")
        print(f"Processing: {item['id']} / {item['name']}")
        print(f"==========================================")
        zip_path = base_zip_dir / f"{item['id']}_{item['name']}.zip"
        
        if not zip_path.exists() or zip_path.stat().st_size < 1000:
            download_file(item["url"], zip_path)
        else:
            print(f"Using already downloaded zip: {zip_path} ({zip_path.stat().st_size / (1024*1024):.1f} MB)")
            
        if not item["dest"].exists() or not any(item["dest"].iterdir()):
            extract_zip(zip_path, item["dest"])
        else:
            print(f"Already extracted at: {item['dest']}")

    print("\nAll 3 Kaggle datasets downloaded and extracted successfully!")


if __name__ == "__main__":
    main()

