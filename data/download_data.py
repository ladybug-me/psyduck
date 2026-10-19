#!/usr/bin/env python3
"""
Psyduck - Fashion Dataset Downloader
Automates downloading the zara2099/fashion-style-image-dataset via kagglehub
and places data.csv and images into the correct data/ paths.
"""

import os
import sys
import shutil
import glob
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
CSV_DEST = os.path.join(BASE_DIR, "data.csv")
DATABASE_DIR = os.path.join(BASE_DIR, "database")


def check_existing_data():
    """Check if data.csv and database/ directory are already populated."""
    csv_ok = os.path.exists(CSV_DEST) and os.path.getsize(CSV_DEST) > 1024
    img_count = 0
    if os.path.isdir(DATABASE_DIR):
        img_count = len(os.listdir(DATABASE_DIR))
    db_ok = img_count > 0
    return csv_ok, db_ok, img_count


def organize_downloaded_data(download_path):
    """Organize downloaded dataset files into data/data.csv and data/database/."""
    print(f"[INFO] Organizing downloaded files from: {download_path}")
    os.makedirs(DATABASE_DIR, exist_ok=True)

    # 1. Locate data.csv
    csv_candidates = glob.glob(os.path.join(download_path, "**", "data.csv"), recursive=True)
    if not csv_candidates:
        csv_candidates = glob.glob(os.path.join(download_path, "*.csv"))

    if csv_candidates:
        src_csv = csv_candidates[0]
        if os.path.abspath(src_csv) != os.path.abspath(CSV_DEST):
            print(f"[INFO] Placing CSV: {src_csv} -> {CSV_DEST}")
            shutil.copy2(src_csv, CSV_DEST)
    else:
        if not os.path.exists(CSV_DEST):
            print("[WARNING] data.csv not found in downloaded files.")

    # 2. Locate image files
    found_images = False
    for sub in ["database", "images", "img"]:
        candidate_dir = os.path.join(download_path, sub)
        if os.path.isdir(candidate_dir):
            print(f"[INFO] Moving image directory contents ({candidate_dir}) -> {DATABASE_DIR}...")
            for fname in os.listdir(candidate_dir):
                src_file = os.path.join(candidate_dir, fname)
                dst_file = os.path.join(DATABASE_DIR, fname)
                if not os.path.exists(dst_file):
                    shutil.move(src_file, dst_file)
            found_images = True
            break

    if not found_images:
        jpgs = glob.glob(os.path.join(download_path, "*.jpg"))
        if jpgs:
            print(f"[INFO] Moving {len(jpgs)} image files -> {DATABASE_DIR}...")
            for src_file in jpgs:
                dst_file = os.path.join(DATABASE_DIR, os.path.basename(src_file))
                if not os.path.exists(dst_file):
                    shutil.move(src_file, dst_file)
            found_images = True

    total_images = len(os.listdir(DATABASE_DIR)) if os.path.exists(DATABASE_DIR) else 0
    print(f"[SUCCESS] Data files ready: {CSV_DEST} and {DATABASE_DIR} ({total_images} images).")


def download(force=False):
    csv_ok, db_ok, img_count = check_existing_data()
    if csv_ok and db_ok and not force:
        print(f"[INFO] Dataset is already present in {BASE_DIR}:")
        print(f"  - data.csv: Found ({os.path.getsize(CSV_DEST):,} bytes)")
        print(f"  - database/: Found ({img_count:,} images)")
        print("Use --force if you wish to re-download.")
        return

    try:
        import kagglehub
    except ImportError:
        print("[ERROR] 'kagglehub' is required to download data. Install dependencies first.")
        sys.exit(1)

    print("[INFO] Downloading 'zara2099/fashion-style-image-dataset' via kagglehub...")
    try:
        path = kagglehub.dataset_download("zara2099/fashion-style-image-dataset")
        print(f"[SUCCESS] KaggleHub download finished: {path}")
        organize_downloaded_data(path)
    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download and organize Psyduck fashion dataset.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if dataset already exists")
    args = parser.parse_args()
    download(force=args.force)


if __name__ == "__main__":
    main()
