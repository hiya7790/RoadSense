"""
prepare_dataset.py — Clean and reorganize RTK dataset for RoadSense

Maps RTK categories to our 4 classes:
    asphaltGood + asphaltRegular → smooth
    pavedRegular + unpavedRegular → gravel
    asphaltBad + pavedBad + unpavedBad → pothole
    (synthetic from smooth via augmentation) → wet

Cleaning steps:
    1. Verify every image can be opened by OpenCV
    2. Remove corrupted / zero-byte / unreadable files
    3. Resize and copy to data/raw/{smooth,gravel,pothole,wet}
    4. Generate synthetic "wet" images from smooth using rain augmentation
"""

import os
import sys
import shutil
import cv2
import numpy as np
from tqdm import tqdm

try:
    import albumentations as A
except ImportError:
    print("albumentations not installed. Run: pip install albumentations")
    sys.exit(1)


# RTK → RoadSense class mapping
CLASS_MAPPING = {
    os.path.join("asphalt", "asphaltGood"): "smooth",
    os.path.join("asphalt", "asphaltRegular"): "smooth",
    os.path.join("asphalt", "asphaltBad"): "pothole",
    os.path.join("paved", "pavedRegular"): "gravel",
    os.path.join("paved", "pavedBad"): "pothole",
    os.path.join("upaved", "unpavedRegular"): "gravel",
    os.path.join("upaved", "unpavedBad"): "pothole",
}

RTK_DIR = "RTK_Dataset"
OUTPUT_DIR = os.path.join("data", "raw")
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Augmentation pipeline for generating synthetic "wet" images
WET_AUGMENTATION = A.Compose([
    A.RandomRain(
        brightness_coefficient=0.8,
        drop_width=2,
        blur_value=3,
        p=1.0,
    ),
    A.RandomBrightnessContrast(
        brightness_limit=(-0.2, 0.0),
        contrast_limit=(-0.1, 0.1),
        p=0.8,
    ),
    A.GaussNoise(p=0.5),
])


def is_valid_image(filepath):
    """Check if a file is a valid, readable image."""
    try:
        if os.path.getsize(filepath) == 0:
            return False
        img = cv2.imread(filepath)
        if img is None:
            return False
        if img.shape[0] < 10 or img.shape[1] < 10:
            return False
        return True
    except Exception:
        return False


def clean_and_copy():
    """Clean RTK dataset and copy to data/raw/ with our class structure."""
    # Create output directories
    for cls in ["smooth", "gravel", "pothole", "wet"]:
        os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

    stats = {"total": 0, "valid": 0, "corrupted": 0, "per_class": {}}
    class_counters = {"smooth": 0, "gravel": 0, "pothole": 0}

    for rtk_subdir, target_class in CLASS_MAPPING.items():
        src_dir = os.path.join(RTK_DIR, rtk_subdir)
        if not os.path.exists(src_dir):
            print(f"  WARNING: {src_dir} not found, skipping.")
            continue

        files = [f for f in os.listdir(src_dir)
                 if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]

        print(f"\n  {rtk_subdir} -> {target_class} ({len(files)} files)")

        valid_count = 0
        corrupt_count = 0

        for fname in tqdm(files, desc=f"    Cleaning {rtk_subdir}"):
            filepath = os.path.join(src_dir, fname)
            stats["total"] += 1

            if not is_valid_image(filepath):
                corrupt_count += 1
                stats["corrupted"] += 1
                continue

            # Copy with unique name to avoid collisions
            class_counters[target_class] += 1
            ext = os.path.splitext(fname)[1].lower()
            new_name = f"{target_class}_{class_counters[target_class]:05d}{ext}"
            dst_path = os.path.join(OUTPUT_DIR, target_class, new_name)
            shutil.copy2(filepath, dst_path)

            valid_count += 1
            stats["valid"] += 1

        stats["per_class"][f"{rtk_subdir} -> {target_class}"] = {
            "valid": valid_count, "corrupted": corrupt_count
        }
        print(f"    ✓ {valid_count} valid, ✗ {corrupt_count} corrupted")

    return stats, class_counters


def generate_wet_images(num_target=400):
    """Generate synthetic 'wet' images from smooth images using rain augmentation."""
    smooth_dir = os.path.join(OUTPUT_DIR, "smooth")
    wet_dir = os.path.join(OUTPUT_DIR, "wet")

    smooth_files = [f for f in os.listdir(smooth_dir)
                    if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]

    if not smooth_files:
        print("  No smooth images to generate wet images from!")
        return 0

    print(f"\n  Generating {num_target} synthetic wet images from {len(smooth_files)} smooth images...")

    count = 0
    idx = 0
    while count < num_target:
        fname = smooth_files[idx % len(smooth_files)]
        filepath = os.path.join(smooth_dir, fname)

        img = cv2.imread(filepath)
        if img is None:
            idx += 1
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = WET_AUGMENTATION(image=img_rgb)["image"]
        aug_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        out_name = f"wet_{count + 1:05d}.jpg"
        cv2.imwrite(os.path.join(wet_dir, out_name), aug_bgr)

        count += 1
        idx += 1

    print(f"    ✓ Generated {count} wet images")
    return count


def main():
    print("=" * 60)
    print("  RoadSense — Dataset Preparation")
    print("=" * 60)

    if not os.path.exists(RTK_DIR):
        print(f"\nERROR: '{RTK_DIR}' not found.")
        sys.exit(1)

    # Step 1: Clean and reorganize
    print("\n[1/3] Cleaning and reorganizing RTK dataset...")
    stats, class_counts = clean_and_copy()

    # Step 2: Generate wet images
    print("\n[2/3] Generating synthetic 'wet' images...")
    wet_count = generate_wet_images(num_target=400)

    # Step 3: Summary
    print("\n[3/3] Final dataset summary:")
    print("-" * 40)
    for cls in ["smooth", "gravel", "pothole", "wet"]:
        cls_dir = os.path.join(OUTPUT_DIR, cls)
        count = len(os.listdir(cls_dir)) if os.path.exists(cls_dir) else 0
        print(f"  {cls:10s}: {count:5d} images")

    total = sum(
        len(os.listdir(os.path.join(OUTPUT_DIR, c)))
        for c in ["smooth", "gravel", "pothole", "wet"]
        if os.path.exists(os.path.join(OUTPUT_DIR, c))
    )
    print(f"  {'TOTAL':10s}: {total:5d} images")
    print(f"\n  Corrupted/removed: {stats['corrupted']}")
    print(f"\n  Dataset ready at: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
