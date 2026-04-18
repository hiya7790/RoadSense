"""
augment.py — Data Augmentation Pipeline for VisionSuspend
Uses Albumentations to generate augmented training images.
"""

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import albumentations as A

CLASSES = ["smooth", "gravel", "pothole", "wet"]

AUGMENTATION_PIPELINE = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.MotionBlur(blur_limit=7, p=0.4),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    A.RandomRain(p=0.2),
    A.Rotate(limit=15, p=0.5),
    A.Resize(224, 224),
])


def augment_class(input_dir, output_dir, class_name, num_augments=5):
    src = os.path.join(input_dir, class_name)
    dst = os.path.join(output_dir, class_name)
    os.makedirs(dst, exist_ok=True)

    images = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"[{class_name}] Found {len(images)} images. Generating {num_augments}x augmentations...")

    for img_file in tqdm(images):
        img_path = os.path.join(src, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Save original (resized)
        orig_resized = cv2.resize(image, (224, 224))
        cv2.imwrite(os.path.join(dst, img_file), cv2.cvtColor(orig_resized, cv2.COLOR_RGB2BGR))

        # Save augmented copies
        base_name = os.path.splitext(img_file)[0]
        for i in range(num_augments):
            augmented = AUGMENTATION_PIPELINE(image=image)["image"]
            out_name = f"{base_name}_aug{i+1}.jpg"
            cv2.imwrite(os.path.join(dst, out_name), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))


def main(args):
    for cls in CLASSES:
        class_src = os.path.join(args.input, cls)
        if not os.path.exists(class_src):
            print(f"Warning: Class folder '{class_src}' not found. Skipping.")
            continue
        augment_class(args.input, args.output, cls, num_augments=args.num_augments)
    print("\nAugmentation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionSuspend Data Augmentation Pipeline")
    parser.add_argument("--input", type=str, default="data/raw", help="Input data directory")
    parser.add_argument("--output", type=str, default="data/augmented", help="Output directory")
    parser.add_argument("--num_augments", type=int, default=5, help="Augmented copies per image")
    args = parser.parse_args()
    main(args)
