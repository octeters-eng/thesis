"""
Prepare the seed dataset for YOLO classification training.

Source structure (data/SEEED/):
    ClassA/
        Single/  *.jpg
        multiple/ *.jpg
        Gerard/  *.jpg
    ClassB/
        ...

Target structure (dataset/):
    train/ClassA/*.jpg  train/ClassB/*.jpg  ...
    val/ClassA/*.jpg    val/ClassB/*.jpg    ...
    test/ClassA/*.jpg   test/ClassB/*.jpg   ...

Usage: python prepare_dataset.py
"""

import random
import shutil
import sys
from collections import Counter
from pathlib import Path

from config import (
    DATA_SOURCE_DIR, DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    TRAIN_SPLIT, VAL_SPLIT, SEED_CLASSES, create_directories,
)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def collect_images_by_class(source_dir: Path) -> dict:
    """Collect all images per class, flattening subfolders."""
    class_images = {}

    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        images = sorted(
            f for f in class_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            class_images[class_name] = images

    return class_images


def split_and_copy(class_images: dict, seed: int = 42):
    """Split images into train/val/test and copy to dataset directory."""
    random.seed(seed)

    stats = {"train": Counter(), "val": Counter(), "test": Counter()}

    for class_name, images in class_images.items():
        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_SPLIT)
        n_val = int(n * VAL_SPLIT)

        splits = {
            "train": (TRAIN_DIR, images[:n_train]),
            "val": (VAL_DIR, images[n_train:n_train + n_val]),
            "test": (TEST_DIR, images[n_train + n_val:]),
        }

        for split_name, (split_dir, split_images) in splits.items():
            class_out = split_dir / class_name
            class_out.mkdir(parents=True, exist_ok=True)

            for idx, img_path in enumerate(split_images):
                # Use unique name to avoid collisions from subfolders
                out_name = f"{img_path.stem}{img_path.suffix}"
                out_path = class_out / out_name
                if out_path.exists():
                    out_name = f"{img_path.stem}_{idx}{img_path.suffix}"
                    out_path = class_out / out_name
                shutil.copy2(img_path, out_path)

            stats[split_name][class_name] = len(split_images)

    return stats


def main():
    print("=" * 60)
    print("PREPARE SEED DATASET FOR CLASSIFICATION")
    print("=" * 60)

    if not DATA_SOURCE_DIR.exists():
        print(f"ERROR: Source data not found at {DATA_SOURCE_DIR}")
        print("Place your data in: data/SEEED/<class_name>/<images>")
        sys.exit(1)

    # Check if already prepared
    if TRAIN_DIR.exists() and any(TRAIN_DIR.iterdir()):
        print(f"Dataset already exists at {DATASET_DIR}")
        response = input("Re-create? This will delete existing splits. (y/n): ").strip().lower()
        if response != 'y':
            print("Skipping. Use existing dataset.")
            return
        shutil.rmtree(DATASET_DIR)

    create_directories()

    # Collect images
    print(f"\nScanning {DATA_SOURCE_DIR} ...")
    class_images = collect_images_by_class(DATA_SOURCE_DIR)

    if not class_images:
        print("ERROR: No class folders with images found.")
        sys.exit(1)

    total = sum(len(v) for v in class_images.values())
    print(f"Found {len(class_images)} classes, {total} total images\n")

    for cls, imgs in sorted(class_images.items()):
        print(f"  {cls:20s}: {len(imgs):>4d} images")

    # Split and copy
    print(f"\nSplitting: train={TRAIN_SPLIT:.0%} / val={VAL_SPLIT:.0%} / test={1-TRAIN_SPLIT-VAL_SPLIT:.0%}")
    stats = split_and_copy(class_images)

    # Summary
    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    header = f"{'Class':20s} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>6}"
    print(header)
    print("-" * len(header))

    totals = Counter()
    for cls in sorted(class_images.keys()):
        tr = stats["train"].get(cls, 0)
        va = stats["val"].get(cls, 0)
        te = stats["test"].get(cls, 0)
        print(f"{cls:20s} {tr:>6} {va:>6} {te:>6} {tr+va+te:>6}")
        totals["train"] += tr
        totals["val"] += va
        totals["test"] += te

    print("-" * len(header))
    print(f"{'TOTAL':20s} {totals['train']:>6} {totals['val']:>6} {totals['test']:>6} {sum(totals.values()):>6}")

    print(f"\nDataset ready at: {DATASET_DIR}")
    print("Next: python train_ultralytics.py --all")


if __name__ == "__main__":
    main()
