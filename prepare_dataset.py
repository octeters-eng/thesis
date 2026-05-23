"""
Prepare the seed dataset for YOLO classification training.

Source layout (data/SEEED/):
    <ClassName>/
        Single/   *.jpg
        multiple/ *.jpg
        Gerard/   *.jpg
        ...

Target layout (dataset/):
    train/<ClassName>/*.jpg
    val/<ClassName>/*.jpg
    test/<ClassName>/*.jpg

Usage:
    python prepare_dataset.py           # interactive
    python prepare_dataset.py --yes     # non-interactive (overwrite without prompt)
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

from config import (
    DATA_SOURCE_DIR, DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    TRAIN_SPLIT, VAL_SPLIT, create_directories,
)

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
)

_SEPARATOR = "=" * 60


def collect_images_by_class(source_dir: Path) -> dict[str, list[Path]]:
    """Recursively collect all images per class, flattening sub-folders.

    Args:
        source_dir: Root directory whose immediate children are class folders.

    Returns:
        Mapping of class name → sorted list of image paths.
    """
    class_images: dict[str, list[Path]] = {}

    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        images = sorted(
            f for f in class_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            class_images[class_dir.name] = images

    return class_images


def split_and_copy(class_images: dict[str, list[Path]], seed: int = 42) -> dict:
    """Split images into train / val / test and copy them to the dataset directory.

    Filenames are made unique across sub-folders by encoding the relative path
    into the filename, preventing collisions.

    Args:
        class_images: Output of :func:`collect_images_by_class`.
        seed: Random seed for reproducible splits.

    Returns:
        Nested counter ``{split_name: {class_name: count}}``.
    """
    random.seed(seed)
    stats: dict[str, Counter] = {"train": Counter(), "val": Counter(), "test": Counter()}

    for class_name, images in class_images.items():
        random.shuffle(images)

        n       = len(images)
        n_train = int(n * TRAIN_SPLIT)
        n_val   = int(n * VAL_SPLIT)

        splits = {
            "train": (TRAIN_DIR, images[:n_train]),
            "val":   (VAL_DIR,   images[n_train : n_train + n_val]),
            "test":  (TEST_DIR,  images[n_train + n_val :]),
        }

        for split_name, (split_dir, split_images) in splits.items():
            class_out = split_dir / class_name
            class_out.mkdir(parents=True, exist_ok=True)

            for img_path in split_images:
                # Encode the sub-folder path into the filename to avoid
                # collisions when multiple sub-folders contain same-named files.
                rel     = img_path.relative_to(DATA_SOURCE_DIR / class_name)
                stem    = "_".join(rel.with_suffix("").parts)
                out_name = f"{stem}{img_path.suffix}".replace(" ", "_")
                shutil.copy2(img_path, class_out / out_name)

            stats[split_name][class_name] = len(split_images)

    return stats


def main(force_yes: bool = False) -> None:
    print(_SEPARATOR)
    print("PREPARE SEED DATASET FOR CLASSIFICATION")
    print(_SEPARATOR)

    if not DATA_SOURCE_DIR.exists():
        print(f"ERROR: Source data not found at {DATA_SOURCE_DIR}")
        print("Place your data in: data/SEEED/<ClassName>/<images>")
        sys.exit(1)

    # Prompt before overwriting an existing split.
    if TRAIN_DIR.exists() and any(TRAIN_DIR.iterdir()):
        print(f"Dataset already exists at {DATASET_DIR}")
        if force_yes:
            recreate = True
        else:
            recreate = input("Re-create? This will delete existing splits. (y/n): ").strip().lower() == "y"
        if not recreate:
            print("Skipping — using existing dataset.")
            return
        shutil.rmtree(DATASET_DIR)

    create_directories()

    print(f"\nScanning {DATA_SOURCE_DIR} ...")
    class_images = collect_images_by_class(DATA_SOURCE_DIR)

    if not class_images:
        print("ERROR: No class folders with images found.")
        sys.exit(1)

    total = sum(len(v) for v in class_images.values())
    print(f"Found {len(class_images)} classes, {total} total images\n")
    for cls, imgs in sorted(class_images.items()):
        print(f"  {cls:20s}: {len(imgs):>4d} images")

    test_split = 1 - TRAIN_SPLIT - VAL_SPLIT
    print(f"\nSplitting: train={TRAIN_SPLIT:.0%} / val={VAL_SPLIT:.0%} / test={test_split:.0%}")
    stats = split_and_copy(class_images)

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + _SEPARATOR)
    print("SPLIT SUMMARY")
    print(_SEPARATOR)
    header = f"{'Class':20s} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>6}"
    print(header)
    print("-" * len(header))

    totals: Counter = Counter()
    for cls in sorted(class_images):
        tr = stats["train"][cls]
        va = stats["val"][cls]
        te = stats["test"][cls]
        print(f"{cls:20s} {tr:>6} {va:>6} {te:>6} {tr + va + te:>6}")
        totals["train"] += tr
        totals["val"]   += va
        totals["test"]  += te

    print("-" * len(header))
    print(
        f"{'TOTAL':20s} {totals['train']:>6} {totals['val']:>6} "
        f"{totals['test']:>6} {sum(totals.values()):>6}"
    )
    print(f"\nDataset ready at: {DATASET_DIR}")
    print("Next: python train_ultralytics.py --all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare seed dataset for YOLO classification.")
    parser.add_argument("--yes", action="store_true", help="Overwrite existing dataset without prompting.")
    args = parser.parse_args()
    main(force_yes=args.yes)
