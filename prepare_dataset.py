"""
Prepare dataset for YOLO training.
Handles multiple input formats and creates proper YOLO directory structure.

Expected YOLO label format (per line): class_id x_center y_center width height
All values normalized to [0, 1].

Usage: python prepare_dataset.py
"""

import os
import sys
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import yaml

from config import (
    RAW_DATA_DIR, PROCESSED_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, IMG_SIZE,
    SEED_CLASSES, PROJECT_ROOT, create_directories,
)


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def find_images_and_labels(data_dir: Path):
    """
    Auto-detect dataset structure and find image-label pairs.
    Supports: YOLO txt, Pascal VOC xml, and already-split datasets.
    """
    images = []
    for f in data_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(f)
    
    if not images:
        print(f"ERROR: No images found in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(images)} images")
    
    # Try to find corresponding labels
    pairs = []
    label_format = None
    
    for img_path in images:
        stem = img_path.stem
        parent = img_path.parent
        
        # Look for YOLO format (.txt) label
        txt_candidates = [
            parent / f"{stem}.txt",
            parent.parent / "labels" / f"{stem}.txt",
            parent / "labels" / f"{stem}.txt",
        ]
        for txt_path in txt_candidates:
            if txt_path.exists():
                pairs.append((img_path, txt_path))
                label_format = "yolo"
                break
        else:
            # Look for Pascal VOC format (.xml) label
            xml_candidates = [
                parent / f"{stem}.xml",
                parent.parent / "annotations" / f"{stem}.xml",
                parent / "annotations" / f"{stem}.xml",
            ]
            for xml_path in xml_candidates:
                if xml_path.exists():
                    pairs.append((img_path, xml_path))
                    label_format = "voc"
                    break
            else:
                # Image without label — skip with warning
                print(f"  Warning: No label found for {img_path.name}")
    
    print(f"Found {len(pairs)} image-label pairs (format: {label_format})")
    return pairs, label_format


def detect_classes_from_dataset(data_dir: Path, pairs, label_format):
    """Auto-detect class names from the dataset."""
    classes = []
    
    # Check for classes file
    for name in ["classes.txt", "classes.names", "obj.names"]:
        found = list(data_dir.rglob(name))
        if found:
            with open(found[0], 'r') as f:
                classes = [line.strip() for line in f if line.strip()]
            print(f"Classes loaded from {found[0]}: {classes}")
            return classes
    
    # Check for data.yaml or dataset.yaml
    for name in ["data.yaml", "dataset.yaml"]:
        found = list(data_dir.rglob(name))
        if found:
            with open(found[0], 'r') as f:
                data = yaml.safe_load(f)
            if "names" in data:
                if isinstance(data["names"], dict):
                    classes = [data["names"][k] for k in sorted(data["names"].keys())]
                else:
                    classes = list(data["names"])
                print(f"Classes loaded from {found[0]}: {classes}")
                return classes
    
    # Infer from VOC XML
    if label_format == "voc":
        class_set = set()
        for _, label_path in pairs[:100]:  # Sample first 100
            tree = ET.parse(label_path)
            for obj in tree.findall(".//object"):
                class_set.add(obj.find("name").text)
        classes = sorted(class_set)
        print(f"Classes detected from VOC annotations: {classes}")
        return classes
    
    # Infer from directory names (if images are organized by class)
    # Check if images are in class-named subdirectories
    img_parents = set(p.parent.name for p, _ in pairs)
    if len(img_parents) > 1 and "images" not in img_parents:
        classes = sorted(img_parents)
        print(f"Classes inferred from directory structure: {classes}")
        return classes
    
    # Auto-detect from YOLO labels (class IDs only)
    if label_format == "yolo":
        class_ids = set()
        for _, label_path in pairs:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_ids.add(int(parts[0]))
        num_classes = max(class_ids) + 1 if class_ids else 0
        classes = [f"seed_class_{i}" for i in range(num_classes)]
        print(f"Detected {num_classes} class IDs from YOLO labels: {sorted(class_ids)}")
        print("  Using placeholder names. Update config.py SEED_CLASSES for proper names.")
    
    return classes


def convert_voc_to_yolo(xml_path: Path, class_names: list) -> list:
    """Convert Pascal VOC XML annotation to YOLO format lines."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)
    
    lines = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in class_names:
            continue
        class_id = class_names.index(class_name)
        
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        
        # Convert to YOLO format (normalized x_center, y_center, width, height)
        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h
        
        # Clamp values to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))
        
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return lines


def check_dataset_already_split(data_dir: Path):
    """Check if dataset is already split into train/val/test."""
    has_train = any((data_dir / d / "images").exists() or (data_dir / d).exists()
                    for d in ["train", "training"])
    has_val = any((data_dir / d / "images").exists() or (data_dir / d).exists()
                  for d in ["val", "valid", "validation"])
    has_test = any((data_dir / d / "images").exists() or (data_dir / d).exists()
                   for d in ["test", "testing"])
    return has_train, has_val, has_test


def copy_pair_to_split(img_path: Path, label_path: Path, split_dir: Path,
                       label_format: str, class_names: list, idx: int):
    """Copy an image-label pair to the target split directory."""
    # Determine output filename (avoid collisions)
    out_name = img_path.stem
    out_img = split_dir / "images" / f"{out_name}{img_path.suffix}"
    out_lbl = split_dir / "labels" / f"{out_name}.txt"
    
    # Handle filename collisions
    if out_img.exists():
        out_name = f"{out_name}_{idx}"
        out_img = split_dir / "images" / f"{out_name}{img_path.suffix}"
        out_lbl = split_dir / "labels" / f"{out_name}.txt"
    
    # Copy image
    shutil.copy2(img_path, out_img)
    
    # Copy or convert label
    if label_format == "yolo":
        shutil.copy2(label_path, out_lbl)
    elif label_format == "voc":
        lines = convert_voc_to_yolo(label_path, class_names)
        with open(out_lbl, 'w') as f:
            f.write("\n".join(lines))


def create_data_yaml(class_names: list, output_path: Path):
    """Create YOLO data.yaml configuration file."""
    data = {
        "path": str(PROCESSED_DIR),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created data.yaml at {output_path}")
    return output_path


def get_label_distribution(split_dir: Path) -> Counter:
    """Count class distribution in a split."""
    counter = Counter()
    labels_dir = split_dir / "labels"
    if labels_dir.exists():
        for txt_file in labels_dir.glob("*.txt"):
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        counter[int(parts[0])] += 1
    return counter


def main():
    create_directories()
    
    print("=" * 60)
    print("DATASET PREPARATION FOR YOLO TRAINING")
    print("=" * 60)
    
    if not RAW_DATA_DIR.exists() or not any(RAW_DATA_DIR.iterdir()):
        print(f"ERROR: No data found in {RAW_DATA_DIR}")
        print("Run 'python download_dataset.py' first.")
        sys.exit(1)
    
    # Check if already in YOLO split format
    has_train, has_val, has_test = check_dataset_already_split(RAW_DATA_DIR)
    
    if has_train and has_val:
        print("Dataset appears to be already split! Checking structure...")
        # Find the existing split and link/copy
        for dirname in ["train", "training"]:
            d = RAW_DATA_DIR / dirname
            if d.exists():
                train_src = d
                break
        for dirname in ["val", "valid", "validation"]:
            d = RAW_DATA_DIR / dirname
            if d.exists():
                val_src = d
                break
    
    # Find images and labels
    pairs, label_format = find_images_and_labels(RAW_DATA_DIR)
    
    if not pairs:
        print("ERROR: No image-label pairs found.")
        print("Please ensure your dataset has images with corresponding label files.")
        sys.exit(1)
    
    # Detect or load classes
    class_names = SEED_CLASSES if SEED_CLASSES else detect_classes_from_dataset(
        RAW_DATA_DIR, pairs, label_format
    )
    
    if not class_names:
        print("ERROR: Could not determine class names.")
        print("Please set SEED_CLASSES in config.py manually.")
        sys.exit(1)
    
    print(f"\nClasses ({len(class_names)}): {class_names}")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(pairs)
    
    n = len(pairs)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)
    
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]
    
    print(f"\nSplit: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
    
    # Clear processed directory
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for sub in ["images", "labels"]:
            d = split_dir / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
    
    # Copy files to splits
    print("\nCopying files...")
    for split_name, split_pairs, split_dir in [
        ("train", train_pairs, TRAIN_DIR),
        ("val", val_pairs, VAL_DIR),
        ("test", test_pairs, TEST_DIR),
    ]:
        print(f"  {split_name}: {len(split_pairs)} pairs")
        for idx, (img_path, lbl_path) in enumerate(split_pairs):
            copy_pair_to_split(img_path, lbl_path, split_dir,
                               label_format, class_names, idx)
    
    # Create data.yaml
    data_yaml_path = create_data_yaml(class_names, PROCESSED_DIR / "data.yaml")
    
    # Also create YOLOv7-compatible data.yaml
    yolov7_data = {
        "train": str(TRAIN_DIR / "images"),
        "val": str(VAL_DIR / "images"),
        "test": str(TEST_DIR / "images"),
        "nc": len(class_names),
        "names": class_names,
    }
    yolov7_yaml_path = PROCESSED_DIR / "data_yolov7.yaml"
    with open(yolov7_yaml_path, 'w') as f:
        yaml.dump(yolov7_data, f, default_flow_style=False, sort_keys=False)
    print(f"Created YOLOv7 data config at {yolov7_yaml_path}")
    
    # Show distribution
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    for split_name, split_dir in [("Train", TRAIN_DIR), ("Val", VAL_DIR), ("Test", TEST_DIR)]:
        dist = get_label_distribution(split_dir)
        n_images = len(list((split_dir / "images").glob("*")))
        n_objects = sum(dist.values())
        print(f"\n{split_name}: {n_images} images, {n_objects} objects")
        for cls_id in sorted(dist.keys()):
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            print(f"  {cls_name}: {dist[cls_id]} objects")
    
    print(f"\nDataset ready! data.yaml: {data_yaml_path}")
    print("Next: Run training scripts.")


if __name__ == "__main__":
    main()
