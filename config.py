"""
Configuration file for Seed Recognition - YOLO Model Comparison
Thesis Project
"""

import os
from pathlib import Path

# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed"
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
TEST_DIR = PROCESSED_DIR / "test"

YOLOV7_DIR = PROJECT_ROOT / "yolov7"
RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"

# ============================================================
# Google Drive Dataset
# ============================================================
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/10qBfsN3tiT9ph_qEIwam7l_3m22SPE3o"
# Extract folder ID from the URL
GDRIVE_FOLDER_ID = "10qBfsN3tiT9ph_qEIwam7l_3m22SPE3o"

# ============================================================
# Dataset Configuration
# ============================================================
# Update these after inspecting your dataset
SEED_CLASSES = []  # Will be auto-detected from dataset, or set manually e.g. ["wheat", "rice", "corn"]
NUM_CLASSES = 0    # Will be set after auto-detection
IMG_SIZE = 640     # Input image size for YOLO models
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ============================================================
# Training Configuration
# ============================================================
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.01
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
DEVICE = "0"  # GPU device, use "cpu" if no GPU available

# ============================================================
# YOLO Model Versions to Compare
# ============================================================
YOLO_VERSIONS = {
    "yolov7": {
        "weights": "yolov7.pt",
        "repo": "https://github.com/WongKinYiu/yolov7.git",
        "type": "yolov7",
    },
    "yolov7-tiny": {
        "weights": "yolov7-tiny.pt",
        "repo": "https://github.com/WongKinYiu/yolov7.git",
        "type": "yolov7",
    },
    "yolov8n": {
        "weights": "yolov8n.pt",
        "type": "ultralytics",
    },
    "yolov8s": {
        "weights": "yolov8s.pt",
        "type": "ultralytics",
    },
    "yolov9c": {
        "weights": "yolov9c.pt",
        "type": "ultralytics",
    },
    "yolov9e": {
        "weights": "yolov9e.pt",
        "type": "ultralytics",
    },
    "yolov10n": {
        "weights": "yolov10n.pt",
        "type": "ultralytics",
    },
    "yolov10s": {
        "weights": "yolov10s.pt",
        "type": "ultralytics",
    },
    "yolov11n": {
        "weights": "yolo11n.pt",
        "type": "ultralytics",
    },
    "yolov11s": {
        "weights": "yolo11s.pt",
        "type": "ultralytics",
    },
}

# ============================================================
# Evaluation Metrics
# ============================================================
METRICS_TO_COMPARE = [
    "precision",
    "recall",
    "mAP50",
    "mAP50-95",
    "f1_score",
    "inference_time_ms",
    "model_size_mb",
    "parameters",
    "flops",
]


def create_directories():
    """Create all necessary project directories."""
    for d in [DATASET_DIR, RAW_DATA_DIR, PROCESSED_DIR, TRAIN_DIR, VAL_DIR,
              TEST_DIR, RUNS_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        # Create images and labels subdirectories for YOLO format
        if d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
            (d / "images").mkdir(exist_ok=True)
            (d / "labels").mkdir(exist_ok=True)


if __name__ == "__main__":
    create_directories()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset directory: {DATASET_DIR}")
    print("All directories created successfully.")
