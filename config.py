"""
Configuration file for Seed Recognition (Classification) - YOLO Model Comparison
Thesis Project
"""

from pathlib import Path

# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_SOURCE_DIR = PROJECT_ROOT / "data" / "SEEED"  # Raw downloaded data
DATASET_DIR = PROJECT_ROOT / "dataset"              # Prepared classification data
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"

# ============================================================
# Seed Classes (16 classes from dataset)
# ============================================================
SEED_CLASSES = [
    "Borboti",
    "Boro_Shosha",
    "Chal_Kumra",
    "Chichinga",
    "Dhonepata",
    "Hatikani_Shim",
    "Jhinga",
    "Korola",
    "Lau",
    "Mishti_Kumra",
    "Monica_Dhedosh",
    "Palong_Shak",
    "Shim",
    "Shosha_Mohabir",
    "pepe",
    "pui_shak",
]
NUM_CLASSES = len(SEED_CLASSES)

# ============================================================
# Dataset Configuration
# ============================================================
IMG_SIZE = 224     # Input image size for YOLO classification models
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ============================================================
# Training Configuration
# ============================================================
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = "mps"  # "0" for CUDA GPU, "mps" for Apple Silicon, "cpu" for CPU

# ============================================================
# YOLO Model Versions to Compare (Classification)
# ============================================================
YOLO_VERSIONS = {
    "yolov8n-cls": {
        "weights": "yolov8n-cls.pt",
        "type": "ultralytics",
    },
    "yolov8s-cls": {
        "weights": "yolov8s-cls.pt",
        "type": "ultralytics",
    },
    "yolov11n-cls": {
        "weights": "yolo11n-cls.pt",
        "type": "ultralytics",
    },
    "yolov11s-cls": {
        "weights": "yolo11s-cls.pt",
        "type": "ultralytics",
    },
}

# ============================================================
# Evaluation Metrics (Classification)
# ============================================================
METRICS_TO_COMPARE = [
    "top1_accuracy",
    "top5_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "inference_time_ms",
    "model_size_mb",
    "parameters",
]


def create_directories():
    """Create all necessary project directories."""
    for d in [DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, RUNS_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_directories()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Classes ({NUM_CLASSES}): {SEED_CLASSES}")
    print("Directories created.")
    print("All directories created successfully.")
