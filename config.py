"""
Central configuration for the Seed Recognition — YOLO Classification project.

Import from here rather than hard-coding paths, splits, or hyper-parameters
anywhere else in the codebase.
"""

import os
from pathlib import Path

# ── Project Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.resolve()
DATA_SOURCE_DIR = PROJECT_ROOT / "data" / "SEEED"   # Raw class folders
DATASET_DIR    = PROJECT_ROOT / "dataset"            # Prepared train/val/test tree
TRAIN_DIR      = DATASET_DIR / "train"
VAL_DIR        = DATASET_DIR / "val"
TEST_DIR       = DATASET_DIR / "test"
RUNS_DIR       = PROJECT_ROOT / "runs"               # Ultralytics training outputs
RESULTS_DIR    = PROJECT_ROOT / "results"            # Evaluation JSON, reports
FIGURES_DIR    = RESULTS_DIR / "figures"             # Generated plots

# Legacy alias kept for backward compatibility with older scripts.
RAW_DATA_DIR = DATA_SOURCE_DIR

# Optional Google Drive folder ID used by download_dataset.py.
GDRIVE_FOLDER_ID = os.getenv("SEED_GDRIVE_FOLDER_ID", "").strip()

# ── Seed Classes ──────────────────────────────────────────────────────────────
# Informational — the actual class list is derived from the dataset directory
# structure at runtime. Keep this in sync with data/SEEED/ folder names.
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

# ── Dataset Splits ────────────────────────────────────────────────────────────
IMG_SIZE     = 224    # Input resolution for all YOLO classification models
TRAIN_SPLIT  = 0.70
VAL_SPLIT    = 0.15
# TEST_SPLIT is implied: 1 - TRAIN_SPLIT - VAL_SPLIT = 0.15

# ── Training Hyper-parameters ─────────────────────────────────────────────────
BATCH_SIZE    = 32
EPOCHS        = 2
LEARNING_RATE = 0.001
DEVICE        = "mps"   # "0" = CUDA GPU | "mps" = Apple Silicon | "cpu"

# ── YOLO Model Variants ───────────────────────────────────────────────────────
YOLO_VERSIONS: dict[str, dict] = {
    "yolov8n-cls":  {"weights": "yolov8n-cls.pt",  "type": "ultralytics"},
    "yolov8s-cls":  {"weights": "yolov8s-cls.pt",  "type": "ultralytics"},
    "yolov11n-cls": {"weights": "yolo11n-cls.pt",  "type": "ultralytics"},
    "yolov11s-cls": {"weights": "yolo11s-cls.pt",  "type": "ultralytics"},
}

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI    = str(PROJECT_ROOT / "mlruns")
MLFLOW_EXPERIMENT_NAME = "seed-recognition-yolo"


def create_directories() -> None:
    """Create all required project directories if they do not already exist."""
    for directory in [DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
                      RUNS_DIR, RESULTS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_directories()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset dir  : {DATASET_DIR}")
    print(f"Classes ({NUM_CLASSES}): {SEED_CLASSES}")
    print("All directories created.")
