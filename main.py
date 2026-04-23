"""
Main pipeline orchestrator for Seed Recognition — YOLO Classification Comparison.

Steps:
  1. Prepare dataset (flatten subfolders, split into train/val/test)
  2. Train all YOLO classification models
  3. Evaluate all trained models on the test set

All metrics and artifacts are logged to MLflow.
Use `mlflow ui` to compare models interactively.

Usage:
    python main.py                   # Full pipeline
    python main.py --step prepare    # Only prepare dataset
    python main.py --step train      # Only train
    python main.py --step evaluate   # Only evaluate
"""

import argparse
import sys
from pathlib import Path

import mlflow

from config import (
    DATA_SOURCE_DIR, DATASET_DIR, RUNS_DIR,
    YOLO_VERSIONS, create_directories,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
)


def step_prepare():
    """Prepare the dataset."""
    print("\n" + "=" * 60)
    print("STEP 1: PREPARE DATASET")
    print("=" * 60)

    if not DATA_SOURCE_DIR.exists():
        print(f"ERROR: Source data not found at {DATA_SOURCE_DIR}")
        print("Put your seed images in data/SEEED/ with one folder per class.")
        sys.exit(1)

    # Check if already prepared
    if (DATASET_DIR / "train").exists():
        n_train = sum(1 for _ in (DATASET_DIR / "train").rglob("*") if _.is_file())
        print(f"Dataset already prepared ({n_train} training images).")
        resp = input("Re-prepare? [y/N]: ").strip().lower()
        if resp != "y":
            print("Skipping.")
            return

    from prepare_dataset import main as prepare_main
    prepare_main()


def step_train():
    """Train all models."""
    print("\n" + "=" * 60)
    print("STEP 2: TRAIN ALL MODELS")
    print("=" * 60)

    from train_ultralytics import train_all
    train_all()


def step_evaluate():
    """Evaluate all trained models."""
    print("\n" + "=" * 60)
    print("STEP 3: EVALUATE ALL MODELS")
    print("=" * 60)

    from evaluate import evaluate_all
    evaluate_all()


STEPS = {
    "prepare": step_prepare,
    "train": step_train,
    "evaluate": step_evaluate,
}


def main():
    parser = argparse.ArgumentParser(
        description="Seed Recognition — YOLO Classification Pipeline"
    )
    parser.add_argument(
        "--step",
        choices=list(STEPS.keys()),
        default=None,
        help="Run a specific step only. Default: run all steps.",
    )
    args = parser.parse_args()

    create_directories()

    # Initialize MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("=" * 60)
    print("SEED RECOGNITION — YOLO CLASSIFICATION COMPARISON")
    print("=" * 60)
    print(f"Models: {list(YOLO_VERSIONS.keys())}")
    print(f"Source: {DATA_SOURCE_DIR}")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Runs: {RUNS_DIR}")
    print(f"MLflow: {MLFLOW_TRACKING_URI}")

    if args.step:
        STEPS[args.step]()
    else:
        step_prepare()
        step_train()
        step_evaluate()

    print("\n" + "=" * 60)
    print("DONE — View results: mlflow ui")
    print("=" * 60)


if __name__ == "__main__":
    main()
