"""
Main pipeline orchestrator for Seed Recognition — YOLO Classification Comparison.

Steps:
  0. (Optional) Download dataset from Google Drive
  1. Prepare dataset (flatten subfolders, split into train/val/test)
  2. Train all YOLO classification models
  3. Evaluate all trained models on the test set
  4. Generate comparison plots and report

All metrics and artifacts are logged to MLflow.
Use `mlflow ui` to compare models interactively.

Usage:
    python main.py                   # Full pipeline
    python main.py --step download   # Download only
    python main.py --step prepare    # Only prepare dataset
    python main.py --step train      # Only train
    python main.py --step evaluate   # Only evaluate
    python main.py --step compare    # Only compare/report
"""

import argparse
import subprocess
import sys
from typing import List, Optional

import mlflow

from config import (
    DATA_SOURCE_DIR, DATASET_DIR, RUNS_DIR,
    YOLO_VERSIONS, create_directories,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
)


def run_script(script_name: str, args: Optional[List[str]] = None):
    """Run a project script as a subprocess and exit on failure."""
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"ERROR: {script_name} failed.")
        sys.exit(result.returncode)


def step_download(force_yes: bool = False):
    """Download dataset."""
    print("\n" + "=" * 60)
    print("STEP 0: DOWNLOAD DATASET")
    print("=" * 60)
    args = ["--yes"] if force_yes else []
    run_script("download_dataset.py", args)


def step_prepare(force_yes: bool = False):
    """Prepare the dataset."""
    print("\n" + "=" * 60)
    print("STEP 1: PREPARE DATASET")
    print("=" * 60)

    args = ["--yes"] if force_yes else []
    run_script("prepare_dataset.py", args)


def step_train():
    """Train all models."""
    print("\n" + "=" * 60)
    print("STEP 2: TRAIN ALL MODELS")
    print("=" * 60)

    run_script("train_ultralytics.py", ["--all"])


def step_evaluate():
    """Evaluate all trained models."""
    print("\n" + "=" * 60)
    print("STEP 3: EVALUATE ALL MODELS")
    print("=" * 60)

    run_script("evaluate.py")


def step_compare():
    """Generate comparison figures and summary report."""
    print("\n" + "=" * 60)
    print("STEP 4: COMPARE MODELS")
    print("=" * 60)
    run_script("compare_results.py")


STEPS = {
    "download": step_download,
    "prepare": step_prepare,
    "train": step_train,
    "evaluate": step_evaluate,
    "compare": step_compare,
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
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Assume yes for prompts where available.",
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
        if args.step in {"download", "prepare"}:
            STEPS[args.step](force_yes=args.yes)
        else:
            STEPS[args.step]()
    else:
        step_prepare(force_yes=args.yes)
        step_train()
        step_evaluate()
        step_compare()

    print("\n" + "=" * 60)
    print("DONE — View results: mlflow ui")
    print("=" * 60)


if __name__ == "__main__":
    main()
