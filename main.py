"""
Pipeline orchestrator for Seed Recognition — YOLO Classification Comparison.

Pipeline steps:
    0. (optional) download  — Download raw dataset from Google Drive
    1.            prepare   — Flatten sub-folders and split into train/val/test
    2.            train     — Train all YOLO classification models
    3.            evaluate  — Evaluate trained models on the test set
    4.            compare   — Generate comparison plots and thesis report

All metrics and artefacts are logged to MLflow.
Run `mlflow ui` after the pipeline to compare models interactively.

Usage:
    python main.py                    # run prepare → train → evaluate → compare
    python main.py --step download    # run only the download step
    python main.py --step prepare
    python main.py --step train
    python main.py --step evaluate
    python main.py --step compare
    python main.py --yes              # suppress interactive prompts
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Optional

import mlflow

from config import (
    DATA_SOURCE_DIR, DATASET_DIR, RUNS_DIR,
    YOLO_VERSIONS, create_directories,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
)

_SEPARATOR = "=" * 60


def _run_script(script_name: str, args: Optional[list[str]] = None) -> None:
    """Run a project script as a subprocess and exit the pipeline on failure.

    Args:
        script_name: Filename of the script relative to the project root.
        args:        Optional list of CLI arguments to forward.
    """
    cmd    = [sys.executable, script_name] + (args or [])
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"ERROR: {script_name} exited with code {result.returncode}.")
        sys.exit(result.returncode)


# ── Pipeline steps ─────────────────────────────────────────────────────────────

def step_download(force_yes: bool = False) -> None:
    print(f"\n{_SEPARATOR}\nSTEP 0: DOWNLOAD DATASET\n{_SEPARATOR}")
    _run_script("download_dataset.py", ["--yes"] if force_yes else [])


def step_prepare(force_yes: bool = False) -> None:
    print(f"\n{_SEPARATOR}\nSTEP 1: PREPARE DATASET\n{_SEPARATOR}")
    _run_script("prepare_dataset.py", ["--yes"] if force_yes else [])


def step_train() -> None:
    print(f"\n{_SEPARATOR}\nSTEP 2: TRAIN ALL MODELS\n{_SEPARATOR}")
    _run_script("train_ultralytics.py", ["--all"])


def step_evaluate() -> None:
    print(f"\n{_SEPARATOR}\nSTEP 3: EVALUATE ALL MODELS\n{_SEPARATOR}")
    _run_script("evaluate.py")


def step_compare() -> None:
    print(f"\n{_SEPARATOR}\nSTEP 4: COMPARE MODELS & GENERATE REPORT\n{_SEPARATOR}")
    _run_script("compare_results.py")


# Steps that accept the --yes flag receive it; others do not.
_STEPS_WITH_YES = {"download", "prepare"}

STEPS = {
    "download": step_download,
    "prepare":  step_prepare,
    "train":    step_train,
    "evaluate": step_evaluate,
    "compare":  step_compare,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed Recognition — YOLO Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--step",
        choices=list(STEPS.keys()),
        default=None,
        help="Run a single pipeline step. Default: run all steps.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Assume 'yes' to all interactive prompts.",
    )
    args = parser.parse_args()

    create_directories()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print(_SEPARATOR)
    print("SEED RECOGNITION — YOLO CLASSIFICATION COMPARISON")
    print(_SEPARATOR)
    print(f"Models  : {list(YOLO_VERSIONS.keys())}")
    print(f"Source  : {DATA_SOURCE_DIR}")
    print(f"Dataset : {DATASET_DIR}")
    print(f"Runs    : {RUNS_DIR}")
    print(f"MLflow  : {MLFLOW_TRACKING_URI}")

    if args.step:
        fn = STEPS[args.step]
        fn(force_yes=args.yes) if args.step in _STEPS_WITH_YES else fn()
    else:
        step_prepare(force_yes=args.yes)
        step_train()
        step_evaluate()
        step_compare()

    print(f"\n{_SEPARATOR}")
    print("DONE — view results with: mlflow ui")
    print(_SEPARATOR)


if __name__ == "__main__":
    main()
