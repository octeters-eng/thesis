"""
Train Ultralytics YOLO classification models (YOLOv8-cls, YOLOv11-cls)
for seed recognition and log results to MLflow.

Usage:
    python train_ultralytics.py --all                    # train every model
    python train_ultralytics.py --model yolov8n-cls      # single model
    python train_ultralytics.py --model yolov11s-cls --epochs 50
"""

from __future__ import annotations

import argparse
import time

import mlflow
from ultralytics import YOLO

from config import (
    DATASET_DIR, RUNS_DIR,
    BATCH_SIZE, EPOCHS, IMG_SIZE, DEVICE,
    YOLO_VERSIONS, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
)

_SEPARATOR = "=" * 60


def train_model(
    model_name: str,
    weights: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    img_size: int = IMG_SIZE,
    device: str = DEVICE,
) -> dict | None:
    """Train a single YOLO classification model and log results to MLflow.

    Args:
        model_name: Identifier used for the run directory and MLflow tags.
        weights:    Pretrained weight file (e.g. ``"yolov8n-cls.pt"``).
        epochs:     Number of training epochs.
        batch_size: Mini-batch size.
        img_size:   Input image resolution.
        device:     Compute device (``"mps"``, ``"cpu"``, or CUDA index).

    Returns:
        Dictionary of training metadata and metrics, or ``None`` on failure.
    """
    if not DATASET_DIR.exists() or not (DATASET_DIR / "train").exists():
        print(f"ERROR: Dataset not found at {DATASET_DIR}. Run prepare_dataset.py first.")
        return None

    print(f"\n{_SEPARATOR}")
    print(f"TRAINING {model_name.upper()}")
    print(_SEPARATOR)
    print(f"  Weights  : {weights}")
    print(f"  Epochs   : {epochs}")
    print(f"  Batch    : {batch_size}")
    print(f"  Img size : {img_size}")
    print(f"  Device   : {device}")
    print(f"  Dataset  : {DATASET_DIR}")
    print(_SEPARATOR + "\n")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    model = YOLO(weights)

    start_time = time.time()
    results = model.train(
        data=str(DATASET_DIR),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=str(RUNS_DIR),
        name=model_name,
        exist_ok=True,
        patience=15,
        save=True,
        plots=True,
        verbose=True,
    )
    training_time = time.time() - start_time

    # Ultralytics' built-in MLflow callback may leave an active run open.
    if mlflow.active_run() is not None:
        mlflow.end_run()

    run_dir      = RUNS_DIR / model_name
    best_weights = run_dir / "weights" / "best.pt"

    # ── Collect metrics ───────────────────────────────────────────────────────
    val_metrics: dict = {}
    if results and hasattr(results, "results_dict"):
        rd = results.results_dict
        val_metrics["top1_accuracy"] = float(rd.get("metrics/accuracy_top1", 0))
        val_metrics["top5_accuracy"] = float(rd.get("metrics/accuracy_top5", 0))

    model_size_mb = (
        best_weights.stat().st_size / (1024 * 1024) if best_weights.exists() else 0.0
    )

    params = 0
    if best_weights.exists():
        try:
            info = YOLO(str(best_weights)).info(verbose=False)
            if info and len(info) > 1:
                params = info[1]
        except Exception:
            pass

    # ── Log summary run to MLflow ─────────────────────────────────────────────
    with mlflow.start_run(run_name=f"train-{model_name}") as run:
        mlflow.set_tag("stage", "training")
        mlflow.set_tag("model_type", "ultralytics")

        mlflow.log_params({
            "model_name": model_name,
            "weights":    weights,
            "epochs":     epochs,
            "batch_size": batch_size,
            "img_size":   img_size,
            "device":     device,
            "patience":   15,
        })
        mlflow.log_metrics({
            "training_time_seconds": round(training_time, 1),
            "model_size_mb":         round(model_size_mb, 2),
            "parameters":            params,
            **{k: float(v) for k, v in val_metrics.items()},
        })

        if best_weights.exists():
            mlflow.log_artifact(str(best_weights), artifact_path="weights")

        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            mlflow.log_artifact(str(results_csv), artifact_path="training")

        run_id = run.info.run_id

    result_data = {
        "model":                  model_name,
        "weights":                weights,
        "weights_path":           str(best_weights),
        "run_dir":                str(run_dir),
        "training_time_seconds":  round(training_time, 1),
        "epochs":                 epochs,
        "batch_size":             batch_size,
        "img_size":               img_size,
        "model_size_mb":          round(model_size_mb, 2),
        "parameters":             params,
        "mlflow_run_id":          run_id,
        **val_metrics,
    }

    print(f"\nTraining complete for {model_name}  (MLflow run: {run_id})")
    return result_data


def train_all(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    img_size: int = IMG_SIZE,
    device: str = DEVICE,
) -> dict:
    """Train every model defined in :data:`~config.YOLO_VERSIONS`.

    Args:
        epochs:     Training epochs for each model.
        batch_size: Mini-batch size.
        img_size:   Input image resolution.
        device:     Compute device.

    Returns:
        Mapping of model name → result dictionary (or error entry).
    """
    print(f"\nTraining {len(YOLO_VERSIONS)} models:")
    for name in YOLO_VERSIONS:
        print(f"  - {name}")

    all_results: dict = {}
    for model_name, cfg in YOLO_VERSIONS.items():
        try:
            result = train_model(
                model_name=model_name,
                weights=cfg["weights"],
                epochs=epochs,
                batch_size=batch_size,
                img_size=img_size,
                device=device,
            )
            if result:
                all_results[model_name] = result
        except Exception as exc:
            print(f"\nERROR training {model_name}: {exc}")
            all_results[model_name] = {"model": model_name, "error": str(exc)}

    print(f"\nTraining complete. {len(all_results)} models logged to MLflow.")
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLO classification models for seed recognition."
    )
    parser.add_argument("--model",      type=str, default=None,
                        help="Model name to train (e.g. yolov8n-cls, yolov11s-cls).")
    parser.add_argument("--all",        action="store_true",
                        help="Train all models defined in config.YOLO_VERSIONS.")
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--img-size",   type=int, default=IMG_SIZE)
    parser.add_argument("--device",     type=str, default=DEVICE)
    args = parser.parse_args()

    if args.all:
        train_all(args.epochs, args.batch_size, args.img_size, args.device)

    elif args.model:
        if args.model not in YOLO_VERSIONS:
            print(f"Unknown model: '{args.model}'")
            print(f"Available: {list(YOLO_VERSIONS.keys())}")
            return
        result = train_model(
            model_name=args.model,
            weights=YOLO_VERSIONS[args.model]["weights"],
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device,
        )
        if result:
            print("\nResult summary:")
            for key, val in result.items():
                print(f"  {key}: {val}")

    else:
        print("Specify --model <name> or --all")
        print(f"Available models: {list(YOLO_VERSIONS.keys())}")


if __name__ == "__main__":
    main()
