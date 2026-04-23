"""
Train Ultralytics YOLO classification models (YOLOv8-cls, YOLOv11-cls)
for seed recognition.

Usage:
    python train_ultralytics.py --model yolov8n-cls
    python train_ultralytics.py --model yolov11n-cls
    python train_ultralytics.py --all
"""

import argparse
import json
import time
from pathlib import Path

import mlflow
from ultralytics import YOLO

from config import (
    DATASET_DIR, RUNS_DIR, RESULTS_DIR,
    BATCH_SIZE, EPOCHS, IMG_SIZE, DEVICE,
    YOLO_VERSIONS, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
)


def train_model(model_name: str, weights: str, epochs: int = EPOCHS,
                batch_size: int = BATCH_SIZE, img_size: int = IMG_SIZE,
                device: str = DEVICE):
    """
    Train a single YOLO classification model.

    Returns:
        dict with training results
    """
    if not DATASET_DIR.exists() or not (DATASET_DIR / "train").exists():
        print(f"ERROR: Dataset not found at {DATASET_DIR}. Run prepare_dataset.py first.")
        return None

    print("\n" + "=" * 60)
    print(f"TRAINING {model_name.upper()}")
    print("=" * 60)
    print(f"  Weights : {weights}")
    print(f"  Epochs  : {epochs}")
    print(f"  Batch   : {batch_size}")
    print(f"  ImgSize : {img_size}")
    print(f"  Device  : {device}")
    print(f"  Dataset : {DATASET_DIR}")
    print("=" * 60 + "\n")

    model = YOLO(weights)

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"train-{model_name}") as run:
        # Log training parameters
        mlflow.log_params({
            "model_name": model_name,
            "weights": weights,
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "device": device,
            "patience": 15,
        })

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

        run_dir = RUNS_DIR / model_name
        best_weights = run_dir / "weights" / "best.pt"

        # Extract metrics
        metrics = {}
        if results and hasattr(results, 'results_dict'):
            rd = results.results_dict
            metrics["top1_accuracy"] = rd.get("metrics/accuracy_top1", 0)
            metrics["top5_accuracy"] = rd.get("metrics/accuracy_top5", 0)

        # Model size
        model_size_mb = 0
        if best_weights.exists():
            model_size_mb = best_weights.stat().st_size / (1024 * 1024)

        # Parameter count
        params = 0
        try:
            m = YOLO(str(best_weights))
            info = m.info(verbose=False)
            if info and len(info) > 1:
                params = info[1]
        except Exception:
            pass

        # Log metrics to MLflow
        mlflow.log_metrics({
            "training_time_seconds": round(training_time, 1),
            "model_size_mb": round(model_size_mb, 2),
            "parameters": params,
            **{k: float(v) for k, v in metrics.items()},
        })

        # Log best weights as artifact
        if best_weights.exists():
            mlflow.log_artifact(str(best_weights), artifact_path="weights")

        # Log training curves CSV if available
        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            mlflow.log_artifact(str(results_csv), artifact_path="training")

        mlflow.set_tag("stage", "training")
        mlflow.set_tag("model_type", "ultralytics")

        result_data = {
            "model": model_name,
            "weights": weights,
            "weights_path": str(best_weights),
            "run_dir": str(run_dir),
            "training_time_seconds": round(training_time, 1),
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "model_size_mb": round(model_size_mb, 2),
            "parameters": params,
            "mlflow_run_id": run.info.run_id,
            **metrics,
        }

    results_json = run_dir / "training_results.json"
    with open(results_json, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"\nResults saved to {results_json}")

    return result_data


def train_all(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
              img_size: int = IMG_SIZE, device: str = DEVICE):
    """Train all configured YOLO classification models."""
    all_results = {}

    models = YOLO_VERSIONS
    print(f"\nTraining {len(models)} models:")
    for name in models:
        print(f"  - {name}")

    for model_name, cfg in models.items():
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
        except Exception as e:
            print(f"\nERROR training {model_name}: {e}")
            all_results[model_name] = {"model": model_name, "error": str(e)}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "training_results.json"
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll training results saved to {out}")
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO classification models for seed recognition"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (e.g. yolov8n-cls, yolov11s-cls)")
    parser.add_argument("--all", action="store_true", help="Train all models")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()

    if args.all:
        train_all(args.epochs, args.batch_size, args.img_size, args.device)
    elif args.model:
        if args.model not in YOLO_VERSIONS:
            print(f"Unknown model: {args.model}")
            print(f"Available: {list(YOLO_VERSIONS.keys())}")
            return
        cfg = YOLO_VERSIONS[args.model]
        result = train_model(
            model_name=args.model,
            weights=cfg["weights"],
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device,
        )
        if result:
            print("\nDone. Results:")
            for k, v in result.items():
                print(f"  {k}: {v}")
    else:
        print("Specify --model <name> or --all")
        print(f"Available: {list(YOLO_VERSIONS.keys())}")


if __name__ == "__main__":
    main()
