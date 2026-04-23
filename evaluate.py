"""
Evaluate all trained YOLO classification models on the test set.
Collects: accuracy, precision, recall, F1, confusion matrix, inference speed.

Usage: python evaluate.py
"""

import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import mlflow
from PIL import Image
from ultralytics import YOLO

from config import (
    DATASET_DIR, RUNS_DIR, RESULTS_DIR, IMG_SIZE, DEVICE,
    SEED_CLASSES, YOLO_VERSIONS, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
)


def find_trained_models() -> dict:
    """Find all trained model best.pt files."""
    trained = {}
    for model_name in YOLO_VERSIONS:
        best = RUNS_DIR / model_name / "weights" / "best.pt"
        if best.exists():
            trained[model_name] = str(best)
    return trained


def evaluate_model(model_name: str, weights_path: str,
                   img_size: int = IMG_SIZE, device: str = DEVICE) -> dict:
    """Evaluate a single classification model on the test set."""
    test_dir = DATASET_DIR / "test"
    if not test_dir.exists():
        print(f"  Test dir not found: {test_dir}")
        return None

    print(f"\n  Evaluating {model_name}...")

    model = YOLO(weights_path)

    # Run validation on test split
    results = model.val(
        data=str(DATASET_DIR),
        split="test",
        imgsz=img_size,
        device=device,
        project=str(RUNS_DIR / "eval"),
        name=model_name,
        exist_ok=True,
        plots=True,
        verbose=False,
    )

    metrics = {
        "top1_accuracy": 0.0,
        "top5_accuracy": 0.0,
    }

    if results and hasattr(results, 'results_dict'):
        rd = results.results_dict
        metrics["top1_accuracy"] = float(rd.get("metrics/accuracy_top1", 0))
        metrics["top5_accuracy"] = float(rd.get("metrics/accuracy_top5", 0))

    # Per-class predictions for precision/recall/F1
    class_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    y_true = []
    y_pred = []

    for cls_idx, cls_dir in enumerate(class_dirs):
        images = sorted(
            f for f in cls_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
        )
        for img_path in images:
            pred = model.predict(str(img_path), imgsz=img_size, device=device, verbose=False)
            if pred and pred[0].probs is not None:
                pred_cls = int(pred[0].probs.top1)
                y_true.append(cls_idx)
                y_pred.append(pred_cls)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute per-class and macro metrics
    n_classes = len(class_names)
    per_class = {}
    precisions, recalls, f1s = [], [], []

    for i in range(n_classes):
        tp = int(np.sum((y_pred == i) & (y_true == i)))
        fp = int(np.sum((y_pred == i) & (y_true != i)))
        fn = int(np.sum((y_pred != i) & (y_true == i)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[class_names[i]] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(np.sum(y_true == i)),
        }
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    metrics["precision_macro"] = round(float(np.mean(precisions)), 4)
    metrics["recall_macro"] = round(float(np.mean(recalls)), 4)
    metrics["f1_macro"] = round(float(np.mean(f1s)), 4)
    metrics["overall_accuracy"] = round(float(np.mean(y_true == y_pred)), 4)
    metrics["per_class"] = per_class

    # Confusion matrix (as nested list for JSON)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    metrics["confusion_matrix"] = cm.tolist()
    metrics["class_names"] = class_names

    # Model size
    metrics["model_size_mb"] = round(Path(weights_path).stat().st_size / (1024 * 1024), 2)

    # Parameter count
    try:
        info = model.info(verbose=False)
        if info and len(info) > 1:
            metrics["parameters"] = info[1]
    except Exception:
        metrics["parameters"] = 0

    # Inference speed benchmark
    metrics["inference_time_ms"] = benchmark_inference(model, img_size, device)

    return metrics


def benchmark_inference(model, img_size: int, device: str, n_runs: int = 100) -> float:
    """Benchmark average inference time in ms."""
    dummy = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

    # Warmup
    for _ in range(10):
        model.predict(dummy, verbose=False, device=device)

    times = []
    for _ in range(n_runs):
        t0 = time.time()
        model.predict(dummy, verbose=False, device=device)
        times.append((time.time() - t0) * 1000)

    return round(float(np.mean(times)), 2)


def evaluate_all():
    """Evaluate all trained models."""
    trained = find_trained_models()

    if not trained:
        print("No trained models found in runs/. Train models first.")
        return {}

    print("=" * 60)
    print("EVALUATING ALL TRAINED MODELS")
    print("=" * 60)
    print(f"Found {len(trained)} models: {list(trained.keys())}")

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    all_results = {}
    for model_name, weights_path in trained.items():
        try:
            m = evaluate_model(model_name, weights_path)
            if m:
                m["model"] = model_name
                all_results[model_name] = m

                # Log evaluation metrics to MLflow
                with mlflow.start_run(run_name=f"eval-{model_name}") as run:
                    mlflow.set_tag("stage", "evaluation")
                    mlflow.set_tag("model_name", model_name)
                    mlflow.log_metrics({
                        "eval_top1_accuracy": m["top1_accuracy"],
                        "eval_top5_accuracy": m["top5_accuracy"],
                        "eval_precision_macro": m["precision_macro"],
                        "eval_recall_macro": m["recall_macro"],
                        "eval_f1_macro": m["f1_macro"],
                        "eval_overall_accuracy": m["overall_accuracy"],
                        "inference_time_ms": m["inference_time_ms"],
                        "model_size_mb": m["model_size_mb"],
                        "parameters": m.get("parameters", 0),
                    })
                    # Log per-class F1 scores
                    for cls_name, cls_metrics in m.get("per_class", {}).items():
                        mlflow.log_metric(
                            f"f1_{cls_name}", cls_metrics["f1"]
                        )

                print(f"\n  {model_name}:")
                print(f"    Top-1 Acc  : {m['top1_accuracy']:.4f}")
                print(f"    Top-5 Acc  : {m['top5_accuracy']:.4f}")
                print(f"    Precision  : {m['precision_macro']:.4f}")
                print(f"    Recall     : {m['recall_macro']:.4f}")
                print(f"    F1 (macro) : {m['f1_macro']:.4f}")
                print(f"    Inference  : {m['inference_time_ms']:.1f} ms")
                print(f"    Size       : {m['model_size_mb']:.1f} MB")
        except Exception as e:
            print(f"\n  ERROR {model_name}: {e}")
            all_results[model_name] = {"model": model_name, "error": str(e)}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "evaluation_results.json"
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {out}")

    return all_results


def main():
    results = evaluate_all()

    if results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        header = f"{'Model':<16} {'Top1':>6} {'Top5':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'ms':>7} {'MB':>7}"
        print(header)
        print("-" * len(header))
        for name, m in sorted(results.items()):
            if "error" in m:
                print(f"{name:<16} ERROR")
                continue
            print(
                f"{name:<16} "
                f"{m.get('top1_accuracy',0):>6.3f} "
                f"{m.get('top5_accuracy',0):>6.3f} "
                f"{m.get('precision_macro',0):>6.3f} "
                f"{m.get('recall_macro',0):>6.3f} "
                f"{m.get('f1_macro',0):>6.3f} "
                f"{m.get('inference_time_ms',0):>7.1f} "
                f"{m.get('model_size_mb',0):>7.1f}"
            )


if __name__ == "__main__":
    main()
