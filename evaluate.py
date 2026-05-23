"""
Evaluate all trained YOLO classification models on the test set.

Collected metrics per model:
    - Top-1 / Top-5 accuracy (from Ultralytics validation)
    - Macro precision, recall, F1  (computed from per-image predictions)
    - Confusion matrix
    - Inference speed (ms)
    - Model size (MB) and parameter count

All metrics and the confusion matrix figure are logged to MLflow.
Results are also saved to results/evaluation_results.json.

Usage:
    python evaluate.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import mlflow
from ultralytics import YOLO

from config import (
    DATASET_DIR, RUNS_DIR, RESULTS_DIR,
    IMG_SIZE, DEVICE, YOLO_VERSIONS,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
)

_SEPARATOR = "=" * 60
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})


# ── Model helpers ──────────────────────────────────────────────────────────────

def find_trained_models() -> dict[str, str]:
    """Return a mapping of model name → path to best.pt for all trained models."""
    return {
        name: str(RUNS_DIR / name / "weights" / "best.pt")
        for name in YOLO_VERSIONS
        if (RUNS_DIR / name / "weights" / "best.pt").exists()
    }


def get_label_map(model: YOLO) -> dict[int, str]:
    """Return a prediction-index → class-name mapping from model metadata.

    Args:
        model: A loaded Ultralytics YOLO model.

    Returns:
        Dict mapping integer prediction indices to class name strings.
    """
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


# ── Core evaluation ────────────────────────────────────────────────────────────

def evaluate_model(
    model_name: str,
    weights_path: str,
    img_size: int = IMG_SIZE,
    device: str = DEVICE,
) -> dict | None:
    """Evaluate a single trained classification model on the test split.

    Args:
        model_name:    Human-readable model identifier.
        weights_path:  Absolute path to the ``best.pt`` weights file.
        img_size:      Input resolution used during inference.
        device:        Compute device (``"mps"``, ``"cpu"``, or CUDA index).

    Returns:
        Dictionary of evaluation metrics, or ``None`` if the test directory is
        missing.
    """
    test_dir = DATASET_DIR / "test"
    if not test_dir.exists():
        print(f"  Test directory not found: {test_dir}")
        return None

    print(f"\n  Evaluating {model_name} ...")
    model = YOLO(weights_path)

    # ── Ultralytics validation ─────────────────────────────────────────────────
    val_results = model.val(
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

    metrics: dict = {"top1_accuracy": 0.0, "top5_accuracy": 0.0}
    if val_results and hasattr(val_results, "results_dict"):
        rd = val_results.results_dict
        metrics["top1_accuracy"] = float(rd.get("metrics/accuracy_top1", 0))
        metrics["top5_accuracy"] = float(rd.get("metrics/accuracy_top5", 0))

    # ── Per-image predictions for precision / recall / F1 ─────────────────────
    class_dirs  = sorted(d for d in test_dir.iterdir() if d.is_dir())
    class_names = [d.name for d in class_dirs]
    label_map   = get_label_map(model)

    y_true: list[str] = []
    y_pred: list[str] = []

    for cls_dir in class_dirs:
        images = sorted(
            f for f in cls_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        for img_path in images:
            pred = model.predict(str(img_path), imgsz=img_size, device=device, verbose=False)
            if pred and pred[0].probs is not None:
                pred_idx = int(pred[0].probs.top1)
                y_true.append(cls_dir.name)
                y_pred.append(label_map.get(pred_idx, f"idx_{pred_idx}"))

    y_true_arr = np.array(y_true, dtype=object)
    y_pred_arr = np.array(y_pred, dtype=object)

    # Warn if dataset class names and model label names do not align exactly.
    dataset_classes = set(class_names)
    model_classes   = set(label_map.values())
    if dataset_classes != model_classes:
        metrics["label_alignment_warning"] = {
            "missing_in_model":   sorted(dataset_classes - model_classes),
            "extra_in_model":     sorted(model_classes - dataset_classes),
        }
        print("  WARNING: dataset and model class names do not match exactly.")

    # Build ordered label list (dataset classes first, then any unseen predictions).
    label_order = list(class_names)
    for label in sorted(set(y_pred)):
        if label not in label_order:
            label_order.append(label)

    # ── Per-class metrics ──────────────────────────────────────────────────────
    per_class: dict = {}
    precisions, recalls, f1_scores = [], [], []

    for label in label_order:
        tp = int(np.sum((y_pred_arr == label) & (y_true_arr == label)))
        fp = int(np.sum((y_pred_arr == label) & (y_true_arr != label)))
        fn = int(np.sum((y_pred_arr != label) & (y_true_arr == label)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[label] = {
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "support":   int(np.sum(y_true_arr == label)),
        }
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

    metrics["precision_macro"]    = round(float(np.mean(precisions)), 4)
    metrics["recall_macro"]       = round(float(np.mean(recalls)),    4)
    metrics["f1_macro"]           = round(float(np.mean(f1_scores)),  4)
    metrics["overall_accuracy"]   = round(float(np.mean(y_true_arr == y_pred_arr)), 4)
    metrics["per_class"]          = per_class

    # ── Confusion matrix ───────────────────────────────────────────────────────
    label_to_idx = {name: i for i, name in enumerate(label_order)}
    cm = np.zeros((len(label_order), len(label_order)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[label_to_idx[true], label_to_idx[pred]] += 1
    metrics["confusion_matrix"] = cm.tolist()
    metrics["class_names"]      = label_order

    # ── Model size and parameter count ─────────────────────────────────────────
    metrics["model_size_mb"] = round(Path(weights_path).stat().st_size / (1024 * 1024), 2)
    try:
        info = model.info(verbose=False)
        metrics["parameters"] = info[1] if (info and len(info) > 1) else 0
    except Exception:
        metrics["parameters"] = 0

    metrics["inference_time_ms"] = benchmark_inference(model, img_size, device)
    return metrics


def benchmark_inference(model: YOLO, img_size: int, device: str, n_runs: int = 100) -> float:
    """Measure mean inference latency over *n_runs* iterations.

    A 10-image warmup is performed first to exclude JIT / device-init overhead.

    Args:
        model:   Loaded Ultralytics YOLO model.
        img_size: Spatial resolution of the dummy input image.
        device:  Compute device string.
        n_runs:  Number of timed inference calls.

    Returns:
        Mean inference time in milliseconds.
    """
    dummy = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

    for _ in range(10):                     # warmup
        model.predict(dummy, verbose=False, device=device)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=False, device=device)
        times.append((time.perf_counter() - t0) * 1000)

    return round(float(np.mean(times)), 2)


# ── Orchestration ──────────────────────────────────────────────────────────────

def evaluate_all() -> dict:
    """Evaluate every trained model and log results to MLflow.

    Returns:
        Mapping of model name → metrics dictionary (may contain an ``"error"``
        key for models that failed evaluation).
    """
    trained = find_trained_models()
    if not trained:
        print("No trained models found in runs/. Train models first.")
        return {}

    print(_SEPARATOR)
    print("EVALUATING ALL TRAINED MODELS")
    print(_SEPARATOR)
    print(f"Found {len(trained)} models: {list(trained.keys())}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    all_results: dict = {}

    for model_name, weights_path in trained.items():
        try:
            m = evaluate_model(model_name, weights_path)
            if m is None:
                continue
            m["model"] = model_name
            all_results[model_name] = m

            # ── MLflow logging ────────────────────────────────────────────────
            with mlflow.start_run(run_name=f"eval-{model_name}"):
                mlflow.set_tag("stage", "evaluation")
                mlflow.set_tag("model_name", model_name)

                mlflow.log_metrics({
                    "eval_top1_accuracy":    m["top1_accuracy"],
                    "eval_top5_accuracy":    m["top5_accuracy"],
                    "eval_precision_macro":  m["precision_macro"],
                    "eval_recall_macro":     m["recall_macro"],
                    "eval_f1_macro":         m["f1_macro"],
                    "eval_overall_accuracy": m["overall_accuracy"],
                    "inference_time_ms":     m["inference_time_ms"],
                    "model_size_mb":         m["model_size_mb"],
                    "parameters":            m.get("parameters", 0),
                })

                for cls_name, cls_m in m.get("per_class", {}).items():
                    mlflow.log_metric(f"f1_{cls_name}",        cls_m["f1"])
                    mlflow.log_metric(f"precision_{cls_name}", cls_m["precision"])
                    mlflow.log_metric(f"recall_{cls_name}",    cls_m["recall"])

                # Confusion matrix figure
                cm          = m.get("confusion_matrix")
                class_names = m.get("class_names")
                if cm is not None and class_names is not None:
                    fig, ax = plt.subplots(figsize=(12, 10))
                    sns.heatmap(
                        np.array(cm), annot=True, fmt="d", cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names, ax=ax,
                    )
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    ax.set_title(f"Confusion Matrix — {model_name}")
                    plt.xticks(rotation=45, ha="right", fontsize=7)
                    plt.yticks(fontsize=7)
                    plt.tight_layout()
                    mlflow.log_figure(fig, f"confusion_matrix_{model_name}.png")
                    plt.close(fig)

            print(f"\n  {model_name}:")
            print(f"    Top-1 Acc  : {m['top1_accuracy']:.4f}")
            print(f"    Top-5 Acc  : {m['top5_accuracy']:.4f}")
            print(f"    Precision  : {m['precision_macro']:.4f}")
            print(f"    Recall     : {m['recall_macro']:.4f}")
            print(f"    F1 (macro) : {m['f1_macro']:.4f}")
            print(f"    Inference  : {m['inference_time_ms']:.1f} ms")
            print(f"    Size       : {m['model_size_mb']:.1f} MB")

        except Exception as exc:
            print(f"\n  ERROR evaluating {model_name}: {exc}")
            all_results[model_name] = {"model": model_name, "error": str(exc)}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "evaluation_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)

    print(f"\nEvaluation complete. {len(all_results)} models logged to MLflow.")
    print(f"Saved: {out_path}")
    return all_results


def main() -> None:
    results = evaluate_all()

    if not results:
        return

    print("\n" + _SEPARATOR)
    print("SUMMARY")
    print(_SEPARATOR)
    header = f"{'Model':<16} {'Top1':>6} {'Top5':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'ms':>7} {'MB':>7}"
    print(header)
    print("-" * len(header))

    for name, m in sorted(results.items()):
        if "error" in m:
            print(f"{name:<16} ERROR: {m['error']}")
            continue
        print(
            f"{name:<16} "
            f"{m.get('top1_accuracy', 0):>6.3f} "
            f"{m.get('top5_accuracy', 0):>6.3f} "
            f"{m.get('precision_macro', 0):>6.3f} "
            f"{m.get('recall_macro', 0):>6.3f} "
            f"{m.get('f1_macro', 0):>6.3f} "
            f"{m.get('inference_time_ms', 0):>7.1f} "
            f"{m.get('model_size_mb', 0):>7.1f}"
        )


if __name__ == "__main__":
    main()
