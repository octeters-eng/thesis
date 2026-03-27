"""
Train Ultralytics YOLO models (YOLOv8, YOLOv9, YOLOv10, YOLOv11) for seed recognition.
Uses the unified Ultralytics API.

Usage:
    python train_ultralytics.py --model yolov8n
    python train_ultralytics.py --model yolov9c
    python train_ultralytics.py --model yolov10n
    python train_ultralytics.py --model yolov11n
    python train_ultralytics.py --all
"""

import argparse
import json
import os
import time
from pathlib import Path

from ultralytics import YOLO

from config import (
    PROCESSED_DIR, RUNS_DIR, RESULTS_DIR,
    BATCH_SIZE, EPOCHS, IMG_SIZE, DEVICE,
    CONFIDENCE_THRESHOLD, IOU_THRESHOLD,
    YOLO_VERSIONS,
)


def get_ultralytics_models():
    """Get all Ultralytics-based model configurations."""
    return {k: v for k, v in YOLO_VERSIONS.items() if v["type"] == "ultralytics"}


def train_model(model_name: str, weights: str, epochs: int = EPOCHS,
                batch_size: int = BATCH_SIZE, img_size: int = IMG_SIZE,
                device: str = DEVICE):
    """
    Train a single Ultralytics YOLO model.
    
    Args:
        model_name: Name for this experiment (e.g., 'yolov8n')
        weights: Pretrained weights file (e.g., 'yolov8n.pt')
        epochs: Number of training epochs
        batch_size: Training batch size
        img_size: Input image size
        device: Device to use ('0' for GPU, 'cpu' for CPU)
    
    Returns:
        dict with training results
    """
    data_yaml = PROCESSED_DIR / "data.yaml"
    if not data_yaml.exists():
        print(f"ERROR: {data_yaml} not found. Run prepare_dataset.py first.")
        return None
    
    print("\n" + "=" * 60)
    print(f"TRAINING {model_name.upper()}")
    print("=" * 60)
    print(f"  Weights: {weights}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")
    print(f"  Device: {device}")
    print(f"  Data: {data_yaml}")
    print("=" * 60 + "\n")
    
    # Load model
    model = YOLO(weights)
    
    # Train
    start_time = time.time()
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=str(RUNS_DIR / "ultralytics"),
        name=model_name,
        exist_ok=True,
        patience=20,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )
    
    training_time = time.time() - start_time
    
    # Get best weights path
    run_dir = RUNS_DIR / "ultralytics" / model_name
    best_weights = run_dir / "weights" / "best.pt"
    
    # Extract metrics from training results
    metrics = {}
    if results and hasattr(results, 'results_dict'):
        rd = results.results_dict
        metrics = {
            "precision": rd.get("metrics/precision(B)", 0),
            "recall": rd.get("metrics/recall(B)", 0),
            "mAP50": rd.get("metrics/mAP50(B)", 0),
            "mAP50-95": rd.get("metrics/mAP50-95(B)", 0),
        }
    
    # Get model info
    model_info = {}
    try:
        best_model = YOLO(str(best_weights))
        info = best_model.info(verbose=False)
        if info:
            model_info["parameters"] = info[1] if len(info) > 1 else 0
            model_info["flops"] = info[2] if len(info) > 2 else 0
    except Exception:
        pass
    
    # Model file size
    model_size_mb = 0
    if best_weights.exists():
        model_size_mb = best_weights.stat().st_size / (1024 * 1024)
    
    result_data = {
        "model": model_name,
        "type": "ultralytics",
        "weights": weights,
        "weights_path": str(best_weights),
        "run_dir": str(run_dir),
        "training_time_seconds": training_time,
        "epochs": epochs,
        "batch_size": batch_size,
        "img_size": img_size,
        "model_size_mb": round(model_size_mb, 2),
        **metrics,
        **model_info,
    }
    
    # Save results
    results_json = run_dir / "training_results.json"
    with open(results_json, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"\nResults saved to {results_json}")
    
    return result_data


def validate_model(model_name: str, weights_path: str, img_size: int = IMG_SIZE,
                   device: str = DEVICE):
    """
    Validate a trained Ultralytics YOLO model on the test set.
    
    Returns:
        dict with validation metrics
    """
    data_yaml = PROCESSED_DIR / "data.yaml"
    
    print(f"\nValidating {model_name}...")
    
    model = YOLO(weights_path)
    
    # Validate on test split
    start_time = time.time()
    results = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=img_size,
        device=device,
        project=str(RUNS_DIR / "ultralytics_val"),
        name=model_name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    val_time = time.time() - start_time
    
    metrics = {}
    if results:
        metrics = {
            "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
            "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
        }
        
        # Calculate F1
        p = metrics["precision"]
        r = metrics["recall"]
        metrics["f1_score"] = 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    metrics["validation_time"] = val_time
    
    return metrics


def benchmark_inference(model_name: str, weights_path: str,
                        img_size: int = IMG_SIZE, device: str = DEVICE,
                        n_runs: int = 100):
    """
    Benchmark inference speed of a model.
    
    Returns:
        Average inference time in milliseconds
    """
    import numpy as np
    
    model = YOLO(weights_path)
    
    # Create dummy input for warmup and benchmark
    dummy_img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(10):
        model.predict(dummy_img, verbose=False, device=device)
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.time()
        model.predict(dummy_img, verbose=False, device=device)
        times.append((time.time() - start) * 1000)  # ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"  {model_name}: {avg_time:.2f} +/- {std_time:.2f} ms")
    return avg_time


def train_all(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
              img_size: int = IMG_SIZE, device: str = DEVICE):
    """Train all Ultralytics YOLO models."""
    models = get_ultralytics_models()
    all_results = {}
    
    print(f"\nTraining {len(models)} Ultralytics models:")
    for name in models:
        print(f"  - {name}")
    
    for model_name, model_config in models.items():
        try:
            result = train_model(
                model_name=model_name,
                weights=model_config["weights"],
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
    
    # Save all results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "ultralytics_training_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {results_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Train Ultralytics YOLO models (v8-v11) for seed recognition"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Model to train (e.g., yolov8n, yolov9c, yolov10n, yolov11n)")
    parser.add_argument("--all", action="store_true",
                        help="Train all Ultralytics models")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()
    
    if args.all:
        results = train_all(
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device,
        )
    elif args.model:
        models = get_ultralytics_models()
        if args.model not in models:
            print(f"Unknown model: {args.model}")
            print(f"Available: {list(models.keys())}")
            return
        
        result = train_model(
            model_name=args.model,
            weights=models[args.model]["weights"],
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device,
        )
        
        if result:
            print("\n" + "=" * 60)
            print("TRAINING COMPLETE")
            print("=" * 60)
            for key, value in result.items():
                print(f"  {key}: {value}")
    else:
        print("Specify --model <name> or --all")
        print(f"Available models: {list(get_ultralytics_models().keys())}")


if __name__ == "__main__":
    main()
