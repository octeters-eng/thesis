"""
Evaluate all trained YOLO models on the test set.
Collects metrics: precision, recall, mAP50, mAP50-95, F1, inference time, model size.

Usage: python evaluate.py
"""

import json
import os
import sys
import time
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

from config import (
    PROJECT_ROOT, PROCESSED_DIR, RUNS_DIR, RESULTS_DIR,
    YOLOV7_DIR, IMG_SIZE, DEVICE, CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD, YOLO_VERSIONS,
)


def evaluate_yolov7(model_name: str, weights_path: str,
                    img_size: int = IMG_SIZE, device: str = DEVICE):
    """Evaluate a YOLOv7 model on the test set."""
    data_yaml = str(PROCESSED_DIR / "data_yolov7.yaml")
    
    if not Path(weights_path).exists():
        print(f"  Weights not found: {weights_path}")
        return None
    
    test_dir = RUNS_DIR / "yolov7_eval" / model_name
    test_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, str(YOLOV7_DIR / "test.py"),
        "--weights", weights_path,
        "--data", data_yaml,
        "--img-size", str(img_size),
        "--device", device,
        "--task", "test",
        "--project", str(RUNS_DIR / "yolov7_eval"),
        "--name", model_name,
        "--exist-ok",
        "--verbose",
        "--save-json",
        "--save-txt",
    ]
    
    print(f"\n  Evaluating {model_name}...")
    result = subprocess.run(
        cmd, cwd=str(YOLOV7_DIR),
        capture_output=True, text=True
    )
    
    output = result.stdout + result.stderr
    
    # Parse metrics from output
    metrics = parse_yolov7_output(output)
    
    # Get model size
    model_size = Path(weights_path).stat().st_size / (1024 * 1024)
    metrics["model_size_mb"] = round(model_size, 2)
    
    # Benchmark inference speed
    metrics["inference_time_ms"] = benchmark_yolov7_inference(
        weights_path, img_size, device
    )
    
    # Get parameter count
    metrics["parameters"] = count_yolov7_parameters(weights_path)
    
    return metrics


def parse_yolov7_output(output: str) -> dict:
    """Parse YOLOv7 test.py output for metrics."""
    metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "mAP50": 0.0,
        "mAP50-95": 0.0,
        "f1_score": 0.0,
    }
    
    for line in output.split("\n"):
        # Look for the "all" summary line
        if " all " in line and "AP" not in line:
            parts = line.strip().split()
            try:
                # Format: Class, Images, Labels, P, R, mAP@.5, mAP@.5:.95
                for i, part in enumerate(parts):
                    if part == "all":
                        vals = [float(x) for x in parts[i+1:] if _is_float(x)]
                        if len(vals) >= 4:
                            metrics["precision"] = vals[-4] if len(vals) >= 4 else 0
                            metrics["recall"] = vals[-3] if len(vals) >= 3 else 0
                            metrics["mAP50"] = vals[-2] if len(vals) >= 2 else 0
                            metrics["mAP50-95"] = vals[-1] if len(vals) >= 1 else 0
                        break
            except (ValueError, IndexError):
                pass
    
    p, r = metrics["precision"], metrics["recall"]
    metrics["f1_score"] = 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    return metrics


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def benchmark_yolov7_inference(weights_path: str, img_size: int = IMG_SIZE,
                                device: str = DEVICE, n_runs: int = 50):
    """Benchmark YOLOv7 inference speed."""
    try:
        sys.path.insert(0, str(YOLOV7_DIR))
        from models.experimental import attempt_load
        from utils.general import check_img_size
        
        dev = torch.device(f"cuda:{device}" if device.isdigit() and torch.cuda.is_available() else "cpu")
        model = attempt_load(weights_path, map_location=dev)
        model.eval()
        
        stride = int(model.stride.max())
        imgsz = check_img_size(img_size, s=stride)
        
        dummy = torch.zeros(1, 3, imgsz, imgsz).to(dev)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                model(dummy)
        
        # Benchmark
        times = []
        for _ in range(n_runs):
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                model(dummy)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)
        
        sys.path.pop(0)
        return round(float(np.mean(times)), 2)
    
    except Exception as e:
        print(f"    Inference benchmark failed: {e}")
        return 0.0


def count_yolov7_parameters(weights_path: str):
    """Count parameters in YOLOv7 model."""
    try:
        ckpt = torch.load(weights_path, map_location="cpu")
        if "model" in ckpt:
            model = ckpt["model"]
            if hasattr(model, 'parameters'):
                return sum(p.numel() for p in model.parameters())
        return 0
    except Exception:
        return 0


def evaluate_ultralytics(model_name: str, weights_path: str,
                         img_size: int = IMG_SIZE, device: str = DEVICE):
    """Evaluate an Ultralytics YOLO model on the test set."""
    from ultralytics import YOLO
    
    if not Path(weights_path).exists():
        print(f"  Weights not found: {weights_path}")
        return None
    
    data_yaml = str(PROCESSED_DIR / "data.yaml")
    
    print(f"\n  Evaluating {model_name}...")
    
    model = YOLO(weights_path)
    
    # Validate on test split
    results = model.val(
        data=data_yaml,
        split="test",
        imgsz=img_size,
        device=device,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        project=str(RUNS_DIR / "ultralytics_eval"),
        name=model_name,
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    
    # Extract metrics
    metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "mAP50": 0.0,
        "mAP50-95": 0.0,
        "f1_score": 0.0,
    }
    
    if results and hasattr(results, 'results_dict'):
        rd = results.results_dict
        metrics["precision"] = float(rd.get("metrics/precision(B)", 0))
        metrics["recall"] = float(rd.get("metrics/recall(B)", 0))
        metrics["mAP50"] = float(rd.get("metrics/mAP50(B)", 0))
        metrics["mAP50-95"] = float(rd.get("metrics/mAP50-95(B)", 0))
        
        p, r = metrics["precision"], metrics["recall"]
        metrics["f1_score"] = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0
    
    # Per-class metrics
    if results and hasattr(results, 'box'):
        box = results.box
        if hasattr(box, 'ap_class_index') and hasattr(box, 'ap50'):
            # Load class names
            with open(data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
            class_names = data_config.get("names", [])
            
            per_class = {}
            for i, cls_idx in enumerate(box.ap_class_index):
                cls_name = class_names[int(cls_idx)] if int(cls_idx) < len(class_names) else f"class_{cls_idx}"
                per_class[cls_name] = {
                    "ap50": float(box.ap50[i]) if i < len(box.ap50) else 0,
                    "ap": float(box.ap[i]) if hasattr(box, 'ap') and i < len(box.ap) else 0,
                }
            metrics["per_class"] = per_class
    
    # Model size
    model_size = Path(weights_path).stat().st_size / (1024 * 1024)
    metrics["model_size_mb"] = round(model_size, 2)
    
    # Parameter count and FLOPs
    try:
        info = model.info(verbose=False)
        if info and len(info) > 1:
            metrics["parameters"] = info[1]
        if info and len(info) > 2:
            metrics["flops"] = info[2]
    except Exception:
        pass
    
    # Inference speed benchmark
    metrics["inference_time_ms"] = benchmark_ultralytics_inference(
        weights_path, img_size, device
    )
    
    return metrics


def benchmark_ultralytics_inference(weights_path: str, img_size: int = IMG_SIZE,
                                     device: str = DEVICE, n_runs: int = 100):
    """Benchmark Ultralytics model inference speed."""
    from ultralytics import YOLO
    
    try:
        model = YOLO(weights_path)
        dummy = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(10):
            model.predict(dummy, verbose=False, device=device)
        
        # Benchmark
        times = []
        for _ in range(n_runs):
            t0 = time.time()
            model.predict(dummy, verbose=False, device=device)
            times.append((time.time() - t0) * 1000)
        
        return round(float(np.mean(times)), 2)
    except Exception as e:
        print(f"    Inference benchmark failed: {e}")
        return 0.0


def find_trained_models():
    """Find all trained model weights."""
    trained = {}
    
    # Check YOLOv7 runs
    yolov7_dir = RUNS_DIR / "yolov7"
    if yolov7_dir.exists():
        for run in yolov7_dir.iterdir():
            if run.is_dir():
                best = run / "weights" / "best.pt"
                if best.exists():
                    trained[run.name] = {
                        "weights_path": str(best),
                        "type": "yolov7",
                    }
    
    # Check Ultralytics runs
    ultra_dir = RUNS_DIR / "ultralytics"
    if ultra_dir.exists():
        for run in ultra_dir.iterdir():
            if run.is_dir():
                best = run / "weights" / "best.pt"
                if best.exists():
                    trained[run.name] = {
                        "weights_path": str(best),
                        "type": "ultralytics",
                    }
    
    return trained


def evaluate_all():
    """Evaluate all trained models."""
    trained = find_trained_models()
    
    if not trained:
        print("No trained models found!")
        print("Run training scripts first:")
        print("  python train_yolov7.py --weights yolov7.pt")
        print("  python train_ultralytics.py --all")
        return {}
    
    print("=" * 60)
    print("EVALUATING ALL TRAINED MODELS")
    print("=" * 60)
    print(f"Found {len(trained)} trained models:")
    for name, info in trained.items():
        print(f"  - {name} ({info['type']})")
    
    all_results = {}
    
    for model_name, model_info in trained.items():
        try:
            if model_info["type"] == "yolov7":
                metrics = evaluate_yolov7(
                    model_name, model_info["weights_path"]
                )
            else:
                metrics = evaluate_ultralytics(
                    model_name, model_info["weights_path"]
                )
            
            if metrics:
                metrics["model"] = model_name
                metrics["type"] = model_info["type"]
                all_results[model_name] = metrics
                
                print(f"\n  {model_name} results:")
                for k, v in metrics.items():
                    if k != "per_class":
                        print(f"    {k}: {v}")
        
        except Exception as e:
            print(f"\n  ERROR evaluating {model_name}: {e}")
            all_results[model_name] = {
                "model": model_name,
                "error": str(e),
            }
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n\nAll evaluation results saved to {results_file}")
    return all_results


def main():
    results = evaluate_all()
    
    if results:
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        # Quick comparison table
        header = f"{'Model':<15} {'Prec':>6} {'Rec':>6} {'mAP50':>7} {'mAP50-95':>9} {'F1':>6} {'ms':>7} {'MB':>7}"
        print(header)
        print("-" * len(header))
        
        for name, m in sorted(results.items()):
            if "error" in m:
                print(f"{name:<15} ERROR: {m['error']}")
                continue
            print(
                f"{name:<15} "
                f"{m.get('precision', 0):>6.3f} "
                f"{m.get('recall', 0):>6.3f} "
                f"{m.get('mAP50', 0):>7.3f} "
                f"{m.get('mAP50-95', 0):>9.3f} "
                f"{m.get('f1_score', 0):>6.3f} "
                f"{m.get('inference_time_ms', 0):>7.1f} "
                f"{m.get('model_size_mb', 0):>7.1f}"
            )


if __name__ == "__main__":
    main()
