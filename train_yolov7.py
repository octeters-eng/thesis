"""
Train YOLOv7 model for seed recognition.
Clones the official YOLOv7 repo and runs training.

Usage: python train_yolov7.py [--weights yolov7.pt] [--epochs 100]
"""

import os
import sys
import subprocess
import argparse
import time
import json
from pathlib import Path

from config import (
    PROJECT_ROOT, YOLOV7_DIR, PROCESSED_DIR, RUNS_DIR,
    BATCH_SIZE, EPOCHS, IMG_SIZE, DEVICE,
)


def setup_yolov7():
    """Clone and set up YOLOv7 repository."""
    if YOLOV7_DIR.exists() and (YOLOV7_DIR / "train.py").exists():
        print("YOLOv7 repository already exists.")
        return True
    
    print("Cloning YOLOv7 repository...")
    result = subprocess.run(
        ["git", "clone", "https://github.com/WongKinYiu/yolov7.git", str(YOLOV7_DIR)],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print(f"Error cloning YOLOv7: {result.stderr}")
        return False
    
    # Install YOLOv7 requirements
    req_file = YOLOV7_DIR / "requirements.txt"
    if req_file.exists():
        print("Installing YOLOv7 requirements...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
            capture_output=True, text=True
        )
    
    return True


def download_weights(weights_name: str):
    """Download pretrained YOLOv7 weights."""
    weights_path = YOLOV7_DIR / weights_name
    
    if weights_path.exists():
        print(f"Weights already exist: {weights_path}")
        return weights_path
    
    base_url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1"
    url = f"{base_url}/{weights_name}"
    
    print(f"Downloading {weights_name}...")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(weights_path))
        print(f"Downloaded to {weights_path}")
    except Exception as e:
        print(f"Error downloading weights: {e}")
        print(f"Please download manually from: {url}")
        print(f"Place in: {weights_path}")
        return None
    
    return weights_path


def train_yolov7(weights: str = "yolov7.pt", epochs: int = EPOCHS,
                 batch_size: int = BATCH_SIZE, img_size: int = IMG_SIZE,
                 device: str = DEVICE, name: str = None):
    """
    Train YOLOv7 model.
    
    Args:
        weights: Pretrained weights filename
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Input image size
        device: GPU device
        name: Experiment name
    
    Returns:
        dict with training results and paths
    """
    if not setup_yolov7():
        return None
    
    weights_path = download_weights(weights)
    if weights_path is None:
        return None
    
    data_yaml = PROCESSED_DIR / "data_yolov7.yaml"
    if not data_yaml.exists():
        print(f"ERROR: {data_yaml} not found. Run prepare_dataset.py first.")
        return None
    
    if name is None:
        name = weights.replace(".pt", "")
    
    run_dir = RUNS_DIR / "yolov7" / name
    
    # Select config based on model variant
    if "tiny" in weights:
        cfg = YOLOV7_DIR / "cfg" / "training" / "yolov7-tiny.yaml"
    elif "x" in weights:
        cfg = YOLOV7_DIR / "cfg" / "training" / "yolov7-x.yaml"
    elif "w6" in weights:
        cfg = YOLOV7_DIR / "cfg" / "training" / "yolov7-w6.yaml"
    else:
        cfg = YOLOV7_DIR / "cfg" / "training" / "yolov7.yaml"
    
    # Update cfg with correct number of classes
    import yaml
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    nc = data_config.get("nc", 1)
    
    # Read and modify the model config for correct nc
    custom_cfg = run_dir / "model.yaml"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    if cfg.exists():
        with open(cfg, 'r') as f:
            model_cfg = yaml.safe_load(f)
        model_cfg['nc'] = nc
        with open(custom_cfg, 'w') as f:
            yaml.dump(model_cfg, f)
    
    # Build training command
    cmd = [
        sys.executable, str(YOLOV7_DIR / "train.py"),
        "--weights", str(weights_path),
        "--data", str(data_yaml),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--img-size", str(img_size), str(img_size),
        "--device", device,
        "--project", str(RUNS_DIR / "yolov7"),
        "--name", name,
        "--exist-ok",
        "--hyp", str(YOLOV7_DIR / "data" / "hyp.scratch.p5.yaml"),
    ]
    
    if custom_cfg.exists():
        cmd.extend(["--cfg", str(custom_cfg)])
    
    print("\n" + "=" * 60)
    print(f"TRAINING YOLOv7 ({name})")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, Image size: {img_size}")
    print(f"Data: {data_yaml}")
    print(f"Output: {run_dir}")
    print("=" * 60 + "\n")
    
    start_time = time.time()
    
    result = subprocess.run(
        cmd,
        cwd=str(YOLOV7_DIR),
        text=True,
    )
    
    training_time = time.time() - start_time
    
    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        return None
    
    # Collect results
    results = {
        "model": name,
        "type": "yolov7",
        "weights_path": str(run_dir / "weights" / "best.pt"),
        "run_dir": str(run_dir),
        "training_time_seconds": training_time,
        "epochs": epochs,
        "batch_size": batch_size,
        "img_size": img_size,
    }
    
    # Parse results.txt if exists
    results_file = run_dir / "results.txt"
    if results_file.exists():
        with open(results_file, 'r') as f:
            lines = f.readlines()
        if lines:
            # Last line has final metrics
            last_line = lines[-1].strip().split()
            if len(last_line) >= 14:
                results["precision"] = float(last_line[8])
                results["recall"] = float(last_line[9])
                results["mAP50"] = float(last_line[10])
                results["mAP50-95"] = float(last_line[11])
    
    # Save results
    results_json = run_dir / "training_results.json"
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_json}")
    
    return results


def test_yolov7(weights_path: str, data_yaml: str = None,
                img_size: int = IMG_SIZE, device: str = DEVICE,
                name: str = "test"):
    """Run YOLOv7 testing/evaluation."""
    if data_yaml is None:
        data_yaml = str(PROCESSED_DIR / "data_yolov7.yaml")
    
    cmd = [
        sys.executable, str(YOLOV7_DIR / "test.py"),
        "--weights", weights_path,
        "--data", data_yaml,
        "--img-size", str(img_size),
        "--device", device,
        "--task", "test",
        "--project", str(RUNS_DIR / "yolov7_test"),
        "--name", name,
        "--exist-ok",
        "--save-json",
        "--verbose",
    ]
    
    print(f"\nTesting YOLOv7: {name}")
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=str(YOLOV7_DIR), capture_output=True, text=True)
    inference_time = time.time() - start_time
    
    output = result.stdout + result.stderr
    print(output)
    
    return {
        "model": name,
        "total_time": inference_time,
        "output": output,
    }


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv7 for seed recognition")
    parser.add_argument("--weights", type=str, default="yolov7.pt",
                        help="Pretrained weights (yolov7.pt or yolov7-tiny.pt)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()
    
    results = train_yolov7(
        weights=args.weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        name=args.name,
    )
    
    if results:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        for key, value in results.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
