"""
Main orchestrator for the Seed Recognition YOLO Comparison project.
Runs the complete pipeline: download → prepare → train → evaluate → compare.

Usage:
    python main.py                    # Run full pipeline
    python main.py --step download    # Run specific step
    python main.py --step prepare
    python main.py --step train
    python main.py --step evaluate
    python main.py --step compare
    python main.py --step train --model yolov8n   # Train single model
"""

import argparse
import json
import sys
import time
from pathlib import Path

from config import (
    PROJECT_ROOT, PROCESSED_DIR, RUNS_DIR, RESULTS_DIR,
    EPOCHS, BATCH_SIZE, IMG_SIZE, DEVICE,
    YOLO_VERSIONS, create_directories,
)


def step_download():
    """Step 1: Download dataset from Google Drive."""
    print("\n" + "=" * 70)
    print("STEP 1: DOWNLOAD DATASET")
    print("=" * 70)
    from download_dataset import main as download_main
    download_main()


def step_prepare():
    """Step 2: Prepare dataset for YOLO training."""
    print("\n" + "=" * 70)
    print("STEP 2: PREPARE DATASET")
    print("=" * 70)
    from prepare_dataset import main as prepare_main
    prepare_main()


def step_train(model_name: str = None, epochs: int = EPOCHS,
               batch_size: int = BATCH_SIZE, img_size: int = IMG_SIZE,
               device: str = DEVICE):
    """Step 3: Train YOLO models."""
    print("\n" + "=" * 70)
    print("STEP 3: TRAIN MODELS")
    print("=" * 70)
    
    data_yaml = PROCESSED_DIR / "data.yaml"
    if not data_yaml.exists():
        print("ERROR: Dataset not prepared. Run step 'prepare' first.")
        return
    
    all_results = {}
    
    if model_name:
        # Train specific model
        models_to_train = {model_name: YOLO_VERSIONS[model_name]}
    else:
        # Train all models
        models_to_train = YOLO_VERSIONS
    
    for name, config in models_to_train.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        try:
            if config["type"] == "yolov7":
                from train_yolov7 import train_yolov7
                result = train_yolov7(
                    weights=config["weights"],
                    epochs=epochs,
                    batch_size=batch_size,
                    img_size=img_size,
                    device=device,
                    name=name,
                )
            else:
                from train_ultralytics import train_model
                result = train_model(
                    model_name=name,
                    weights=config["weights"],
                    epochs=epochs,
                    batch_size=batch_size,
                    img_size=img_size,
                    device=device,
                )
            
            elapsed = time.time() - start_time
            
            if result:
                result["total_time"] = elapsed
                all_results[name] = result
                print(f"\n  {name} completed in {elapsed/60:.1f} minutes")
            else:
                print(f"\n  {name} training failed")
                all_results[name] = {"model": name, "error": "Training returned None"}
        
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n  ERROR training {name}: {e}")
            all_results[name] = {"model": name, "error": str(e)}
    
    # Save training summary
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_file = RESULTS_DIR / "training_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nTraining summary saved to {summary_file}")
    
    return all_results


def step_evaluate():
    """Step 4: Evaluate all trained models."""
    print("\n" + "=" * 70)
    print("STEP 4: EVALUATE MODELS")
    print("=" * 70)
    from evaluate import evaluate_all
    return evaluate_all()


def step_compare():
    """Step 5: Generate comparison charts and tables."""
    print("\n" + "=" * 70)
    print("STEP 5: GENERATE COMPARISON RESULTS")
    print("=" * 70)
    from compare_results import main as compare_main
    compare_main()


def run_full_pipeline(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
                      img_size: int = IMG_SIZE, device: str = DEVICE):
    """Run the complete pipeline."""
    total_start = time.time()
    
    print("=" * 70)
    print("SEED RECOGNITION - YOLO MODEL COMPARISON")
    print("Full Pipeline")
    print("=" * 70)
    print(f"\nSettings:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")
    print(f"  Device: {device}")
    print(f"  Models: {list(YOLO_VERSIONS.keys())}")
    
    create_directories()
    
    # Step 1: Download
    step_download()
    
    # Step 2: Prepare
    step_prepare()
    
    # Step 3: Train
    step_train(epochs=epochs, batch_size=batch_size,
               img_size=img_size, device=device)
    
    # Step 4: Evaluate
    step_evaluate()
    
    # Step 5: Compare
    step_compare()
    
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Seed Recognition YOLO Comparison - Main Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  download   - Download dataset from Google Drive
  prepare    - Prepare dataset (split, convert to YOLO format)
  train      - Train all YOLO models (v7 through v11)
  evaluate   - Evaluate all trained models on test set
  compare    - Generate comparison charts, tables, analysis

Examples:
  python main.py                          # Run full pipeline
  python main.py --step train             # Train all models
  python main.py --step train --model yolov8n  # Train one model
  python main.py --step evaluate          # Evaluate trained models
  python main.py --step compare           # Generate reports
        """
    )
    parser.add_argument("--step", type=str, default=None,
                        choices=["download", "prepare", "train", "evaluate", "compare"],
                        help="Run a specific step only")
    parser.add_argument("--model", type=str, default=None,
                        help="Train a specific model (used with --step train)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()
    
    if args.step:
        if args.step == "download":
            step_download()
        elif args.step == "prepare":
            step_prepare()
        elif args.step == "train":
            step_train(
                model_name=args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                img_size=args.img_size,
                device=args.device,
            )
        elif args.step == "evaluate":
            step_evaluate()
        elif args.step == "compare":
            step_compare()
    else:
        run_full_pipeline(
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device,
        )


if __name__ == "__main__":
    main()
