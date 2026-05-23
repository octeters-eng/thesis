# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Benchmarks four Ultralytics YOLO image-classification models on a 16-class Bangladeshi seed/vegetable dataset. The 2×2 study design (YOLOv8 vs YOLO11, nano vs small) isolates the effect of architecture and model size under identical hyperparameters.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The default compute device is `mps` (Apple Silicon). Change `DEVICE` in `config.py` to `"0"` for CUDA or `"cpu"` for CPU.

## Pipeline Commands

```bash
# Full pipeline (prepare → train → evaluate → compare)
python main.py
python main.py --yes          # skip interactive prompts

# Individual steps
python main.py --step download   # fetch dataset from Google Drive (needs SEED_GDRIVE_FOLDER_ID env var)
python main.py --step prepare
python main.py --step train
python main.py --step evaluate
python main.py --step compare

# Train directly
python train_ultralytics.py --all
python train_ultralytics.py --model yolov8n-cls --epochs 100 --batch-size 16 --device mps

# View experiment tracking
mlflow ui
```

## Architecture

### Configuration as Single Source of Truth

`config.py` is the only place to change: paths, the 16 class names, model definitions (`YOLO_VERSIONS` dict), image size (224), train/val/test split ratios (70/15/15), and training hyperparameters (`EPOCHS`, `BATCH_SIZE`, `DEVICE`). Every other script imports from it.

### Pipeline Flow

`main.py` orchestrates four scripts as subprocesses in sequence:

1. **`prepare_dataset.py`** — Recursively collects images from `data/SEEED/<class>/`, flattens subfolders, and copies into `dataset/train|val|test/<class>/` using a deterministic unique naming scheme to avoid collisions. Prompts before overwriting an existing split (`--yes` skips).

2. **`train_ultralytics.py`** — Calls `model.train()` from Ultralytics for each entry in `YOLO_VERSIONS`. Checkpoints land at `runs/<model_name>/weights/best.pt`. Ultralytics starts its own internal MLflow callback; the script ends that run then opens a fresh `train-<model_name>` MLflow run to log a clean summary (params, timing, model size, parameter count, `best.pt` artifact, `results.csv`).

3. **`evaluate.py`** — Loads each `best.pt`, runs `model.val(split="test")` for Top-1/Top-5, then runs `model.predict()` per image to build `y_true`/`y_pred` arrays for per-class precision/recall/F1 computed manually. Benchmarks inference with 100 warmup-then-timed runs. Writes `results/evaluation_results.json` and logs an MLflow `eval-<model_name>` run with per-class metrics and a confusion matrix figure.

4. **`compare_results.py`** — Reads `evaluation_results.json`, generates eight comparison charts (accuracy bars, P/R/F1, speed-vs-accuracy scatter, per-class F1, radar, training curves, confusion matrices, model size), a LaTeX table, and a text summary report. Logs all artifacts to MLflow under a `comparison-report` run.

### MLflow Tracking

Each stage creates its own named MLflow run (`train-*`, `eval-*`, `comparison-report`) within the `seed-recognition-yolo` experiment stored locally at `mlruns/`. Ultralytics also auto-creates runs internally during training — these are ended before the project's own run is opened to prevent nesting issues.

### Dataset

- **Source**: `data/SEEED/<class_name>/` (subfolders allowed; `prepare_dataset.py` flattens them)
- **Prepared split**: `dataset/train|val|test/<class_name>/` — 4,507 images total across 16 classes
- **Classes**: Borboti, Boro_Shosha, Chal_Kumra, Chichinga, Dhonepata, Hatikani_Shim, Jhinga, Korola, Lau, Mishti_Kumra, Monica_Dhedosh, Palong_Shak, Shim, Shosha_Mohabir, pepe, pui_shak
- **Imbalance**: Largest class (Boro_Shosha, 343) is ~2.7× the smallest (Shim, 126)

### Models

| Key in `YOLO_VERSIONS` | Weights file |
|---|---|
| `yolov8n-cls` | `yolov8n-cls.pt` |
| `yolov8s-cls` | `yolov8s-cls.pt` |
| `yolov11n-cls` | `yolo11n-cls.pt` |
| `yolov11s-cls` | `yolo11s-cls.pt` |

Pre-trained `.pt` files must be present at the project root before training.

## Outputs

| Path | Contents |
|---|---|
| `runs/<model>/weights/best.pt` | Best model checkpoint |
| `runs/<model>/results.csv` | Per-epoch training metrics |
| `results/evaluation_results.json` | Full evaluation data (requires `evaluate.py` to complete) |
| `results/figures/*.png` | Comparison charts |
| `results/summary_report.txt` | Text summary |
| `results/comparison_table.tex` | LaTeX table for thesis |
| `mlruns/` | MLflow local tracking store |

## Known Gaps (from `analysis/YOLO_Classification_Comparison_Overview.md`)

- `EPOCHS = 2` in `config.py` is a smoke-test value. For final thesis results, use 50–100 epochs; the `patience=15` early-stopping already in `train_ultralytics.py` will stop training automatically.
- The test-set evaluation (`evaluate.py` → `results/evaluation_results.json`) must complete before `compare_results.py` can generate precision/recall/F1 results.
- For a defensible comparison, report test-split metrics (not validation), and consider repeating runs with multiple seeds to confirm small gaps between models are statistically real.
