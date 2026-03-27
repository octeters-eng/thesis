# Seed Recognition: YOLO Model Version Comparison

## Thesis Project

Comparing seed recognition performance across YOLO model versions: **YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLOv11**.

## Project Structure

```
thesis/
├── config.py                # Configuration (paths, hyperparameters, model list)
├── main.py                  # Main pipeline orchestrator
├── download_dataset.py      # Download dataset from Google Drive
├── prepare_dataset.py       # Prepare & split dataset for YOLO format
├── train_yolov7.py          # YOLOv7 training (separate codebase)
├── train_ultralytics.py     # YOLOv8-v11 training (Ultralytics API)
├── evaluate.py              # Evaluate all models on test set
├── compare_results.py       # Generate comparison charts & tables
├── requirements.txt         # Python dependencies
├── dataset/
│   ├── raw/                 # Downloaded raw data
│   └── processed/           # YOLO-formatted data
│       ├── data.yaml        # Ultralytics data config
│       ├── data_yolov7.yaml # YOLOv7 data config
│       ├── train/images & labels/
│       ├── val/images & labels/
│       └── test/images & labels/
├── runs/                    # Training runs
│   ├── yolov7/
│   └── ultralytics/
└── results/                 # Final results
    ├── figures/             # Charts (PNG + PDF)
    └── tables/              # LaTeX & Markdown tables
```

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt
```

## Quick Start (Full Pipeline)

```bash
# Run everything: download → prepare → train → evaluate → compare
python main.py --epochs 100 --batch-size 16 --device 0
```

## Step-by-Step Execution

```bash
# Step 1: Download dataset
python main.py --step download

# Step 2: Prepare dataset (split into train/val/test, convert labels)
python main.py --step prepare

# Step 3: Train all models
python main.py --step train --epochs 100

# Or train individual models:
python main.py --step train --model yolov7
python main.py --step train --model yolov8n
python main.py --step train --model yolov9c
python main.py --step train --model yolov10n
python main.py --step train --model yolov11n

# Step 4: Evaluate all trained models on test set
python main.py --step evaluate

# Step 5: Generate comparison figures and tables
python main.py --step compare
```

## Models Compared

| Model       | Version | Variant  | Source      |
| ----------- | ------- | -------- | ----------- |
| YOLOv7      | v7      | base     | WongKinYiu  |
| YOLOv7-tiny | v7      | tiny     | WongKinYiu  |
| YOLOv8n     | v8      | nano     | Ultralytics |
| YOLOv8s     | v8      | small    | Ultralytics |
| YOLOv9c     | v9      | compact  | Ultralytics |
| YOLOv9e     | v9      | extended | Ultralytics |
| YOLOv10n    | v10     | nano     | Ultralytics |
| YOLOv10s    | v10     | small    | Ultralytics |
| YOLOv11n    | v11     | nano     | Ultralytics |
| YOLOv11s    | v11     | small    | Ultralytics |

## Metrics Evaluated

- **Precision** — correct detections / total detections
- **Recall** — correct detections / total ground truth
- **mAP@50** — mean Average Precision at IoU 0.50
- **mAP@50-95** — mean AP averaged over IoU 0.50 to 0.95
- **F1 Score** — harmonic mean of precision and recall
- **Inference Time** — milliseconds per image
- **Model Size** — file size in MB
- **Parameters** — total learnable parameters
- **FLOPs** — floating point operations

## Output Figures

After running the comparison step, these figures are generated:

- `metrics_comparison_bar.png` — Bar charts for P, R, mAP50, mAP50-95
- `map_comparison.png` — Grouped bar chart for mAP
- `speed_vs_accuracy.png` — Scatter plot (key thesis figure)
- `model_efficiency.png` — Model size, params, FLOPs comparison
- `f1_comparison.png` — F1 score comparison
- `radar_comparison.png` — Multi-metric radar/spider chart
- `training_curves.png` — Loss & mAP curves during training
- `per_class_heatmap.png` — Per-class AP heatmap
- `results_table.tex` — LaTeX table for thesis
- `analysis_summary.txt` — Text summary with recommendations

## Configuration

Edit `config.py` to adjust:

- `SEED_CLASSES` — list of seed class names (auto-detected from data)
- `EPOCHS` — training epochs (default: 100)
- `BATCH_SIZE` — batch size (default: 16, reduce if GPU OOM)
- `IMG_SIZE` — input image size (default: 640)
- `DEVICE` — GPU device ("0") or "cpu"
- `TRAIN_SPLIT / VAL_SPLIT / TEST_SPLIT` — data split ratios (70/15/15)

## Requirements

- Python 3.8+
- PyTorch with CUDA (recommended for GPU training)
- ~10GB disk space for models and dataset
