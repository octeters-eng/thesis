# Seed Recognition: YOLO Classification Comparison

Thesis repository for benchmarking multiple Ultralytics YOLO classification models on a seed image dataset.

## What This Project Does

- Prepares a folder-based classification dataset from raw images.
- Trains multiple YOLO classification variants.
- Evaluates trained models on the test split.
- Logs metrics and artifacts to MLflow.
- Generates comparison plots and report files.

## Repository Layout

```text
thesis/
├── config.py
├── main.py
├── download_dataset.py
├── prepare_dataset.py
├── train_ultralytics.py
├── evaluate.py
├── compare_results.py
├── requirements.txt
├── data/SEEED/                 # raw images, one folder per class (runtime)
├── dataset/                    # prepared train/val/test split (runtime)
├── runs/                       # model checkpoints + training artifacts (runtime)
├── results/                    # evaluation JSON + charts/reports (runtime)
└── mlruns/                     # MLflow local tracking store (runtime)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset Expectations

Place source images under:

```text
data/SEEED/<class_name>/**/*.jpg
```

Subfolders are allowed and will be flattened during preparation.

## Optional Dataset Download

If your dataset is in a Google Drive folder:

1. Set folder ID:
   ```bash
   export SEED_GDRIVE_FOLDER_ID="<your_folder_id>"
   ```
2. Run:
   ```bash
   python main.py --step download
   ```

## Run the Pipeline

### Full pipeline (prepare -> train -> evaluate -> compare)

```bash
python main.py
```

### Non-interactive run (auto-accept prompts)

```bash
python main.py --yes
```

### Single step

```bash
python main.py --step prepare
python main.py --step train
python main.py --step evaluate
python main.py --step compare
```

## Train Directly

```bash
python train_ultralytics.py --all
python train_ultralytics.py --model yolov8n-cls --epochs 100 --batch-size 16 --device mps
```

## Outputs

- Model checkpoints: `runs/<model>/weights/best.pt`
- Evaluation results JSON: `results/evaluation_results.json`
- Comparison figures: `results/figures/*.png`
- Summary report: `results/summary_report.txt`
- LaTeX table: `results/comparison_table.tex`
- Experiment tracking: `mlruns/` (view via `mlflow ui`)

## MLflow

```bash
mlflow ui
```

Then open the local MLflow URL shown in your terminal.
