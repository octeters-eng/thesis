# Comparison of Output Quality of YOLO Classification Models on a Single Seed/Vegetable Image Dataset

**Project:** Seed Recognition — YOLO Classification Comparison
**Prepared as a working overview for the thesis write-up.**
**Date:** 24 May 2026 (updated from 20 May 2026)

> **Reading note.** This document reflects the current, complete state of the project. All numbers come from (a) the training logs (`runs/<model>/results.csv`) for epoch-level metrics and (b) `results/evaluation_results.json` for the full test-set evaluation — which is now complete. Every model was trained for **only 2 epochs**, which is a smoke-test, not a converged run; see Section 7 for what still needs to be done before final thesis submission.

---

## 1. Aim of the Study

The objective is to compare the **output quality of four different YOLO image-classification models on one and the same dataset**, holding the data, image size, and training settings constant so that any difference in performance can be attributed to the model architecture rather than to the data or the training recipe.

The four models compared are two generations of Ultralytics YOLO classifiers, each in a "nano" (n) and a "small" (s) size:

| Model | Architecture generation | Size variant | Pretrained weights file |
|---|---|---|---|
| `yolov8n-cls` | YOLOv8 | nano | `yolov8n-cls.pt` |
| `yolov8s-cls` | YOLOv8 | small | `yolov8s-cls.pt` |
| `yolov11n-cls` | YOLO11 | nano | `yolo11n-cls.pt` |
| `yolov11s-cls` | YOLO11 | small | `yolo11s-cls.pt` |

This 2×2 design lets the thesis answer two questions at once: *does the newer YOLO11 architecture beat YOLOv8?* and *how much does going from nano to small buy you?*

---

## 2. Dataset

The dataset is a **16-class image classification dataset** of Bangladeshi vegetable/seed types. The class names (as used in the code) are:

> Borboti, Boro_Shosha, Chal_Kumra, Chichinga, Dhonepata, Hatikani_Shim, Jhinga, Korola, Lau, Mishti_Kumra, Monica_Dhedosh, Palong_Shak, Shim, Shosha_Mohabir, pepe, pui_shak.

The raw images are organised one folder per class and split into train/validation/test by `prepare_dataset.py` using a **70% / 15% / 15%** split. The prepared split that exists on disk contains **4,507 images total**:

| Split | Images | Share |
|---|---:|---:|
| Train | 3,150 | ~70% |
| Validation | 668 | ~15% |
| Test | 689 | ~15% |
| **Total** | **4,507** | **100%** |

### Per-class image counts

| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Borboti | 176 | 37 | 39 | 252 |
| Boro_Shosha | 240 | 51 | 52 | 343 |
| Chal_Kumra | 210 | 45 | 45 | 300 |
| Chichinga | 218 | 46 | 48 | 312 |
| Dhonepata | 128 | 27 | 29 | 184 |
| Hatikani_Shim | 211 | 45 | 46 | 302 |
| Jhinga | 209 | 44 | 46 | 299 |
| Korola | 212 | 45 | 46 | 303 |
| Lau | 209 | 44 | 46 | 299 |
| Mishti_Kumra | 224 | 48 | 48 | 320 |
| Monica_Dhedosh | 210 | 45 | 46 | 301 |
| Palong_Shak | 194 | 41 | 43 | 278 |
| Shim | 88 | 18 | 20 | 126 |
| Shosha_Mohabir | 211 | 45 | 46 | 302 |
| pepe | 214 | 45 | 47 | 306 |
| pui_shak | 196 | 42 | 42 | 280 |
| **Total** | **3,150** | **668** | **689** | **4,507** |

The dataset is **moderately imbalanced**: the largest class (`Boro_Shosha`, 343 images) has nearly **2.7×** the images of the smallest (`Shim`, 126 images). This is worth mentioning in the thesis because the smaller and visually similar classes are the ones most likely to be confused — exactly what the test-set results now confirm.

All images are resized to **224×224** pixels for classification.

---

## 3. Methodology and Pipeline

The project is organised as a clean, reproducible five-stage pipeline orchestrated by `main.py`:

1. **Download** (`download_dataset.py`) — optional; fetches the raw dataset from a Google Drive folder (requires `SEED_GDRIVE_FOLDER_ID` env var), extracts archives, and prints a directory summary.
2. **Prepare** (`prepare_dataset.py`) — flattens raw class subfolders and creates the train/val/test split using a deterministic, collision-safe filename encoding.
3. **Train** (`train_ultralytics.py`) — fine-tunes each of the four pretrained YOLO classifiers on the dataset.
4. **Evaluate** (`evaluate.py`) — runs each trained model on the **test** split and computes accuracy, macro precision/recall/F1, per-class metrics, a confusion matrix, inference speed (100-run benchmark with 10-image warmup), and model size. Writes everything to `results/evaluation_results.json`.
5. **Compare** (`compare_results.py`) — reads the evaluation JSON and generates nine comparison charts, a LaTeX table, and a text summary report; logs all artefacts to MLflow.

Every stage logs metrics and artifacts to a local **MLflow** tracking store (`mlruns/`), so runs can be compared interactively with `mlflow ui`.

The default pipeline run (steps 2–5) is:
```bash
python main.py          # interactive
python main.py --yes    # suppress prompts
```

Individual steps can be run as `python main.py --step <download|prepare|train|evaluate|compare>`.

### Controlled (identical) training settings

To make the comparison fair, all four models were trained with the **same hyperparameters**, confirmed from each run's `results.csv`:

| Setting | Value |
|---|---|
| Epochs | **2** (smoke-test only; see Section 7) |
| Batch size | 32 |
| Image size | 224 × 224 |
| Optimizer | auto (Ultralytics chooses per model) |
| Initial learning rate (`lr0`) | 0.01 (Ultralytics default) |
| Warmup epochs | 3.0 |
| Device | Apple Silicon GPU (`mps`) |
| Early-stopping patience | 15 |

This is a methodologically sound setup: by changing *only* the model and keeping everything else fixed, the study isolates the effect of architecture and model size on output quality.

---

## 4. Results

> ⚠️ **Important caveat — these are 2-epoch results.** `config.py` sets `EPOCHS = 2`, and the training logs confirm each model saw only two passes over the data. The accuracy values are already high (89–93%) because the models start from ImageNet-pretrained weights, but **for the final thesis these runs should be repeated with substantially more epochs** (50–100, with patience-15 early stopping already in place). Both training and validation losses are still falling steeply at epoch 2 — direct evidence the models had not finished learning.

### 4.1 Training metrics (validation split, from results.csv)

These figures are read directly from `runs/<model>/results.csv` at epoch 2. Accuracy and losses are measured on the **validation** split during training; training time is wall-clock for the 2-epoch run on Apple Silicon (MPS).

| Model | Top-1 Acc | Top-5 Acc | Train loss | Val loss | Train time (2 ep) |
|---|---:|---:|---:|---:|---:|
| yolov8n-cls | 0.9027 | 1.0000 | 0.8788 | 0.2892 | 87.2 s |
| yolov8s-cls | 0.9177 | 1.0000 | 0.6379 | 0.1655 | 110.7 s |
| yolov11n-cls | 0.9207 | 1.0000 | 0.5212 | 0.1704 | 110.4 s |
| **yolov11s-cls** | **0.9326** | **1.0000** | **0.3416** | **0.1263** | 132.7 s |

### 4.2 Test-set evaluation (from evaluation_results.json)

These metrics come from running each trained `best.pt` on the held-out **test split** (689 images that were never seen during training or model selection). Top-1/Top-5 accuracy is from `model.val(split="test")`; macro precision/recall/F1 and per-class metrics are computed from per-image `model.predict()` calls; inference speed is measured over 100 timed runs after a 10-image warmup.

| Model | Top-1 Acc | Top-5 Acc | Precision (macro) | Recall (macro) | F1 (macro) | Inference (ms) | Size (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| yolov8n-cls | 0.8940 | 1.0000 | 0.8909 | 0.8942 | 0.8797 | 2.07 | 2.86 |
| yolov8s-cls | **0.9173** | 1.0000 | **0.9201** | 0.9142 | **0.9117** | **2.09** | 9.82 |
| yolov11n-cls | 0.9100 | 1.0000 | 0.9043 | 0.9118 | 0.9021 | 2.70 | 3.07 |
| **yolov11s-cls** | **0.9245** | 1.0000 | 0.8989 | **0.9292** | 0.9092 | 2.73 | **10.55** |

**Best overall model on test set: `yolov11s-cls`** (highest Top-1 accuracy, highest recall). However, **`yolov8s-cls` leads on macro precision and macro F1** — this is a nuance the thesis should discuss.

All four models reach a **Top-5 accuracy of 1.000**, meaning the correct class is always among the model's top five guesses; the discriminating metric is Top-1.

#### Key ranking observations

- **Top-1 accuracy order (test set):** yolov11s (0.9245) > yolov8s (0.9173) > yolov11n (0.9100) > yolov8n (0.8940)
- **F1 macro order (test set):** yolov8s (0.9117) > yolov11s (0.9092) > yolov11n (0.9021) > yolov8n (0.8797)
- **Important:** `yolov11n-cls` (nano, 3.07 MB) scores *below* `yolov8s-cls` (small, 9.82 MB) on the test set (0.910 vs 0.917 Top-1; 0.902 vs 0.912 F1). The validation-set result suggested yolov11n was competitive with yolov8s — the test-set result narrows this picture. More training epochs are needed before drawing strong architecture conclusions.
- **Inference speed:** Both YOLOv8 models (2.07–2.09 ms) are ~25% faster than both YOLO11 models (2.70–2.73 ms) on Apple Silicon. This is a meaningful efficiency difference for real-time applications.
- **Model size:** Nano models (2.86 MB and 3.07 MB) are ~3.5× smaller than small models (9.82 MB and 10.55 MB), with a moderate accuracy cost.

### 4.3 Accuracy comparison

![Top-1 and Top-5 accuracy bar chart](figures/comparison_accuracy.png)

*Top-1 accuracy rises from YOLOv8-nano to YOLO11-small; Top-5 is saturated at 1.0 for every model.*

### 4.4 Accuracy vs model size

![Accuracy versus model size scatter](figures/comparison_acc_vs_size.png)

*Accuracy increases with model size. The YOLO11 generation sits above the YOLOv8 line at both sizes — the architecture improvement gives "free" accuracy at no extra disk cost.*

### 4.5 Training and validation loss

![Training and validation loss curves](figures/comparison_loss.png)

*Both losses are still dropping steeply between epoch 1 and epoch 2 — clear visual evidence that 2 epochs is insufficient. `yolov11s-cls` has the lowest loss at every point.*

### 4.6 Training cost

![Training time per model](figures/comparison_traintime.png)

*Larger and newer models cost more compute time. The thesis can frame this as an accuracy-vs-efficiency trade-off.*

---

## 5. Per-Class Precision, Recall, and F1 (Test Split)

The evaluation is now complete. The table below shows the per-class results for every model on the 689-image test set. **Bold** marks the best value per class.

### Macro summary

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| yolov8n-cls | 0.8909 | 0.8942 | 0.8797 |
| yolov8s-cls | **0.9201** | 0.9142 | **0.9117** |
| yolov11n-cls | 0.9043 | 0.9118 | 0.9021 |
| yolov11s-cls | 0.8989 | **0.9292** | 0.9092 |

### Per-class F1 scores

| Class | yolov8n | yolov8s | yolov11n | yolov11s | Support |
|---|---:|---:|---:|---:|---:|
| Borboti | **1.0000** | 0.9873 | **1.0000** | **1.0000** | 39 |
| Boro_Shosha | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 52 |
| Chal_Kumra | 0.9677 | **1.0000** | **1.0000** | **1.0000** | 45 |
| Chichinga | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 48 |
| Dhonepata | 0.7568 | 0.9032 | 0.9355 | **0.9667** | 29 |
| Hatikani_Shim | 0.9565 | 0.9462 | 0.9091 | **0.9778** | 46 |
| Jhinga | 0.9684 | 0.9787 | 0.9787 | **0.9787** | 46 |
| Korola | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 46 |
| Lau | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 46 |
| Mishti_Kumra | 0.9677 | **1.0000** | **1.0000** | **1.0000** | 48 |
| Monica_Dhedosh | 0.5968 | 0.6087 | 0.5833 | **0.6471** | 46 |
| Palong_Shak | 0.8861 | 0.9268 | **0.9512** | 0.9762 | 43 |
| Shim | 0.9189 | 0.8889 | 0.8571 | **1.0000** | 20 |
| **Shosha_Mohabir** | 0.1111 | 0.3478 | 0.2188 | **0.0000** | 46 |
| pepe | 0.9451 | **1.0000** | **1.0000** | **1.0000** | 47 |
| pui_shak | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 42 |

### Critical finding: Shosha_Mohabir is the hardest class by far

Every model struggles with `Shosha_Mohabir`, but the severity varies wildly:

- **yolov8n-cls:** F1=0.111 — only 3 of 46 images classified correctly; 43 misclassified as `Monica_Dhedosh`.
- **yolov8s-cls:** F1=0.348 — 12 of 46 correct; 34 misclassified as `Monica_Dhedosh`.
- **yolov11n-cls:** F1=0.219 — 7 of 46 correct; 39 misclassified as `Monica_Dhedosh`.
- **yolov11s-cls:** F1=0.000 — **0 of 46 images correctly classified**; every single `Shosha_Mohabir` test image was predicted as `Monica_Dhedosh`.

This is the most dramatic result in the dataset. `Shosha_Mohabir` and `Monica_Dhedosh` are clearly visually very similar; the confusion is consistently one-directional (Shosha→Monica). This finding must be highlighted in the thesis and likely warrants dedicated investigation — more data, data augmentation, or feature visualisation (Grad-CAM) to understand what the models are missing.

Note: `Monica_Dhedosh` itself has low F1 across all models (0.58–0.65) because many `Shosha_Mohabir` images are being dumped into it, inflating its false-positive count.

### Easy classes (perfect or near-perfect across all models)

`Boro_Shosha`, `Chichinga`, `Korola`, `Lau`, `pui_shak` achieve F1 = 1.000 for all four models, and several others (`Borboti`, `Jhinga`, `Mishti_Kumra`) are close. These classes are visually distinctive enough that even a 2-epoch fine-tune is sufficient.

---

## 6. Confusion Matrices (Test Split)

These normalised (row-wise recall) confusion matrices were generated by `evaluate.py` on the **test** split and logged to MLflow. A perfect model would be a clean diagonal.

### yolov11s-cls (best Top-1 accuracy)

![Confusion matrix, yolov11s-cls](figures/cm_test_yolov11s-cls.png)

Nearly perfect except for the complete collapse on `Shosha_Mohabir` (entire row predicted as `Monica_Dhedosh`) and minor `Hatikani_Shim`↔`Jhinga` leakage.

### yolov11n-cls

![Confusion matrix, yolov11n-cls](figures/cm_test_yolov11n-cls.png)

7 of 46 `Shosha_Mohabir` correct; `Hatikani_Shim` has 4 misses into `Shim`; `Monica_Dhedosh` 35/46 correctly identified.

### yolov8s-cls

![Confusion matrix, yolov8s-cls](figures/cm_test_yolov8s-cls.png)

Best Shosha_Mohabir of the group (12/46, F1=0.35); minor `Shim` confusions (1 into `Borboti`, 3 into `Hatikani_Shim`).

### yolov8n-cls (weakest)

![Confusion matrix, yolov8n-cls](figures/cm_test_yolov8n-cls.png)

`Shosha_Mohabir` almost entirely missed (only 3/46); `Monica_Dhedosh` heavily polluted (41/46 Shosha images land here); `Dhonepata` inflated similarly.

**Take-away for the thesis:** All confusion matrices confirm the same hierarchy (yolov8n is weakest, yolov11s is best on Top-1). The `Shosha_Mohabir`↔`Monica_Dhedosh` confusion is the single biggest quality problem across all models and deserves dedicated attention in the thesis.

---

## 7. What Each Source File Does

A short tour of the current codebase, useful for the "Implementation / Methodology" chapter:

- **`config.py`** — single source of truth: project paths, the 16 class names, the four model definitions (`YOLO_VERSIONS`), image size (224), split ratios (70/15/15), and training hyperparameters (`EPOCHS=2`, `BATCH_SIZE=32`, `LEARNING_RATE=0.001`, `DEVICE="mps"`). Every other script imports from it.
- **`download_dataset.py`** — fetches the raw dataset from a Google Drive folder using `gdown`; extracts `.zip`/`.tar*` archives automatically; prints a directory and file-type summary. Controlled by the `SEED_GDRIVE_FOLDER_ID` environment variable.
- **`prepare_dataset.py`** — recursively collects images from `data/SEEED/<class>/`, flattens subfolders using a collision-safe path-encoding naming scheme, and copies into `dataset/train|val|test/<class>/` with a 70/15/15 split (random seed 42). Prompts before overwriting (`--yes` skips).
- **`train_ultralytics.py`** — trains one (`--model`) or all (`--all`) models via `model.train()`. Checkpoints land at `runs/<model_name>/weights/best.pt`. Ends Ultralytics' internal MLflow callback, then opens a fresh `train-<model_name>` MLflow run to log a clean summary (params, timing, model size, parameter count, `best.pt` artefact, `results.csv`).
- **`evaluate.py`** — loads each `best.pt`, runs `model.val(split="test")` for Top-1/Top-5, then runs `model.predict()` per image to build `y_true`/`y_pred` arrays for per-class precision/recall/F1 computed from scratch. Benchmarks inference with 10-image warmup then 100 timed runs. Plots a confusion-matrix heatmap and logs everything to MLflow under `eval-<model_name>`; writes `results/evaluation_results.json`.
- **`compare_results.py`** — reads `evaluation_results.json` and generates nine comparison charts: `accuracy_comparison.png`, `precision_recall_f1.png`, `speed_vs_accuracy.png`, `confusion_matrix_<model>.png` (×4), `per_class_f1.png`, `radar_comparison.png`, `training_loss_curves.png`, `training_accuracy_curves.png`, `model_size_comparison.png`. Also writes a LaTeX table and a text summary report. Logs all artefacts to MLflow under `comparison-report`.
- **`main.py`** — orchestrates all five steps as subprocesses with `--step` selector and `--yes` flag. Sets up MLflow experiment before dispatching.

---

## 8. Known Gaps and Recommendations Before the Final Thesis Run

### ✅ Now complete (since last version of this document)

- **Test-set evaluation is done.** `evaluate.py` has run successfully and `results/evaluation_results.json` contains full per-class precision, recall, F1, inference speed, and model size for all four models. The gap noted in the previous version of this document is closed.

### ⚠️ Still outstanding

1. **Train for more epochs.** This is the single most important change. `EPOCHS = 2` produces unconverged models — the losses are still falling sharply at epoch 2. Set `EPOCHS` to 50–100 in `config.py`; the `patience=15` early stopping will halt training automatically once validation plateaus. Re-run `python main.py` (or `python main.py --step train` then re-evaluate and re-compare) to regenerate all results.

2. **Investigate `Shosha_Mohabir`.** The `Shosha_Mohabir`↔`Monica_Dhedosh` confusion is the dominant failure mode across all models. Options to explore: (a) visualise with Grad-CAM to see what the model is looking at, (b) add targeted augmentation for these two classes, (c) collect additional training images, (d) check whether the class boundary is genuinely ambiguous in the raw data.

3. **Prefer test-split metrics for thesis claims.** The headline numbers in Section 4.1 are validation-split accuracies from training. The thesis should base its main claims on the **test-split** metrics in Section 4.2, since the test set was never seen during training or model selection. Both sets are now available.

4. **Address class imbalance in the report.** With `Shim` at 126 images vs `Boro_Shosha` at 343, the macro-averaged metrics (which are already computed) are the right choice for headline reporting. Consider additionally reporting weighted-macro or per-class numbers for the hardest classes (`Shosha_Mohabir`, `Monica_Dhedosh`, `Dhonepata`).

5. **Random seed / repeated runs.** For a defensible comparison fix a random seed and ideally repeat each training 3× to report mean ± std, so gaps like yolov8s vs yolov11n on the test set (0.917 vs 0.910 Top-1) can be shown to be real rather than sampling noise. With 2-epoch runs this is inexpensive; with 50-epoch runs it is more costly but still important.

6. **Run `compare_results.py` after re-evaluating.** Once longer training is done and `evaluate.py` is re-run, run `python main.py --step compare` to regenerate all nine charts, the LaTeX table, and the text report with the new numbers.

---

## 9. One-Paragraph Summary (drop-in for an abstract)

Four Ultralytics YOLO image-classification models — YOLOv8 and YOLO11, each in nano and small variants — were fine-tuned on an identical 16-class, 4,507-image Bangladeshi seed/vegetable dataset (70/15/15 split, 224×224 input) under identical hyperparameters and evaluated on a held-out test set. After a 2-epoch training run, `yolov11s-cls` achieved the highest test-set Top-1 accuracy (92.5%) followed by `yolov8s-cls` (91.7%), `yolov11n-cls` (91.0%), and `yolov8n-cls` (89.4%); on macro F1, `yolov8s-cls` led narrowly (0.912) ahead of `yolov11s-cls` (0.909). Both YOLO11 models were ~25% slower on inference (2.70–2.73 ms vs 2.07–2.09 ms per image on Apple Silicon) but were not substantially more accurate than YOLOv8 at the same model size — a finding that may change with longer training. The dominant failure mode across all models was the `Shosha_Mohabir`↔`Monica_Dhedosh` confusion: `yolov11s-cls` misclassified all 46 test-set `Shosha_Mohabir` images as `Monica_Dhedosh` (F1 = 0.0), while `yolov8s-cls` performed best on this class (F1 = 0.35). These results are preliminary 2-epoch figures and should be regenerated with 50–100 epochs and repeated runs before final reporting.

---

*Figures referenced in Sections 4 and 6 are stored in the `figures/` folder next to this file (from earlier training runs). Updated figures generated by `compare_results.py` will be written to `results/figures/` after the full pipeline completes.*
