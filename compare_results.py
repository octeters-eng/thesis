"""
Compare and visualise results across all trained YOLO classification models.

Outputs (saved to results/figures/ and logged to MLflow):
    - accuracy_comparison.png       — Top-1 / Top-5 bar chart
    - precision_recall_f1.png       — Macro P / R / F1 grouped bars
    - speed_vs_accuracy.png         — Scatter: latency vs accuracy
    - confusion_matrix_<model>.png  — Heatmap per model
    - per_class_f1.png              — Per-class F1 grouped bars
    - radar_comparison.png          — Radar / spider chart
    - training_loss_curves.png      — Loss over epochs
    - training_accuracy_curves.png  — Val accuracy over epochs (if available)
    - model_size_comparison.png     — Horizontal size bar chart

Reports (saved to results/ and logged to MLflow):
    - comparison_table.tex          — LaTeX table for thesis
    - summary_report.txt            — Plain-text summary

Usage:
    python compare_results.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import mlflow

from config import (
    RESULTS_DIR, RUNS_DIR, FIGURES_DIR,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
)

RESULTS_FILE = RESULTS_DIR / "evaluation_results.json"
_SEPARATOR   = "=" * 60


# ── Data loading ───────────────────────────────────────────────────────────────

def load_results() -> dict:
    """Load the evaluation results JSON produced by evaluate.py.

    Returns:
        Parsed results dictionary, or an empty dict if the file does not exist.
    """
    if not RESULTS_FILE.exists():
        print(f"Results file not found: {RESULTS_FILE}")
        print("Run evaluate.py first.")
        return {}
    with RESULTS_FILE.open() as fh:
        return json.load(fh)


def results_to_dataframe(results: dict) -> pd.DataFrame:
    """Convert the results dict to a tidy DataFrame for plotting.

    Error entries (models that failed evaluation) are silently skipped.

    Args:
        results: Mapping of model name → metrics dict.

    Returns:
        DataFrame with one row per successfully evaluated model.
    """
    rows = [
        {
            "Model":          name,
            "Top-1 Acc":      m.get("top1_accuracy",    0),
            "Top-5 Acc":      m.get("top5_accuracy",    0),
            "Precision":      m.get("precision_macro",  0),
            "Recall":         m.get("recall_macro",     0),
            "F1 (macro)":     m.get("f1_macro",         0),
            "Inference (ms)": m.get("inference_time_ms",0),
            "Size (MB)":      m.get("model_size_mb",    0),
            "Parameters":     m.get("parameters",       0),
        }
        for name, m in results.items()
        if "error" not in m
    ]
    return pd.DataFrame(rows)


# ── Chart helpers ──────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, filename: str) -> None:
    """Save *fig* to FIGURES_DIR and print a confirmation line."""
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_accuracy_comparison(df: pd.DataFrame) -> None:
    """Grouped bar chart: Top-1 and Top-5 accuracy per model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x, w = np.arange(len(df)), 0.35

    ax.bar(x - w / 2, df["Top-1 Acc"], w, label="Top-1 Accuracy", color="#2196F3")
    ax.bar(x + w / 2, df["Top-5 Acc"], w, label="Top-5 Accuracy", color="#4CAF50")

    for i, v in enumerate(df["Top-1 Acc"]):
        ax.text(i - w / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(df["Top-5 Acc"]):
        ax.text(i + w / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    ax.set(xlabel="Model", ylabel="Accuracy",
           title="Top-1 and Top-5 Classification Accuracy", ylim=(0, 1.05))
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()
    _save(fig, "accuracy_comparison.png")


def plot_precision_recall_f1(df: pd.DataFrame) -> None:
    """Grouped bar chart: macro Precision, Recall, and F1 per model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x, w = np.arange(len(df)), 0.25

    ax.bar(x - w, df["Precision"],  w, label="Precision", color="#FF9800")
    ax.bar(x,     df["Recall"],     w, label="Recall",    color="#9C27B0")
    ax.bar(x + w, df["F1 (macro)"], w, label="F1 Score",  color="#F44336")

    ax.set(xlabel="Model", ylabel="Score",
           title="Precision, Recall, and F1 Score (Macro)", ylim=(0, 1.05))
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()
    _save(fig, "precision_recall_f1.png")


def plot_speed_vs_accuracy(df: pd.DataFrame) -> None:
    """Scatter plot: inference latency vs Top-1 accuracy; bubble size = model size."""
    fig, ax = plt.subplots(figsize=(10, 7))
    sizes = df["Size (MB)"] / df["Size (MB)"].max() * 500 + 50

    ax.scatter(
        df["Inference (ms)"], df["Top-1 Acc"],
        s=sizes, alpha=0.7, c=range(len(df)), cmap="Set2", edgecolors="black",
    )
    for _, row in df.iterrows():
        ax.annotate(
            row["Model"],
            (row["Inference (ms)"], row["Top-1 Acc"]),
            textcoords="offset points", xytext=(8, 8), fontsize=9,
        )

    ax.set(xlabel="Inference Time (ms)", ylabel="Top-1 Accuracy",
           title="Speed vs Accuracy  (bubble size ∝ model size)")
    plt.tight_layout()
    _save(fig, "speed_vs_accuracy.png")


def plot_confusion_matrices(results: dict) -> None:
    """Save a heatmap confusion matrix for each model in *results*."""
    for name, m in results.items():
        cm          = m.get("confusion_matrix")
        class_names = m.get("class_names")
        if cm is None or class_names is None:
            continue

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            np.array(cm), annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {name}")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        _save(fig, f"confusion_matrix_{name}.png")


def plot_per_class_f1(results: dict) -> None:
    """Grouped bar chart: per-class F1 score for every model."""
    models_data = {
        name: m["per_class"]
        for name, m in results.items()
        if m.get("per_class")
    }
    if not models_data:
        return

    classes   = sorted({cls for pc in models_data.values() for cls in pc})
    n_models  = len(models_data)
    w         = 0.8 / max(n_models, 1)
    x         = np.arange(len(classes))

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (mname, pc) in enumerate(models_data.items()):
        f1_vals = [pc.get(c, {}).get("f1", 0) for c in classes]
        ax.bar(x + i * w - (n_models - 1) * w / 2, f1_vals, w, label=mname)

    ax.set(xlabel="Seed Class", ylabel="F1 Score",
           title="Per-Class F1 Score Comparison", ylim=(0, 1.1))
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, "per_class_f1.png")


def plot_radar_chart(df: pd.DataFrame) -> None:
    """Radar (spider) chart comparing models across key metrics."""
    metrics   = [m for m in ("Top-1 Acc", "Precision", "Recall", "F1 (macro)") if m in df.columns]
    if len(metrics) < 3:
        return

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    colors  = plt.cm.Set2(np.linspace(0, 1, len(df)))

    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[m] for m in metrics] + [row[metrics[0]]]
        ax.plot(angles, values, "o-", linewidth=2, label=row["Model"], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title("Model Comparison — Radar Chart", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    _save(fig, "radar_comparison.png")


def plot_training_curves(results: dict) -> None:
    """Plot training loss and validation accuracy curves from results.csv files."""
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    fig_acc,  ax_acc  = plt.subplots(figsize=(10, 6))
    acc_found = False

    for model_name in results:
        csv_path = RUNS_DIR / model_name / "results.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()

            loss_cols = [c for c in df.columns if "loss" in c.lower()]
            if loss_cols:
                ax_loss.plot(df["epoch"], df[loss_cols[0]], label=model_name)

            acc_cols = [c for c in df.columns
                        if "accuracy_top1" in c.lower() or "top1" in c.lower()]
            if acc_cols:
                ax_acc.plot(df["epoch"], df[acc_cols[0]], label=model_name)
                acc_found = True

        except Exception as exc:
            print(f"  Warning: could not parse {csv_path.name}: {exc}")

    ax_loss.set(xlabel="Epoch", ylabel="Loss", title="Training Loss Curves")
    ax_loss.legend()
    plt.figure(fig_loss.number)
    plt.tight_layout()
    _save(fig_loss, "training_loss_curves.png")

    if acc_found:
        ax_acc.set(xlabel="Epoch", ylabel="Top-1 Accuracy", title="Validation Accuracy Curves")
        ax_acc.legend()
        plt.figure(fig_acc.number)
        plt.tight_layout()
        _save(fig_acc, "training_accuracy_curves.png")
    else:
        plt.close(fig_acc)   # prevent resource leak when no accuracy data found


def plot_model_size_comparison(df: pd.DataFrame) -> None:
    """Horizontal bar chart of model sizes in MB."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = plt.cm.Set2(np.linspace(0, 1, len(df)))
    ax.barh(df["Model"], df["Size (MB)"], color=colors)
    for i, v in enumerate(df["Size (MB)"]):
        ax.text(v + 0.1, i, f"{v:.1f} MB", va="center", fontsize=9)
    ax.set(xlabel="Model Size (MB)", title="Model Size Comparison")
    plt.tight_layout()
    _save(fig, "model_size_comparison.png")


# ── Report generation ──────────────────────────────────────────────────────────

def generate_latex_table(df: pd.DataFrame) -> str:
    """Return a LaTeX booktabs table string suitable for a thesis chapter.

    Args:
        df: Results DataFrame from :func:`results_to_dataframe`.

    Returns:
        LaTeX table as a string.
    """
    cols = ["Model", "Top-1 Acc", "Top-5 Acc", "Precision",
            "Recall", "F1 (macro)", "Inference (ms)", "Size (MB)"]
    return df[cols].to_latex(index=False, float_format="%.4f", escape=False)


def generate_summary_report(results: dict, df: pd.DataFrame) -> str:
    """Generate a plain-text summary report for console output and archiving.

    Args:
        results: Raw results dictionary (may include error entries).
        df:      Cleaned DataFrame from :func:`results_to_dataframe`.

    Returns:
        Formatted multi-line report string.
    """
    first_key  = next(iter(results), None)
    n_classes  = len(results[first_key].get("class_names", [])) if first_key else 0

    lines = [
        "=" * 70,
        "SEED RECOGNITION — YOLO CLASSIFICATION MODEL COMPARISON",
        "=" * 70,
        f"\nModels evaluated : {len(df)}",
        f"Number of classes: {n_classes}",
    ]

    if not df.empty:
        best_acc  = df.loc[df["Top-1 Acc"].idxmax()]
        best_f1   = df.loc[df["F1 (macro)"].idxmax()]
        fastest   = df.loc[df["Inference (ms)"].idxmin()]
        smallest  = df.loc[df["Size (MB)"].idxmin()]

        lines += [
            f"\nBest Top-1 Accuracy : {best_acc['Model']} ({best_acc['Top-1 Acc']:.4f})",
            f"Best F1 Score       : {best_f1['Model']}  ({best_f1['F1 (macro)']:.4f})",
            f"Fastest Inference   : {fastest['Model']}  ({fastest['Inference (ms)']:.1f} ms)",
            f"Smallest Model      : {smallest['Model']} ({smallest['Size (MB)']:.1f} MB)",
        ]

    header = f"\n{'Model':<16} {'Top1':>7} {'Top5':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'ms':>8} {'MB':>8}"
    lines += ["\n" + "-" * 70, "DETAILED METRICS", "-" * 70, header, "-" * len(header.strip())]

    for _, row in df.iterrows():
        lines.append(
            f"{row['Model']:<16} "
            f"{row['Top-1 Acc']:>7.4f} "
            f"{row['Top-5 Acc']:>7.4f} "
            f"{row['Precision']:>7.4f} "
            f"{row['Recall']:>7.4f} "
            f"{row['F1 (macro)']:>7.4f} "
            f"{row['Inference (ms)']:>8.1f} "
            f"{row['Size (MB)']:>8.1f}"
        )

    return "\n".join(lines)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    results = load_results()
    if not results:
        return

    valid = {k: v for k, v in results.items() if "error" not in v}
    if not valid:
        print("No valid results to compare.")
        return

    df = results_to_dataframe(valid)
    if df.empty:
        print("DataFrame is empty — nothing to plot.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(_SEPARATOR)
    print("GENERATING COMPARISON CHARTS & REPORT")
    print(_SEPARATOR)

    plot_accuracy_comparison(df)
    plot_precision_recall_f1(df)
    plot_speed_vs_accuracy(df)
    plot_confusion_matrices(valid)
    plot_per_class_f1(valid)
    plot_radar_chart(df)
    plot_training_curves(valid)
    plot_model_size_comparison(df)

    # ── Write text reports ─────────────────────────────────────────────────────
    latex_path  = RESULTS_DIR / "comparison_table.tex"
    report_path = RESULTS_DIR / "summary_report.txt"

    with latex_path.open("w") as fh:
        fh.write(generate_latex_table(df))
    print(f"  Saved {latex_path.name}")

    report = generate_summary_report(valid, df)
    with report_path.open("w") as fh:
        fh.write(report)
    print(f"  Saved {report_path.name}")

    # ── Log everything to MLflow ───────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="comparison-report"):
        mlflow.set_tag("stage", "comparison")
        for fig_file in FIGURES_DIR.glob("*.png"):
            mlflow.log_artifact(str(fig_file), artifact_path="figures")
        mlflow.log_artifact(str(latex_path),  artifact_path="reports")
        mlflow.log_artifact(str(report_path), artifact_path="reports")
        eval_json = RESULTS_DIR / "evaluation_results.json"
        if eval_json.exists():
            mlflow.log_artifact(str(eval_json), artifact_path="reports")
        print("  Logged all artifacts to MLflow")

    print("\n" + report)


if __name__ == "__main__":
    main()
