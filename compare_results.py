"""
Compare and visualize results across all trained YOLO classification models.
Generates charts, tables, and a summary report for the thesis.

Usage: python compare_results.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

from config import RESULTS_DIR, RUNS_DIR, METRICS_TO_COMPARE, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME


RESULTS_FILE = RESULTS_DIR / "evaluation_results.json"
FIGURES_DIR = RESULTS_DIR / "figures"


def load_results() -> dict:
    """Load evaluation results JSON."""
    if not RESULTS_FILE.exists():
        print(f"Results file not found: {RESULTS_FILE}")
        print("Run evaluate.py first.")
        return {}
    with open(RESULTS_FILE) as f:
        return json.load(f)


def results_to_dataframe(results: dict) -> pd.DataFrame:
    """Convert results dict to a DataFrame for analysis."""
    rows = []
    for name, m in results.items():
        if "error" in m:
            continue
        rows.append({
            "Model": name,
            "Top-1 Acc": m.get("top1_accuracy", 0),
            "Top-5 Acc": m.get("top5_accuracy", 0),
            "Precision": m.get("precision_macro", 0),
            "Recall": m.get("recall_macro", 0),
            "F1 (macro)": m.get("f1_macro", 0),
            "Inference (ms)": m.get("inference_time_ms", 0),
            "Size (MB)": m.get("model_size_mb", 0),
            "Parameters": m.get("parameters", 0),
        })
    return pd.DataFrame(rows)


# ── Chart functions ─────────────────────────────────────────

def plot_accuracy_comparison(df: pd.DataFrame):
    """Bar chart comparing Top-1 and Top-5 accuracy."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df["Top-1 Acc"], w, label="Top-1 Accuracy", color="#2196F3")
    ax.bar(x + w / 2, df["Top-5 Acc"], w, label="Top-5 Accuracy", color="#4CAF50")

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_title("Top-1 and Top-5 Classification Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()

    for i, v in enumerate(df["Top-1 Acc"]):
        ax.text(i - w / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(df["Top-5 Acc"]):
        ax.text(i + w / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "accuracy_comparison.png", dpi=150)
    plt.close(fig)
    print("  Saved accuracy_comparison.png")


def plot_precision_recall_f1(df: pd.DataFrame):
    """Grouped bar chart for Precision, Recall, F1."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    w = 0.25
    ax.bar(x - w, df["Precision"], w, label="Precision", color="#FF9800")
    ax.bar(x, df["Recall"], w, label="Recall", color="#9C27B0")
    ax.bar(x + w, df["F1 (macro)"], w, label="F1 Score", color="#F44336")

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Precision, Recall, and F1 Score (Macro)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "precision_recall_f1.png", dpi=150)
    plt.close(fig)
    print("  Saved precision_recall_f1.png")


def plot_speed_vs_accuracy(df: pd.DataFrame):
    """Scatter plot: inference time vs accuracy, bubble size = model size."""
    fig, ax = plt.subplots(figsize=(10, 7))
    sizes = df["Size (MB)"] / df["Size (MB)"].max() * 500 + 50

    scatter = ax.scatter(
        df["Inference (ms)"], df["Top-1 Acc"],
        s=sizes, alpha=0.7, c=range(len(df)), cmap="Set2", edgecolors="black",
    )
    for i, row in df.iterrows():
        ax.annotate(
            row["Model"],
            (row["Inference (ms)"], row["Top-1 Acc"]),
            textcoords="offset points", xytext=(8, 8), fontsize=9,
        )

    ax.set_xlabel("Inference Time (ms)")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Speed vs Accuracy (bubble size = model size)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "speed_vs_accuracy.png", dpi=150)
    plt.close(fig)
    print("  Saved speed_vs_accuracy.png")


def plot_confusion_matrices(results: dict):
    """Plot confusion matrix heatmap for each model."""
    for name, m in results.items():
        cm = m.get("confusion_matrix")
        class_names = m.get("class_names")
        if cm is None or class_names is None:
            continue

        cm = np.array(cm)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {name}")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()

        fig.savefig(FIGURES_DIR / f"confusion_matrix_{name}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved confusion_matrix_{name}.png")


def plot_per_class_f1(results: dict):
    """Grouped bar chart showing per-class F1 for all models."""
    models_data = {}
    all_classes = set()

    for name, m in results.items():
        pc = m.get("per_class")
        if pc:
            models_data[name] = pc
            all_classes.update(pc.keys())

    if not models_data:
        return

    classes = sorted(all_classes)
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(classes))
    n_models = len(models_data)
    w = 0.8 / max(n_models, 1)

    for i, (mname, pc) in enumerate(models_data.items()):
        f1_vals = [pc.get(c, {}).get("f1", 0) for c in classes]
        ax.bar(x + i * w - (n_models - 1) * w / 2, f1_vals, w, label=mname)

    ax.set_xlabel("Seed Class")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "per_class_f1.png", dpi=150)
    plt.close(fig)
    print("  Saved per_class_f1.png")


def plot_radar_chart(df: pd.DataFrame):
    """Radar (spider) chart comparing models across metrics."""
    metrics = ["Top-1 Acc", "Precision", "Recall", "F1 (macro)"]
    available = [m for m in metrics if m in df.columns]
    if len(available) < 3:
        return

    angles = np.linspace(0, 2 * np.pi, len(available), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))

    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[m] for m in available]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=row["Model"], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title("Model Comparison — Radar Chart", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "radar_comparison.png", dpi=150)
    plt.close(fig)
    print("  Saved radar_comparison.png")


def plot_training_curves(results: dict):
    """Plot training curves from Ultralytics results.csv files."""
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
    found = False

    for model_name in results:
        csv_path = RUNS_DIR / model_name / "results.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()

            # Loss
            loss_cols = [c for c in df.columns if "loss" in c.lower()]
            if loss_cols:
                ax_loss.plot(df["epoch"], df[loss_cols[0]], label=model_name)

            # Accuracy
            acc_cols = [c for c in df.columns if "accuracy_top1" in c.lower() or "top1" in c.lower()]
            if acc_cols:
                ax_acc.plot(df["epoch"], df[acc_cols[0]], label=model_name)
                found = True
        except Exception as e:
            print(f"  Warning: could not parse {csv_path}: {e}")

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Loss Curves")
    ax_loss.legend()
    plt.tight_layout()
    fig_loss.savefig(FIGURES_DIR / "training_loss_curves.png", dpi=150)
    plt.close(fig_loss)
    print("  Saved training_loss_curves.png")

    if found:
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Top-1 Accuracy")
        ax_acc.set_title("Validation Accuracy Curves")
        ax_acc.legend()
        plt.tight_layout()
        fig_acc.savefig(FIGURES_DIR / "training_accuracy_curves.png", dpi=150)
        plt.close(fig_acc)
        print("  Saved training_accuracy_curves.png")


def plot_model_size_comparison(df: pd.DataFrame):
    """Horizontal bar chart of model sizes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
    ax.barh(df["Model"], df["Size (MB)"], color=colors)
    for i, v in enumerate(df["Size (MB)"]):
        ax.text(v + 0.1, i, f"{v:.1f} MB", va="center", fontsize=9)
    ax.set_xlabel("Model Size (MB)")
    ax.set_title("Model Size Comparison")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "model_size_comparison.png", dpi=150)
    plt.close(fig)
    print("  Saved model_size_comparison.png")


# ── Tables & Report ─────────────────────────────────────────

def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate a LaTeX table for the thesis."""
    cols = ["Model", "Top-1 Acc", "Top-5 Acc", "Precision", "Recall",
            "F1 (macro)", "Inference (ms)", "Size (MB)"]
    tex = df[cols].to_latex(index=False, float_format="%.4f", escape=False)
    return tex


def generate_summary_report(results: dict, df: pd.DataFrame) -> str:
    """Generate a text summary report."""
    lines = []
    lines.append("=" * 70)
    lines.append("SEED RECOGNITION — YOLO CLASSIFICATION MODEL COMPARISON")
    lines.append("=" * 70)
    lines.append(f"\nModels evaluated: {len(df)}")
    lines.append(f"Number of classes: {len(results[list(results.keys())[0]].get('class_names', []))}")

    if not df.empty:
        best_acc = df.loc[df["Top-1 Acc"].idxmax()]
        lines.append(f"\nBest Top-1 Accuracy: {best_acc['Model']} ({best_acc['Top-1 Acc']:.4f})")

        best_f1 = df.loc[df["F1 (macro)"].idxmax()]
        lines.append(f"Best F1 Score:       {best_f1['Model']} ({best_f1['F1 (macro)']:.4f})")

        fastest = df.loc[df["Inference (ms)"].idxmin()]
        lines.append(f"Fastest Inference:   {fastest['Model']} ({fastest['Inference (ms)']:.1f} ms)")

        smallest = df.loc[df["Size (MB)"].idxmin()]
        lines.append(f"Smallest Model:      {smallest['Model']} ({smallest['Size (MB)']:.1f} MB)")

    lines.append("\n" + "-" * 70)
    lines.append("DETAILED METRICS")
    lines.append("-" * 70)

    header = f"{'Model':<16} {'Top1':>7} {'Top5':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'ms':>8} {'MB':>8}"
    lines.append(header)
    lines.append("-" * len(header))

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


# ── Main ────────────────────────────────────────────────────

def main():
    results = load_results()
    if not results:
        return

    # Filter out error entries
    valid = {k: v for k, v in results.items() if "error" not in v}
    if not valid:
        print("No valid results to compare.")
        return

    df = results_to_dataframe(valid)
    if df.empty:
        print("DataFrame is empty.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATING COMPARISON CHARTS & REPORT")
    print("=" * 60)

    plot_accuracy_comparison(df)
    plot_precision_recall_f1(df)
    plot_speed_vs_accuracy(df)
    plot_confusion_matrices(valid)
    plot_per_class_f1(valid)
    plot_radar_chart(df)
    plot_training_curves(valid)
    plot_model_size_comparison(df)

    # LaTeX table
    latex = generate_latex_table(df)
    latex_path = RESULTS_DIR / "comparison_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"  Saved {latex_path}")

    # Summary report
    report = generate_summary_report(valid, df)
    report_path = RESULTS_DIR / "summary_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved {report_path}")

    # Log all figures and reports to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="comparison-report"):
        mlflow.set_tag("stage", "comparison")
        # Log all generated figures
        for fig_file in FIGURES_DIR.glob("*.png"):
            mlflow.log_artifact(str(fig_file), artifact_path="figures")
        # Log LaTeX table and summary report
        mlflow.log_artifact(str(latex_path), artifact_path="reports")
        mlflow.log_artifact(str(report_path), artifact_path="reports")
        # Log the evaluation results JSON
        eval_json = RESULTS_DIR / "evaluation_results.json"
        if eval_json.exists():
            mlflow.log_artifact(str(eval_json), artifact_path="reports")
        print("  Logged all artifacts to MLflow")

    # Print to console
    print("\n" + report)


if __name__ == "__main__":
    main()
