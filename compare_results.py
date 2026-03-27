"""
Generate comparison charts, tables, and analysis for thesis.
Produces publication-quality figures comparing YOLO model versions.

Usage: python compare_results.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

from config import RESULTS_DIR, RUNS_DIR


# Plotting style for thesis
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette for YOLO versions
VERSION_COLORS = {
    'yolov7': '#e74c3c',
    'yolov7-tiny': '#c0392b',
    'yolov8n': '#3498db',
    'yolov8s': '#2980b9',
    'yolov9c': '#2ecc71',
    'yolov9e': '#27ae60',
    'yolov10n': '#f39c12',
    'yolov10s': '#e67e22',
    'yolov11n': '#9b59b6',
    'yolov11s': '#8e44ad',
}


def load_results():
    """Load evaluation results from JSON file."""
    results_file = RESULTS_DIR / "evaluation_results.json"
    
    if not results_file.exists():
        print(f"ERROR: {results_file} not found.")
        print("Run 'python evaluate.py' first.")
        sys.exit(1)
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Filter out errored results
    clean = {k: v for k, v in results.items() if "error" not in v}
    
    if not clean:
        print("No valid results found.")
        sys.exit(1)
    
    return clean


def results_to_dataframe(results: dict) -> pd.DataFrame:
    """Convert results dict to a pandas DataFrame."""
    rows = []
    for model_name, metrics in results.items():
        row = {
            "Model": model_name,
            "Type": metrics.get("type", "unknown"),
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "mAP@50": metrics.get("mAP50", 0),
            "mAP@50-95": metrics.get("mAP50-95", 0),
            "F1 Score": metrics.get("f1_score", 0),
            "Inference (ms)": metrics.get("inference_time_ms", 0),
            "Model Size (MB)": metrics.get("model_size_mb", 0),
            "Parameters (M)": metrics.get("parameters", 0) / 1e6 if metrics.get("parameters", 0) > 0 else 0,
            "FLOPs (G)": metrics.get("flops", 0) / 1e9 if metrics.get("flops", 0) > 0 else 0,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by YOLO version
    version_order = ['yolov7', 'yolov7-tiny', 'yolov8n', 'yolov8s',
                     'yolov9c', 'yolov9e', 'yolov10n', 'yolov10s',
                     'yolov11n', 'yolov11s']
    df['sort_key'] = df['Model'].map({v: i for i, v in enumerate(version_order)})
    df = df.sort_values('sort_key').drop('sort_key', axis=1).reset_index(drop=True)
    
    return df


def get_color(model_name: str) -> str:
    """Get color for a model."""
    return VERSION_COLORS.get(model_name, '#95a5a6')


def plot_metrics_bar_chart(df: pd.DataFrame, output_dir: Path):
    """Bar chart comparing precision, recall, mAP50, mAP50-95 across models."""
    metrics = ['Precision', 'Recall', 'mAP@50', 'mAP@50-95']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        colors = [get_color(m) for m in df['Model']]
        bars = ax.bar(range(len(df)), df[metric], color=colors, edgecolor='white', linewidth=0.5)
        
        ax.set_title(metric, fontweight='bold')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score')
        ax.grid(axis='y', alpha=0.3)
        
        # Value labels
        for bar, val in zip(bars, df[metric]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Seed Recognition: Detection Metrics Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison_bar.png')
    plt.savefig(output_dir / 'metrics_comparison_bar.pdf')
    plt.close()
    print("  Saved: metrics_comparison_bar.png/pdf")


def plot_map_comparison(df: pd.DataFrame, output_dir: Path):
    """Grouped bar chart for mAP50 and mAP50-95."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['mAP@50'], width, label='mAP@50',
                   color='#3498db', edgecolor='white')
    bars2 = ax.bar(x + width/2, df['mAP@50-95'], width, label='mAP@50-95',
                   color='#e74c3c', edgecolor='white')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('mAP Score')
    ax.set_title('Mean Average Precision: YOLO Version Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'map_comparison.png')
    plt.savefig(output_dir / 'map_comparison.pdf')
    plt.close()
    print("  Saved: map_comparison.png/pdf")


def plot_speed_vs_accuracy(df: pd.DataFrame, output_dir: Path):
    """Scatter plot: inference speed vs mAP50-95 (key thesis figure)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for _, row in df.iterrows():
        color = get_color(row['Model'])
        ax.scatter(row['Inference (ms)'], row['mAP@50-95'],
                   s=row['Model Size (MB)'] * 3 + 50,  # Size proportional to model size
                   c=color, edgecolors='black', linewidth=0.5,
                   alpha=0.8, zorder=5)
        ax.annotate(row['Model'],
                     (row['Inference (ms)'], row['mAP@50-95']),
                     textcoords="offset points", xytext=(8, 5),
                     fontsize=9)
    
    ax.set_xlabel('Inference Time (ms)')
    ax.set_ylabel('mAP@50-95')
    ax.set_title('Speed vs Accuracy Trade-off\n(bubble size = model file size)',
                 fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'speed_vs_accuracy.png')
    plt.savefig(output_dir / 'speed_vs_accuracy.pdf')
    plt.close()
    print("  Saved: speed_vs_accuracy.png/pdf")


def plot_model_efficiency(df: pd.DataFrame, output_dir: Path):
    """Compare model size, parameters, and FLOPs."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    metrics = [
        ('Model Size (MB)', 'Model Size (MB)', '#3498db'),
        ('Parameters (M)', 'Parameters (Millions)', '#2ecc71'),
        ('FLOPs (G)', 'FLOPs (GFLOPs)', '#e74c3c'),
    ]
    
    for ax, (col, ylabel, color) in zip(axes, metrics):
        values = df[col]
        colors = [get_color(m) for m in df['Model']]
        bars = ax.barh(range(len(df)), values, color=colors, edgecolor='white')
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['Model'])
        ax.set_xlabel(ylabel)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2.,
                        f'{val:.1f}', va='center', fontsize=9)
    
    plt.suptitle('Model Efficiency Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_efficiency.png')
    plt.savefig(output_dir / 'model_efficiency.pdf')
    plt.close()
    print("  Saved: model_efficiency.png/pdf")


def plot_f1_comparison(df: pd.DataFrame, output_dir: Path):
    """F1 Score comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [get_color(m) for m in df['Model']]
    bars = ax.bar(range(len(df)), df['F1 Score'], color=colors, edgecolor='white')
    
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score Comparison Across YOLO Versions', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, df['F1 Score']):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Add best model highlight
    best_idx = df['F1 Score'].idxmax()
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'f1_comparison.png')
    plt.savefig(output_dir / 'f1_comparison.pdf')
    plt.close()
    print("  Saved: f1_comparison.png/pdf")


def plot_radar_chart(df: pd.DataFrame, output_dir: Path):
    """Radar/spider chart comparing all metrics (normalized)."""
    metrics = ['Precision', 'Recall', 'mAP@50', 'mAP@50-95', 'F1 Score']
    
    # Add inverted inference time (faster = better)
    if df['Inference (ms)'].max() > 0:
        df_plot = df.copy()
        max_inf = df_plot['Inference (ms)'].max()
        df_plot['Speed'] = 1 - (df_plot['Inference (ms)'] / max_inf) if max_inf > 0 else 0
        metrics.append('Speed')
    else:
        df_plot = df.copy()
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for _, row in df_plot.iterrows():
        values = [row.get(m, 0) for m in metrics]
        values += values[:1]
        color = get_color(row['Model'])
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_title('Multi-Metric Comparison\n(higher = better for all axes)',
                 fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'radar_comparison.png')
    plt.savefig(output_dir / 'radar_comparison.pdf')
    plt.close()
    print("  Saved: radar_comparison.png/pdf")


def plot_training_curves(output_dir: Path):
    """Plot training loss curves for all models (if available)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    found_any = False
    
    # Check Ultralytics CSV results
    ultra_dir = RUNS_DIR / "ultralytics"
    if ultra_dir.exists():
        for run in sorted(ultra_dir.iterdir()):
            csv_file = run / "results.csv"
            if csv_file.exists():
                try:
                    run_df = pd.read_csv(csv_file)
                    run_df.columns = run_df.columns.str.strip()
                    
                    color = get_color(run.name)
                    
                    # Box loss
                    if 'train/box_loss' in run_df.columns:
                        axes[0].plot(run_df['train/box_loss'], label=run.name,
                                     color=color, linewidth=1.5)
                        found_any = True
                    
                    # mAP50
                    if 'metrics/mAP50(B)' in run_df.columns:
                        axes[1].plot(run_df['metrics/mAP50(B)'], label=run.name,
                                     color=color, linewidth=1.5)
                except Exception as e:
                    print(f"  Could not parse {csv_file}: {e}")
    
    if found_any:
        axes[0].set_title('Training Box Loss', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title('Validation mAP@50', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('mAP@50')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Training Progress Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves.png')
        plt.savefig(output_dir / 'training_curves.pdf')
        plt.close()
        print("  Saved: training_curves.png/pdf")
    else:
        plt.close()
        print("  No training curves found (results.csv files missing)")


def plot_per_class_comparison(results: dict, output_dir: Path):
    """Compare per-class AP across models (if available)."""
    # Collect per-class data
    per_class_data = {}
    for model_name, metrics in results.items():
        if "per_class" in metrics:
            for cls_name, cls_metrics in metrics["per_class"].items():
                if cls_name not in per_class_data:
                    per_class_data[cls_name] = {}
                per_class_data[cls_name][model_name] = cls_metrics.get("ap50", 0)
    
    if not per_class_data:
        print("  No per-class data available")
        return
    
    # Create heatmap
    classes = sorted(per_class_data.keys())
    models = sorted(set(m for cls in per_class_data.values() for m in cls.keys()))
    
    data = np.zeros((len(classes), len(models)))
    for i, cls in enumerate(classes):
        for j, model in enumerate(models):
            data[i, j] = per_class_data.get(cls, {}).get(model, 0)
    
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.5), max(6, len(classes) * 0.8)))
    
    sns.heatmap(data, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=models, yticklabels=classes,
                ax=ax, vmin=0, vmax=1)
    
    ax.set_title('Per-Class AP@50 Comparison', fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('Seed Class')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_heatmap.png')
    plt.savefig(output_dir / 'per_class_heatmap.pdf')
    plt.close()
    print("  Saved: per_class_heatmap.png/pdf")


def generate_latex_table(df: pd.DataFrame, output_dir: Path):
    """Generate LaTeX table for thesis."""
    # Select columns for the table
    cols = ['Model', 'Precision', 'Recall', 'mAP@50', 'mAP@50-95',
            'F1 Score', 'Inference (ms)', 'Model Size (MB)', 'Parameters (M)']
    
    available_cols = [c for c in cols if c in df.columns]
    table_df = df[available_cols].copy()
    
    # Format numbers
    for col in available_cols:
        if col != 'Model':
            if col in ['Inference (ms)', 'Model Size (MB)', 'Parameters (M)']:
                table_df[col] = table_df[col].apply(lambda x: f'{x:.1f}')
            else:
                table_df[col] = table_df[col].apply(lambda x: f'{x:.4f}')
    
    # Bold the best value in each column
    latex = table_df.to_latex(index=False, escape=False, column_format='l' + 'c' * (len(available_cols) - 1))
    
    # Write to file
    with open(output_dir / 'results_table.tex', 'w') as f:
        f.write("% Auto-generated comparison table\n")
        f.write("% Include in thesis with \\input{results_table.tex}\n\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparison of YOLO Model Versions for Seed Recognition}\n")
        f.write("\\label{tab:yolo_comparison}\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write(latex)
        f.write("}\n")
        f.write("\\end{table}\n")
    
    print("  Saved: results_table.tex")
    
    # Also generate markdown table
    md_table = tabulate(table_df, headers='keys', tablefmt='pipe', showindex=False)
    with open(output_dir / 'results_table.md', 'w') as f:
        f.write("# YOLO Model Comparison Results\n\n")
        f.write(md_table)
        f.write("\n")
    
    print("  Saved: results_table.md")


def generate_analysis_summary(df: pd.DataFrame, output_dir: Path):
    """Generate a text summary of the analysis."""
    summary = []
    summary.append("=" * 70)
    summary.append("SEED RECOGNITION: YOLO VERSION COMPARISON - ANALYSIS SUMMARY")
    summary.append("=" * 70)
    
    summary.append(f"\nModels compared: {len(df)}")
    summary.append(f"Models: {', '.join(df['Model'].tolist())}")
    
    # Best models
    summary.append("\n--- BEST PERFORMERS ---")
    
    for metric in ['Precision', 'Recall', 'mAP@50', 'mAP@50-95', 'F1 Score']:
        best_idx = df[metric].idxmax()
        best_model = df.loc[best_idx, 'Model']
        best_val = df.loc[best_idx, metric]
        summary.append(f"  Best {metric}: {best_model} ({best_val:.4f})")
    
    # Fastest
    if df['Inference (ms)'].max() > 0:
        fast_idx = df.loc[df['Inference (ms)'] > 0, 'Inference (ms)'].idxmin()
        fast_model = df.loc[fast_idx, 'Model']
        fast_val = df.loc[fast_idx, 'Inference (ms)']
        summary.append(f"  Fastest inference: {fast_model} ({fast_val:.1f} ms)")
    
    # Smallest
    if df['Model Size (MB)'].max() > 0:
        small_idx = df.loc[df['Model Size (MB)'] > 0, 'Model Size (MB)'].idxmin()
        small_model = df.loc[small_idx, 'Model']
        small_val = df.loc[small_idx, 'Model Size (MB)']
        summary.append(f"  Smallest model: {small_model} ({small_val:.1f} MB)")
    
    # Version progression
    summary.append("\n--- VERSION PROGRESSION ---")
    versions = ['yolov7', 'yolov8', 'yolov9', 'yolov10', 'yolov11']
    for ver in versions:
        ver_models = df[df['Model'].str.startswith(ver)]
        if not ver_models.empty:
            best = ver_models.loc[ver_models['mAP@50-95'].idxmax()]
            summary.append(f"  {ver.upper()} best: {best['Model']} "
                           f"(mAP50-95={best['mAP@50-95']:.4f})")
    
    # Overall recommendation
    summary.append("\n--- RECOMMENDATION ---")
    best_map = df.loc[df['mAP@50-95'].idxmax()]
    summary.append(f"  Best overall accuracy: {best_map['Model']} "
                   f"(mAP@50-95={best_map['mAP@50-95']:.4f})")
    
    if df['Inference (ms)'].max() > 0:
        # Best accuracy-speed trade-off (simple efficiency score)
        df_eff = df[df['Inference (ms)'] > 0].copy()
        if not df_eff.empty:
            df_eff['efficiency'] = df_eff['mAP@50-95'] / df_eff['Inference (ms)']
            best_eff = df_eff.loc[df_eff['efficiency'].idxmax()]
            summary.append(f"  Best speed/accuracy trade-off: {best_eff['Model']} "
                           f"(mAP@50-95={best_eff['mAP@50-95']:.4f}, "
                           f"{best_eff['Inference (ms)']:.1f} ms)")
    
    summary_text = "\n".join(summary)
    
    with open(output_dir / 'analysis_summary.txt', 'w') as f:
        f.write(summary_text)
    
    print(summary_text)
    return summary_text


def main():
    print("=" * 60)
    print("GENERATING COMPARISON RESULTS & VISUALIZATIONS")
    print("=" * 60)
    
    # Create output directories
    figures_dir = RESULTS_DIR / "figures"
    tables_dir = RESULTS_DIR / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results = load_results()
    df = results_to_dataframe(results)
    
    print(f"\nLoaded results for {len(df)} models")
    print(df[['Model', 'mAP@50', 'mAP@50-95']].to_string(index=False))
    
    # Generate all plots
    print("\nGenerating figures...")
    plot_metrics_bar_chart(df, figures_dir)
    plot_map_comparison(df, figures_dir)
    plot_speed_vs_accuracy(df, figures_dir)
    plot_model_efficiency(df, figures_dir)
    plot_f1_comparison(df, figures_dir)
    plot_radar_chart(df, figures_dir)
    plot_training_curves(figures_dir)
    plot_per_class_comparison(results, figures_dir)
    
    # Generate tables
    print("\nGenerating tables...")
    generate_latex_table(df, tables_dir)
    
    # Generate analysis
    print("\nGenerating analysis...")
    generate_analysis_summary(df, tables_dir)
    
    print(f"\nAll outputs saved to {RESULTS_DIR}")
    print(f"  Figures: {figures_dir}")
    print(f"  Tables: {tables_dir}")


if __name__ == "__main__":
    main()
