"""
evaluator.py — Evaluation and visualization for the IDS pipeline.

Phase 10:
- Comparison tables (Accuracy, Precision, Recall, F1)
- Bar charts comparing before/after SMOTE, before/after FS
- Confusion matrix heatmaps
- Save all plots and tables to results/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import RESULTS_DIR, MODEL_NAMES, CATEGORIES


def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def print_results_table(results, title="Model Comparison"):
    """
    Print a formatted results table to console.
    """
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(f"  {'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Time(s)':>8}")
    print(f"  {'-' * 72}")

    for name in MODEL_NAMES:
        if name in results:
            r = results[name]
            print(f"  {name:<22} {r['accuracy']:>9}% {r['precision']:>9}% "
                  f"{r['recall']:>9}% {r['f1']:>9}% {r['train_time']:>7}")

    print(f"  {'-' * 72}")


def save_results_csv(all_results, filename="comparison_results.csv"):
    """
    Save all results as a CSV table.

    Args:
        all_results: dict of {config_name: {model_name: {metrics}}}
    """
    ensure_results_dir()
    rows = []

    for config_name, results in all_results.items():
        for model_name, metrics in results.items():
            rows.append({
                'Configuration': config_name,
                'Model': model_name,
                'Accuracy (%)': metrics['accuracy'],
                'Precision (%)': metrics['precision'],
                'Recall (%)': metrics['recall'],
                'F1-Score (%)': metrics['f1'],
                'Train Time (s)': metrics['train_time'],
            })

    df = pd.DataFrame(rows)
    filepath = os.path.join(RESULTS_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"\n  Results saved to: {filepath}")
    return df


def plot_metric_comparison(all_results, metric='accuracy', filename=None):
    """
    Bar chart comparing a single metric across all configurations and models.
    """
    ensure_results_dir()

    configs = list(all_results.keys())
    models = MODEL_NAMES

    x = np.arange(len(models))
    width = 0.8 / len(configs)

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, config in enumerate(configs):
        values = []
        for model in models:
            if model in all_results[config]:
                values.append(all_results[config][model][metric])
            else:
                values.append(0)

        bars = ax.bar(x + i * width, values, width, label=config, alpha=0.85)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(f'{metric.capitalize()} (%)', fontsize=12)
    ax.set_title(f'{metric.capitalize()} Comparison Across Configurations', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(configs) - 1) / 2)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if filename is None:
        filename = f"{metric}_comparison.png"
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {filepath}")


def plot_confusion_matrix(cm, labels, title, filename):
    """
    Plot a single confusion matrix heatmap.
    """
    ensure_results_dir()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    plt.tight_layout()
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_confusion_matrices(results, task='binary', config_name='baseline'):
    """
    Plot confusion matrices for all models in a config.
    """
    if task == 'binary':
        labels = ['Normal', 'Attack']
    else:
        labels = CATEGORIES

    for name in MODEL_NAMES:
        if name in results and 'confusion_matrix' in results[name]:
            cm = results[name]['confusion_matrix']
            safe_name = name.replace(' ', '_').lower()
            filename = f"cm_{config_name}_{safe_name}.png"
            title = f"{name} - {config_name}"
            plot_confusion_matrix(cm, labels, title, filename)


def plot_before_after_comparison(results_before, results_after, label_before, label_after, metric='f1', filename=None):
    """
    Side-by-side bar chart comparing a metric before and after a treatment (SMOTE or FS).
    """
    ensure_results_dir()

    models = [m for m in MODEL_NAMES if m in results_before and m in results_after]
    before_vals = [results_before[m][metric] for m in models]
    after_vals = [results_after[m][metric] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, before_vals, width, label=label_before, color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width / 2, after_vals, width, label=label_after, color='#2ecc71', alpha=0.8)

    # Value labels
    for bar, val in zip(bars1, before_vals):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, after_vals):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(f'{metric.capitalize()} (%)', fontsize=12)
    ax.set_title(f'{metric.capitalize()}: {label_before} vs {label_after}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if filename is None:
        safe_label = label_after.replace(' ', '_').lower()
        filename = f"{metric}_{safe_label}_comparison.png"
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {filepath}")


def generate_all_plots(all_results):
    """
    Generate all comparison plots from the full results dictionary.

    Args:
        all_results: dict of {config_name: {model_name: {metrics}}}
    """
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 70)

    # Metric comparison across all configs
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        plot_metric_comparison(all_results, metric=metric)

    # Before/After comparisons
    configs = list(all_results.keys())

    if 'Baseline' in all_results and 'With SMOTE' in all_results:
        for metric in ['accuracy', 'f1', 'recall']:
            plot_before_after_comparison(
                all_results['Baseline'], all_results['With SMOTE'],
                'Before SMOTE', 'After SMOTE', metric=metric
            )

    if 'Baseline' in all_results and 'With FS' in all_results:
        for metric in ['accuracy', 'f1']:
            plot_before_after_comparison(
                all_results['Baseline'], all_results['With FS'],
                'Before FS', 'After FS', metric=metric
            )

    if 'With SMOTE' in all_results and 'SMOTE + FS' in all_results:
        for metric in ['accuracy', 'f1']:
            plot_before_after_comparison(
                all_results['With SMOTE'], all_results['SMOTE + FS'],
                'SMOTE Only', 'SMOTE + FS', metric=metric
            )

    print("\n  All plots generated successfully!")


def find_best_model(all_results, metric='f1'):
    """
    Find the best model across all configurations by a given metric.

    Returns:
        best_config, best_model_name, best_score
    """
    best_config = None
    best_model = None
    best_score = -1

    for config_name, results in all_results.items():
        for model_name, metrics in results.items():
            if metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model = model_name
                best_config = config_name

    return best_config, best_model, best_score
