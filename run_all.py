#!/usr/bin/env python3
"""
Run both baseline and BERT models, save metrics to separate files, and generate
a Markdown comparison table in results/comparison_table.md.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import load_dataset, get_train_test_split, ensure_results_dir
from src.baseline_model import train_baseline, evaluate_baseline
from src.bert_model import train_bert, evaluate_bert


def save_metrics_txt(path: Path, title: str, metrics: dict, report: str, extra: str = "") -> None:
    """Write metrics and classification report to a text file."""
    with open(path, "w") as f:
        f.write(f"{title}\n")
        f.write("=" * 50 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        if extra:
            f.write(extra)
    print(f"Metrics saved to: {path}")


def write_comparison_table(path: Path, baseline_metrics: dict, bert_metrics: dict) -> None:
    """Generate a Markdown comparison table and write to path."""
    rows = [
        ("Metric", "Baseline (LR + TF-IDF)", "BERT"),
        ("---", "---", "---"),
        ("Accuracy", f"{baseline_metrics['accuracy']:.4f}", f"{bert_metrics['accuracy']:.4f}"),
        ("Precision", f"{baseline_metrics['precision']:.4f}", f"{bert_metrics['precision']:.4f}"),
        ("Recall", f"{baseline_metrics['recall']:.4f}", f"{bert_metrics['recall']:.4f}"),
        ("F1-score", f"{baseline_metrics['f1']:.4f}", f"{bert_metrics['f1']:.4f}"),
    ]
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    table = "\n".join(lines)
    with open(path, "w") as f:
        f.write("# Model Comparison (Test Set)\n\n")
        f.write(table)
        f.write("\n")
    print(f"Comparison table saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline and BERT, then generate comparison table.")
    parser.add_argument(
        "--data",
        default="data/dataset.csv",
        help="Path to CSV with 'text' and 'label' columns (default: data/dataset.csv)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for outputs (default: results)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="BERT training epochs (default: 3)",
    )
    args = parser.parse_args()

    results_dir = ensure_results_dir(args.results_dir)
    baseline_metrics_path = results_dir / "baseline_metrics.txt"
    bert_metrics_path = results_dir / "bert_metrics.txt"
    baseline_plot_path = results_dir / "confusion_matrix_baseline.png"
    bert_plot_path = results_dir / "confusion_matrix_bert.png"
    comparison_path = results_dir / "comparison_table.md"

    print("Loading dataset...")
    df = load_dataset(args.data)
    X_train, X_test, y_train, y_test = get_train_test_split(df, test_size=0.2, stratify=True)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}\n")

    # --- Baseline ---
    print("Training baseline (TF-IDF + Logistic Regression)...")
    model, vectorizer = train_baseline(X_train, y_train)
    baseline_metrics, baseline_report = evaluate_baseline(
        model, vectorizer, X_test, y_test, save_plot_path=str(baseline_plot_path)
    )
    print("\n--- Baseline Classification Report ---\n")
    print(baseline_report)
    save_metrics_txt(
        baseline_metrics_path,
        "Baseline (TF-IDF + Logistic Regression) - Test Set Metrics",
        baseline_metrics,
        baseline_report,
        extra=f"\nConfusion matrix plot: {baseline_plot_path}\n",
    )

    # --- BERT ---
    print("\nFine-tuning BERT (bert-base-uncased)...")
    bert_model, tokenizer = train_bert(
        X_train, y_train, X_test, y_test,
        epochs=args.epochs,
    )
    bert_metrics, bert_report = evaluate_bert(
        bert_model, tokenizer, X_test, y_test, save_plot_path=str(bert_plot_path)
    )
    print("\n--- BERT Classification Report ---\n")
    print(bert_report)
    save_metrics_txt(
        bert_metrics_path,
        "BERT (bert-base-uncased) - Test Set Metrics",
        bert_metrics,
        bert_report,
        extra=f"\nConfusion matrix plot: {bert_plot_path}\n",
    )

    # --- Comparison table ---
    write_comparison_table(comparison_path, baseline_metrics, bert_metrics)
    print("\nDone. Run complete.")


if __name__ == "__main__":
    main()
