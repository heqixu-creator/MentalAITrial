#!/usr/bin/env python3
"""
Entry point: train and evaluate the baseline (TF-IDF + Logistic Regression) model.
Saves metrics to results/baseline_metrics.txt and confusion matrix to results/confusion_matrix_baseline.png.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import load_dataset, get_train_test_split, ensure_results_dir
from src.baseline_model import train_baseline, evaluate_baseline


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate baseline (TF-IDF + LR) model.")
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
    args = parser.parse_args()

    results_dir = ensure_results_dir(args.results_dir)
    metrics_path = results_dir / "baseline_metrics.txt"
    plot_path = results_dir / "confusion_matrix_baseline.png"

    print("Loading dataset...")
    df = load_dataset(args.data)
    X_train, X_test, y_train, y_test = get_train_test_split(df, test_size=0.2, stratify=True)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    print("Training baseline (TF-IDF + Logistic Regression)...")
    model, vectorizer = train_baseline(X_train, y_train)

    print("Evaluating on test set...")
    metrics, report = evaluate_baseline(
        model, vectorizer, X_test, y_test, save_plot_path=str(plot_path)
    )

    print("\n--- Classification Report (Baseline) ---\n")
    print(report)
    print("\n--- Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nConfusion matrix plot saved to: {plot_path}")

    # Save metrics to file
    with open(metrics_path, "w") as f:
        f.write("Baseline (TF-IDF + Logistic Regression) - Test Set Metrics\n")
        f.write("=" * 50 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write(f"\nConfusion matrix plot: {plot_path}\n")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
