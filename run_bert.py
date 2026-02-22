#!/usr/bin/env python3
"""
Entry point: fine-tune and evaluate the BERT model for binary depression classification.
Saves metrics to results/bert_metrics.txt.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import load_dataset, get_train_test_split, ensure_results_dir
from src.bert_model import train_bert, evaluate_bert


def main():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate BERT model.")
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
        help="Number of training epochs (default: 3)",
    )
    args = parser.parse_args()

    results_dir = ensure_results_dir(args.results_dir)
    metrics_path = results_dir / "bert_metrics.txt"
    plot_path = results_dir / "confusion_matrix_bert.png"

    print("Loading dataset...")
    df = load_dataset(args.data)
    X_train, X_test, y_train, y_test = get_train_test_split(df, test_size=0.2, stratify=True)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    print("Fine-tuning BERT (bert-base-uncased)...")
    model, tokenizer = train_bert(
        X_train, y_train, X_test, y_test,
        epochs=args.epochs,
    )

    print("Evaluating on test set...")
    metrics, report = evaluate_bert(model, tokenizer, X_test, y_test, save_plot_path=str(plot_path))

    print("\n--- Classification Report (BERT) ---\n")
    print(report)
    print("\n--- Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save metrics to file
    with open(metrics_path, "w") as f:
        f.write("BERT (bert-base-uncased) - Test Set Metrics\n")
        f.write("=" * 50 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write(f"\nConfusion matrix plot: {plot_path}\n")
    print(f"Confusion matrix plot saved to: {plot_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
