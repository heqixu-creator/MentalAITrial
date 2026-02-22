#!/usr/bin/env python3
"""
Generate results/confusion_matrix_bert.png from existing results/bert_metrics.txt
by parsing the classification report (support and recall) to reconstruct the 2x2 matrix.
Run this if BERT was evaluated before confusion matrix plotting was added.
"""

import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_classification_report(text: str) -> tuple:
    """Parse support and recall for non-depression (0) and depression (1). Returns (supports, recalls)."""
    # Report order: non-depression then depression. Match by start of line content to avoid "depression" matching "non-depression".
    supports = []
    recalls = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            if stripped.startswith("non-depression"):
                rec = float(parts[2])
                sup = int(parts[4])
                supports.append(sup)
                recalls.append(rec)
            elif stripped.startswith("depression") and "non-depression" not in line:
                rec = float(parts[2])
                sup = int(parts[4])
                supports.append(sup)
                recalls.append(rec)
        except (ValueError, IndexError):
            continue
    if len(supports) != 2 or len(recalls) != 2:
        return None, None
    return supports, recalls


def confusion_matrix_from_report(support0: int, support1: int, recall0: float, recall1: float) -> np.ndarray:
    """Reconstruct 2x2 confusion matrix [ [TN, FP], [FN, TP] ] from binary report."""
    # For class 0: recall_0 = TN / (TN + FN) = TN / support0  =>  TN = recall0 * support0
    # For class 1: recall_1 = TP / (TP + FN) = TP / support1   =>  TP = recall1 * support1
    TN = int(round(recall0 * support0))
    FN = support0 - TN
    TP = int(round(recall1 * support1))
    FP = support1 - TP
    return np.array([[TN, FP], [FN, TP]])


def plot_and_save(cm: np.ndarray, save_path: str) -> None:
    """Plot confusion matrix heatmap and save."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["non-depression", "depression"],
        yticklabels=["non-depression", "depression"],
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, format(int(cm[i, j]), "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    root = Path(__file__).resolve().parent.parent
    metrics_path = root / "results" / "bert_metrics.txt"
    out_path = root / "results" / "confusion_matrix_bert.png"
    if not metrics_path.exists():
        print(f"Not found: {metrics_path}. Run run_bert.py first.")
        return 1
    text = metrics_path.read_text()
    supports, recalls = parse_classification_report(text)
    if not supports or not recalls:
        print("Could not parse classification report from bert_metrics.txt.")
        return 1
    cm = confusion_matrix_from_report(supports[0], supports[1], recalls[0], recalls[1])
    plot_and_save(cm, str(out_path))
    return 0


if __name__ == "__main__":
    exit(main())
