#!/usr/bin/env python3
"""
Build results/comparison_table.md from existing results/baseline_metrics.txt
and results/bert_metrics.txt. Run this after both run_baseline.py and run_bert.py
have been executed (e.g. after BERT finishes when run in background).
"""

import re
from pathlib import Path

def parse_metrics(path: Path) -> dict:
    text = path.read_text()
    metrics = {}
    for name in ("accuracy", "precision", "recall", "f1"):
        m = re.search(rf"{name}\s*:\s*([\d.]+)", text, re.I)
        if m:
            metrics[name] = float(m.group(1))
    if len(metrics) != 4:
        raise ValueError(f"Could not parse all four metrics from {path}. Got: {list(metrics.keys())}")
    return metrics

def main():
    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results"
    baseline_path = results_dir / "baseline_metrics.txt"
    bert_path = results_dir / "bert_metrics.txt"
    out_path = results_dir / "comparison_table.md"

    if not baseline_path.exists():
        print(f"Missing {baseline_path}. Run run_baseline.py first.")
        return 1
    if not bert_path.exists():
        print(f"Missing {bert_path}. Run run_bert.py first.")
        return 1

    baseline_metrics = parse_metrics(baseline_path)
    bert_metrics = parse_metrics(bert_path)
    if len(baseline_metrics) != 4 or len(bert_metrics) != 4:
        print("Could not parse all four metrics from one or both files.")
        return 1

    rows = [
        ("Metric", "Baseline (LR + TF-IDF)", "BERT"),
        ("---", "---", "---"),
        ("Accuracy", f"{baseline_metrics['accuracy']:.4f}", f"{bert_metrics['accuracy']:.4f}"),
        ("Precision", f"{baseline_metrics['precision']:.4f}", f"{bert_metrics['precision']:.4f}"),
        ("Recall", f"{baseline_metrics['recall']:.4f}", f"{bert_metrics['recall']:.4f}"),
        ("F1-score", f"{baseline_metrics['f1']:.4f}", f"{bert_metrics['f1']:.4f}"),
    ]
    table = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    out_path.write_text("# Model Comparison (Test Set)\n\n" + table + "\n")
    print(f"Wrote {out_path}")
    return 0

if __name__ == "__main__":
    exit(main())
