#!/usr/bin/env python3
"""
Download the Reddit depression dataset from HuggingFace and save as data/dataset.csv
with columns 'text' and 'label' (0 = non-depression, 1 = depression).
"""

import sys
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    from datasets import load_dataset
    import pandas as pd

    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "dataset.csv"

    print("Loading hugginglearners/reddit-depression-cleaned from HuggingFace...")
    ds = load_dataset("hugginglearners/reddit-depression-cleaned", trust_remote_code=True)
    # Use train split (or first available)
    split = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()

    # Map to expected columns: text, label (0/1)
    if "clean_text" in df.columns and "is_depression" in df.columns:
        df = df.rename(columns={"clean_text": "text", "is_depression": "label"})
    elif "text" not in df.columns or "label" not in df.columns:
        # Try common variants
        text_col = next((c for c in df.columns if "text" in c.lower() or c == "post"), df.columns[0])
        label_col = next((c for c in df.columns if "depress" in c.lower() or "label" in c), None)
        if label_col is None:
            raise ValueError(f"Could not find label column. Columns: {list(df.columns)}")
        df = df.rename(columns={text_col: "text", label_col: "label"})

    df = df[["text", "label"]].dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    if set(df["label"].unique()) - {0, 1}:
        # Map to binary if needed (e.g. 0/1 or True/False)
        df["label"] = (df["label"] != 0).astype(int)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
