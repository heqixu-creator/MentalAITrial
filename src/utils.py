"""
Shared utilities for data loading, train/test split, and results directory.
Uses a fixed random state (42) for reproducibility.
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Reproducibility: same split and seeds across baseline and BERT
RANDOM_STATE = 42
DEFAULT_DATA_PATH = "data/dataset.csv"
DEFAULT_RESULTS_DIR = "results"


def load_dataset(path: str = None) -> pd.DataFrame:
    """
    Load the labelled CSV dataset and validate columns and labels.

    Expected columns: 'text' (str), 'label' (int: 0 = non-depression, 1 = depression).
    Raises ValueError if columns are missing or labels are not binary.

    Args:
        path: Path to CSV file. Defaults to data/dataset.csv.

    Returns:
        DataFrame with columns 'text' and 'label'.
    """
    path = path or DEFAULT_DATA_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place your CSV with columns 'text' and 'label' there."
        )
    df = pd.read_csv(path)
    # Support Reddit depression cleaned format: clean_text, is_depression
    if "text" not in df.columns and "clean_text" in df.columns:
        df = df.rename(columns={"clean_text": "text"})
    if "label" not in df.columns and "is_depression" in df.columns:
        df = df.rename(columns={"is_depression": "label"})
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Dataset must have columns 'text' and 'label' (or 'clean_text' and 'is_depression'). Found: {list(df.columns)}"
        )
    df = df[["text", "label"]].dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    unique = df["label"].unique()
    if not (set(unique) <= {0, 1}):
        raise ValueError(
            f"Labels must be binary (0 and 1). Found unique values: {sorted(unique)}"
        )
    return df


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: int = None,
):
    """
    Split DataFrame into train and test sets (80/20 by default, stratified).

    Args:
        df: DataFrame with 'text' and 'label' columns.
        test_size: Fraction for test set (default 0.2).
        stratify: If True, stratify by 'label' for balanced splits.
        random_state: Random seed (default: RANDOM_STATE).

    Returns:
        Tuple (X_train, X_test, y_train, y_test) where X are text series, y are label arrays.
    """
    random_state = random_state if random_state is not None else RANDOM_STATE
    X = df["text"]
    y = df["label"].values
    stratify_arg = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify_arg, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def ensure_results_dir(results_dir: str = None) -> Path:
    """
    Create the results directory if it does not exist.

    Args:
        results_dir: Path to results folder (default: results/).

    Returns:
        Path to the results directory.
    """
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    path = Path(results_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int = None):
    """
    Set random seeds for numpy (and optionally torch if available) for reproducibility.
    """
    seed = seed or RANDOM_STATE
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
