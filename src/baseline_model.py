"""
Baseline model: text preprocessing, TF-IDF vectorisation, and Logistic Regression.
Evaluation includes accuracy, precision, recall, F1, confusion matrix, and classification report.
"""

import re
import string
from pathlib import Path
from typing import Dict, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from .utils import RANDOM_STATE


def preprocess_text(series: pd.Series) -> pd.Series:
    """
    Preprocess text: lowercase and remove punctuation.

    Args:
        series: Pandas Series of raw text strings.

    Returns:
        Series of preprocessed strings.
    """
    def _clean(s):
        if pd.isna(s) or not isinstance(s, str):
            return ""
        s = s.lower().strip()
        s = re.sub(f"[{re.escape(string.punctuation)}]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    return series.apply(_clean)


def train_baseline(
    X_train: pd.Series,
    y_train: np.ndarray,
    max_features: int = 15000,
    ngram_range: Tuple[int, int] = (1, 2),
    random_state: int = None,
) -> Tuple[Any, TfidfVectorizer]:
    """
    Train TF-IDF vectoriser and Logistic Regression on preprocessed text.

    Args:
        X_train: Training text (will be preprocessed inside).
        y_train: Training labels (0/1).
        max_features: Max vocabulary size for TF-IDF.
        ngram_range: (min_n, max_n) for n-grams.
        random_state: Seed for LogisticRegression.

    Returns:
        Tuple (fitted LogisticRegression model, fitted TfidfVectorizer).
    """
    random_state = random_state or RANDOM_STATE
    X_clean = preprocess_text(X_train)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_vec = vectorizer.fit_transform(X_clean)

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_vec, y_train)
    return model, vectorizer


def evaluate_baseline(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    X_test: pd.Series,
    y_test: np.ndarray,
    save_plot_path: str = None,
) -> Tuple[Dict[str, float], str]:
    """
    Evaluate the baseline model: compute metrics, optionally plot confusion matrix,
    and return metrics dict plus classification report string.

    Args:
        model: Fitted LogisticRegression.
        vectorizer: Fitted TfidfVectorizer.
        X_test: Test text (will be preprocessed).
        y_test: Test labels.
        save_plot_path: If set, save confusion matrix plot to this path.

    Returns:
        Tuple (metrics_dict, classification_report_str).
        metrics_dict has keys: accuracy, precision, recall, f1.
    """
    X_clean = preprocess_text(X_test)
    X_vec = vectorizer.transform(X_clean)
    y_pred = model.predict(X_vec)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_test, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)

    report = classification_report(
        y_test, y_pred, target_names=["non-depression", "depression"], zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    if save_plot_path:
        _plot_confusion_matrix(cm, save_plot_path)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    return metrics, report


def _plot_confusion_matrix(cm: np.ndarray, save_path: str) -> None:
    """Plot confusion matrix as heatmap and save to save_path."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["non-depression", "depression"],
        yticklabels=["non-depression", "depression"],
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
