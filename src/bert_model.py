"""
Deep learning model: fine-tune BERT (bert-base-uncased) for binary sequence classification.
Uses HuggingFace transformers, AutoTokenizer, AdamW, and 2-3 epochs.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from .utils import RANDOM_STATE, set_seed

# Model and training defaults
BERT_MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5


class TextClassificationDataset(Dataset):
    """PyTorch Dataset yielding input_ids, attention_mask, and labels for BERT."""

    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int = MAX_LENGTH,
    ):
        self.texts = list(texts)
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx] if isinstance(self.texts[idx], str) else str(self.texts[idx])
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_bert(
    X_train: pd.Series,
    y_train: np.ndarray,
    X_test: pd.Series,
    y_test: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    max_length: int = MAX_LENGTH,
    random_state: int = None,
) -> Tuple[Any, Any]:
    """
    Load BERT tokenizer and model, create datasets, and fine-tune for 2-3 epochs with AdamW.

    Args:
        X_train, y_train: Training text and labels.
        X_test, y_test: Test text and labels (used only for evaluation after training).
        epochs: Number of training epochs (default 3).
        batch_size: Batch size (default 16).
        learning_rate: AdamW learning rate (default 2e-5).
        max_length: Max token length for tokenizer.
        random_state: Seed for reproducibility.

    Returns:
        Tuple (fitted model, tokenizer). Model is on CPU for easy evaluation.
    """
    set_seed(random_state or RANDOM_STATE)
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=2,
    )
    model.to(device)

    train_dataset = TextClassificationDataset(X_train.values, y_train, tokenizer, max_length)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch + 1}/{epochs}, avg loss: {avg_loss:.4f}")

    model.to("cpu")
    return model, tokenizer


def evaluate_bert(
    model,
    tokenizer,
    X_test: pd.Series,
    y_test: np.ndarray,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    save_plot_path: str = None,
) -> Tuple[Dict[str, float], str]:
    """
    Evaluate the BERT model on the test set: accuracy, precision, recall, F1, classification report.

    Args:
        model: Fine-tuned BERT model (will be moved to CPU if needed).
        tokenizer: BERT tokenizer.
        X_test: Test text.
        y_test: Test labels.
        batch_size: Batch size for inference.
        max_length: Max token length.

    Returns:
        Tuple (metrics_dict, classification_report_str).
    """
    model.eval()
    device = next(model.parameters()).device
    if device.type != "cpu":
        model.to("cpu")
        device = torch.device("cpu")

    test_dataset = TextClassificationDataset(X_test.values, y_test, tokenizer, max_length)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())

    y_pred = np.array(all_preds)

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
