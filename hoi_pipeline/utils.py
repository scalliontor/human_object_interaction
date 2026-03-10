"""
Utility functions for the HOI Training Pipeline.
"""
import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    average_precision_score,
)


def seed_everything(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_avg

    def apply_shadow(self):
        """Apply EMA weights (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights (after evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


def compute_metrics(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    predicate_names: List[str],
    threshold: float = 0.5,
) -> Dict:
    """
    Compute classification metrics.

    Args:
        all_preds: [N, C] — predicted probabilities
        all_labels: [N, C] — ground truth binary labels
        predicate_names: List of predicate names
        threshold: Classification threshold

    Returns:
        Dict with accuracy, per-class P/R/F1, mAP
    """
    binary_preds = (all_preds >= threshold).astype(int)
    metrics = {}

    # Per-class metrics
    for i, name in enumerate(predicate_names):
        p, r, f1, _ = precision_recall_fscore_support(
            all_labels[:, i], binary_preds[:, i], average="binary", zero_division=0
        )
        try:
            ap = average_precision_score(all_labels[:, i], all_preds[:, i])
        except ValueError:
            ap = 0.0
        metrics[f"{name}_precision"] = p
        metrics[f"{name}_recall"] = r
        metrics[f"{name}_f1"] = f1
        metrics[f"{name}_ap"] = ap

    # Overall metrics
    metrics["mAP"] = np.mean([metrics[f"{n}_ap"] for n in predicate_names])
    metrics["macro_f1"] = np.mean([metrics[f"{n}_f1"] for n in predicate_names])

    # Sample-level accuracy (at least one correct prediction per sample)
    correct = 0
    for i in range(len(all_labels)):
        gt_set = set(np.where(all_labels[i] > 0.5)[0])
        pred_set = set(np.where(binary_preds[i] > 0.5)[0])
        if not gt_set and not pred_set:
            correct += 1
        elif gt_set == pred_set:
            correct += 1
    metrics["exact_match_acc"] = correct / max(len(all_labels), 1)

    return metrics


def format_metrics(metrics: Dict, prefix: str = "") -> str:
    """Format metrics dict into a readable string."""
    lines = []
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            lines.append(f"  {prefix}{k}: {v:.4f}")
        else:
            lines.append(f"  {prefix}{k}: {v}")
    return "\n".join(lines)
