"""
Utility functions for the HOI Training Pipeline V3.

New in V3:
  - FocalLoss (multi-class, gamma=0 → standard CE)
  - compute_metrics_v3: macro F1, balanced accuracy, confusion matrix
  - make_balanced_sampler: WeightedRandomSampler by label_now class freq
  - ConfusionMatrixLogger: pretty-print + optional TensorBoard logging

Kept from V2:
  - seed_everything, EMA, EarlyStopping
"""
import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    average_precision_score,
)


# ════════════════════════════════════════════════════════════════════
# Reproducibility
# ════════════════════════════════════════════════════════════════════

def seed_everything(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ════════════════════════════════════════════════════════════════════
# Focal Loss  (multi-class, single-label)
# ════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Multi-class focal loss.
      FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    When gamma=0 this reduces exactly to standard CrossEntropyLoss.
    Use gamma=2.0 for standard focal weighting.

    Args:
        gamma:     focusing parameter (0 = standard CE)
        weight:    per-class weight tensor [C] — same as CE weight
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("weight", weight)  # may be None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C]  raw (un-softmaxed) logits
            targets: [B]     class indices (int64)
        Returns:
            scalar loss
        """
        # Standard CE per sample: [B]
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            reduction="none",
        )
        if self.gamma == 0.0:
            loss = ce
        else:
            p_t = torch.exp(-ce)                        # probability of correct class
            loss = (1.0 - p_t) ** self.gamma * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ════════════════════════════════════════════════════════════════════
# Class-weight computation  (for CE / FocalLoss)
# ════════════════════════════════════════════════════════════════════

def compute_class_weights(labels: np.ndarray, n_classes: int, clip: float = 10.0) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from an array of integer labels.

    weight[c] = N / (n_classes * count[c]),  clipped to [1/clip, clip].

    Args:
        labels:    [N] integer class indices
        n_classes: number of classes
        clip:      maximum weight value
    Returns:
        FloatTensor [n_classes]
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1)          # avoid division by zero
    weights = len(labels) / (n_classes * counts)
    weights = np.clip(weights, 1.0 / clip, clip)
    return torch.tensor(weights, dtype=torch.float32)


# ════════════════════════════════════════════════════════════════════
# Balanced sampler
# ════════════════════════════════════════════════════════════════════

def make_balanced_sampler(label_now_list: List[int]) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that yields each class approximately equally.

    Args:
        label_now_list: list of int class indices, one per dataset sample
    Returns:
        WeightedRandomSampler with replacement=True
    """
    labels = np.array(label_now_list, dtype=np.int64)
    n_classes = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1)
    class_weights = 1.0 / counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(labels),
        replacement=True,
    )


# ════════════════════════════════════════════════════════════════════
# Metrics V3  (single-label, 4-class CE)
# ════════════════════════════════════════════════════════════════════

def compute_metrics_v3(
    all_preds: np.ndarray,       # [N] int predicted class indices
    all_labels: np.ndarray,      # [N] int ground-truth class indices
    class_names: List[str],
) -> Dict:
    """
    Compute classification metrics for a single-label 4-class problem.

    Returns dict with:
      macro_f1, balanced_acc,
      per_class_f1 (dict name→f1),
      commit_recall, hesitate_recall,
      confusion_matrix (numpy [C,C] normalized by row = recall)
    """
    n_classes = len(class_names)
    labels_range = list(range(n_classes))

    macro_f1 = float(f1_score(all_labels, all_preds, average="macro",
                               labels=labels_range, zero_division=0))
    bal_acc   = float(balanced_accuracy_score(all_labels, all_preds))

    per_f1 = f1_score(all_labels, all_preds, average=None,
                      labels=labels_range, zero_division=0)
    per_recall = precision_recall_fscore_support(
        all_labels, all_preds, average=None,
        labels=labels_range, zero_division=0,
    )[1]

    cm_raw = confusion_matrix(all_labels, all_preds, labels=labels_range)
    row_sums = cm_raw.sum(axis=1, keepdims=True).astype(np.float64)
    row_sums = np.maximum(row_sums, 1)
    cm_norm = cm_raw / row_sums       # row-normalized = per-class recall matrix

    metrics = {
        "macro_f1":    macro_f1,
        "balanced_acc": bal_acc,
        "confusion_matrix": cm_norm,
    }

    for i, name in enumerate(class_names):
        metrics[f"{name}_f1"]     = float(per_f1[i])
        metrics[f"{name}_recall"] = float(per_recall[i])

    # Robot-critical highlights
    commit_idx  = class_names.index("commit")  if "commit"  in class_names else -1
    hesit_idx   = class_names.index("hesitate") if "hesitate" in class_names else -1
    if commit_idx >= 0:
        metrics["commit_recall"]   = float(per_recall[commit_idx])
    if hesit_idx >= 0:
        metrics["hesitate_recall"] = float(per_recall[hesit_idx])

    return metrics


def format_metrics_v3(metrics: Dict, prefix: str = "") -> str:
    """Format V3 metrics dict into a readable multi-line string."""
    lines = []
    skip = {"confusion_matrix"}
    for k, v in sorted(metrics.items()):
        if k in skip:
            continue
        if isinstance(v, float):
            lines.append(f"{prefix}{k}: {v:.4f}")
        else:
            lines.append(f"{prefix}{k}: {v}")

    # Pretty confusion matrix
    cm = metrics.get("confusion_matrix")
    if cm is not None:
        lines.append(f"{prefix}confusion_matrix (row=true, col=pred, values=recall):")
        for row in cm:
            lines.append(f"{prefix}  " + "  ".join(f"{v:.2f}" for v in row))

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════
# EMA
# ════════════════════════════════════════════════════════════════════

class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self):
        """Swap in EMA weights (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore training weights (after evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def add_param(self, name: str, param: nn.Parameter):
        """Register a newly unfrozen parameter (Phase B backbone)."""
        if param.requires_grad and name not in self.shadow:
            self.shadow[name] = param.data.clone()


# ════════════════════════════════════════════════════════════════════
# Early stopping
# ════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "min"):
        """
        Args:
            mode: 'min' (track loss) or 'max' (track metric like macro_f1)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best: Optional[float] = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        if self.best is None:
            self.best = value
        elif self._is_better(value):
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def _is_better(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta


# ════════════════════════════════════════════════════════════════════
# Legacy V2 metrics  (kept for backward compat with old checkpoints)
# ════════════════════════════════════════════════════════════════════

def compute_metrics(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    predicate_names: List[str],
    threshold: float = 0.5,
) -> Dict:
    """V2 multi-label metrics. Kept for reference only."""
    binary_preds = (all_preds >= threshold).astype(int)
    metrics = {}
    for i, name in enumerate(predicate_names):
        p, r, f1, _ = precision_recall_fscore_support(
            all_labels[:, i], binary_preds[:, i],
            average="binary", zero_division=0,
        )
        try:
            ap = average_precision_score(all_labels[:, i], all_preds[:, i])
        except ValueError:
            ap = 0.0
        metrics[f"{name}_precision"] = p
        metrics[f"{name}_recall"]    = r
        metrics[f"{name}_f1"]        = f1
        metrics[f"{name}_ap"]        = ap
    metrics["mAP"]      = float(np.mean([metrics[f"{n}_ap"] for n in predicate_names]))
    metrics["macro_f1"] = float(np.mean([metrics[f"{n}_f1"] for n in predicate_names]))
    return metrics


def format_metrics(metrics: Dict, prefix: str = "") -> str:
    lines = []
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            lines.append(f"  {prefix}{k}: {v:.4f}")
        else:
            lines.append(f"  {prefix}{k}: {v}")
    return "\n".join(lines)
