"""
Training script for the HOI Model V3.

Loss:        λ_now·CE(now) + λ_f1·CE(f1) + λ_f2·CE(f2)
             CE = FocalLoss(gamma=0) by default  →  standard CrossEntropyLoss
             λ_now=1.0,  λ_f1=0.5,  λ_f2=0.5

Scheduler:   Cosine with linear warmup (step-level)
Sampler:     WeightedRandomSampler balanced by label_now class freq
Phase A:     freeze backbone, train all other modules   (epochs 1 → phase_b_epoch)
Phase B:     unfreeze last 2 ViT blocks + norm          (epochs phase_b_epoch+1 → end)
Metrics:     macro F1 (primary), balanced accuracy, commit_recall, hesitate_recall
             confusion matrix printed each epoch
Checkpoint:  saved when macro_f1_now improves
EarlyStopping: on val_loss (min mode), patience=15

Usage:
    python -m hoi_pipeline.train \\
        --data_root ./annotated_dataset_export/Training \\
        --anno_root ./annotated_dataset_export/annotations/Training \\
        --cache_dir ./preprocessed_v3 \\
        --ckpt_dir  ./checkpoints_v3
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from .config import PathConfig, ModelConfig, TrainingConfig, PREDICATES, get_config
from .dataset import HOIChunkDataset, train_val_split
from .model import HOIModelV3
from .utils import (
    seed_everything,
    EMA,
    EarlyStopping,
    FocalLoss,
    compute_class_weights,
    make_balanced_sampler,
    compute_metrics_v3,
    format_metrics_v3,
)


# ════════════════════════════════════════════════════════════════════
# Loss helper
# ════════════════════════════════════════════════════════════════════

def compute_loss(
    outputs: dict,
    batch: dict,
    criterion: nn.Module,
    cfg: TrainingConfig,
    device: torch.device,
) -> tuple:
    """
    Returns (total_loss, loss_dict).

    total = λ_now·L(now) + λ_f1·L(f1) + λ_f2·L(f2)
    """
    label_now = batch["label_now"].to(device)
    label_f1  = batch["label_f1"].to(device)
    label_f2  = batch["label_f2"].to(device)

    l_now = criterion(outputs["logits_now"], label_now)
    l_f1  = criterion(outputs["logits_f1"],  label_f1)
    l_f2  = criterion(outputs["logits_f2"],  label_f2)

    total = (
        cfg.loss_now_weight * l_now
        + cfg.loss_f1_weight  * l_f1
        + cfg.loss_f2_weight  * l_f2
    )
    return total, {
        "loss_now": l_now.item(),
        "loss_f1":  l_f1.item(),
        "loss_f2":  l_f2.item(),
        "loss":     total.item(),
    }


# ════════════════════════════════════════════════════════════════════
# Training epoch
# ════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: HOIModelV3,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    criterion: nn.Module,
    cfg: TrainingConfig,
    device: torch.device,
    epoch: int,
    ema: "EMA | None",
    scheduler: "torch.optim.lr_scheduler._LRScheduler | None",
) -> dict:
    model.train()
    total_stats: dict = {"loss": 0.0, "loss_now": 0.0, "loss_f1": 0.0, "loss_f2": 0.0}
    n_batches = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        # Move visual tensors to device
        gpu_batch = {
            k: v.to(device, non_blocking=True)
            for k, v in batch.items()
        }

        amp_dtype = (
            torch.bfloat16
            if (cfg.use_amp and torch.cuda.is_bf16_supported())
            else torch.float16
        )
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=cfg.use_amp):
            outputs = model(gpu_batch)
            loss, loss_dict = compute_loss(outputs, gpu_batch, criterion, cfg, device)
            loss = loss / cfg.grad_accumulation

        scaler.scale(loss).backward()

        if (batch_idx + 1) % cfg.grad_accumulation == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema is not None:
                ema.update()
            if scheduler is not None:
                scheduler.step()

        for k in total_stats:
            total_stats[k] += loss_dict[k]
        n_batches += 1

        if (batch_idx + 1) % 20 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [{epoch}][{batch_idx+1}/{len(loader)}] "
                f"loss={loss_dict['loss']:.4f} "
                f"(now={loss_dict['loss_now']:.3f} "
                f"f1={loss_dict['loss_f1']:.3f} "
                f"f2={loss_dict['loss_f2']:.3f}) "
                f"lr={lr:.2e}"
            )

    return {k: v / max(n_batches, 1) for k, v in total_stats.items()}


# ════════════════════════════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model: HOIModelV3,
    loader: DataLoader,
    criterion: nn.Module,
    cfg: TrainingConfig,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    preds_now, labels_now = [], []
    preds_f1,  labels_f1  = [], []
    preds_f2,  labels_f2  = [], []

    for batch in loader:
        gpu_batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        outputs = model(gpu_batch)
        _, loss_dict = compute_loss(outputs, gpu_batch, criterion, cfg, device)
        total_loss += loss_dict["loss"]
        n_batches  += 1

        preds_now.append(outputs["logits_now"].argmax(dim=1).cpu().numpy())
        preds_f1.append( outputs["logits_f1"].argmax(dim=1).cpu().numpy())
        preds_f2.append( outputs["logits_f2"].argmax(dim=1).cpu().numpy())

        labels_now.append(gpu_batch["label_now"].cpu().numpy())
        labels_f1.append( gpu_batch["label_f1"].cpu().numpy())
        labels_f2.append( gpu_batch["label_f2"].cpu().numpy())

    preds_now  = np.concatenate(preds_now)
    preds_f1   = np.concatenate(preds_f1)
    preds_f2   = np.concatenate(preds_f2)
    labels_now = np.concatenate(labels_now)
    labels_f1  = np.concatenate(labels_f1)
    labels_f2  = np.concatenate(labels_f2)

    metrics_now = compute_metrics_v3(preds_now, labels_now, PREDICATES)
    metrics_f1  = compute_metrics_v3(preds_f1,  labels_f1,  PREDICATES)
    metrics_f2  = compute_metrics_v3(preds_f2,  labels_f2,  PREDICATES)

    return {
        "loss":        total_loss / max(n_batches, 1),
        "metrics_now": metrics_now,
        "metrics_f1":  metrics_f1,
        "metrics_f2":  metrics_f2,
    }


# ════════════════════════════════════════════════════════════════════
# Main training loop
# ════════════════════════════════════════════════════════════════════

def train(args):
    path_cfg, model_cfg, train_cfg = get_config()

    # CLI overrides
    if args.data_root:
        path_cfg.dataset_root = os.path.dirname(args.data_root.rstrip("/"))
    if args.epochs:
        train_cfg.num_epochs = args.epochs
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.lr:
        train_cfg.learning_rate = args.lr

    data_root = args.data_root  or path_cfg.training_dir
    anno_root = args.anno_root  or path_cfg.annotations_dir
    cache_dir = args.cache_dir  or path_cfg.preprocessed_dir
    ckpt_dir  = args.ckpt_dir   or path_cfg.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    seed_everything(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Train / val split ─────────────────────────────────────────
    train_ids, val_ids = train_val_split(anno_root, train_cfg.val_ratio, train_cfg.seed)
    print(f"Split: {len(train_ids)} train videos, {len(val_ids)} val videos")

    # ── Datasets ──────────────────────────────────────────────────
    common_ds_kwargs = dict(
        anno_root=anno_root,
        preprocessed_dir=cache_dir,
        data_root=data_root,
        chunk_length=train_cfg.chunk_length,
        frame_stride=train_cfg.frame_stride,
        future_1_offset=train_cfg.future_1_offset,
        future_2_offset=train_cfg.future_2_offset,
        image_size=train_cfg.image_size,
        preferred_camera=train_cfg.preferred_camera,
    )
    train_dataset = HOIChunkDataset(
        chunk_stride=train_cfg.chunk_stride,
        video_ids=train_ids,
        augment=True,
        **common_ds_kwargs,
    )
    val_dataset = HOIChunkDataset(
        chunk_stride=train_cfg.chunk_length,   # no overlap for val
        video_ids=val_ids,
        augment=False,
        **common_ds_kwargs,
    )
    print(f"Clips: {len(train_dataset)} train, {len(val_dataset)} val")

    if len(train_dataset) == 0:
        print("ERROR: No training samples found. Check preprocessing and paths.")
        sys.exit(1)

    # ── Balanced sampler (by label_now class freq) ────────────────
    if train_cfg.balanced_sampler:
        sampler = make_balanced_sampler(train_dataset.get_label_now_list())
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────
    model = HOIModelV3(model_cfg).to(device)
    param_info = model.count_parameters()
    print(
        f"Parameters: {param_info['total']:,} total, "
        f"{param_info['trainable']:,} trainable, "
        f"{param_info['backbone_frozen']:,} frozen backbone"
    )
    print(f"Mamba available: {model.temporal.using_mamba}")

    # ── Class weights from training label distribution ─────────────
    train_labels = np.array(train_dataset.get_label_now_list(), dtype=np.int64)
    class_weights = compute_class_weights(train_labels, n_classes=len(PREDICATES))
    print(f"Class weights: {dict(zip(PREDICATES, class_weights.tolist()))}")

    # ── Loss criterion ────────────────────────────────────────────
    if train_cfg.use_focal_loss:
        criterion = FocalLoss(gamma=train_cfg.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # ── Optimizer (Phase A: backbone frozen) ──────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.999),
    )

    # ── LR scheduler: cosine with linear warmup (step-level) ──────
    total_steps  = len(train_loader) * train_cfg.num_epochs
    warmup_steps = len(train_loader) * train_cfg.warmup_epochs

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + float(np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # ── AMP scaler ────────────────────────────────────────────────
    scaler = GradScaler(enabled=train_cfg.use_amp)

    # ── EMA ───────────────────────────────────────────────────────
    ema = EMA(model, train_cfg.ema_decay) if train_cfg.use_ema else None

    # ── Early stopping ────────────────────────────────────────────
    early_stopping = EarlyStopping(patience=train_cfg.patience, min_delta=train_cfg.min_delta, mode="min")

    # ── Training loop ─────────────────────────────────────────────
    best_macro_f1 = -1.0
    history = []
    phase_b_started = False

    print(f"\n{'='*65}")
    print(f"Training V3 — {train_cfg.num_epochs} epochs")
    print(f"Horizons: now, +{train_cfg.future_1_offset/30:.1f}s, +{train_cfg.future_2_offset/30:.1f}s")
    print(f"{'='*65}\n")

    for epoch in range(1, train_cfg.num_epochs + 1):
        t0 = time.time()

        # ── Phase B: partially unfreeze backbone ──────────────────
        if epoch == train_cfg.phase_b_epoch + 1 and not phase_b_started:
            print(f"\n[Phase B] Unfreezing last 2 backbone blocks at epoch {epoch}")
            model.set_backbone_trainable(n_unfreeze=2)

            # Rebuild optimizer with two groups
            backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
            other_params    = [
                p for name, p in model.named_parameters()
                if p.requires_grad and not name.startswith("backbone.")
            ]
            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": train_cfg.phase_b_backbone_lr},
                {"params": other_params,    "lr": train_cfg.phase_b_head_lr},
            ], weight_decay=train_cfg.weight_decay)

            # Register newly unfrozen params with EMA
            if ema is not None:
                for name, p in model.backbone.named_parameters():
                    if p.requires_grad:
                        ema.add_param("backbone." + name, p)

            phase_b_started = True

        # ── Train epoch ───────────────────────────────────────────
        train_stats = train_one_epoch(
            model, train_loader, optimizer, scaler,
            criterion, train_cfg, device, epoch, ema, scheduler,
        )

        # ── Evaluate ──────────────────────────────────────────────
        if ema is not None:
            ema.apply_shadow()
        val_result = evaluate(model, val_loader, criterion, train_cfg, device)
        if ema is not None:
            ema.restore()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # ── Log ───────────────────────────────────────────────────
        m_now = val_result["metrics_now"]
        m_f1  = val_result["metrics_f1"]
        m_f2  = val_result["metrics_f2"]

        print(f"\nEpoch {epoch}/{train_cfg.num_epochs}  ({elapsed:.0f}s  lr={lr:.2e})")
        print(f"  Train: loss={train_stats['loss']:.4f}  "
              f"(now={train_stats['loss_now']:.3f} "
              f"f1={train_stats['loss_f1']:.3f} "
              f"f2={train_stats['loss_f2']:.3f})")
        print(f"  Val:   loss={val_result['loss']:.4f}")
        print(f"  Val [now]  macro_f1={m_now['macro_f1']:.4f}  "
              f"bal_acc={m_now['balanced_acc']:.4f}  "
              f"commit_recall={m_now.get('commit_recall',0):.3f}  "
              f"hesitate_recall={m_now.get('hesitate_recall',0):.3f}")
        print(f"  Val [+{train_cfg.future_1_offset/30:.1f}s] macro_f1={m_f1['macro_f1']:.4f}")
        print(f"  Val [+{train_cfg.future_2_offset/30:.1f}s] macro_f1={m_f2['macro_f1']:.4f}")

        # Confusion matrix (now head)
        cm = m_now.get("confusion_matrix")
        if cm is not None:
            print("  Confusion (now, rows=true, cols=pred):")
            for row in cm:
                print("   ", "  ".join(f"{v:.2f}" for v in row))

        # ── Save best checkpoint ──────────────────────────────────
        macro_f1_now = m_now["macro_f1"]
        if macro_f1_now > best_macro_f1:
            best_macro_f1 = macro_f1_now
            ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
            state = {
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":         val_result["loss"],
                "macro_f1_now":     macro_f1_now,
                "metrics_now":      {k: v.tolist() if hasattr(v, "tolist") else v
                                     for k, v in m_now.items()},
                "model_config":     model_cfg.__dict__,
                "train_config":     train_cfg.__dict__,
            }
            if ema is not None:
                state["ema_shadow"] = ema.shadow
            torch.save(state, ckpt_path)
            print(f"  ✓ Saved best (macro_f1_now={macro_f1_now:.4f})")

        # ── History ───────────────────────────────────────────────
        history.append({
            "epoch":          epoch,
            "train_loss":     train_stats["loss"],
            "val_loss":       val_result["loss"],
            "macro_f1_now":   m_now["macro_f1"],
            "macro_f1_f1":    m_f1["macro_f1"],
            "macro_f1_f2":    m_f2["macro_f1"],
            "balanced_acc":   m_now["balanced_acc"],
            "commit_recall":  m_now.get("commit_recall", 0.0),
            "hesitate_recall": m_now.get("hesitate_recall", 0.0),
            "lr":             lr,
            "phase_b":        phase_b_started,
        })

        # ── Early stopping ────────────────────────────────────────
        if early_stopping(val_result["loss"]):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # ── Save history ──────────────────────────────────────────────
    hist_path = os.path.join(ckpt_dir, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*65}")
    print(f"Done. Best macro_f1_now: {best_macro_f1:.4f}")
    print(f"Checkpoint: {os.path.join(ckpt_dir, 'best_model.pt')}")
    print(f"History:    {hist_path}")
    print(f"{'='*65}")


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train HOI Model V3")
    parser.add_argument("--data_root",  type=str, default=None,
                        help="Path to Training/ directory")
    parser.add_argument("--anno_root",  type=str, default=None,
                        help="Path to annotations/Training/ directory")
    parser.add_argument("--cache_dir",  type=str, default=None,
                        help="Path to preprocessed_v3/ directory")
    parser.add_argument("--ckpt_dir",   type=str, default=None,
                        help="Checkpoint output directory")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
