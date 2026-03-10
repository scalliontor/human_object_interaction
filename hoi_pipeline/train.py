"""
Training script for the HOI Model.

Multi-Task Loss: BCE(cls, weight=1.0) + 0.3 × BCE(anticipation)
+ Label Smoothing (ε=0.05, train only)
+ Cosine LR + Warmup (5 epochs)
+ EMA (decay=0.999)
+ Grad Accumulation
+ AMP bfloat16

Usage:
    python -m hoi_pipeline.train \\
        --data_root ./Training \\
        --anno_root ./annotations/Training \\
        --cache_dir ./preprocessed \\
        --epochs 100 --batch_size 8
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
from torch.cuda.amp import autocast, GradScaler

from .config import PathConfig, ModelConfig, TrainingConfig, PREDICATES, get_config
from .dataset import HOIChunkDataset, train_val_split
from .model import HOIModel
from .utils import (
    seed_everything,
    EMA,
    EarlyStopping,
    compute_metrics,
    format_metrics,
)


def smooth_labels(labels: torch.Tensor, smoothing: float = 0.05) -> torch.Tensor:
    """Apply label smoothing to binary labels."""
    return labels * (1 - smoothing) + 0.5 * smoothing


def compute_pos_weight(dataset, device) -> torch.Tensor:
    """Compute pos_weight from actual chunk labels for class-balanced BCE.
    pos_weight[c] = num_negative[c] / num_positive[c]
    Higher weight = model penalized more for missing rare classes.
    """
    import numpy as np
    all_labels = []
    for i in range(len(dataset)):
        sample = dataset.samples[i]
        frame_indices = sample["frame_indices"]
        labels = sample["frame_labels"][frame_indices]
        chunk_cls = (labels.sum(axis=0) > 0).astype(np.float32)
        all_labels.append(chunk_cls)
    all_labels = np.stack(all_labels)  # [N, C]
    pos_count = all_labels.sum(axis=0) + 1  # avoid div/0
    neg_count = len(all_labels) - pos_count + 1
    weights = neg_count / pos_count
    # Clip extremes
    weights = np.clip(weights, 0.5, 10.0)
    print(f"  pos_weight: {dict(zip(PREDICATES, weights.tolist()))}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    cfg: TrainingConfig,
    device: torch.device,
    epoch: int,
    ema: EMA = None,
    pos_weight: torch.Tensor = None,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_antic_loss = 0.0
    n_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        # Move to device
        hand_features = batch["hand_features"].to(device)
        hand_confidence = batch["hand_confidence"].to(device)
        obj_crops = batch["obj_crops"].to(device)
        ctx_frames = batch["ctx_frames"].to(device)
        obj_visible = batch["obj_visible"].to(device)
        spatial_features = batch["spatial_features"].to(device)
        cls_labels = batch["cls_labels"].to(device)
        antic_labels = batch["antic_labels"].to(device)

        # Label smoothing (train only)
        cls_targets = smooth_labels(cls_labels, cfg.label_smoothing)
        antic_targets = smooth_labels(antic_labels, cfg.label_smoothing)

        # Forward with AMP
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with autocast(dtype=amp_dtype, enabled=cfg.use_amp):
            cls_logits, antic_logits = model(
                hand_features, hand_confidence,
                obj_crops, ctx_frames,
                obj_visible, spatial_features,
            )

            # Multi-task loss (class-weighted BCE)
            cls_loss = nn.functional.binary_cross_entropy_with_logits(
                cls_logits, cls_targets, pos_weight=pos_weight
            )
            antic_loss = nn.functional.binary_cross_entropy_with_logits(
                antic_logits, antic_targets, pos_weight=pos_weight
            )
            loss = cfg.cls_loss_weight * cls_loss + cfg.antic_loss_weight * antic_loss
            loss = loss / cfg.grad_accumulation

        # Backward with AMP scaling
        scaler.scale(loss).backward()

        # Gradient accumulation step
        if (batch_idx + 1) % cfg.grad_accumulation == 0 or (batch_idx + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if ema is not None:
                ema.update()

        total_loss += loss.item() * cfg.grad_accumulation
        total_cls_loss += cls_loss.item()
        total_antic_loss += antic_loss.item()
        n_batches += 1

        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                  f"loss={avg_loss:.4f} cls={cls_loss.item():.4f} antic={antic_loss.item():.4f}")

    return {
        "loss": total_loss / max(n_batches, 1),
        "cls_loss": total_cls_loss / max(n_batches, 1),
        "antic_loss": total_antic_loss / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    cfg: TrainingConfig,
    device: torch.device,
    pos_weight: torch.Tensor = None,
) -> dict:
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_cls_preds = []
    all_cls_labels = []
    all_antic_preds = []
    all_antic_labels = []

    for batch in loader:
        hand_features = batch["hand_features"].to(device)
        hand_confidence = batch["hand_confidence"].to(device)
        obj_crops = batch["obj_crops"].to(device)
        ctx_frames = batch["ctx_frames"].to(device)
        obj_visible = batch["obj_visible"].to(device)
        spatial_features = batch["spatial_features"].to(device)
        cls_labels = batch["cls_labels"].to(device)
        antic_labels = batch["antic_labels"].to(device)

        cls_logits, antic_logits = model(
            hand_features, hand_confidence,
            obj_crops, ctx_frames,
            obj_visible, spatial_features,
        )

        cls_loss = nn.functional.binary_cross_entropy_with_logits(
            cls_logits, cls_labels, pos_weight=pos_weight
        )
        antic_loss = nn.functional.binary_cross_entropy_with_logits(
            antic_logits, antic_labels, pos_weight=pos_weight
        )
        loss = cfg.cls_loss_weight * cls_loss + cfg.antic_loss_weight * antic_loss

        total_loss += loss.item()
        n_batches += 1

        all_cls_preds.append(torch.sigmoid(cls_logits).cpu().numpy())
        all_cls_labels.append(cls_labels.cpu().numpy())
        all_antic_preds.append(torch.sigmoid(antic_logits).cpu().numpy())
        all_antic_labels.append(antic_labels.cpu().numpy())

    # Compute metrics
    cls_preds = np.concatenate(all_cls_preds)
    cls_labels_np = np.concatenate(all_cls_labels)
    antic_preds = np.concatenate(all_antic_preds)
    antic_labels_np = np.concatenate(all_antic_labels)

    cls_metrics = compute_metrics(cls_preds, cls_labels_np, PREDICATES)
    antic_metrics = compute_metrics(antic_preds, antic_labels_np, PREDICATES)

    return {
        "loss": total_loss / max(n_batches, 1),
        "cls_metrics": cls_metrics,
        "antic_metrics": antic_metrics,
    }


def train(args):
    """Main training loop."""
    path_cfg, model_cfg, train_cfg = get_config()

    # Override from args
    if args.data_root:
        path_cfg.dataset_root = os.path.dirname(args.data_root)
    if args.epochs:
        train_cfg.num_epochs = args.epochs
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.lr:
        train_cfg.learning_rate = args.lr

    data_root = args.data_root or path_cfg.training_dir
    anno_root = args.anno_root or path_cfg.annotations_dir
    cache_dir = args.cache_dir or path_cfg.preprocessed_dir
    ckpt_dir = args.ckpt_dir or path_cfg.checkpoint_dir

    os.makedirs(ckpt_dir, exist_ok=True)

    # Seed
    seed_everything(train_cfg.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Train/val split
    train_ids, val_ids = train_val_split(anno_root, train_cfg.val_ratio, train_cfg.seed)
    print(f"Train: {len(train_ids)} videos, Val: {len(val_ids)} videos")

    # Datasets
    train_dataset = HOIChunkDataset(
        anno_root=anno_root,
        preprocessed_dir=cache_dir,
        data_root=data_root,
        chunk_length=train_cfg.chunk_length,
        chunk_stride=train_cfg.chunk_stride,
        frame_stride=train_cfg.frame_stride,
        anticipation_offset=train_cfg.anticipation_offset,
        image_size=train_cfg.image_size,
        preferred_camera=train_cfg.preferred_camera,
        video_ids=train_ids,
        augment=True,
    )
    val_dataset = HOIChunkDataset(
        anno_root=anno_root,
        preprocessed_dir=cache_dir,
        data_root=data_root,
        chunk_length=train_cfg.chunk_length,
        chunk_stride=train_cfg.chunk_length,  # No overlap for val
        frame_stride=train_cfg.frame_stride,
        anticipation_offset=train_cfg.anticipation_offset,
        image_size=train_cfg.image_size,
        preferred_camera=train_cfg.preferred_camera,
        video_ids=val_ids,
        augment=False,
    )

    print(f"Train chunks: {len(train_dataset)}, Val chunks: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("ERROR: No training samples! Check preprocessing and annotations.")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
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

    # Model
    model = HOIModel(model_cfg).to(device)
    param_info = model.count_parameters()
    print(f"Model parameters: {param_info['total']:,} total, "
          f"{param_info['trainable']:,} trainable, "
          f"{param_info['backbone_frozen']:,} frozen backbone")

    # Optimizer — only trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    # Scheduler: Cosine with linear warmup
    total_steps = len(train_loader) * train_cfg.num_epochs
    warmup_steps = len(train_loader) * train_cfg.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler
    scaler = GradScaler(enabled=train_cfg.use_amp)

    # EMA
    ema = EMA(model, train_cfg.ema_decay) if train_cfg.use_ema else None

    # Early stopping
    early_stopping = EarlyStopping(
        patience=train_cfg.patience,
        min_delta=train_cfg.min_delta,
    )

    # Training loop
    best_val_loss = float("inf")
    history = []

    print(f"\n{'='*60}")
    print(f"Starting training: {train_cfg.num_epochs} epochs")
    print(f"{'='*60}\n")

    # Compute class weights from training data
    pos_weight = compute_pos_weight(train_dataset, device)

    for epoch in range(1, train_cfg.num_epochs + 1):
        start_time = time.time()

        # Train
        train_result = train_one_epoch(
            model, train_loader, optimizer, scaler,
            train_cfg, device, epoch, ema, pos_weight,
        )

        # Step scheduler per epoch
        scheduler.step()

        # Evaluate
        if ema is not None:
            ema.apply_shadow()

        val_result = evaluate(model, val_loader, train_cfg, device, pos_weight)

        if ema is not None:
            ema.restore()

        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]["lr"]

        # Log
        print(f"\nEpoch {epoch}/{train_cfg.num_epochs} ({elapsed:.1f}s, lr={lr:.6f})")
        print(f"  Train: loss={train_result['loss']:.4f} "
              f"(cls={train_result['cls_loss']:.4f}, antic={train_result['antic_loss']:.4f})")
        print(f"  Val:   loss={val_result['loss']:.4f}")
        print(f"  Val Classification:")
        print(format_metrics(val_result["cls_metrics"], prefix="    "))
        print(f"  Val Anticipation:")
        print(format_metrics(val_result["antic_metrics"], prefix="    "))

        # Save history
        history.append({
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "val_loss": val_result["loss"],
            "val_cls_mAP": val_result["cls_metrics"].get("mAP", 0),
            "val_antic_mAP": val_result["antic_metrics"].get("mAP", 0),
            "lr": lr,
        })

        # Save best model
        val_loss = val_result["loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_cls_metrics": val_result["cls_metrics"],
                "val_antic_metrics": val_result["antic_metrics"],
                "model_config": model_cfg.__dict__,
                "train_config": train_cfg.__dict__,
            }
            if ema is not None:
                state["ema_shadow"] = ema.shadow
            torch.save(state, ckpt_path)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping at epoch {epoch} (patience={train_cfg.patience})")
            break

    # Save training history
    history_path = os.path.join(ckpt_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {os.path.join(ckpt_dir, 'best_model.pt')}")
    print(f"History: {history_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Train HOI Model")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to Training/ directory")
    parser.add_argument("--anno_root", type=str, default=None,
                        help="Path to annotations/Training/ directory")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Path to preprocessed/ directory")
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="Checkpoint output directory")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
