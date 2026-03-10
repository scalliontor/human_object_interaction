"""
Configuration for the HOI Training Pipeline.
All hyperparameters, paths, and constants are defined here.

Architecture follows the diagram exactly:
  Stream 1: Hand Pose → GCN (MediaPipe right hand, 21 joints + velocity)
  Stream 2: Object Crop 224×224 → DINOv3 ViT-S/16 (shared backbone)
  Stream 3: Context Full Frame 224×224 → DINOv3 ViT-S/16 (shared backbone)
  Stream 4: Spatial Features 7-dim → Spatial Bottleneck
  → Pose-Query Cross-Attention Fusion
  → Temporal Attention Encoder (3 layers)
  → Classification Head + Anticipation Head
  → Multi-Task Loss
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class PathConfig:
    """Paths configuration."""
    dataset_root: str = "/media/hung/30fcade7-362c-4b72-9673-ca95aa0e6a9d/Dataset"

    @property
    def training_dir(self) -> str:
        return os.path.join(self.dataset_root, "Training")

    @property
    def annotations_dir(self) -> str:
        return os.path.join(self.dataset_root, "annotations", "Training")

    @property
    def preprocessed_dir(self) -> str:
        return os.path.join(self.dataset_root, "preprocessed")

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.dataset_root, "checkpoints")


@dataclass
class ModelConfig:
    """Model architecture configuration — matches diagram exactly."""
    # ── DINOv3 ViT-S/16 backbone ──
    backbone_name: str = "vit_small_patch16_dinov3"
    backbone_frozen: bool = True        # Freeze backbone, use as feature extractor
    backbone_dim: int = 384             # ViT-S embedding dim
    num_register_tokens: int = 4        # DINOv3 register tokens
    num_patch_tokens: int = 196         # 224/16 = 14, 14×14 = 196 patches
    # CLS token output = 384-dim

    # ── Hand landmarks (Stream 1: replaces YOLO Pose) ──
    num_landmarks: int = 21             # MediaPipe right hand
    landmark_dim: int = 3               # x, y, z per landmark
    use_velocity: bool = True           # Add velocity features
    input_dim: int = 6                  # position (3) + velocity (3)

    # ── Model dimensions ──
    d_model: int = 384                  # Matches backbone_dim
    n_heads: int = 6                    # Temporal attention heads
    n_cross_heads: int = 8             # Cross-attention fusion heads
    n_temporal_layers: int = 3          # Temporal attention encoder layers
    ffn_dim: int = 1536                 # 4× d_model
    dropout: float = 0.3               # Dropout (higher for small dataset)
    spatial_dropout: float = 0.3        # 30% randomly zeros spatial features

    # ── Spatial features (Stream 4) ──
    spatial_dim: int = 7                # dx, dy, log(w_ratio), log(h_ratio), IoU, distance, angle
    spatial_hidden: int = 64            # Bottleneck hidden dim

    # ── GCN (Stream 1) ──
    gcn_layers: int = 2

    # ── Output ──
    num_predicates: int = 4             # commit, hesitate, abort, waiting


@dataclass
class TrainingConfig:
    """Training configuration."""
    # ── Temporal chunking ──
    chunk_length: int = 8               # T=8 frames per chunk
    chunk_stride: int = 4               # Sliding window stride
    frame_stride: int = 8              # Sample every 8th frame (8 × 8 = 64 frames ≈ 2.1s @30fps)
    anticipation_offset: int = 30       # 1 second ahead at 30fps

    # ── Training ──
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    grad_accumulation: int = 2

    # ── Loss ──
    cls_loss_weight: float = 1.0
    antic_loss_weight: float = 0.3
    label_smoothing: float = 0.05

    # ── EMA ──
    ema_decay: float = 0.999
    use_ema: bool = True

    # ── Early stopping ──
    patience: int = 15
    min_delta: float = 1e-4

    # ── Image preprocessing ──
    image_size: int = 224               # Input size for DINOv3

    # ── Misc ──
    num_workers: int = 4
    seed: int = 42
    val_ratio: float = 0.2              # 80/20 train/val split
    use_amp: bool = True
    preferred_camera: str = "cam_832112070255"


# Predicate names matching annotation format
PREDICATES = ["commit", "hesitate", "abort", "waiting"]
PREDICATE_TO_IDX = {p: i for i, p in enumerate(PREDICATES)}

# MediaPipe Hand Skeleton — adjacency list for GCN
# Wrist (0) is the hub connecting to each finger base
HAND_SKELETON_EDGES = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Inter-finger connections (palm)
    (5, 9), (9, 13), (13, 17),
]


def get_config():
    """Get default configuration."""
    return PathConfig(), ModelConfig(), TrainingConfig()
