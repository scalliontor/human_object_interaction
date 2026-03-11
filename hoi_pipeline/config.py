"""
Configuration for the HOI Training Pipeline V2.
All hyperparameters, paths, and constants are defined here.

Architecture V2:
  Stream 1: Body Pose → GCN (YOLOv26 Pose, 17 COCO joints + velocity)
  Stream 2: Object Crop 224×224 → DINOv3 ViT-S/16 (custom YOLO object bbox)
  Stream 3: Context Full Frame 224×224 → DINOv3 ViT-S/16 (shared backbone)
  Stream 4: Spatial Features 12-dim → Spatial Bottleneck (person-object relative)
  → Pose-Query Cross-Attention Fusion
  → Temporal Attention Encoder (3 layers)
  → Classification Head + Anticipation Head
  → Multi-Task Loss
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict


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
        return os.path.join(self.dataset_root, "preprocessed_v2")

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.dataset_root, "checkpoints_v2")


@dataclass
class ModelConfig:
    """Model architecture configuration — V2 with YOLOv26 Pose + custom YOLO OD."""
    # ── DINOv3 ViT-S/16 backbone ──
    backbone_name: str = "vit_small_patch16_dinov3"
    backbone_frozen: bool = True
    backbone_dim: int = 384

    # ── Body Pose (Stream 1: YOLOv26 Pose 17 COCO keypoints) ──
    num_joints: int = 17
    joint_dim: int = 3                  # x, y, confidence per joint
    use_velocity: bool = True
    input_dim: int = 6                  # position (3) + velocity (3)

    # ── Model dimensions ──
    d_model: int = 384
    n_heads: int = 6
    n_cross_heads: int = 8
    n_temporal_layers: int = 3
    ffn_dim: int = 1536
    dropout: float = 0.3
    spatial_dropout: float = 0.3

    # ── Spatial features (Stream 4: person-object relative) ──
    spatial_dim: int = 12
    spatial_hidden: int = 64

    # ── GCN (Stream 1) ──
    gcn_layers: int = 2

    # ── Output ──
    num_predicates: int = 4             # commit, hesitate, abort, waiting


@dataclass
class TrainingConfig:
    """Training configuration."""
    # ── Temporal chunking ──
    chunk_length: int = 8
    chunk_stride: int = 4
    frame_stride: int = 8
    anticipation_offset: int = 30

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
    image_size: int = 224

    # ── Misc ──
    num_workers: int = 4
    seed: int = 42
    val_ratio: float = 0.2
    use_amp: bool = True
    preferred_camera: str = "cam_832112070255"

    # ── YOLOv26 Pose ──
    yolo_pose_model: str = "yolo26x-pose.pt"
    yolo_pose_conf: float = 0.3

    # ── Custom YOLO Object Detection ──
    yolo_obj_model: str = "best.pt"     # Custom trained: bottle, cake box, plastic bottle
    yolo_obj_conf: float = 0.3


# Predicate names matching annotation format
PREDICATES = ["commit", "hesitate", "abort", "waiting"]
PREDICATE_TO_IDX = {p: i for i, p in enumerate(PREDICATES)}

# Object class names from custom YOLO model
OBJECT_CLASSES = {0: "bottle", 1: "cake box", 2: "plastic bottle"}

# COCO Body Skeleton — 17 keypoints adjacency for GCN
# 0:nose  1:L_eye  2:R_eye  3:L_ear  4:R_ear  5:L_shoulder  6:R_shoulder
# 7:L_elbow  8:R_elbow  9:L_wrist  10:R_wrist  11:L_hip  12:R_hip
# 13:L_knee  14:R_knee  15:L_ankle  16:R_ankle
BODY_SKELETON_EDGES = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Left arm
    (5, 7), (7, 9),
    # Right arm
    (6, 8), (8, 10),
    # Left leg
    (11, 13), (13, 15),
    # Right leg
    (12, 14), (14, 16),
    # Shoulder-nose
    (0, 5), (0, 6),
]


def get_config():
    """Get default configuration."""
    return PathConfig(), ModelConfig(), TrainingConfig()
