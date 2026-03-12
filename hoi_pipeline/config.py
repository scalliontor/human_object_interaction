"""
Configuration for the HOI Training Pipeline V3.

Architecture V3:
  Stream 1: Upper-body Pose (9 joints × 5-dim) → MLP → 256
  Stream 2: Full Frame → DINOv3 ViT-S/16 → PatchROIPool by bbox → 256
  Stream 3: Relation/Kinematic (20-dim) → MLP → 256
  → Pose-Query Cross-Attention Fusion (Q=pose, KV=[obj, rel])
  → Temporal Mamba (2 blocks, d_model=256)
  → Head_now + Head_f1 + Head_f2  (3 × 4-class CE)
  → Rule-based Policy Trigger

Key changes from V2:
  - Dropped full-body GCN → upper-body MLP (9 joints, simpler + better for data size)
  - Dropped dual DINOv3 (obj_crop + ctx_frame) → single-pass + ROI pool
  - Dropped multi-label BCE → single-label CE (multi-label rate = 0% in dataset)
  - Dropped 1 anticipation head → 3 heads (now, +0.5s, +1.0s)
  - d_model: 384 → 256, dropout: 0.3 → 0.1, Transformer → Mamba
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict


# ════════════════════════════════════════════════════════════════════
# Label schema
# ════════════════════════════════════════════════════════════════════

PREDICATES = ["waiting", "abort", "hesitate", "commit"]
PREDICATE_TO_IDX = {p: i for i, p in enumerate(PREDICATES)}

# Priority for resolving single label when frame has multiple annotations
# (dataset shows 0% multi-predicate frames, but kept for safety)
# commit(3) > hesitate(2) > abort(1) > waiting(0)
LABEL_PRIORITY = {"commit": 3, "hesitate": 2, "abort": 1, "waiting": 0}

# Object class names from custom YOLO model
OBJECT_CLASSES = {0: "bottle", 1: "cake box", 2: "plastic bottle"}

# Object class from scenario name
SCENARIO_TO_OBJECT_CLASS = {
    "bottle_01": 2,  # plastic bottle
    "bottle_02": 0,  # metallic bottle
    "box": 1,        # cake box
}


# ════════════════════════════════════════════════════════════════════
# Skeleton / joint constants
# ════════════════════════════════════════════════════════════════════

# Full COCO 17-joint skeleton edges (kept for preprocess/live_inference visualization)
# 0:nose  1:L_eye  2:R_eye  3:L_ear  4:R_ear  5:L_shoulder  6:R_shoulder
# 7:L_elbow  8:R_elbow  9:L_wrist  10:R_wrist  11:L_hip  12:R_hip
# 13:L_knee  14:R_knee  15:L_ankle  16:R_ankle
BODY_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6), (5, 11), (6, 12), (11, 12),       # torso
    (5, 7), (7, 9),                            # left arm
    (6, 8), (8, 10),                           # right arm
    (11, 13), (13, 15),                        # left leg
    (12, 14), (14, 16),                        # right leg
    (0, 5), (0, 6),                            # shoulder-nose
]

# Upper-body joint selection from COCO 17-joint output
# nose(0), L_shoulder(5), R_shoulder(6), L_elbow(7), R_elbow(8),
# L_wrist(9), R_wrist(10), L_hip(11), R_hip(12)
UPPER_BODY_JOINT_INDICES = [0, 5, 6, 7, 8, 9, 10, 11, 12]  # 9 joints
N_UPPER_JOINTS = len(UPPER_BODY_JOINT_INDICES)               # 9

# Per-joint features: x, y, conf, dx (velocity x), dy (velocity y)
JOINT_FEAT_DIM = 5
POSE_FLAT_DIM = N_UPPER_JOINTS * JOINT_FEAT_DIM              # 9 × 5 = 45

# Wrist indices in the FULL 17-joint array (used for relation feature computation)
WRIST_L_IDX = 9
WRIST_R_IDX = 10


# ════════════════════════════════════════════════════════════════════
# Relation / kinematic feature layout (20-dim)
# ════════════════════════════════════════════════════════════════════

RELATION_DIM = 20
# [0-1]   dx_wr, dy_wr       right wrist → obj center displacement
# [2-3]   dx_wl, dy_wl       left wrist → obj center displacement
# [4]     dist_r             ||right wrist – obj||
# [5]     dist_l             ||left wrist – obj||
# [6-7]   vx_obj, vy_obj     object center velocity (finite diff)
# [8-9]   vx_wr, vy_wr       right wrist velocity
# [10-11] vx_wl, vy_wl       left wrist velocity
# [12]    d_dist_r           approach speed: dist_r[t] – dist_r[t-1]
# [13]    d_dist_l           approach speed: dist_l[t] – dist_l[t-1]
# [14]    iou                person_bbox ∩ obj_bbox / union
# [15]    overlap            intersection / min(area_p, area_o)
# [16]    ttc_r              time-to-contact right wrist (clipped 0–100)
# [17]    ttc_l              time-to-contact left wrist  (clipped 0–100)
# [18]    angle              atan2(dy_wr, dx_wr) / π   ∈ [-1, 1]
# [19]    norm_dist          dist_r / sqrt(2)  (img-diagonal normalised)


# ════════════════════════════════════════════════════════════════════
# Data-driven horizon constants (from dataset analysis)
# ════════════════════════════════════════════════════════════════════
# Dataset: 30 FPS, phase lengths ~2.5–3.5 s
# Robot prep needs ~0.5–1.0 s lead time
# future_1_offset_frames = 15 → 0.5 s
# future_2_offset_frames = 30 → 1.0 s
FUTURE_1_OFFSET_FRAMES = 15   # raw frame offset (+0.5 s at 30 fps)
FUTURE_2_OFFSET_FRAMES = 30   # raw frame offset (+1.0 s at 30 fps)
FUTURE_1_SEC = FUTURE_1_OFFSET_FRAMES / 30.0
FUTURE_2_SEC = FUTURE_2_OFFSET_FRAMES / 30.0


# ════════════════════════════════════════════════════════════════════
# Path config
# ════════════════════════════════════════════════════════════════════

@dataclass
class PathConfig:
    """Paths configuration — override dataset_root for your machine."""
    dataset_root: str = "/media/hung/30fcade7-362c-4b72-9673-ca95aa0e6a9d/Dataset"

    @property
    def training_dir(self) -> str:
        return os.path.join(self.dataset_root, "Training")

    @property
    def annotations_dir(self) -> str:
        return os.path.join(self.dataset_root, "annotations", "Training")

    @property
    def preprocessed_dir(self) -> str:
        return os.path.join(self.dataset_root, "preprocessed_v3")

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.dataset_root, "checkpoints_v3")


# ════════════════════════════════════════════════════════════════════
# Model config
# ════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """
    Model architecture configuration — V3.

    Streams:
      1. PoseMLP:       flatten(9 joints × 5) = 45 → 128 → 256
      2. PatchROIPool:  ViT-S/16 full frame → patch tokens → ROI pool → Linear 384→256
      3. RelationMLP:   20-dim kinematic → 64 → 256

    Fusion:  CrossAttn (Q=pose, KV=[obj,rel], 4 heads, d=256)
    Temporal: Mamba 2 blocks  (fallback: causal transformer if mamba_ssm unavailable)
    Heads:   3 × MLP(256 → 4)  for label_now / label_f1 / label_f2
    """

    # ── Shared dimension ──────────────────────────────────────────
    d_model: int = 256
    dropout: float = 0.1

    # ── Pose stream ───────────────────────────────────────────────
    pose_flat_dim: int = POSE_FLAT_DIM      # 45
    pose_mlp_hidden: int = 128

    # ── Relation stream ───────────────────────────────────────────
    relation_dim: int = RELATION_DIM        # 20
    rel_mlp_hidden: int = 64
    ttc_clip_max: float = 100.0             # clamp for time-to-contact

    # ── Visual backbone (DINOv3 ViT-S/16, frozen Phase A) ─────────
    backbone_name: str = "vit_small_patch16_dinov3"
    backbone_frozen: bool = True            # Phase A: freeze
    backbone_dim: int = 384                 # ViT-S token width
    patch_size: int = 16
    img_size: int = 224
    n_patches_side: int = 14               # img_size // patch_size

    # ── Cross-attention fusion (pose-query, 2 KV tokens) ──────────
    n_cross_heads: int = 4
    ffn_ratio: float = 2.0                 # FFN hidden = d_model × ffn_ratio

    @property
    def ffn_dim(self) -> int:
        return int(self.d_model * self.ffn_ratio)

    # ── Temporal Mamba ────────────────────────────────────────────
    mamba_n_layers: int = 2
    mamba_d_state: int = 16
    mamba_expand: int = 2
    mamba_d_conv: int = 4
    mamba_fallback_heads: int = 4          # used when mamba_ssm unavailable

    # ── Output ────────────────────────────────────────────────────
    num_classes: int = 4                   # waiting / abort / hesitate / commit

    # ── Backbone trainability flag (toggled at Phase B) ───────────
    # Note: actual requires_grad toggling done via HOIModelV3.set_backbone_trainable()


# ════════════════════════════════════════════════════════════════════
# Training config
# ════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """Training configuration V3."""

    # ── Temporal chunking ─────────────────────────────────────────
    chunk_length: int = 8          # T frames per clip
    chunk_stride: int = 4          # stride for training clips (overlap)
    frame_stride: int = 8          # sub-sample every N-th frame from video

    # ── Future horizons (raw frame offsets at 30 fps) ─────────────
    future_1_offset: int = FUTURE_1_OFFSET_FRAMES   # 15 frames = 0.5 s
    future_2_offset: int = FUTURE_2_OFFSET_FRAMES   # 30 frames = 1.0 s

    # ── Training ──────────────────────────────────────────────────
    batch_size: int = 8
    num_epochs: int = 60
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    grad_accumulation: int = 2

    # ── Loss weights ──────────────────────────────────────────────
    loss_now_weight: float = 1.0
    loss_f1_weight: float = 0.5
    loss_f2_weight: float = 0.5

    # ── Focal loss (gamma=0 → standard CE) ────────────────────────
    use_focal_loss: bool = False
    focal_gamma: float = 2.0

    # ── Phase B fine-tuning ───────────────────────────────────────
    phase_b_epoch: int = 40            # start backbone fine-tune after this epoch
    phase_b_backbone_lr: float = 1e-5
    phase_b_head_lr: float = 5e-5

    # ── EMA ───────────────────────────────────────────────────────
    ema_decay: float = 0.999
    use_ema: bool = True

    # ── Balanced sampler ─────────────────────────────────────────
    balanced_sampler: bool = True

    # ── Early stopping ────────────────────────────────────────────
    patience: int = 15
    min_delta: float = 1e-4

    # ── Image preprocessing ───────────────────────────────────────
    image_size: int = 224

    # ── AMP / misc ────────────────────────────────────────────────
    use_amp: bool = True
    num_workers: int = 4
    seed: int = 42
    val_ratio: float = 0.2

    # ── Camera preference ─────────────────────────────────────────
    preferred_camera: str = "cam_832112070255"

    # ── YOLOv26 Pose ──────────────────────────────────────────────
    yolo_pose_model: str = "yolo26x-pose.pt"
    yolo_pose_conf: float = 0.3

    # ── Custom YOLO Object Detection ──────────────────────────────
    yolo_obj_model: str = "best.pt"
    yolo_obj_conf: float = 0.3


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

def get_config():
    """Return default (PathConfig, ModelConfig, TrainingConfig) tuple."""
    return PathConfig(), ModelConfig(), TrainingConfig()
