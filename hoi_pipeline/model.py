"""
HOI Model V3 — 3-Stream Anticipation Architecture

Streams:
  1. PoseMLP       : upper-body pose (9 joints × 5-dim) → 256
  2. PatchROIPooler: full-frame ViT-S/16 → patch ROI pool → 256
  3. RelationEncoder: 20-dim kinematic features → 256

Pipeline:
  [pose, obj_roi, relation] → CrossAttentionFusion (Q=pose, KV=[obj,rel])
  → TemporalMamba (2 blocks, d=256)
  → h_last = Z[:, -1]
  → head_now / head_f1 / head_f2  (4-class CE logits)

Phase A: backbone fully frozen
Phase B: call model.set_backbone_trainable(n_unfreeze=2) to unfreeze last blocks
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import timm

from .config import ModelConfig

# ════════════════════════════════════════════════════════════════════
# Mamba availability check (once at import time)
# ════════════════════════════════════════════════════════════════════

_MAMBA_AVAILABLE = False
try:
    from mamba_ssm import Mamba as _MambaSSM   # type: ignore[import]
    _MAMBA_AVAILABLE = True
except (ImportError, RuntimeError):
    pass


# ════════════════════════════════════════════════════════════════════
# Stream 1: Pose MLP
# ════════════════════════════════════════════════════════════════════

class PoseMLP(nn.Module):
    """
    Upper-body pose encoder.

    Input:  [B, T, 9, 5]  — 9 joints × (x, y, conf, dx, dy)
    Output: [B, T, d_model]

    Architecture:
        flatten → LN → Linear(45, 128) → GELU → Dropout → Linear(128, 256) → LN
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        in_dim = cfg.pose_flat_dim   # 45
        d      = cfg.d_model         # 256
        h      = cfg.pose_mlp_hidden  # 128

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, d),
            nn.LayerNorm(d),
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """[B, T, 9, 5] → [B, T, 256]"""
        B, T, J, C = pose.shape
        x = pose.reshape(B, T, J * C)   # [B, T, 45]
        return self.net(x)


# ════════════════════════════════════════════════════════════════════
# Stream 2: Visual backbone + Patch ROI Pooler
# ════════════════════════════════════════════════════════════════════

class PatchROIPooler(nn.Module):
    """
    Extracts object features from ViT patch tokens using bbox-guided mean pooling.

    ViT-S/16 on 224×224 → 14×14 = 196 spatial patch tokens.
    Patch i covers pixel region [i//14 * 16 : (i//14+1)*16, (i%14)*16 : (i%14+1)*16].

    Input:
        patch_tokens: [B*T, 196, 384]  (CLS token removed)
        bboxes:       [B*T, 4]          normalized [x1,y1,x2,y2] ∈ [0,1]
    Output: [B, T, d_model]

    Invisible-object guard: if obj_visible=0 for a frame, replace pooled
    feature with a learnable mask token.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_side = cfg.n_patches_side   # 14
        self.proj   = nn.Linear(cfg.backbone_dim, cfg.d_model)
        self.norm   = nn.LayerNorm(cfg.d_model)

        # Learnable mask token for occluded / missing objects
        self.mask_token = nn.Parameter(torch.zeros(1, cfg.d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        patch_tokens: torch.Tensor,   # [B*T, 196, 384]
        bboxes: torch.Tensor,          # [B*T, 4]
        obj_visible: torch.Tensor,     # [B*T]  float 0/1
        B: int,
        T: int,
    ) -> torch.Tensor:                 # [B, T, 256]
        BT, N, D = patch_tokens.shape
        n = self.n_side   # 14

        # Reshape to spatial grid [BT, n, n, D]
        grid = patch_tokens.reshape(BT, n, n, D)

        # Map normalised bbox coords → patch grid indices
        x1 = (bboxes[:, 0] * n).clamp(0, n - 1).long()
        y1 = (bboxes[:, 1] * n).clamp(0, n - 1).long()
        x2 = (bboxes[:, 2] * n).clamp(0, n).ceil().long().clamp(1, n)
        y2 = (bboxes[:, 3] * n).clamp(0, n).ceil().long().clamp(1, n)
        # Ensure at least one patch
        x2 = torch.maximum(x2, x1 + 1)
        y2 = torch.maximum(y2, y1 + 1)

        # Mean-pool patches inside ROI (Python loop over BT — acceptable for T=8)
        pooled = torch.zeros(BT, D, device=patch_tokens.device, dtype=patch_tokens.dtype)
        for i in range(BT):
            roi = grid[i, y1[i]:y2[i], x1[i]:x2[i], :]   # [h, w, D]
            if roi.numel() > 0:
                pooled[i] = roi.mean(dim=(0, 1))
            # else: stay zero (rare degenerate bbox)

        # Project 384 → 256, apply LN
        out = self.norm(self.proj(pooled))      # [BT, 256]

        # Apply mask token where object not visible
        # obj_visible: [BT] float, broadcast over feature dim
        vis = obj_visible.unsqueeze(-1)         # [BT, 1]
        out = vis * out + (1.0 - vis) * self.mask_token

        return out.reshape(B, T, -1)            # [B, T, 256]


# ════════════════════════════════════════════════════════════════════
# Stream 3: Relation / Kinematic encoder
# ════════════════════════════════════════════════════════════════════

class RelationEncoder(nn.Module):
    """
    Kinematic relation encoder.

    Input:  [B, T, 20]
    Output: [B, T, d_model]

    Architecture:
        LN → Linear(20, 64) → GELU → Dropout → Linear(64, 256) → LN
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        in_dim = cfg.relation_dim    # 20
        d      = cfg.d_model         # 256
        h      = cfg.rel_mlp_hidden  # 64

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, d),
            nn.LayerNorm(d),
        )

    def forward(self, rel: torch.Tensor) -> torch.Tensor:
        """[B, T, 20] → [B, T, 256]"""
        return self.net(rel)


# ════════════════════════════════════════════════════════════════════
# Fusion: Pose-query Cross-Attention (2 KV tokens)
# ════════════════════════════════════════════════════════════════════

class CrossAttentionFusion(nn.Module):
    """
    Per-frame cross-attention: Q=pose, KV=[obj, rel].

    Includes learnable type embeddings so the model distinguishes
    appearance tokens (obj) from kinematic tokens (rel).

    Input:  h_pose [B,T,d], h_obj [B,T,d], h_rel [B,T,d]
    Output: [B, T, d]

    Architecture:
        kv = h_obj + obj_type_emb  |  h_rel + rel_type_emb   → stack [BT, 2, d]
        MultiheadAttention(Q=pose, KV=kv, n_heads=4)
        Add residual + LN
        FFN (256 → 512 → 256) + Add residual + LN
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model

        # Learnable type embeddings for the 2 KV slots
        self.obj_type_emb = nn.Parameter(torch.zeros(1, 1, d))
        self.rel_type_emb = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.obj_type_emb, std=0.02)
        nn.init.trunc_normal_(self.rel_type_emb, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=cfg.n_cross_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d)

        ffn_dim = cfg.ffn_dim   # 512
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(ffn_dim, d),
            nn.Dropout(cfg.dropout),
        )
        self.norm2 = nn.LayerNorm(d)

    def forward(
        self,
        h_pose: torch.Tensor,   # [B, T, d]
        h_obj:  torch.Tensor,   # [B, T, d]
        h_rel:  torch.Tensor,   # [B, T, d]
    ) -> torch.Tensor:          # [B, T, d]
        B, T, d = h_pose.shape

        # Reshape for per-frame attention
        q  = h_pose.reshape(B * T, 1, d)               # [BT, 1, d]

        # KV: add type embeddings then stack
        kv_obj = (h_obj + self.obj_type_emb).reshape(B * T, 1, d)
        kv_rel = (h_rel + self.rel_type_emb).reshape(B * T, 1, d)
        kv = torch.cat([kv_obj, kv_rel], dim=1)         # [BT, 2, d]

        attn_out, _ = self.cross_attn(q, kv, kv)        # [BT, 1, d]
        x = self.norm1(q + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x.reshape(B, T, d)


# ════════════════════════════════════════════════════════════════════
# Temporal: Mamba (or causal Transformer fallback)
# ════════════════════════════════════════════════════════════════════

class _MambaBlock(nn.Module):
    """Single Mamba block with pre-norm and residual."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = _MambaSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.mamba(self.norm(x)))


class _CausalTransformerBlock(nn.Module):
    """Pre-norm causal Transformer — fallback when mamba_ssm unavailable."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.drop  = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=causal_mask)
        x = x + self.drop(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class TemporalMamba(nn.Module):
    """
    Temporal sequence model: 2 blocks of Mamba (or causal Transformer fallback).

    Input:  [B, T, d_model]
    Output: [B, T, d_model]

    Includes learnable absolute position embedding.
    """

    def __init__(self, cfg: ModelConfig, max_len: int = 64):
        super().__init__()
        self.using_mamba = _MAMBA_AVAILABLE
        d = cfg.d_model

        self.pos_emb = nn.Embedding(max_len, d)

        if self.using_mamba:
            self.blocks = nn.ModuleList([
                _MambaBlock(
                    d_model=d,
                    d_state=cfg.mamba_d_state,
                    d_conv=cfg.mamba_d_conv,
                    expand=cfg.mamba_expand,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.mamba_n_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                _CausalTransformerBlock(
                    d_model=d,
                    n_heads=cfg.mamba_fallback_heads,
                    ffn_dim=cfg.ffn_dim,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.mamba_n_layers)
            ])

        self.out_norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, d] → [B, T, d]"""
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        x = x + self.pos_emb(pos).unsqueeze(0)   # broadcast over B
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x)


# ════════════════════════════════════════════════════════════════════
# Classification Head  (shared structure for all 3 heads)
# ════════════════════════════════════════════════════════════════════

class ClassificationHead(nn.Module):
    """
    MLP classifier head.

    Input:  [B, d_model]
    Output: [B, num_classes]  (raw logits)

    LN → Dropout → Linear(d, d) → GELU → Dropout → Linear(d, C)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ════════════════════════════════════════════════════════════════════
# Full HOI Model V3
# ════════════════════════════════════════════════════════════════════

class HOIModelV3(nn.Module):
    """
    Human-Object Interaction Anticipation Model V3.

    Input batch dict:
        pose_features:      [B, T, 9, 5]
        ctx_frames:         [B, T, 3, 224, 224]
        obj_bbox_norm:      [B, T, 4]
        relation_features:  [B, T, 20]
        obj_visible:        [B, T]

    Output dict:
        logits_now: [B, 4]   current state
        logits_f1:  [B, 4]   +0.5 s future
        logits_f2:  [B, 4]   +1.0 s future
    """

    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg

        # ── Visual backbone (DINOv3 ViT-S/16) ────────────────────
        self.backbone = timm.create_model(
            cfg.backbone_name,
            pretrained=True,
            num_classes=0,   # remove classifier head → CLS token output
        )
        self._backbone_trainable = False
        self._freeze_backbone()

        # ── Stream encoders ───────────────────────────────────────
        self.pose_enc = PoseMLP(cfg)
        self.roi_pool = PatchROIPooler(cfg)
        self.rel_enc  = RelationEncoder(cfg)

        # ── Fusion ────────────────────────────────────────────────
        self.fusion = CrossAttentionFusion(cfg)

        # ── Temporal ──────────────────────────────────────────────
        self.temporal = TemporalMamba(cfg)

        # ── Prediction heads ──────────────────────────────────────
        self.head_now = ClassificationHead(cfg)
        self.head_f1  = ClassificationHead(cfg)
        self.head_f2  = ClassificationHead(cfg)

        # Weight initialisation (excludes frozen backbone)
        self._init_weights()

    # ── Backbone freeze/unfreeze ──────────────────────────────────

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self._backbone_trainable = False

    def set_backbone_trainable(self, n_unfreeze: int = 2):
        """
        Phase B: unfreeze the last n_unfreeze transformer blocks of the ViT backbone.
        All other backbone parameters stay frozen.

        Args:
            n_unfreeze: number of final transformer blocks to unfreeze (default 2)
        """
        # First, ensure everything is frozen
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Unfreeze last n blocks
        blocks = list(self.backbone.blocks)  # type: ignore[attr-defined]
        for block in blocks[-n_unfreeze:]:
            for p in block.parameters():
                p.requires_grad = True

        # Also unfreeze the final norm if present
        if hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

        self._backbone_trainable = True

    def train(self, mode: bool = True):
        """Keep backbone eval unless it has been partially unfrozen (Phase B)."""
        super().train(mode)
        if not self._backbone_trainable:
            self.backbone.eval()
        return self

    # ── Weight init ───────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pose        = batch["pose_features"]      # [B, T, 9, 5]
        ctx         = batch["ctx_frames"]          # [B, T, 3, 224, 224]
        bboxes      = batch["obj_bbox_norm"]       # [B, T, 4]
        rel         = batch["relation_features"]   # [B, T, 20]
        obj_visible = batch["obj_visible"]         # [B, T]

        B, T = pose.shape[:2]

        # ── Backbone: one forward pass for all frames ─────────────
        ctx_flat = ctx.reshape(B * T, 3, 224, 224)
        grad_ctx = torch.enable_grad if self._backbone_trainable else torch.no_grad
        with grad_ctx():
            feats = self.backbone.forward_features(ctx_flat)  # [BT, 197, 384]
        patch_tokens = feats[:, 1:, :]     # drop CLS → [BT, 196, 384]

        # ── Encode 3 streams ──────────────────────────────────────
        h_pose = self.pose_enc(pose)                          # [B, T, 256]
        h_obj  = self.roi_pool(
            patch_tokens,
            bboxes.reshape(B * T, 4),
            obj_visible.reshape(B * T),
            B, T,
        )                                                      # [B, T, 256]
        h_rel  = self.rel_enc(rel)                            # [B, T, 256]

        # ── Cross-attention fusion ────────────────────────────────
        z = self.fusion(h_pose, h_obj, h_rel)                 # [B, T, 256]

        # ── Temporal Mamba ────────────────────────────────────────
        Z      = self.temporal(z)                              # [B, T, 256]
        h_last = Z[:, -1, :]                                   # [B, 256]

        # ── 3 prediction heads ────────────────────────────────────
        return {
            "logits_now": self.head_now(h_last),   # [B, 4]
            "logits_f1":  self.head_f1(h_last),
            "logits_f2":  self.head_f2(h_last),
        }

    # ── Parameter count utility ───────────────────────────────────

    def count_parameters(self) -> Dict[str, int]:
        total      = sum(p.numel() for p in self.parameters())
        trainable  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone   = sum(p.numel() for p in self.backbone.parameters())
        return {
            "total":           total,
            "trainable":       trainable,
            "backbone_total":  backbone,
            "backbone_frozen": sum(
                p.numel() for p in self.backbone.parameters()
                if not p.requires_grad
            ),
        }
