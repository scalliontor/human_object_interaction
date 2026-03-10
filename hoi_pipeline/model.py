"""
Full 4-Stream HOI Model following the architecture diagram EXACTLY:

  Stream 1: Hand Pose → GCN (MediaPipe right hand 21 joints + velocity → 384-dim)
  Stream 2: Object Crop 224×224 → DINOv3 ViT-S/16 → CLS 384-dim
  Stream 3: Context Full Frame 224×224 → DINOv3 ViT-S/16 (shared backbone) → CLS 384-dim
  Stream 4: Spatial 7-dim → Bottleneck → 384-dim
  → Pose-Query Cross-Attention Fusion (Q=h_pose, KV=[h_obj, h_ctx, h_spatial])
  → Temporal Attention Encoder (3 layers, CLS token)
  → Classification Head + Anticipation Head
  → Multi-Task Loss: BCE(cls) + 0.3 × BCE(antic)

Changes from original diagram:
  - YOLO Pose (17 body joints) → MediaPipe right hand (21 landmarks + velocity)
  - Object bbox not in annotations → use hand-centric crop as object region
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import timm

from .config import ModelConfig, HAND_SKELETON_EDGES


# ════════════════════════════════════════════════════════════════════════════
# Graph Convolution Layer
# ════════════════════════════════════════════════════════════════════════════

class GraphConvLayer(nn.Module):
    """Graph Convolution: X' = σ(D^{-1/2} A D^{-1/2} X W)."""

    def __init__(self, in_features: int, out_features: int, adj: torch.Tensor):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.register_buffer("adj", self._normalize_adj(adj))

    @staticmethod
    def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        degree = adj.sum(dim=1)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat = torch.diag(d_inv_sqrt)
        return d_mat @ adj @ d_mat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B*T, N, C] → [B*T, N, C_out]"""
        return torch.matmul(self.adj, self.linear(x))


# ════════════════════════════════════════════════════════════════════════════
# Stream 1: Hand Graph Encoder (HumanGraphEncoder in diagram)
# Replaces: Pose GCN with 17 body joints → 21 right-hand landmarks + velocity
# ════════════════════════════════════════════════════════════════════════════

class HandGraphEncoder(nn.Module):
    """
    2-layer Graph Conv + LayerNorm on 21-node hand skeleton.
    Input: [B, T, 21, 6] (x,y,z + vx,vy,vz)
    Output: h_hand [B, T, 384]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        num_nodes = cfg.num_landmarks  # 21
        input_dim = cfg.input_dim      # 6
        d = cfg.d_model                # 384

        adj = torch.zeros(num_nodes, num_nodes)
        for i, j in HAND_SKELETON_EDGES:
            adj[i, j] = 1.0
            adj[j, i] = 1.0

        hidden = d // 2  # 192
        self.gcn1 = GraphConvLayer(input_dim, hidden, adj)
        self.gcn2 = GraphConvLayer(hidden, d, adj)
        self.norm = nn.LayerNorm(d)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)
        self.pool_proj = nn.Linear(d, d)

    def forward(self, hand_features: torch.Tensor) -> torch.Tensor:
        """[B, T, 21, 6] → [B, T, 384]"""
        B, T, N, C = hand_features.shape
        x = hand_features.reshape(B * T, N, C)
        x = self.act(self.gcn1(x))
        x = self.dropout(x)
        x = self.gcn2(x)
        x = self.norm(x)
        x = x.mean(dim=1)         # Pool over nodes → [B*T, d]
        x = self.pool_proj(x)
        return x.reshape(B, T, -1)


# ════════════════════════════════════════════════════════════════════════════
# Pose Fallback — learnable [1, 384] (trunc_normal init)
# Protects Q when hand is not detected
# Blend: h = conf * h_gcn + (1-conf) * fallback
# ════════════════════════════════════════════════════════════════════════════

class PoseFallback(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fallback = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.fallback, std=0.02)

    def forward(self, h_gcn: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        """h_gcn [B,T,d], confidence [B,T] → [B,T,d]"""
        conf = confidence.unsqueeze(-1)
        return conf * h_gcn + (1 - conf) * self.fallback


# ════════════════════════════════════════════════════════════════════════════
# Stream 2 & 3: DINOv3 ViT-S/16 Visual Backbone (SHARED weights)
# Stream 2: Object Crop 224×224 → CLS 384-dim
# Stream 3: Context Full Frame 224×224 → CLS 384-dim
# Loaded via timm: vit_small_patch16_dinov3
# ════════════════════════════════════════════════════════════════════════════

class DINOv3Backbone(nn.Module):
    """
    DINOv3 ViT-S/16 shared backbone loaded via timm.
    21M params, 384-dim CLS token → feature vector.
    Backbone is FROZEN (used as feature extractor).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Load pretrained DINOv3 via timm
        # num_classes=0 removes the classification head → outputs CLS features
        self.backbone = timm.create_model(
            cfg.backbone_name,
            pretrained=True,
            num_classes=0,
        )

        # Freeze backbone
        if cfg.backbone_frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Get the output feature dimension from timm
        self.feat_dim = self.backbone.num_features  # Should be 384 for ViT-S

        # Linear projection: backbone_dim → d_model
        # Diagram: "Linear Proj 384 → 384"
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def train(self, mode=True):
        """Override train to keep backbone always in eval mode."""
        super().train(mode)
        if self.cfg.backbone_frozen:
            self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B*T, 3, 224, 224] — ImageNet-normalized images
        Returns:
            features: [B*T, d_model] — CLS token features
        """
        with torch.no_grad():
            features = self.backbone(pixel_values)  # [B*T, feat_dim]
        return self.proj(features)                   # [B*T, d_model]


# ════════════════════════════════════════════════════════════════════════════
# Mask Token — learnable [1, 384] (trunc_normal init)
# Replaces zero-tensor when object is occluded/tiny/missing
# Prevents temporal flicker
# ════════════════════════════════════════════════════════════════════════════

class MaskToken(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self, h_obj: torch.Tensor, obj_visible: torch.Tensor
    ) -> torch.Tensor:
        """
        h_obj: [B, T, d], obj_visible: [B, T] boolean flag
        Returns: h_obj with mask token substituted where not visible
        """
        mask = obj_visible.unsqueeze(-1)  # [B, T, 1]
        return mask * h_obj + (1 - mask) * self.mask_token.unsqueeze(0)


# ════════════════════════════════════════════════════════════════════════════
# Stream 4: Spatial Bottleneck
# 7-dim → Linear(7→64) + GELU + Linear(64→384) + Dropout(0.3)
# ════════════════════════════════════════════════════════════════════════════

class SpatialEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.spatial_dim, cfg.spatial_hidden),
            nn.GELU(),
            nn.Linear(cfg.spatial_hidden, cfg.d_model),
            nn.Dropout(cfg.spatial_dropout),
        )

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:
        """[B, T, 7] → [B, T, 384]"""
        return self.net(spatial)


# ════════════════════════════════════════════════════════════════════════════
# Pose-Query Cross-Attention Fusion (per frame)
# Q = h_pose [1, 384] (fallback-blended)
# KV = [h_obj, h_ctx, h_spatial] [3, 384]
# (h_obj may be mask token)
# → Pose attends to object appearance, scene context, AND spatial location
# ════════════════════════════════════════════════════════════════════════════

class CrossAttentionFusion(nn.Module):
    """
    Multi-Head Cross-Attention (8 heads, 384-dim)
    + LayerNorm + Residual
    + FFN (Linear(384→1536) + GELU + Linear(1536→384))
    + LayerNorm + Residual
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=cfg.n_cross_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d)

        self.ffn = nn.Sequential(
            nn.Linear(d, cfg.ffn_dim),
            nn.GELU(),
            nn.Linear(cfg.ffn_dim, d),
            nn.Dropout(cfg.dropout),
        )
        self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        h_pose: torch.Tensor,      # [B, T, d]
        h_obj: torch.Tensor,        # [B, T, d]
        h_ctx: torch.Tensor,        # [B, T, d]
        h_spatial: torch.Tensor,    # [B, T, d]
    ) -> torch.Tensor:
        """
        Per-frame cross attention: Q=pose, KV=[obj, ctx, spatial]
        Returns: fused [B, T, d]
        """
        B, T, d = h_pose.shape

        # Reshape for per-frame attention
        q = h_pose.reshape(B * T, 1, d)                # [B*T, 1, d]
        kv = torch.stack([h_obj, h_ctx, h_spatial], dim=2)  # [B, T, 3, d]
        kv = kv.reshape(B * T, 3, d)                   # [B*T, 3, d]

        attn_out, _ = self.cross_attn(q, kv, kv)       # [B*T, 1, d]
        attn_out = self.dropout(attn_out)

        # Residual + LayerNorm
        x = self.norm1(q + attn_out)

        # FFN + Residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x.reshape(B, T, d)


# ════════════════════════════════════════════════════════════════════════════
# Temporal Attention Encoder (3 layers)
# CLS token (384-dim, learnable) + Position Embedding [1, T+1, 384]
# 3× Temporal Attention Block:
#   Self-Attention (6 heads) + Relative Position Bias + LayerNorm + Residual
#   FFN (384→1536→384) + LayerNorm + Residual
# → CLS[0] = Pooled CLS [B, 384], tokens[1:] = Frame Features [B, T, 384]
# ════════════════════════════════════════════════════════════════════════════

class TemporalAttentionBlock(nn.Module):
    """Single temporal attention block with relative position bias."""

    def __init__(self, cfg: ModelConfig, max_len: int = 64):
        super().__init__()
        d = cfg.d_model
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        # Learnable relative position bias
        self.rel_pos_bias = nn.Parameter(torch.zeros(cfg.n_heads, max_len, max_len))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        self.norm1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, cfg.ffn_dim),
            nn.GELU(),
            nn.Linear(cfg.ffn_dim, d),
            nn.Dropout(cfg.dropout),
        )
        self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, d] → [B, S, d] where S = T+1"""
        B, S, d = x.shape
        n_heads = self.rel_pos_bias.size(0)

        # Relative position bias as attention bias
        bias = self.rel_pos_bias[:, :S, :S]  # [n_heads, S, S]
        # Expand for batch: [B * n_heads, S, S]
        attn_mask = bias.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * n_heads, S, S)

        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        attn_out = self.dropout(attn_out)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class TemporalAttentionEncoder(nn.Module):
    """CLS token + position embedding + 3 temporal attention blocks + LayerNorm."""

    def __init__(self, cfg: ModelConfig, max_len: int = 64):
        super().__init__()
        d = cfg.d_model

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, d))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.layers = nn.ModuleList([
            TemporalAttentionBlock(cfg, max_len + 1)
            for _ in range(cfg.n_temporal_layers)
        ])
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [B, T, d] → (cls_feat [B, d], frame_feat [B, T, d])
        """
        B, T, d = x.shape

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)          # [B, T+1, d]
        x = x + self.pos_embed[:, :T + 1, :]

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        return x[:, 0], x[:, 1:]  # cls [B,d], frames [B,T,d]


# ════════════════════════════════════════════════════════════════════════════
# Output Heads
# ════════════════════════════════════════════════════════════════════════════

class ClassificationHead(nn.Module):
    """MLP Classifier: Linear(384→384) + GELU + Dropout + Linear(384→C)
    → 4 HOI predicates logits (BCEWithLogitsLoss)"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, cfg.num_predicates),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class AnticipationHead(nn.Module):
    """Anticipation MLP: Linear(384→384) + GELU + Dropout + Linear(384→C)
    → predict interactions ~1 second in the future (BCEWithLogitsLoss)"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, cfg.num_predicates),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ════════════════════════════════════════════════════════════════════════════
# Full HOI Model — combining all 4 streams
# ════════════════════════════════════════════════════════════════════════════

class HOIModel(nn.Module):
    """
    Full Human-Object Interaction Model.

    Input per chunk:
        hand_features:    [B, T, 21, 6]  — MediaPipe right hand (pos + vel)
        hand_confidence:  [B, T]         — detection confidence
        obj_crops:        [B, T, 3, 224, 224] — object region crops
        ctx_frames:       [B, T, 3, 224, 224] — full context frames
        obj_visible:      [B, T]         — whether object is visible
        spatial_features: [B, T, 7]      — spatial vector

    Output:
        cls_logits:  [B, num_predicates]
        antic_logits: [B, num_predicates]
    """

    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg

        # Stream 1: Hand Pose GCN
        self.hand_encoder = HandGraphEncoder(cfg)
        self.pose_fallback = PoseFallback(cfg.d_model)

        # Stream 2 & 3: DINOv3 ViT-S/16 (SHARED backbone)
        self.dino_backbone = DINOv3Backbone(cfg)

        # Linear projections for each visual stream (diagram shows separate Linear Proj 384→384)
        self.obj_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.ctx_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # Mask Token for occluded/missing objects
        self.mask_token = MaskToken(cfg.d_model)

        # Stream 4: Spatial Bottleneck
        self.spatial_encoder = SpatialEncoder(cfg)

        # Cross-Attention Fusion
        self.cross_attention = CrossAttentionFusion(cfg)

        # Temporal Attention Encoder (3 layers)
        self.temporal_encoder = TemporalAttentionEncoder(cfg)

        # Output Heads
        self.cls_head = ClassificationHead(cfg)
        self.antic_head = AnticipationHead(cfg)

        # Init weights (excluding frozen backbone)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        hand_features: torch.Tensor,      # [B, T, 21, 6]
        hand_confidence: torch.Tensor,     # [B, T]
        obj_crops: torch.Tensor,           # [B, T, 3, 224, 224]
        ctx_frames: torch.Tensor,          # [B, T, 3, 224, 224]
        obj_visible: torch.Tensor,         # [B, T]
        spatial_features: torch.Tensor,    # [B, T, 7]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = hand_features.shape[:2]
        d = self.cfg.d_model

        # ── Stream 1: Hand GCN ──
        h_hand = self.hand_encoder(hand_features)         # [B, T, d]
        h_hand = self.pose_fallback(h_hand, hand_confidence)

        # ── Stream 2: Object Crop → DINOv3 → proj ──
        obj_flat = obj_crops.reshape(B * T, 3, 224, 224)
        h_obj = self.dino_backbone(obj_flat)              # [B*T, d]
        h_obj = self.obj_proj(h_obj).reshape(B, T, d)     # [B, T, d]
        h_obj = self.mask_token(h_obj, obj_visible)       # Apply mask token if needed

        # ── Stream 3: Context Full Frame → DINOv3 (shared) → proj ──
        ctx_flat = ctx_frames.reshape(B * T, 3, 224, 224)
        h_ctx = self.dino_backbone(ctx_flat)              # [B*T, d]
        h_ctx = self.ctx_proj(h_ctx).reshape(B, T, d)     # [B, T, d]

        # ── Stream 4: Spatial ──
        h_spatial = self.spatial_encoder(spatial_features)  # [B, T, d]

        # ── Cross-Attention Fusion ──
        # Q = h_hand (pose, fallback-blended)
        # KV = [h_obj (or mask token), h_ctx, h_spatial]
        fused = self.cross_attention(h_hand, h_obj, h_ctx, h_spatial)  # [B, T, d]

        # ── Temporal Attention Encoder ──
        cls_feat, frame_feat = self.temporal_encoder(fused)  # [B, d], [B, T, d]

        # ── Output Heads ──
        cls_logits = self.cls_head(cls_feat)       # [B, C]
        antic_logits = self.antic_head(cls_feat)    # [B, C]

        return cls_logits, antic_logits

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone = sum(p.numel() for p in self.dino_backbone.backbone.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "backbone_frozen": backbone,
            "head_trainable": trainable,
        }
