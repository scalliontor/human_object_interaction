# HOI Pipeline V3 — Human-Object Interaction Anticipation for Robot Assistance

Robot-centric anticipation model that predicts **current and future interaction states** between a person and an object, producing signals that drive a robot's `wait / prepare / assist / cancel` policy.

---

## Architecture Overview

```
─────────── Input (T=8 frames, stride=8, ~2.1 s clip) ───────────

 Stream 1 ── Upper-body Pose ─────────────────────────────────────
   9 joints × 5 features (x, y, conf, dx, dy)
   [B, T, 9, 5] → flatten → PoseMLP (45→128→256) → [B, T, 256]

 Stream 2 ── Object Feature (ROI from full frame) ────────────────
   Full frame [B, T, 3, 224, 224]
   → DINOv3 ViT-S/16 (frozen Phase A) → patch tokens [BT, 196, 384]
   → PatchROIPooler (bbox-guided mean pool → Linear 384→256) → [B, T, 256]
   + MaskToken when object not visible

 Stream 3 ── Relation / Kinematic ────────────────────────────────
   20-dim kinematic features (wrist↔obj vectors, velocities, IoU, TTC …)
   [B, T, 20] → RelationEncoder (20→64→256) → [B, T, 256]

─────────── Fusion ───────────────────────────────────────────────
   CrossAttentionFusion  Q=pose, KV=[obj+type_emb, rel+type_emb]
   4 heads, d=256, FFN 256→512→256
   → z [B, T, 256]

─────────── Temporal ─────────────────────────────────────────────
   TemporalMamba  2 blocks, d=256, d_state=16, expand=2
   (fallback: causal Transformer if mamba_ssm unavailable)
   → Z [B, T, 256]  →  h_last = Z[:, -1]  [B, 256]

─────────── 3 Prediction Heads ──────────────────────────────────
   head_now → logits [B, 4]   current state
   head_f1  → logits [B, 4]   +0.5 s future
   head_f2  → logits [B, 4]   +1.0 s future
```

### Key Design Choices vs V2

| Aspect | V2 | V3 |
|---|---|---|
| Pose encoder | GCN (17 joints full-body) | MLP (9 joints upper-body) |
| Visual feature | 2× DINOv3 (crop + full frame) | 1× DINOv3 + ROI pool |
| Relation features | 12-dim spatial | 20-dim kinematic (wrist vel, TTC, …) |
| Temporal module | Transformer 3 layers + CLS | Mamba 2 blocks |
| d_model | 384 | 256 |
| dropout | 0.3 | 0.1 |
| Label type | Multi-label BCE | Single-label CE (verified: 0% multi-label in data) |
| Prediction heads | 2 (now + 1 antic) | 3 (now + 0.5 s + 1.0 s) |
| Trainable params | ~8 M | ~5–8 M |
| Frozen backbone | ~21 M DINOv3 | ~21 M DINOv3 |

---

## Dataset

```
annotated_dataset_export/
├── Training/                         # .mp4 video files
│   ├── Person01/                     # 62 videos, 10 scenario types
│   │   ├── fully_abort_bottle_01/    # 15 videos
│   │   ├── fully_abort_bottle_02/    # 15 videos
│   │   ├── fully_abort_box/          # 13 videos
│   │   ├── hesitate_abort_bottle_01/ # 12 videos
│   │   └── ...
│   └── Person02/                     # 11 videos, 3 scenario types
└── annotations/Training/             # 73 × .json annotation files
```

### Statistics

| | |
|---|---|
| **Total videos** | 73 (Person01: 62, Person02: 11) |
| **Total frames** | 25,944 |
| **Annotated frames** | 19,750 (76.1%) |
| **FPS** | 30 |
| **Resolution** | 640 × 384 |
| **Multi-label rate** | 0% (confirmed single-label per frame) |

### Class Distribution

| Class | Frames | % annotated |
|---|---|---|
| `commit` | 7,803 | 39.5% |
| `abort` | 5,067 | 25.7% |
| `waiting` | 4,406 | 22.3% |
| `hesitate` | 2,474 | 12.5% |

### Dominant Transition Pattern

```
commit → waiting → abort   (60% of videos)
commit → hesitate → abort  (23% of videos)
```

Phase lengths: ~2.5–3.5 s per phase. T=8 clip at stride=8 ≈ 2.1 s covers roughly one phase.

### Label Schema

```python
# Priority: commit(3) > hesitate(2) > abort(1) > waiting(0)
# Empty annotation frames → waiting (0)
PREDICATES = ["waiting", "abort", "hesitate", "commit"]
```

### Relation Features (20-dim)

```
[0-1]   dx_wr, dy_wr       right wrist → object center
[2-3]   dx_wl, dy_wl       left wrist  → object center
[4]     dist_r              ||right wrist – obj||
[5]     dist_l              ||left wrist  – obj||
[6-7]   vx_obj, vy_obj      object velocity
[8-9]   vx_wr,  vy_wr       right wrist velocity
[10-11] vx_wl,  vy_wl       left wrist velocity
[12]    d_dist_r             approach speed (delta dist_r per frame)
[13]    d_dist_l
[14]    iou                  person ∩ object / union
[15]    overlap              intersection / min(area_p, area_o)
[16]    ttc_r                time-to-contact right wrist (clipped 0–100)
[17]    ttc_l
[18]    angle                atan2(dy_wr, dx_wr) / pi in [-1,1]
[19]    norm_dist            dist_r / sqrt(2)
```

---

## Usage

### 1. Preprocess

```bash
python -m hoi_pipeline.preprocess \
    --data_root  ./annotated_dataset_export/Training \
    --anno_root  ./annotated_dataset_export/annotations/Training \
    --output_dir ./preprocessed_v3 \
    --pose_model yolo26x-pose.pt \
    --obj_model  ./annotated_dataset_export/Training/best.pt
```

### 2. Train

```bash
python -m hoi_pipeline.train \
    --data_root ./annotated_dataset_export/Training \
    --anno_root ./annotated_dataset_export/annotations/Training \
    --cache_dir ./preprocessed_v3 \
    --ckpt_dir  ./checkpoints_v3 \
    --epochs 60 --batch_size 8
```

Phase B (backbone partial fine-tune) starts automatically at epoch 40.

### 3. Inference

```bash
# Batch video
python -m hoi_pipeline.inference \
    --video_path ./my_video.mp4 \
    --model_path ./checkpoints_v3/best_model.pt

# Live webcam
python -m hoi_pipeline.live_inference \
    --model_path ./checkpoints_v3/best_model.pt
```

---

## Training Protocol

### Phase A (epochs 1–40): backbone frozen

- Train: PoseMLP, RelationEncoder, PatchROIPooler projection, CrossAttentionFusion, TemporalMamba, 3 heads
- Optimizer: AdamW, lr=1e-4, wd=1e-4
- Scheduler: step-level cosine with 5-epoch linear warmup

### Phase B (epochs 41–60): partial backbone unfreeze

- Unfreeze last 2 ViT blocks + final norm
- Backbone lr: 1e-5, head lr: 5e-5

### Loss

```
L = 1.0 × CE(now) + 0.5 × CE(f1) + 0.5 × CE(f2)
```

Class-weighted CrossEntropyLoss (set `use_focal_loss=True` to switch to FocalLoss γ=2).

### Sampler

`WeightedRandomSampler` balanced by `label_now` class frequency — ensures all 4 classes appear roughly equally per mini-batch.

---

## Metrics

| Metric | Description |
|---|---|
| `macro_f1_now` | **Primary** — macro F1 across 4 classes, current state |
| `balanced_acc` | Mean per-class recall |
| `commit_recall` | Robot-critical: high → trigger `prepare` |
| `hesitate_recall` | Anticipation quality |
| `macro_f1_f1` | F1 for +0.5 s head |
| `macro_f1_f2` | F1 for +1.0 s head |
| confusion matrix | Printed every epoch (row=true, col=pred, values=recall) |

Best checkpoint saved when `macro_f1_now` improves. Early stopping on `val_loss` (patience=15).

---

## Rule-based Policy (v0.1)

No supervised policy head in v0.1. Robot action driven by softmax probabilities from `head_now` and `head_f1`:

```python
P_now    = softmax(logits_now)   # [4]
P_future = softmax(logits_f1)    # [4]
dist_r   = relation_features[4]  # right wrist distance to object

if   P_future[commit] > 0.6:                  action = "prepare"
elif P_now[commit]    > 0.7 and dist_r < 0.1: action = "assist"
elif P_now[abort]     > 0.6:                  action = "cancel"
else:                                          action = "wait"
```

---

## Known Limitations (v0.1)

- Single object per frame (highest-confidence YOLO detection)
- Fixed camera viewpoint (`cam_832112070255`)
- 2 subjects only — limited cross-person generalisation
- No supervised policy head (rule-based policy only)
- Full-frame resize 640×384 → 224×224 (aspect ratio distortion; letterbox is future work)

---

## File Structure

```
hoi_pipeline/
├── config.py          # Constants, ModelConfig, TrainingConfig
├── model.py           # HOIModelV3: PoseMLP, PatchROIPooler, RelationEncoder,
│                      #   CrossAttentionFusion, TemporalMamba, 3 ClassificationHeads
├── dataset.py         # HOIChunkDataset V3 (3-stream, single-label, 20-dim relation)
├── train.py           # Phase A/B training, balanced sampler, macro F1 checkpoint
├── utils.py           # FocalLoss, compute_metrics_v3, EMA, EarlyStopping, make_balanced_sampler
├── preprocess.py      # YOLOv26 Pose + custom YOLO OD → .npz feature cache
├── inference.py       # Batch video inference + visualisation
└── live_inference.py  # Real-time webcam inference
```
