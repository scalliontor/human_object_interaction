# Human-Object Interaction (HOI) Recognition Pipeline

4-stream encoder architecture for recognizing human-object interactions from egocentric video, with both **classification** and **anticipation** heads.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     4-Stream Encoder                            │
│                                                                 │
│  Stream 1: Hand Pose ──► GCN (21 joints + velocity) ──► 384d   │
│  Stream 2: Object Crop ──► DINOv3 ViT-S/16 (frozen) ──► 384d  │
│  Stream 3: Context Frame ──► DINOv3 ViT-S/16 (shared) ──► 384d│
│  Stream 4: Spatial (7-dim) ──► MLP Bottleneck ──► 384d         │
│                                                                 │
│  ──► Cross-Attention Fusion (Q=pose, KV=[obj,ctx,spatial])     │
│  ──► Temporal Attention Encoder (3 layers + CLS token)          │
│  ──► Classification Head (4 actions)                            │
│  ──► Anticipation Head (predict ~1s ahead)                      │
│                                                                 │
│  Loss: BCE(cls) + 0.3 × BCE(antic) + class weights             │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Choices

| Component | Original Diagram | This Implementation |
|-----------|-----------------|---------------------|
| Pose | YOLO Pose (17 body joints) | **MediaPipe Right Hand (21 landmarks + velocity)** |
| Visual Backbone | DINOv3 ViT-S/16 | DINOv3 ViT-S/16 via **timm** (frozen, 21M params) |
| Object Crop | Ground-truth bbox | **Hand-centric crop** (padded bbox around hand) |
| Temporal | 8 frames × stride 8 ≈ 2.1s | Same |

## Actions

| Action | Description |
|--------|-------------|
| `commit` | Person completes the intended interaction |
| `hesitate` | Person shows uncertainty before acting |
| `abort` | Person starts but cancels the interaction |
| `waiting` | Idle / no active interaction |

## Project Structure

```
hoi_pipeline/
├── config.py        # Hyperparameters, paths, hand skeleton graph
├── preprocess.py    # MediaPipe hand landmark extraction → .npz cache
├── dataset.py       # 4-stream PyTorch Dataset with temporal chunking
├── model.py         # Full model: GCN + DINOv3 + Spatial + CrossAttn + Temporal
├── train.py         # Training loop: AMP, EMA, early stopping, class weights
└── utils.py         # Metrics (mAP, F1), EMA, seeding
```

## Setup

```bash
pip install torch torchvision timm mediapipe opencv-python numpy scikit-learn

# Download MediaPipe hand model
wget -O hoi_pipeline/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## Usage

### 1. Preprocess (extract hand landmarks)

```bash
python3 -m hoi_pipeline.preprocess \
  --data_root ./Training \
  --anno_root ./annotations/Training \
  --output_dir ./preprocessed
```

### 2. Train

```bash
python3 -m hoi_pipeline.train \
  --data_root ./Training \
  --anno_root ./annotations/Training \
  --cache_dir ./preprocessed \
  --epochs 100 --batch_size 8
```

## Model Stats

| Metric | Value |
|--------|-------|
| Total parameters | 29.8M |
| Trainable parameters | 8.2M |
| Frozen backbone (DINOv3) | 21.6M |
| Input: temporal chunk | T=8 frames, stride=8 (≈2.1s @30fps) |
| Output | 4-class logits (classification + anticipation) |

## Data Format

**Annotations** (`annotations/Training/<person>/<action>/videoXXX.json`):
```json
{
  "video_id": "Training/Person01/fully_abort_bottle_01/video000",
  "relation_instances": [
    [{"predicate": "waiting"}],
    [{"predicate": "hesitate"}],
    [{"predicate": "commit"}],
    ...
  ]
}
```

**Videos**: `Training/<person>/<action>/videoXXX/cam_832112070255/rgb/*.mp4`
