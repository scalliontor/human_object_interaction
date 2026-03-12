"""
Dataset V3 for the HOI Training Pipeline.

3-stream data contract:
  pose_features:      [T, 9, 5]        upper-body joints (x,y,conf,dx,dy)
  ctx_frames:         [T, 3, 224, 224] full context frames (for ViT backbone)
  obj_bbox_norm:      [T, 4]           normalized [x1,y1,x2,y2] for ROI pool
  relation_features:  [T, 20]          kinematic / spatial features
  obj_visible:        [T]              object detection confidence flag

Single-label targets:
  label_now:  scalar int64   — state at last frame of clip
  label_f1:   scalar int64   — state at last frame + 15 raw frames (+0.5 s)
  label_f2:   scalar int64   — state at last frame + 30 raw frames (+1.0 s)

Changes from V2:
  - Dropped obj_crops (no separate crop pass)
  - Dropped hand_confidence (conf now a feature channel in pose_features)
  - Dropped cls_labels / antic_labels (multi-label BCE) → single-label CE
  - Added obj_bbox_norm for model-side ROI pooling
  - Richer relation features: 20-dim instead of 12
  - Future horizons in raw frame offsets (0.5 s, 1.0 s) instead of sampled steps
  - Label resolution: max-priority rule (commit > hesitate > abort > waiting)
"""
import os
import json
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional
from torchvision import transforms

from .config import (
    PREDICATES,
    PREDICATE_TO_IDX,
    LABEL_PRIORITY,
    UPPER_BODY_JOINT_INDICES,
    WRIST_L_IDX,
    WRIST_R_IDX,
    RELATION_DIM,
    TrainingConfig,
    ModelConfig,
)


# ════════════════════════════════════════════════════════════════════
# Label helpers
# ════════════════════════════════════════════════════════════════════

def resolve_label(annotations: List[str]) -> int:
    """
    Resolve a list of predicate strings for a single frame into one class index.
    Priority: commit(3) > hesitate(2) > abort(1) > waiting(0).
    Empty annotations → waiting (0).
    """
    if not annotations:
        return PREDICATE_TO_IDX["waiting"]
    valid = [a for a in annotations if a in LABEL_PRIORITY]
    if not valid:
        return PREDICATE_TO_IDX["waiting"]
    best = max(valid, key=lambda a: LABEL_PRIORITY[a])
    return PREDICATE_TO_IDX[best]


def build_frame_label_array(relation_instances: List[List[Dict]], n_frames: int) -> np.ndarray:
    """
    Convert raw relation_instances list into integer label array [n_frames].

    Empty frames (no annotation) → 0 (waiting).
    Multi-predicate frames resolved by priority rule.
    """
    labels = np.zeros(n_frames, dtype=np.int64)
    for t, frame_rels in enumerate(relation_instances):
        preds = [r.get("predicate", "") for r in frame_rels]
        labels[t] = resolve_label(preds)
    return labels


# ════════════════════════════════════════════════════════════════════
# Feature computation helpers
# ════════════════════════════════════════════════════════════════════

def compute_pose_features(
    lm_chunk: np.ndarray,    # [T, 17, 3]  (x, y, conf)
    vel_chunk: np.ndarray,   # [T, 17, 3]  (vx, vy, vconf)
) -> np.ndarray:             # [T, 9, 5]
    """
    Filter to upper-body joints and build [T, 9, 5] pose feature tensor.
    Channels: x, y, conf, dx (velocity x), dy (velocity y).
    """
    # Select upper-body joints from full 17-joint array
    upper_lm  = lm_chunk[:, UPPER_BODY_JOINT_INDICES, :]     # [T, 9, 3]
    upper_vel = vel_chunk[:, UPPER_BODY_JOINT_INDICES, :2]    # [T, 9, 2] (vx, vy only)
    # Concatenate: [x, y, conf, dx, dy]
    pose = np.concatenate([upper_lm, upper_vel], axis=-1)     # [T, 9, 5]
    return pose.astype(np.float32)


def compute_relation_features(
    lm_chunk: np.ndarray,         # [T, 17, 3]  full COCO keypoints
    vel_chunk: np.ndarray,        # [T, 17, 3]
    obj_bbox: np.ndarray,         # [T, 4]  normalized [x1,y1,x2,y2]
    person_bbox: np.ndarray,      # [T, 4]
    ttc_clip: float = 100.0,
) -> np.ndarray:                  # [T, 20]
    """
    Compute 20-dim kinematic relation features per frame.

    Layout (see config.py RELATION_DIM comment for full description):
    [0-1]  dx_wr, dy_wr       right wrist → obj center
    [2-3]  dx_wl, dy_wl       left wrist  → obj center
    [4]    dist_r
    [5]    dist_l
    [6-7]  vx_obj, vy_obj     object velocity
    [8-9]  vx_wr, vy_wr       right wrist velocity
    [10-11] vx_wl, vy_wl      left wrist velocity
    [12]   d_dist_r            approach speed
    [13]   d_dist_l
    [14]   iou
    [15]   overlap
    [16]   ttc_r
    [17]   ttc_l
    [18]   angle               atan2(dy_wr, dx_wr)/π
    [19]   norm_dist           dist_r / sqrt(2)
    """
    T = lm_chunk.shape[0]
    EPS = 1e-6

    # Object center
    obj_cx = (obj_bbox[:, 0] + obj_bbox[:, 2]) / 2.0
    obj_cy = (obj_bbox[:, 1] + obj_bbox[:, 3]) / 2.0

    # Wrist positions (already normalized to [0,1] by preprocess)
    wr_x = lm_chunk[:, WRIST_R_IDX, 0]
    wr_y = lm_chunk[:, WRIST_R_IDX, 1]
    wl_x = lm_chunk[:, WRIST_L_IDX, 0]
    wl_y = lm_chunk[:, WRIST_L_IDX, 1]

    # [0-3] Displacement vectors
    dx_wr = obj_cx - wr_x
    dy_wr = obj_cy - wr_y
    dx_wl = obj_cx - wl_x
    dy_wl = obj_cy - wl_y

    # [4-5] Euclidean distance
    dist_r = np.sqrt(dx_wr ** 2 + dy_wr ** 2)
    dist_l = np.sqrt(dx_wl ** 2 + dy_wl ** 2)

    # Finite-difference velocity helper (zero at first frame)
    def _vel(arr: np.ndarray) -> np.ndarray:
        v = np.zeros_like(arr)
        v[1:] = arr[1:] - arr[:-1]
        return v

    # [6-11] Velocities
    vx_obj = _vel(obj_cx);  vy_obj = _vel(obj_cy)
    vx_wr  = _vel(wr_x);    vy_wr  = _vel(wr_y)
    vx_wl  = _vel(wl_x);    vy_wl  = _vel(wl_y)

    # [12-13] Approach speed (rate of change of distance)
    d_dist_r = _vel(dist_r)
    d_dist_l = _vel(dist_l)

    # [14-15] IoU and overlap
    pb, ob = person_bbox, obj_bbox
    ix1 = np.maximum(pb[:, 0], ob[:, 0])
    iy1 = np.maximum(pb[:, 1], ob[:, 1])
    ix2 = np.minimum(pb[:, 2], ob[:, 2])
    iy2 = np.minimum(pb[:, 3], ob[:, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_p = np.maximum((pb[:, 2]-pb[:, 0]) * (pb[:, 3]-pb[:, 1]), EPS)
    area_o = np.maximum((ob[:, 2]-ob[:, 0]) * (ob[:, 3]-ob[:, 1]), EPS)
    union  = area_p + area_o - inter
    iou     = inter / np.maximum(union, EPS)
    overlap = inter / np.minimum(area_p, area_o)

    # [16-17] Time-to-contact (clipped)
    ttc_r = np.clip(dist_r / (np.abs(d_dist_r) + EPS), 0.0, ttc_clip)
    ttc_l = np.clip(dist_l / (np.abs(d_dist_l) + EPS), 0.0, ttc_clip)

    # [18] Angle (right wrist → object, normalized to [-1, 1])
    angle = np.arctan2(dy_wr, dx_wr) / np.pi

    # [19] Normalized distance (by image diagonal ≈ sqrt(2) in [0,1] coords)
    norm_dist = dist_r / (np.sqrt(2.0) + EPS)

    rel = np.stack([
        dx_wr, dy_wr, dx_wl, dy_wl,
        dist_r, dist_l,
        vx_obj, vy_obj,
        vx_wr,  vy_wr,
        vx_wl,  vy_wl,
        d_dist_r, d_dist_l,
        iou, overlap,
        ttc_r, ttc_l,
        angle, norm_dist,
    ], axis=-1)   # [T, 20]

    return rel.astype(np.float32)


# ════════════════════════════════════════════════════════════════════
# Dataset
# ════════════════════════════════════════════════════════════════════

class HOIChunkDataset(Dataset):
    """
    V3 dataset. Produces temporal clips with 3-stream features and
    3 single-label targets (now / +0.5s / +1.0s).
    """

    def __init__(
        self,
        anno_root: str,
        preprocessed_dir: str,
        data_root: str,
        chunk_length: int = 8,
        chunk_stride: int = 4,
        frame_stride: int = 8,
        future_1_offset: int = 15,   # raw frames (+0.5 s at 30 fps)
        future_2_offset: int = 30,   # raw frames (+1.0 s at 30 fps)
        image_size: int = 224,
        preferred_camera: str = "cam_832112070255",
        video_ids: Optional[List[str]] = None,
        augment: bool = False,
        ttc_clip: float = 100.0,
    ):
        self.anno_root        = anno_root
        self.preprocessed_dir = preprocessed_dir
        self.data_root        = data_root
        self.chunk_length     = chunk_length
        self.chunk_stride     = chunk_stride
        self.frame_stride     = frame_stride
        self.future_1_offset  = future_1_offset
        self.future_2_offset  = future_2_offset
        self.image_size       = image_size
        self.preferred_camera = preferred_camera
        self.augment          = augment
        self.ttc_clip         = ttc_clip

        # ImageNet normalisation (ViT trained with ImageNet stats)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.samples: List[Dict] = []
        self._build_index(video_ids)

    # ── Index building ────────────────────────────────────────────

    def _find_video_path(self, video_id: str) -> Optional[str]:
        """Locate the .mp4 file for a given video_id."""
        rel = video_id.removeprefix("Training/")
        video_dir = os.path.join(self.data_root, rel)
        if not os.path.isdir(video_dir):
            return None

        def _scan(base: str) -> Optional[str]:
            for cam in sorted(os.listdir(base)):
                cam_path = os.path.join(base, cam)
                if not os.path.isdir(cam_path):
                    continue
                rgb_dir = os.path.join(cam_path, "rgb")
                if not os.path.isdir(rgb_dir):
                    continue
                mp4s = [f for f in os.listdir(rgb_dir) if f.endswith(".mp4")]
                if mp4s:
                    return os.path.join(rgb_dir, mp4s[0])
            return None

        # Try preferred camera first
        pref = os.path.join(video_dir, self.preferred_camera)
        if os.path.isdir(pref):
            rgb = os.path.join(pref, "rgb")
            mp4s = [f for f in os.listdir(rgb) if f.endswith(".mp4")] if os.path.isdir(rgb) else []
            if mp4s:
                return os.path.join(rgb, mp4s[0])

        return _scan(video_dir)

    def _build_index(self, video_ids: Optional[List[str]]):
        """Scan annotations and build the clip index."""
        anno_files = []
        for root, _, files in os.walk(self.anno_root):
            for f in files:
                if f.endswith(".json"):
                    anno_files.append(os.path.join(root, f))
        anno_files.sort()

        for anno_path in anno_files:
            with open(anno_path) as fp:
                anno = json.load(fp)

            video_id = anno.get("video_id", "")
            if not video_id:
                continue
            if video_ids is not None and video_id not in video_ids:  # type: ignore[operator]
                continue

            # Check preprocessed .npz
            rel_id = video_id.removeprefix("Training/")
            npz_path = os.path.join(self.preprocessed_dir, rel_id + ".npz")
            if not os.path.exists(npz_path):
                continue

            # Check video file
            video_path = self._find_video_path(video_id)
            if video_path is None:
                continue

            # Build frame-level integer label array
            ri = anno.get("relation_instances", [])
            n_frames = len(ri)
            if n_frames < self.chunk_length:
                continue

            frame_labels = build_frame_label_array(ri, n_frames)  # [n_frames] int

            # Load person confidence for chunk filtering
            npz = np.load(npz_path)
            person_conf = npz["person_conf"]          # [total_T]

            # Determine effective stride
            min_needed = self.chunk_length * self.frame_stride
            eff_stride: int = self.frame_stride if n_frames >= min_needed else 1
            max_start: int = (n_frames - min_needed) if n_frames >= min_needed else (n_frames - self.chunk_length)

            clip_step: int = int(self.chunk_stride) * int(eff_stride)
            step = eff_stride  # local alias keeps checker happy

            for start in range(0, max(max_start, 0) + 1, clip_step):
                frame_idx: List[int] = []
                for i in range(self.chunk_length):
                    frame_idx.append(min(start + i * step, n_frames - 1))

                # Skip clips with no person at all
                chunk_conf = (
                    person_conf[frame_idx]
                    if max(frame_idx) < len(person_conf)
                    else np.zeros(len(frame_idx))
                )
                if not np.any(chunk_conf > 0.3):
                    continue

                # Labels at last frame and two future horizons
                last_raw = frame_idx[-1]
                f1_raw   = min(last_raw + self.future_1_offset, n_frames - 1)
                f2_raw   = min(last_raw + self.future_2_offset, n_frames - 1)

                self.samples.append({
                    "npz_path":    npz_path,
                    "video_path":  video_path,
                    "frame_idx":   frame_idx,
                    "label_now":   int(frame_labels[last_raw]),
                    "label_f1":    int(frame_labels[f1_raw]),
                    "label_f2":    int(frame_labels[f2_raw]),
                    "video_id":    video_id,
                    "n_frames":    n_frames,
                    "last_raw":    last_raw,
                })

    # ── Dataset interface ─────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def get_label_now_list(self) -> List[int]:
        """Return all label_now values — used by make_balanced_sampler."""
        return [s["label_now"] for s in self.samples]

    # ── Frame reading ─────────────────────────────────────────────

    def _read_frames(self, video_path: str, frame_idx: List[int]) -> List[np.ndarray]:
        """Read specific frames from video; returns list of RGB arrays."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 384
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640

        for idx in frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    # ── __getitem__ ───────────────────────────────────────────────

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s   = self.samples[idx]
        T   = self.chunk_length
        fi  = s["frame_idx"]

        # ── Load preprocessed .npz ────────────────────────────────
        npz           = np.load(s["npz_path"])
        lm_all        = npz["pose_landmarks"]   # [total_T, 17, 3]
        vel_all       = npz["pose_velocity"]    # [total_T, 17, 3]
        person_conf   = npz["person_conf"]      # [total_T]
        person_bbox   = npz["person_bbox"]      # [total_T, 4]
        obj_bbox      = npz["object_bbox"]      # [total_T, 4]
        obj_conf      = npz["object_conf"]      # [total_T]

        # Clip to chunk indices
        lm_chunk    = lm_all[fi]                # [T, 17, 3]
        vel_chunk   = vel_all[fi]               # [T, 17, 3]
        pbbox_chunk = person_bbox[fi]           # [T, 4]
        obbox_chunk = obj_bbox[fi]              # [T, 4]
        oconf_chunk = obj_conf[fi]              # [T]

        # ── Stream 1: Pose features [T, 9, 5] ────────────────────
        pose_feat = compute_pose_features(lm_chunk, vel_chunk)

        # ── Stream 3: Relation / kinematic features [T, 20] ──────
        rel_feat = compute_relation_features(
            lm_chunk, vel_chunk,
            obbox_chunk, pbbox_chunk,
            self.ttc_clip,
        )

        # ── Stream 2: Full context frames [T, 3, 224, 224] ───────
        raw_frames = self._read_frames(s["video_path"], fi)
        ctx_list = []
        for frame in raw_frames:
            # Resize full frame to 224×224 (letterbox optional future work)
            ctx = cv2.resize(frame, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_LINEAR)
            ctx_list.append(self.img_transform(ctx))
        ctx_frames = torch.stack(ctx_list)    # [T, 3, 224, 224]

        # ── Object visibility flag ────────────────────────────────
        obj_visible = (oconf_chunk > 0.2).astype(np.float32)  # [T]

        # ── Object bbox (normalized, for ROI pooling in model) ────
        # Keep as-is: already normalized [0,1] from YOLO
        obj_bbox_norm = obbox_chunk.astype(np.float32)          # [T, 4]

        # ── Labels (single-class indices, int64) ──────────────────
        return {
            "pose_features":     torch.from_numpy(pose_feat).float(),      # [T,9,5]
            "ctx_frames":        ctx_frames,                                # [T,3,224,224]
            "obj_bbox_norm":     torch.from_numpy(obj_bbox_norm).float(),   # [T,4]
            "relation_features": torch.from_numpy(rel_feat).float(),        # [T,20]
            "obj_visible":       torch.from_numpy(obj_visible).float(),     # [T]
            "label_now":         torch.tensor(s["label_now"], dtype=torch.long),
            "label_f1":          torch.tensor(s["label_f1"],  dtype=torch.long),
            "label_f2":          torch.tensor(s["label_f2"],  dtype=torch.long),
        }


# ════════════════════════════════════════════════════════════════════
# Split utilities
# ════════════════════════════════════════════════════════════════════

def get_video_ids_from_annotations(anno_root: str) -> List[str]:
    """Collect all video_ids from annotation files."""
    video_ids = []
    for root, _, files in os.walk(anno_root):
        for f in files:
            if f.endswith(".json"):
                with open(os.path.join(root, f)) as fp:
                    anno = json.load(fp)
                vid = anno.get("video_id", "")
                if vid:
                    video_ids.append(vid)
    return sorted(video_ids)


def train_val_split(
    anno_root: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Stratified train/val split by scenario type (not by subject).

    Groups videos by their scenario folder (e.g., 'fully_abort_bottle_01').
    Within each scenario group, holds out val_ratio for validation.
    Single-video scenarios go entirely to train.

    This prevents data leakage from the same physical recording session
    appearing in both train and val.
    """
    import random as _rng
    rng = _rng.Random(seed)

    video_ids = get_video_ids_from_annotations(anno_root)

    # Group by scenario: Training/PersonXX/scenario_name
    scenario_groups: Dict[str, List[str]] = {}
    for vid in video_ids:
        parts = vid.split("/")
        # scenario key = person + scenario, e.g. "Person01/fully_abort_bottle_01"
        if len(parts) >= 3:
            scenario_key = parts[1] + "/" + parts[2]
        else:
            scenario_key = vid
        scenario_groups.setdefault(scenario_key, []).append(vid)

    train_ids: List[str] = []
    val_ids: List[str] = []
    for key, vids in sorted(scenario_groups.items()):
        rng.shuffle(vids)
        n_val = max(1, int(len(vids) * val_ratio)) if len(vids) > 1 else 0
        val_ids.extend(vids[:n_val])
        train_ids.extend(vids[n_val:])

    return train_ids, val_ids
