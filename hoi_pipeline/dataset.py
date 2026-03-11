"""
Dataset V2 for the HOI Training Pipeline.
Loads YOLO Pose + Grounding DINO preprocessed data + RGB frames for all 4 streams.

Returns per chunk:
  hand_features:    [T, 17, 6]       — body pose keypoints + velocity
  hand_confidence:  [T]              — person detection confidence
  obj_crops:        [T, 3, 224, 224] — object crops (from Grounding DINO bbox)
  ctx_frames:       [T, 3, 224, 224] — full context frames (resized)
  obj_visible:      [T]              — object detected flag
  spatial_features: [T, 12]          — person-object relative spatial features
  cls_labels:       [4]              — chunk-level classification
  antic_labels:     [4]              — anticipation labels
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
    TrainingConfig,
    ModelConfig,
)


class HOIChunkDataset(Dataset):
    """
    Full 4-stream dataset V2. Loads YOLO Pose body keypoints, Grounding DINO
    object bboxes, and reads RGB frames for DINOv3 visual streams.
    """

    def __init__(
        self,
        anno_root: str,
        preprocessed_dir: str,
        data_root: str,
        chunk_length: int = 8,
        chunk_stride: int = 4,
        frame_stride: int = 8,
        anticipation_offset: int = 30,
        image_size: int = 224,
        preferred_camera: str = "cam_832112070255",
        video_ids: Optional[List[str]] = None,
        augment: bool = False,
    ):
        self.anno_root = anno_root
        self.preprocessed_dir = preprocessed_dir
        self.data_root = data_root
        self.chunk_length = chunk_length
        self.chunk_stride = chunk_stride
        self.frame_stride = frame_stride
        self.anticipation_offset = anticipation_offset
        self.image_size = image_size
        self.preferred_camera = preferred_camera
        self.augment = augment

        # Image transforms (DINOv3 expects ImageNet-normalized 224×224)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.samples = []
        self._build_index(video_ids)

    def _find_video_path(self, video_id: str) -> Optional[str]:
        """Find mp4 video file from video_id."""
        rel_path = video_id.replace("Training/", "", 1) if video_id.startswith("Training/") else video_id
        video_dir = os.path.join(self.data_root, rel_path)
        if not os.path.isdir(video_dir):
            return None

        for cam_dir_name in sorted(os.listdir(video_dir)):
            cam_path = os.path.join(video_dir, cam_dir_name)
            if not os.path.isdir(cam_path):
                continue
            rgb_dir = os.path.join(cam_path, "rgb")
            if not os.path.isdir(rgb_dir):
                continue
            mp4_files = [f for f in os.listdir(rgb_dir) if f.endswith(".mp4")]
            if mp4_files and self.preferred_camera in cam_dir_name:
                return os.path.join(rgb_dir, mp4_files[0])

        # Fallback
        for cam_dir_name in sorted(os.listdir(video_dir)):
            cam_path = os.path.join(video_dir, cam_dir_name)
            if not os.path.isdir(cam_path):
                continue
            rgb_dir = os.path.join(cam_path, "rgb")
            if not os.path.isdir(rgb_dir):
                continue
            mp4_files = [f for f in os.listdir(rgb_dir) if f.endswith(".mp4")]
            if mp4_files:
                return os.path.join(rgb_dir, mp4_files[0])
        return None

    def _build_index(self, video_ids: Optional[List[str]] = None):
        """Scan annotations, match with preprocessed data, create chunk index."""
        anno_files = []
        for root, dirs, files in os.walk(self.anno_root):
            for f in files:
                if f.endswith(".json"):
                    anno_files.append(os.path.join(root, f))
        anno_files.sort()

        for anno_path in anno_files:
            with open(anno_path, "r") as fp:
                anno = json.load(fp)

            video_id = anno.get("video_id", "")
            if not video_id:
                continue
            if video_ids is not None and video_id not in video_ids:
                continue

            # Check preprocessed file
            rel_id = video_id.replace("Training/", "", 1) if video_id.startswith("Training/") else video_id
            npz_path = os.path.join(self.preprocessed_dir, rel_id + ".npz")
            if not os.path.exists(npz_path):
                continue

            # Find video mp4 path
            video_path = self._find_video_path(video_id)
            if video_path is None:
                continue

            # Parse labels
            relation_instances = anno.get("relation_instances", [])
            n_frames = len(relation_instances)

            min_frames_needed = self.chunk_length * self.frame_stride
            if n_frames < min_frames_needed:
                if n_frames < self.chunk_length:
                    continue

            # Create frame-level labels [n_frames, 4]
            frame_labels = np.zeros((n_frames, len(PREDICATES)), dtype=np.float32)
            for t, frame_rels in enumerate(relation_instances):
                if not frame_rels:
                    frame_labels[t, PREDICATE_TO_IDX["waiting"]] = 1.0
                else:
                    for rel in frame_rels:
                        pred = rel.get("predicate", "")
                        if pred in PREDICATE_TO_IDX:
                            frame_labels[t, PREDICATE_TO_IDX[pred]] = 1.0

            # Anticipation labels (shifted by offset)
            antic_labels = np.zeros_like(frame_labels)
            offset = self.anticipation_offset
            if offset < n_frames:
                antic_labels[:n_frames - offset] = frame_labels[offset:]
                antic_labels[n_frames - offset:] = frame_labels[n_frames - offset:]
            else:
                antic_labels = frame_labels.copy()

            # Generate temporal chunks
            max_start = n_frames - self.chunk_length * self.frame_stride
            if max_start < 0:
                effective_stride = 1
                max_start = n_frames - self.chunk_length
            else:
                effective_stride = self.frame_stride

            # Load person confidence for chunk filtering
            npz_data = np.load(npz_path)
            all_person_conf = npz_data["person_conf"]

            for start in range(0, max(max_start, 0) + 1, self.chunk_stride * effective_stride):
                frame_indices = [
                    min(start + i * effective_stride, n_frames - 1)
                    for i in range(self.chunk_length)
                ]

                # Filter: skip chunks with NO person detected
                chunk_conf = all_person_conf[frame_indices] if max(frame_indices) < len(all_person_conf) else np.zeros(len(frame_indices))
                has_person = np.any(chunk_conf > 0.3)

                if not has_person:
                    continue

                self.samples.append({
                    "npz_path": npz_path,
                    "video_path": video_path,
                    "frame_labels": frame_labels,
                    "antic_labels": antic_labels,
                    "frame_indices": frame_indices,
                    "video_id": video_id,
                    "n_frames": n_frames,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def _read_video_frames(
        self,
        video_path: str,
        frame_indices: List[int],
    ) -> List[np.ndarray]:
        """Read specific frames from video. Returns list of RGB frames."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def _crop_and_resize(
        self, frame: np.ndarray, bbox: np.ndarray, size: int
    ) -> np.ndarray:
        """Crop frame using normalized bbox [x1, y1, x2, y2] and resize."""
        h, w = frame.shape[:2]
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            crop = frame
        else:
            crop = frame[y1:y2, x1:x2]
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        frame_indices = sample["frame_indices"]
        T = self.chunk_length

        # ── Load preprocessed data ──
        data = np.load(sample["npz_path"])
        pose_landmarks = data["pose_landmarks"]    # [total_T, 17, 3]
        pose_velocity = data["pose_velocity"]      # [total_T, 17, 3]
        person_conf = data["person_conf"]           # [total_T]
        person_bbox = data["person_bbox"]           # [total_T, 4]
        object_bbox = data["object_bbox"]           # [total_T, 4]
        object_conf = data["object_conf"]           # [total_T]

        # Index into chunk
        lm_chunk = pose_landmarks[frame_indices]    # [T, 17, 3]
        vel_chunk = pose_velocity[frame_indices]    # [T, 17, 3]
        pconf_chunk = person_conf[frame_indices]    # [T]
        pbbox_chunk = person_bbox[frame_indices]    # [T, 4]
        obbox_chunk = object_bbox[frame_indices]    # [T, 4]
        oconf_chunk = object_conf[frame_indices]    # [T]

        # Concatenate position + velocity → [T, 17, 6]
        hand_features = np.concatenate([lm_chunk, vel_chunk], axis=-1)

        # ── Compute spatial features [T, 12] ──
        spatial = self._compute_spatial_features(
            pbbox_chunk, obbox_chunk, pconf_chunk, oconf_chunk
        )

        # ── Read RGB frames ──
        raw_frames = self._read_video_frames(sample["video_path"], frame_indices)

        # ── Process visual streams ──
        obj_crops = []
        ctx_frames = []
        obj_visible = []

        for t in range(T):
            frame = raw_frames[t]
            ctx = cv2.resize(frame, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_LINEAR)
            ctx_frames.append(self.img_transform(ctx))

            # Object crop from Grounding DINO bbox
            if oconf_chunk[t] > 0.2:
                crop = self._crop_and_resize(frame, obbox_chunk[t], self.image_size)
                obj_crops.append(self.img_transform(crop))
                obj_visible.append(1.0)
            else:
                obj_crops.append(self.img_transform(ctx.copy()))
                obj_visible.append(0.0)

        obj_crops = torch.stack(obj_crops)
        ctx_frames = torch.stack(ctx_frames)
        obj_visible = torch.tensor(obj_visible, dtype=torch.float32)

        # ── Labels ──
        labels = sample["frame_labels"][frame_indices]
        antic_labels = sample["antic_labels"][frame_indices]
        chunk_cls = (labels.sum(axis=0) > 0).astype(np.float32)
        chunk_antic = (antic_labels.sum(axis=0) > 0).astype(np.float32)

        return {
            "hand_features": torch.from_numpy(hand_features).float(),
            "hand_confidence": torch.from_numpy(pconf_chunk).float(),
            "obj_crops": obj_crops,
            "ctx_frames": ctx_frames,
            "obj_visible": obj_visible,
            "spatial_features": torch.from_numpy(spatial).float(),
            "cls_labels": torch.from_numpy(chunk_cls).float(),
            "antic_labels": torch.from_numpy(chunk_antic).float(),
        }

    def _compute_spatial_features(
        self,
        person_bbox: np.ndarray,   # [T, 4] x1,y1,x2,y2 normalized
        object_bbox: np.ndarray,   # [T, 4]
        person_conf: np.ndarray,   # [T]
        object_conf: np.ndarray,   # [T]
    ) -> np.ndarray:
        """
        Compute 12-dim person-object relative spatial features per frame.
        [dx, dy, log_w_ratio, log_h_ratio, log_area_ratio, IoU,
         distance, angle, person_h, person_w, obj_h, obj_w]
        """
        T = person_bbox.shape[0]
        spatial = np.zeros((T, 12), dtype=np.float32)

        for t in range(T):
            if person_conf[t] < 0.3 or object_conf[t] < 0.2:
                continue

            pb = person_bbox[t]
            ob = object_bbox[t]

            # Person center and size
            pcx = (pb[0] + pb[2]) / 2
            pcy = (pb[1] + pb[3]) / 2
            pw = max(pb[2] - pb[0], 1e-6)
            ph = max(pb[3] - pb[1], 1e-6)

            # Object center and size
            ocx = (ob[0] + ob[2]) / 2
            ocy = (ob[1] + ob[3]) / 2
            ow = max(ob[2] - ob[0], 1e-6)
            oh = max(ob[3] - ob[1], 1e-6)

            # Relative displacement
            dx = ocx - pcx
            dy = ocy - pcy

            # Size ratios (log scale)
            log_w_ratio = np.log(ow / pw + 1e-6)
            log_h_ratio = np.log(oh / ph + 1e-6)
            log_area_ratio = np.log((ow * oh) / (pw * ph) + 1e-6)

            # IoU
            ix1 = max(pb[0], ob[0])
            iy1 = max(pb[1], ob[1])
            ix2 = min(pb[2], ob[2])
            iy2 = min(pb[3], ob[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = pw * ph + ow * oh - inter
            iou = inter / max(union, 1e-6)

            # Distance and angle
            distance = np.sqrt(dx ** 2 + dy ** 2)
            angle = np.arctan2(dy, dx)

            spatial[t] = [dx, dy, log_w_ratio, log_h_ratio, log_area_ratio, iou,
                          distance, angle, ph, pw, oh, ow]

        return spatial


def get_video_ids_from_annotations(anno_root: str) -> List[str]:
    """Get all video IDs from annotation files."""
    video_ids = []
    for root, dirs, files in os.walk(anno_root):
        for f in files:
            if f.endswith(".json"):
                with open(os.path.join(root, f), "r") as fp:
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
    """Split video IDs into train/val, stratified by action type."""
    import random as _random
    rng = _random.Random(seed)

    video_ids = get_video_ids_from_annotations(anno_root)

    action_groups: Dict[str, List[str]] = {}
    for vid in video_ids:
        parts = vid.split("/")
        action_key = parts[-2] if len(parts) >= 3 else "unknown"
        action_groups.setdefault(action_key, []).append(vid)

    train_ids, val_ids = [], []
    for action_key, vids in sorted(action_groups.items()):
        rng.shuffle(vids)
        n_val = max(1, int(len(vids) * val_ratio))
        val_ids.extend(vids[:n_val])
        train_ids.extend(vids[n_val:])

    return train_ids, val_ids
