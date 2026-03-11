"""
Inference V2: run HOI model on a single video using preprocessed data.
Uses YOLO Pose + Grounding DINO bboxes from preprocessing (ground truth).

Usage:
    python3 -m hoi_pipeline.inference \
        --video ./Training/Person01/.../rgb/video.mp4 \
        --preprocessed ./preprocessed_v2/Person01/.../videoXXX.npz \
        --checkpoint ./checkpoints_v2/best_model.pt \
        --output ./output.mp4
"""
import os
import sys
import json
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d

from .config import ModelConfig, TrainingConfig, PREDICATES, BODY_SKELETON_EDGES, get_config
from .model import HOIModel


# Colors
ACTION_COLORS = {
    "commit": (0, 200, 0),
    "hesitate": (0, 200, 255),
    "abort": (0, 0, 255),
    "waiting": (200, 200, 200),
}


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained V2 model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg_dict = ckpt.get("model_config", {})
    cfg = ModelConfig(**{k: v for k, v in model_cfg_dict.items()
                         if k in ModelConfig.__dataclass_fields__})
    model = HOIModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    print(f"Val loss: {ckpt.get('val_loss', '?'):.4f}")
    return model, cfg


def compute_spatial_12(pbbox, obbox, pconf, oconf):
    """12-dim person-object spatial features."""
    spatial = np.zeros(12, dtype=np.float32)
    if pconf < 0.3 or oconf < 0.2:
        return spatial
    pcx = (pbbox[0] + pbbox[2]) / 2
    pcy = (pbbox[1] + pbbox[3]) / 2
    pw = max(pbbox[2] - pbbox[0], 1e-6)
    ph = max(pbbox[3] - pbbox[1], 1e-6)
    ocx = (obbox[0] + obbox[2]) / 2
    ocy = (obbox[1] + obbox[3]) / 2
    ow = max(obbox[2] - obbox[0], 1e-6)
    oh = max(obbox[3] - obbox[1], 1e-6)
    dx, dy = ocx - pcx, ocy - pcy
    log_w_ratio = np.log(ow / pw + 1e-6)
    log_h_ratio = np.log(oh / ph + 1e-6)
    log_area_ratio = np.log((ow * oh) / (pw * ph) + 1e-6)
    ix1, iy1 = max(pbbox[0], obbox[0]), max(pbbox[1], obbox[1])
    ix2, iy2 = min(pbbox[2], obbox[2]), min(pbbox[3], obbox[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = pw * ph + ow * oh - inter
    iou = inter / max(union, 1e-6)
    distance = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    return np.array([dx, dy, log_w_ratio, log_h_ratio, log_area_ratio, iou,
                     distance, angle, ph, pw, oh, ow], dtype=np.float32)


def crop_and_resize(frame, bbox, size=224):
    """Crop frame using normalized bbox."""
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0] * w))
    y1 = max(0, int(bbox[1] * h))
    x2 = min(w, int(bbox[2] * w))
    y2 = min(h, int(bbox[3] * h))
    if x2 <= x1 or y2 <= y1:
        crop = frame
    else:
        crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def run_inference(video_path, preprocessed_path, checkpoint_path, output_path=None, device=None):
    """Run V2 inference on a single video using preprocessed data."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, train_cfg = get_config()
    model, model_cfg = load_model(checkpoint_path, device)

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load preprocessed data
    data = np.load(preprocessed_path)
    pose_lm = data["pose_landmarks"]    # [T, 17, 3]
    pose_vel = data["pose_velocity"]    # [T, 17, 3]
    person_conf = data["person_conf"]   # [T]
    person_bbox = data["person_bbox"]   # [T, 4]
    object_bbox = data["object_bbox"]   # [T, 4]
    object_conf = data["object_conf"]   # [T]
    total_frames = len(person_conf)

    person_rate = (person_conf > 0.3).mean() * 100
    obj_rate = (object_conf > 0.2).mean() * 100
    print(f"{total_frames} frames, {person_rate:.1f}% person, {obj_rate:.1f}% object")

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Sliding window inference
    T = train_cfg.chunk_length
    stride = train_cfg.frame_stride

    frame_cls_probs = np.zeros((total_frames, len(PREDICATES)), dtype=np.float32)
    frame_antic_probs = np.zeros((total_frames, len(PREDICATES)), dtype=np.float32)
    frame_counts = np.zeros(total_frames, dtype=np.float32)

    max_start = total_frames - T * stride
    if max_start < 0:
        effective_stride = 1
        max_start = total_frames - T
    else:
        effective_stride = stride

    print("Running model inference...")
    chunk_count = 0

    for start in range(0, max(max_start, 0) + 1, 1 * effective_stride):
        frame_indices = [
            min(start + i * effective_stride, total_frames - 1)
            for i in range(T)
        ]

        lm_chunk = pose_lm[frame_indices]
        vel_chunk = pose_vel[frame_indices]
        pconf_chunk = person_conf[frame_indices]
        pbbox_chunk = person_bbox[frame_indices]
        obbox_chunk = object_bbox[frame_indices]
        oconf_chunk = object_conf[frame_indices]

        hand_features = np.concatenate([lm_chunk, vel_chunk], axis=-1)

        spatial = np.stack([
            compute_spatial_12(pbbox_chunk[t], obbox_chunk[t], pconf_chunk[t], oconf_chunk[t])
            for t in range(T)
        ])

        obj_crops = []
        ctx_frames = []
        obj_visible = []

        for t_idx, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            ctx = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            ctx_frames.append(img_transform(ctx))

            if oconf_chunk[t_idx] > 0.2:
                crop = crop_and_resize(frame_rgb, obbox_chunk[t_idx], 224)
                obj_crops.append(img_transform(crop))
                obj_visible.append(1.0)
            else:
                obj_crops.append(img_transform(ctx.copy()))
                obj_visible.append(0.0)

        hand_t = torch.from_numpy(hand_features).float().unsqueeze(0).to(device)
        conf_t = torch.from_numpy(pconf_chunk).float().unsqueeze(0).to(device)
        obj_t = torch.stack(obj_crops).unsqueeze(0).to(device)
        ctx_t = torch.stack(ctx_frames).unsqueeze(0).to(device)
        vis_t = torch.tensor([obj_visible], dtype=torch.float32).to(device)
        spat_t = torch.from_numpy(spatial).float().unsqueeze(0).to(device)

        with torch.no_grad():
            cls_logits, antic_logits = model(hand_t, conf_t, obj_t, ctx_t, vis_t, spat_t)
            cls_probs = torch.sigmoid(cls_logits).cpu().numpy()[0]
            antic_probs = torch.sigmoid(antic_logits).cpu().numpy()[0]

        for fi in frame_indices:
            frame_cls_probs[fi] += cls_probs
            frame_antic_probs[fi] += antic_probs
            frame_counts[fi] += 1

        chunk_count += 1

    # Average
    mask = frame_counts > 0
    frame_cls_probs[mask] /= frame_counts[mask, np.newaxis]
    frame_antic_probs[mask] /= frame_counts[mask, np.newaxis]

    # Fill uncovered frames
    covered = np.where(mask)[0]
    if len(covered) > 0 and len(covered) < total_frames:
        for c in range(len(PREDICATES)):
            f_cls = interp1d(covered, frame_cls_probs[covered, c], kind='nearest', fill_value='extrapolate')
            f_antic = interp1d(covered, frame_antic_probs[covered, c], kind='nearest', fill_value='extrapolate')
            frame_cls_probs[:, c] = f_cls(np.arange(total_frames))
            frame_antic_probs[:, c] = f_antic(np.arange(total_frames))

    # Temporal smoothing
    smooth_window = 15
    for c in range(len(PREDICATES)):
        frame_cls_probs[:, c] = uniform_filter1d(frame_cls_probs[:, c], size=smooth_window, mode='nearest')
        frame_antic_probs[:, c] = uniform_filter1d(frame_antic_probs[:, c], size=smooth_window, mode='nearest')

    print(f"  Processed {chunk_count} chunks (smoothing={smooth_window})")

    # Write output video
    if out:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            cls_p = frame_cls_probs[frame_idx]
            antic_p = frame_antic_probs[frame_idx]

            # Panel background
            cv2.rectangle(frame, (5, 5), (320, 225), (0, 0, 0), -1)

            y_off = 25
            cv2.putText(frame, f"Frame {frame_idx}/{total_frames}", (15, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_off += 25

            cv2.putText(frame, "Classification:", (15, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y_off += 20
            for i, name in enumerate(PREDICATES):
                prob = cls_p[i]
                color = ACTION_COLORS[name]
                cv2.rectangle(frame, (15, y_off - 10), (15 + int(prob * 180), y_off + 4), color, -1)
                cv2.putText(frame, f"{name}: {prob:.2f}", (200, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_off += 20

            y_off += 5
            cv2.putText(frame, "Anticipation:", (15, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y_off += 20
            for i, name in enumerate(PREDICATES):
                prob = antic_p[i]
                color = ACTION_COLORS[name]
                cv2.rectangle(frame, (15, y_off - 10), (15 + int(prob * 180), y_off + 4), color, -1)
                cv2.putText(frame, f"{name}: {prob:.2f}", (200, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_off += 20

            # Draw body skeleton
            if frame_idx < len(person_conf) and person_conf[frame_idx] > 0.3:
                kpts = pose_lm[frame_idx]
                for i, j in BODY_SKELETON_EDGES:
                    if kpts[i, 2] > 0.3 and kpts[j, 2] > 0.3:
                        pt1 = (int(kpts[i, 0] * width), int(kpts[i, 1] * height))
                        pt2 = (int(kpts[j, 0] * width), int(kpts[j, 1] * height))
                        cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
                for k in range(17):
                    if kpts[k, 2] > 0.3:
                        pt = (int(kpts[k, 0] * width), int(kpts[k, 1] * height))
                        cv2.circle(frame, pt, 3, (0, 0, 255), -1)

                # Person bbox
                pb = person_bbox[frame_idx]
                cv2.rectangle(frame,
                              (int(pb[0]*width), int(pb[1]*height)),
                              (int(pb[2]*width), int(pb[3]*height)),
                              (0, 255, 0), 2)

            # Object bbox
            if frame_idx < len(object_conf) and object_conf[frame_idx] > 0.2:
                ob = object_bbox[frame_idx]
                cv2.rectangle(frame,
                              (int(ob[0]*width), int(ob[1]*height)),
                              (int(ob[2]*width), int(ob[3]*height)),
                              (255, 0, 255), 2)

            out.write(frame)

        out.release()
        print(f"Output video saved to: {output_path}")

    cap.release()

    # Summary
    print("\n=== Prediction Summary ===")
    for i, name in enumerate(PREDICATES):
        avg_cls = frame_cls_probs[mask, i].mean() if mask.any() else 0
        avg_antic = frame_antic_probs[mask, i].mean() if mask.any() else 0
        print(f"  {name}: cls={avg_cls:.3f}, antic={avg_antic:.3f}")

    # Save JSON
    json_path = (output_path or video_path).rsplit(".", 1)[0] + "_predictions.json"
    predictions = {
        "video_path": video_path,
        "total_frames": total_frames,
        "person_detection_rate": float(person_rate / 100),
        "object_detection_rate": float(obj_rate / 100),
        "per_frame_classification": frame_cls_probs.tolist(),
        "per_frame_anticipation": frame_antic_probs.tolist(),
    }
    with open(json_path, "w") as f:
        json.dump(predictions, f)
    print(f"Predictions JSON saved to: {json_path}")

    return frame_cls_probs, frame_antic_probs


def main():
    parser = argparse.ArgumentParser(description="HOI V2 Inference")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--preprocessed", type=str, required=True, help="Path to .npz file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.video)[0]
        args.output = base + "_pred_v2.mp4"

    run_inference(args.video, args.preprocessed, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
