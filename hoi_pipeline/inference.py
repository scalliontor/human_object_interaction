"""
Inference script: run HOI model on a single video and visualize predictions.

Usage:
    python3 -m hoi_pipeline.inference \
        --video ./Training/Person01/fully_abort_bottle_01/video000/cam_832112070255/rgb/video.mp4 \
        --checkpoint ./checkpoints/best_model.pt \
        --output ./output_predictions.mp4
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

from .config import ModelConfig, TrainingConfig, PREDICATES, get_config
from .model import HOIModel
from .preprocess import extract_right_hand_landmarks


# Colors for each action (BGR)
ACTION_COLORS = {
    "commit": (0, 200, 0),      # Green
    "hesitate": (0, 200, 255),   # Orange
    "abort": (0, 0, 255),        # Red
    "waiting": (200, 200, 200),  # Gray
}


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
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


def compute_spatial_features(landmarks, velocity, confidence):
    """Compute 7-dim spatial features for a single frame."""
    spatial = np.zeros(7, dtype=np.float32)
    if confidence < 0.5:
        return spatial

    lm = landmarks
    cx = lm[:, 0].mean()
    cy = lm[:, 1].mean()
    dx = cx - 0.5
    dy = cy - 0.5
    x_range = max(lm[:, 0].max() - lm[:, 0].min(), 1e-6)
    y_range = max(lm[:, 1].max() - lm[:, 1].min(), 1e-6)
    log_w_ratio = np.log(x_range + 1e-6)
    log_h_ratio = np.log(y_range + 1e-6)
    hand_area = x_range * y_range
    iou_proxy = min(hand_area, 1.0)
    distance = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx)
    spatial = np.array([dx, dy, log_w_ratio, log_h_ratio, iou_proxy, distance, angle],
                       dtype=np.float32)
    return spatial


def crop_and_resize(frame, bbox, size=224):
    """Crop frame using normalized bbox and resize."""
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


def run_inference(
    video_path: str,
    checkpoint_path: str,
    output_path: str = None,
    device: torch.device = None,
):
    """Run HOI inference on a single video."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, train_cfg = get_config()

    # Load model
    model, model_cfg = load_model(checkpoint_path, device)

    # Image transform
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Extract hand landmarks
    print(f"Extracting hand landmarks from: {video_path}")
    landmarks, velocity, confidence, hand_bbox = extract_right_hand_landmarks(video_path)
    total_frames = len(confidence)
    print(f"  {total_frames} frames, {np.mean(confidence)*100:.1f}% hand detected")

    # Open video for reading frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video writer
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Sliding window inference
    T = train_cfg.chunk_length
    stride = train_cfg.frame_stride
    chunk_stride = train_cfg.chunk_stride

    # Store per-frame predictions (accumulate from overlapping chunks)
    frame_cls_probs = np.zeros((total_frames, len(PREDICATES)), dtype=np.float32)
    frame_antic_probs = np.zeros((total_frames, len(PREDICATES)), dtype=np.float32)
    frame_counts = np.zeros(total_frames, dtype=np.float32)

    # Process chunks
    max_start = total_frames - T * stride
    if max_start < 0:
        effective_stride = 1
        max_start = total_frames - T
    else:
        effective_stride = stride

    print("Running model inference...")
    chunk_count = 0

    for start in range(0, max(max_start, 0) + 1, 1 * effective_stride):  # stride=1 for dense inference
        frame_indices = [
            min(start + i * effective_stride, total_frames - 1)
            for i in range(T)
        ]

        # Prepare hand features
        lm_chunk = landmarks[frame_indices]
        vel_chunk = velocity[frame_indices]
        conf_chunk = confidence[frame_indices]
        bbox_chunk = hand_bbox[frame_indices]

        hand_features = np.concatenate([lm_chunk, vel_chunk], axis=-1)  # [T, 21, 6]

        # Spatial features
        spatial = np.stack([
            compute_spatial_features(lm_chunk[t], vel_chunk[t], conf_chunk[t])
            for t in range(T)
        ])

        # Read RGB frames for visual streams
        obj_crops = []
        ctx_frames = []
        obj_visible = []

        for t_idx, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Context: full frame
            ctx = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            ctx_frames.append(img_transform(ctx))

            # Object crop
            if conf_chunk[t_idx] > 0.5:
                crop = crop_and_resize(frame_rgb, bbox_chunk[t_idx], 224)
                obj_crops.append(img_transform(crop))
                obj_visible.append(1.0)
            else:
                obj_crops.append(img_transform(ctx.copy()))
                obj_visible.append(0.0)

        # Stack and add batch dim
        hand_t = torch.from_numpy(hand_features).float().unsqueeze(0).to(device)
        conf_t = torch.from_numpy(conf_chunk).float().unsqueeze(0).to(device)
        obj_t = torch.stack(obj_crops).unsqueeze(0).to(device)
        ctx_t = torch.stack(ctx_frames).unsqueeze(0).to(device)
        vis_t = torch.tensor([obj_visible], dtype=torch.float32).to(device)
        spat_t = torch.from_numpy(spatial).float().unsqueeze(0).to(device)

        with torch.no_grad():
            cls_logits, antic_logits = model(hand_t, conf_t, obj_t, ctx_t, vis_t, spat_t)
            cls_probs = torch.sigmoid(cls_logits).cpu().numpy()[0]
            antic_probs = torch.sigmoid(antic_logits).cpu().numpy()[0]

        # Accumulate predictions for all frames in the chunk
        for fi in frame_indices:
            frame_cls_probs[fi] += cls_probs
            frame_antic_probs[fi] += antic_probs
            frame_counts[fi] += 1

        chunk_count += 1

    # Average overlapping predictions
    mask = frame_counts > 0
    frame_cls_probs[mask] /= frame_counts[mask, np.newaxis]
    frame_antic_probs[mask] /= frame_counts[mask, np.newaxis]

    # Fill uncovered frames (frame_counts==0) with nearest covered frame's predictions
    covered_indices = np.where(mask)[0]
    if len(covered_indices) > 0 and len(covered_indices) < total_frames:
        from scipy.interpolate import interp1d
        for c in range(len(PREDICATES)):
            # Interpolate/extrapolate from covered frames
            f_cls = interp1d(covered_indices, frame_cls_probs[covered_indices, c],
                             kind='nearest', fill_value='extrapolate')
            f_antic = interp1d(covered_indices, frame_antic_probs[covered_indices, c],
                               kind='nearest', fill_value='extrapolate')
            all_indices = np.arange(total_frames)
            frame_cls_probs[:, c] = f_cls(all_indices)
            frame_antic_probs[:, c] = f_antic(all_indices)

    # Temporal smoothing — moving average to remove jitter
    from scipy.ndimage import uniform_filter1d
    smooth_window = 15  # ~0.5 second at 30fps
    for c in range(len(PREDICATES)):
        frame_cls_probs[:, c] = uniform_filter1d(frame_cls_probs[:, c],
                                                  size=smooth_window, mode='nearest')
        frame_antic_probs[:, c] = uniform_filter1d(frame_antic_probs[:, c],
                                                    size=smooth_window, mode='nearest')

    print(f"  Processed {chunk_count} chunks (smoothing window={smooth_window} frames)")

    # Write output video with overlay
    if output_path:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            cls_p = frame_cls_probs[frame_idx]
            antic_p = frame_antic_probs[frame_idx]

            # Draw prediction overlay
            y_offset = 30
            cv2.putText(frame, f"Frame {frame_idx}/{total_frames}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30

            cv2.putText(frame, "Classification:", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 22

            for i, pred_name in enumerate(PREDICATES):
                prob = cls_p[i]
                color = ACTION_COLORS[pred_name]
                bar_width = int(prob * 200)
                label = f"{pred_name}: {prob:.2f}"

                cv2.rectangle(frame, (10, y_offset - 12), (10 + bar_width, y_offset + 4),
                              color, -1)
                cv2.putText(frame, label, (220, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                y_offset += 22

            y_offset += 10
            cv2.putText(frame, "Anticipation:", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 22

            for i, pred_name in enumerate(PREDICATES):
                prob = antic_p[i]
                color = ACTION_COLORS[pred_name]
                bar_width = int(prob * 200)
                label = f"{pred_name}: {prob:.2f}"

                cv2.rectangle(frame, (10, y_offset - 12), (10 + bar_width, y_offset + 4),
                              color, -1)
                cv2.putText(frame, label, (220, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                y_offset += 22

            # Draw hand landmarks if detected
            if confidence[frame_idx] > 0.5:
                lm = landmarks[frame_idx]
                for j in range(21):
                    x = int(lm[j, 0] * width)
                    y = int(lm[j, 1] * height)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                # Draw hand bbox
                bb = hand_bbox[frame_idx]
                x1 = int(bb[0] * width)
                y1 = int(bb[1] * height)
                x2 = int(bb[2] * width)
                y2 = int(bb[3] * height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            out.write(frame)

        out.release()
        print(f"Output video saved to: {output_path}")

    cap.release()

    # Print summary
    print("\n=== Prediction Summary ===")
    for i, pred_name in enumerate(PREDICATES):
        avg_cls = frame_cls_probs[mask, i].mean()
        avg_antic = frame_antic_probs[mask, i].mean()
        print(f"  {pred_name}: cls={avg_cls:.3f}, antic={avg_antic:.3f}")

    # Save predictions JSON
    json_path = (output_path or video_path).rsplit(".", 1)[0] + "_predictions.json"
    predictions = {
        "video_path": video_path,
        "total_frames": total_frames,
        "detection_rate": float(np.mean(confidence)),
        "per_frame_classification": frame_cls_probs.tolist(),
        "per_frame_anticipation": frame_antic_probs.tolist(),
    }
    with open(json_path, "w") as f:
        json.dump(predictions, f)
    print(f"Predictions JSON saved to: {json_path}")

    return frame_cls_probs, frame_antic_probs


def main():
    parser = argparse.ArgumentParser(description="HOI Inference on a single video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.video)[0]
        args.output = base + "_predictions.mp4"

    run_inference(args.video, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
