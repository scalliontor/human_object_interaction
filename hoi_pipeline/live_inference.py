"""
Live Realtime HOI Inference.

Runs YOLOv26 Pose + Custom YOLO OD + HOI model on live camera/video feed.
Displays body skeleton, object bbox, and action prediction bars in realtime.

Usage:
    # Webcam
    python3 -m hoi_pipeline.live_inference --checkpoint ./checkpoints_v2/best_model.pt

    # Video file
    python3 -m hoi_pipeline.live_inference --checkpoint ./checkpoints_v2/best_model.pt --source video.mp4

    # Save output
    python3 -m hoi_pipeline.live_inference --checkpoint ./checkpoints_v2/best_model.pt --source video.mp4 --save output.mp4
"""
import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from collections import deque
from torchvision import transforms

from .config import ModelConfig, TrainingConfig, PREDICATES, BODY_SKELETON_EDGES, get_config
from .model import HOIModel


# ── Colors ──
ACTION_COLORS = {
    "commit": (0, 200, 0),
    "hesitate": (0, 200, 255),
    "abort": (0, 0, 255),
    "waiting": (200, 200, 200),
}

SKELETON_COLOR = (0, 255, 255)
OBJECT_COLOR = (255, 0, 255)
PERSON_COLOR = (0, 255, 0)


def load_models(checkpoint_path: str, pose_model_path: str, obj_model_path: str, device: torch.device):
    """Load HOI model + YOLOv26 Pose + custom YOLO OD."""
    # ── HOI Model ──
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg_dict = ckpt.get("model_config", {})
    cfg = ModelConfig(**{k: v for k, v in model_cfg_dict.items()
                         if k in ModelConfig.__dataclass_fields__})
    hoi_model = HOIModel(cfg).to(device)
    hoi_model.load_state_dict(ckpt["model_state_dict"])
    hoi_model.eval()
    print(f"HOI model loaded (epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?'):.4f})")

    # ── YOLOv26 Pose ──
    from ultralytics import YOLO
    yolo_pose = YOLO(pose_model_path)
    print(f"YOLO Pose loaded: {pose_model_path}")

    # ── Custom YOLO Object Detection ──
    yolo_obj = YOLO(obj_model_path)
    print(f"YOLO OD loaded: {obj_model_path} (classes: {yolo_obj.names})")

    return hoi_model, cfg, yolo_pose, yolo_obj


def detect_person_yolo(frame_rgb, yolo_pose, w, h):
    """YOLOv26 Pose → keypoints [17,3] + person bbox [4] + confidence."""
    results = yolo_pose(frame_rgb, verbose=False, conf=0.3)
    result = results[0]

    kpts = np.zeros((17, 3), dtype=np.float32)
    pbbox = np.zeros(4, dtype=np.float32)
    pconf = 0.0

    if result.keypoints is not None and len(result.keypoints) > 0:
        if result.boxes is not None and len(result.boxes) > 0:
            confs = result.boxes.conf.cpu().numpy()
            best_idx = confs.argmax()
            pconf = float(confs[best_idx])

            kpts_data = result.keypoints[best_idx].data.cpu().numpy()[0]
            kpts[:, 0] = kpts_data[:, 0] / w
            kpts[:, 1] = kpts_data[:, 1] / h
            kpts[:, 2] = kpts_data[:, 2]

            box = result.boxes[best_idx].xyxyn.cpu().numpy()[0]
            pbbox = box.astype(np.float32)

    return kpts, pbbox, pconf


def detect_object_yolo(frame_rgb, yolo_obj, w, h):
    """Custom YOLO OD → object bbox [4] + confidence + class name."""
    results = yolo_obj(frame_rgb, verbose=False, conf=0.3)
    result = results[0]

    obbox = np.zeros(4, dtype=np.float32)
    oconf = 0.0
    oclass = ""

    if result.boxes is not None and len(result.boxes) > 0:
        confs = result.boxes.conf.cpu().numpy()
        best_idx = confs.argmax()
        oconf = float(confs[best_idx])
        box = result.boxes[best_idx].xyxyn.cpu().numpy()[0]
        obbox = box.astype(np.float32)
        cls_id = int(result.boxes[best_idx].cls.cpu().numpy()[0])
        oclass = result.names.get(cls_id, "")

    return obbox, oconf, oclass


def compute_spatial_12(pbbox, obbox, pconf, oconf):
    """Compute 12-dim person-object spatial features."""
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

    dx = ocx - pcx
    dy = ocy - pcy
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


def draw_overlay(frame, kpts, pbbox, pconf, obbox, oconf, oclass,
                 cls_probs, antic_probs, fps, w, h):
    """Draw body skeleton, object bbox, and prediction bars on frame."""
    # ── Body skeleton ──
    if pconf > 0.3:
        for i, j in BODY_SKELETON_EDGES:
            if kpts[i, 2] > 0.3 and kpts[j, 2] > 0.3:
                pt1 = (int(kpts[i, 0] * w), int(kpts[i, 1] * h))
                pt2 = (int(kpts[j, 0] * w), int(kpts[j, 1] * h))
                cv2.line(frame, pt1, pt2, SKELETON_COLOR, 2)
        for k in range(17):
            if kpts[k, 2] > 0.3:
                pt = (int(kpts[k, 0] * w), int(kpts[k, 1] * h))
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)
        x1, y1 = int(pbbox[0] * w), int(pbbox[1] * h)
        x2, y2 = int(pbbox[2] * w), int(pbbox[3] * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), PERSON_COLOR, 2)
        cv2.putText(frame, f"Person {pconf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, PERSON_COLOR, 1)

    # ── Object bbox ──
    if oconf > 0.2:
        x1, y1 = int(obbox[0] * w), int(obbox[1] * h)
        x2, y2 = int(obbox[2] * w), int(obbox[3] * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), OBJECT_COLOR, 2)
        cv2.putText(frame, f"{oclass} {oconf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, OBJECT_COLOR, 1)

    # ── Prediction bars ──
    panel_h = 220
    cv2.rectangle(frame, (5, 5), (320, 5 + panel_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (5, 5), (320, 5 + panel_h), (100, 100, 100), 1)

    y_off = 25
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_off += 25

    cv2.putText(frame, "Classification:", (15, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y_off += 20

    for i, name in enumerate(PREDICATES):
        prob = cls_probs[i]
        color = ACTION_COLORS[name]
        bar_w = int(prob * 180)
        cv2.rectangle(frame, (15, y_off - 10), (15 + bar_w, y_off + 4), color, -1)
        cv2.putText(frame, f"{name}: {prob:.2f}", (200, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y_off += 20

    y_off += 5
    cv2.putText(frame, "Anticipation:", (15, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y_off += 20

    for i, name in enumerate(PREDICATES):
        prob = antic_probs[i]
        color = ACTION_COLORS[name]
        bar_w = int(prob * 180)
        cv2.rectangle(frame, (15, y_off - 10), (15 + bar_w, y_off + 4), color, -1)
        cv2.putText(frame, f"{name}: {prob:.2f}", (200, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y_off += 20

    return frame


def run_live(source, checkpoint_path, pose_model_path, obj_model_path, save_path=None):
    """Main live inference loop with sliding buffer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hoi_model, model_cfg, yolo_pose, yolo_obj = load_models(
        checkpoint_path, pose_model_path, obj_model_path, device
    )

    _, _, train_cfg = get_config()
    T = train_cfg.chunk_length
    frame_stride = train_cfg.frame_stride

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if source is None or source == "0":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {source}")
        return

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source: {source or 'webcam'} ({w}x{h} @ {fps_src:.0f}fps)")

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps_src, (w, h))

    # Sliding buffer
    buf_kpts = deque(maxlen=T * frame_stride)
    buf_vel = deque(maxlen=T * frame_stride)
    buf_pconf = deque(maxlen=T * frame_stride)
    buf_pbbox = deque(maxlen=T * frame_stride)
    buf_obbox = deque(maxlen=T * frame_stride)
    buf_oconf = deque(maxlen=T * frame_stride)
    buf_frames = deque(maxlen=T * frame_stride)

    smooth_cls = np.zeros(len(PREDICATES), dtype=np.float32)
    smooth_antic = np.zeros(len(PREDICATES), dtype=np.float32)
    ema_alpha = 0.3

    frame_count = 0
    t_start = time.time()
    last_oclass = ""

    print("Running live inference... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── YOLOv26 Pose (every frame) ──
        kpts, pbbox, pconf = detect_person_yolo(frame_rgb, yolo_pose, w, h)

        # ── Custom YOLO OD (every frame — fast!) ──
        obbox, oconf, oclass = detect_object_yolo(frame_rgb, yolo_obj, w, h)
        if oclass:
            last_oclass = oclass

        # ── Velocity ──
        vel = kpts - buf_kpts[-1] if len(buf_kpts) > 0 else np.zeros_like(kpts)

        # ── Buffer ──
        buf_kpts.append(kpts)
        buf_vel.append(vel)
        buf_pconf.append(pconf)
        buf_pbbox.append(pbbox)
        buf_obbox.append(obbox)
        buf_oconf.append(oconf)
        buf_frames.append(frame_rgb.copy())

        # ── HOI prediction when buffer full ──
        if len(buf_kpts) >= T * frame_stride:
            indices = list(range(0, T * frame_stride, frame_stride))

            lm_c = np.stack([buf_kpts[i] for i in indices])
            vel_c = np.stack([buf_vel[i] for i in indices])
            pconf_c = np.array([buf_pconf[i] for i in indices])
            pbbox_c = np.stack([buf_pbbox[i] for i in indices])
            obbox_c = np.stack([buf_obbox[i] for i in indices])
            oconf_c = np.array([buf_oconf[i] for i in indices])
            frames_c = [buf_frames[i] for i in indices]

            hand_features = np.concatenate([lm_c, vel_c], axis=-1)
            spatial = np.stack([
                compute_spatial_12(pbbox_c[t], obbox_c[t], pconf_c[t], oconf_c[t])
                for t in range(T)
            ])

            obj_crops, ctx_frames, obj_visible = [], [], []
            for t in range(T):
                ctx = cv2.resize(frames_c[t], (224, 224))
                ctx_frames.append(img_transform(ctx))
                if oconf_c[t] > 0.2:
                    bb = obbox_c[t]
                    x1 = max(0, int(bb[0]*w)); y1 = max(0, int(bb[1]*h))
                    x2 = min(w, int(bb[2]*w)); y2 = min(h, int(bb[3]*h))
                    if x2>x1 and y2>y1:
                        crop = cv2.resize(frames_c[t][y1:y2, x1:x2], (224, 224))
                    else:
                        crop = ctx.copy()
                    obj_crops.append(img_transform(crop))
                    obj_visible.append(1.0)
                else:
                    obj_crops.append(img_transform(ctx.copy()))
                    obj_visible.append(0.0)

            hand_t = torch.from_numpy(hand_features).float().unsqueeze(0).to(device)
            conf_t = torch.from_numpy(pconf_c).float().unsqueeze(0).to(device)
            obj_t = torch.stack(obj_crops).unsqueeze(0).to(device)
            ctx_t = torch.stack(ctx_frames).unsqueeze(0).to(device)
            vis_t = torch.tensor([obj_visible], dtype=torch.float32).to(device)
            spat_t = torch.from_numpy(spatial).float().unsqueeze(0).to(device)

            with torch.no_grad():
                cls_logits, antic_logits = hoi_model(hand_t, conf_t, obj_t, ctx_t, vis_t, spat_t)
                cls_probs = torch.sigmoid(cls_logits).cpu().numpy()[0]
                antic_probs = torch.sigmoid(antic_logits).cpu().numpy()[0]

            smooth_cls = ema_alpha * cls_probs + (1 - ema_alpha) * smooth_cls
            smooth_antic = ema_alpha * antic_probs + (1 - ema_alpha) * smooth_antic

        # ── Overlay ──
        elapsed = time.time() - t_start
        fps_actual = (frame_count + 1) / max(elapsed, 1e-6)
        frame_out = draw_overlay(frame, kpts, pbbox, pconf, obbox, oconf,
                                 last_oclass, smooth_cls, smooth_antic, fps_actual, w, h)

        cv2.imshow("HOI Live Inference", frame_out)
        if writer:
            writer.write(frame_out)

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    if writer:
        writer.release()
        print(f"Output saved: {save_path}")
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames in {elapsed:.1f}s ({fps_actual:.1f} fps)")


def main():
    parser = argparse.ArgumentParser(description="HOI Live Realtime Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--pose-model", type=str, default="yolo26x-pose.pt")
    parser.add_argument("--obj-model", type=str, default="./Training/best.pt")
    args = parser.parse_args()

    run_live(args.source, args.checkpoint, args.pose_model, args.obj_model, args.save)


if __name__ == "__main__":
    main()
