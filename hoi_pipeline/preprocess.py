"""
Preprocessing V2: YOLOv26 Pose + Custom YOLO Object Detection.

Extracts:
  - Body pose: 17 COCO keypoints + velocity via YOLOv26 Pose
  - Object bbox: custom YOLO model (bottle, cake box, plastic bottle)
  - Person bbox: from YOLOv26 Pose detection

Saves per-video .npz with:
  pose_landmarks:  [T, 17, 3]  (x, y, conf)
  pose_velocity:   [T, 17, 3]
  person_bbox:     [T, 4]      (x1, y1, x2, y2 normalized)
  person_conf:     [T]
  object_bbox:     [T, 4]
  object_conf:     [T]

Usage:
    python3 -m hoi_pipeline.preprocess \
        --data_root ./Training \
        --anno_root ./annotations/Training \
        --output_dir ./preprocessed_v2 \
        --pose_model yolo26x-pose.pt \
        --obj_model ./Training/best.pt
"""
import os
import sys
import json
import glob
import argparse
import numpy as np
import cv2


def load_models(pose_model_path: str, obj_model_path: str):
    """Load YOLOv26 Pose + custom YOLO object detection models."""
    from ultralytics import YOLO

    print(f"Loading pose model: {pose_model_path}")
    pose_model = YOLO(pose_model_path)

    print(f"Loading object model: {obj_model_path}")
    obj_model = YOLO(obj_model_path)
    print(f"  Object classes: {obj_model.names}")

    return pose_model, obj_model


def extract_pose_and_object(video_path: str, pose_model, obj_model):
    """
    Extract body pose + object bbox from video using both YOLO models.

    Returns:
        pose_landmarks: [T, 17, 3]
        pose_velocity:  [T, 17, 3]
        person_bbox:    [T, 4]
        person_conf:    [T]
        object_bbox:    [T, 4]
        object_conf:    [T]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pose_list = []
    person_bbox_list = []
    person_conf_list = []
    object_bbox_list = []
    object_conf_list = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # ── YOLOv26 Pose ──
        kpts = np.zeros((17, 3), dtype=np.float32)
        pbbox = np.zeros(4, dtype=np.float32)
        pconf = 0.0

        pose_results = pose_model(frame_rgb, verbose=False, conf=0.3)
        pose_result = pose_results[0]

        if pose_result.keypoints is not None and len(pose_result.keypoints) > 0:
            if pose_result.boxes is not None and len(pose_result.boxes) > 0:
                confs = pose_result.boxes.conf.cpu().numpy()
                best_idx = confs.argmax()
                pconf = float(confs[best_idx])

                kpts_data = pose_result.keypoints[best_idx].data.cpu().numpy()[0]
                kpts[:, 0] = kpts_data[:, 0] / w
                kpts[:, 1] = kpts_data[:, 1] / h
                kpts[:, 2] = kpts_data[:, 2]

                box = pose_result.boxes[best_idx].xyxyn.cpu().numpy()[0]
                pbbox = box.astype(np.float32)

        pose_list.append(kpts)
        person_bbox_list.append(pbbox)
        person_conf_list.append(pconf)

        # ── Custom YOLO Object Detection ──
        obbox = np.zeros(4, dtype=np.float32)
        oconf = 0.0

        obj_results = obj_model(frame_rgb, verbose=False, conf=0.3)
        obj_result = obj_results[0]

        if obj_result.boxes is not None and len(obj_result.boxes) > 0:
            obj_confs = obj_result.boxes.conf.cpu().numpy()
            best_obj_idx = obj_confs.argmax()
            oconf = float(obj_confs[best_obj_idx])
            obj_box = obj_result.boxes[best_obj_idx].xyxyn.cpu().numpy()[0]
            obbox = obj_box.astype(np.float32)

        object_bbox_list.append(obbox)
        object_conf_list.append(oconf)

        frame_idx += 1

    cap.release()

    # Stack
    pose_landmarks = np.stack(pose_list)
    person_bbox = np.stack(person_bbox_list)
    person_conf = np.array(person_conf_list)
    object_bbox = np.stack(object_bbox_list)
    object_conf = np.array(object_conf_list)

    # Compute velocity
    pose_velocity = np.zeros_like(pose_landmarks)
    pose_velocity[1:] = pose_landmarks[1:] - pose_landmarks[:-1]

    return pose_landmarks, pose_velocity, person_bbox, person_conf, object_bbox, object_conf


def preprocess_dataset(
    data_root: str,
    anno_root: str,
    output_dir: str,
    pose_model_path: str,
    obj_model_path: str,
    preferred_camera: str = "cam_832112070255",
):
    """Preprocess all annotated videos."""
    pose_model, obj_model = load_models(pose_model_path, obj_model_path)

    # Find all annotated videos
    anno_files = sorted(glob.glob(os.path.join(anno_root, "**", "*.json"), recursive=True))
    print(f"Found {len(anno_files)} annotation files")

    for idx, anno_path in enumerate(anno_files):
        with open(anno_path) as f:
            anno = json.load(f)

        video_id = anno.get("video_id", "")
        if not video_id:
            continue

        rel_id = video_id.replace("Training/", "", 1) if video_id.startswith("Training/") else video_id

        # Check if already done
        out_path = os.path.join(output_dir, rel_id + ".npz")
        if os.path.exists(out_path):
            print(f"  [{idx+1}/{len(anno_files)}] SKIP (exists): {rel_id}")
            continue

        # Find video file
        video_dir = os.path.join(data_root, rel_id)
        video_path = None
        if os.path.isdir(video_dir):
            cam_dir = os.path.join(video_dir, preferred_camera, "rgb")
            if os.path.isdir(cam_dir):
                mp4s = [f for f in os.listdir(cam_dir) if f.endswith(".mp4")]
                if mp4s:
                    video_path = os.path.join(cam_dir, mp4s[0])

        if video_path is None:
            print(f"  [{idx+1}/{len(anno_files)}] SKIP (no video): {rel_id}")
            continue

        try:
            pose_lm, pose_vel, pbbox, pconf, obbox, oconf = extract_pose_and_object(
                video_path, pose_model, obj_model,
            )

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.savez_compressed(
                out_path,
                pose_landmarks=pose_lm,
                pose_velocity=pose_vel,
                person_bbox=pbbox,
                person_conf=pconf,
                object_bbox=obbox,
                object_conf=oconf,
            )

            person_rate = (pconf > 0.3).mean() * 100
            obj_rate = (oconf > 0.3).mean() * 100
            print(f"  [{idx+1}/{len(anno_files)}] OK: {rel_id} "
                  f"({len(pconf)} frames, {person_rate:.1f}% person, {obj_rate:.1f}% object)")

        except Exception as e:
            print(f"  [{idx+1}/{len(anno_files)}] ERROR: {rel_id}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess V2: YOLOv26 Pose + Custom YOLO OD")
    parser.add_argument("--data_root", type=str, required=True, help="Training/ dir")
    parser.add_argument("--anno_root", type=str, required=True, help="annotations/Training/ dir")
    parser.add_argument("--output_dir", type=str, required=True, help="Output preprocessed dir")
    parser.add_argument("--camera", type=str, default="cam_832112070255")
    parser.add_argument("--pose_model", type=str, default="yolo26x-pose.pt",
                        help="YOLOv26 Pose model path")
    parser.add_argument("--obj_model", type=str, default="./Training/best.pt",
                        help="Custom YOLO object detection model")
    args = parser.parse_args()

    preprocess_dataset(
        args.data_root, args.anno_root, args.output_dir,
        args.pose_model, args.obj_model, args.camera,
    )


if __name__ == "__main__":
    main()
