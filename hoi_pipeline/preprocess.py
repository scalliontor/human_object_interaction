"""
Preprocessing: Extract MediaPipe right hand landmarks + velocity from videos.

Uses MediaPipe Tasks API (v0.10+) with HandLandmarker.

Saves per-video .npz with:
  - landmarks: [T, 21, 3]  — (x, y, z) normalized coords
  - velocity: [T, 21, 3]   — frame-to-frame delta
  - confidence: [T]         — 1.0 if hand detected
  - hand_bbox: [T, 4]      — hand bounding box (x1,y1,x2,y2) normalized
  - frame_indices: [T]      — original frame indices

Usage:
    python -m hoi_pipeline.preprocess --data_root ./Training \\
        --anno_root ./annotations/Training --output_dir ./preprocessed
"""
import os
import json
import argparse
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from typing import Optional, Tuple


# Path to the hand_landmarker.task model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def extract_right_hand_landmarks(
    video_path: str,
    crop_padding: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract right hand landmarks from a video using MediaPipe HandLandmarker (Tasks API).

    Returns:
        landmarks: [T, 21, 3] — (x, y, z) normalized coords
        velocity: [T, 21, 3] — frame-to-frame delta
        confidence: [T] — 1.0 if hand detected
        hand_bbox: [T, 4] — hand bounding box (x1, y1, x2, y2) normalized, padded
    """
    # Create HandLandmarker
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.2,
        min_hand_presence_confidence=0.2,
        min_tracking_confidence=0.2,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    landmarks_list = []
    confidence_list = []
    bbox_list = []

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR → RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Process frame (VIDEO mode requires timestamp in ms)
            timestamp_ms = int(frame_idx * 1000 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Only take "Right" hand (never "Left").
            detected_hand = None
            if result.hand_landmarks and result.handedness:
                for idx, h in enumerate(result.handedness):
                    if h[0].category_name == "Right":
                        detected_hand = result.hand_landmarks[idx]
                        break

            if detected_hand is not None:
                lm = np.array(
                    [[p.x, p.y, p.z] for p in detected_hand],
                    dtype=np.float32,
                )
                landmarks_list.append(lm)
                confidence_list.append(1.0)

                # Compute hand bounding box with padding
                xs = lm[:, 0]
                ys = lm[:, 1]
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                w = x2 - x1
                h = y2 - y1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                size = max(w, h) * (1 + crop_padding * 2)
                x1_pad = max(0, cx - size / 2)
                y1_pad = max(0, cy - size / 2)
                x2_pad = min(1, cx + size / 2)
                y2_pad = min(1, cy + size / 2)
                bbox_list.append([x1_pad, y1_pad, x2_pad, y2_pad])
            else:
                landmarks_list.append(np.zeros((21, 3), dtype=np.float32))
                confidence_list.append(0.0)
                bbox_list.append([0.0, 0.0, 1.0, 1.0])

            frame_idx += 1

    cap.release()

    landmarks = np.stack(landmarks_list, axis=0)  # [T, 21, 3]
    confidence = np.array(confidence_list, dtype=np.float32)
    hand_bbox = np.array(bbox_list, dtype=np.float32)  # [T, 4]

    # Compute velocity: v[t] = landmarks[t] - landmarks[t-1]
    velocity = np.zeros_like(landmarks)
    velocity[1:] = landmarks[1:] - landmarks[:-1]
    for t in range(1, len(confidence)):
        if confidence[t] < 0.5 or confidence[t - 1] < 0.5:
            velocity[t] = 0.0

    return landmarks, velocity, confidence, hand_bbox


def find_video_path(
    data_root: str,
    video_id: str,
    preferred_camera: str = "cam_832112070255",
) -> Optional[str]:
    """Find the RGB mp4 video file for a given video_id."""
    rel_path = video_id.replace("Training/", "", 1) if video_id.startswith("Training/") else video_id
    video_dir = os.path.join(data_root, rel_path)

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
        if mp4_files and preferred_camera in cam_dir_name:
            return os.path.join(rgb_dir, mp4_files[0])

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


def preprocess_single_video(video_path: str, output_path: str) -> dict:
    """Process a single video: extract landmarks + velocity + bbox, save as .npz."""
    landmarks, velocity, confidence, hand_bbox = extract_right_hand_landmarks(video_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        landmarks=landmarks,
        velocity=velocity,
        confidence=confidence,
        hand_bbox=hand_bbox,
        video_path=video_path,
    )

    return {
        "frame_count": len(confidence),
        "detection_rate": float(np.mean(confidence)),
        "output_path": output_path,
    }


def preprocess_dataset(
    data_root: str,
    anno_root: str,
    output_dir: str,
    preferred_camera: str = "cam_832112070255",
    max_videos: int = -1,
):
    """Preprocess all annotated videos."""
    os.makedirs(output_dir, exist_ok=True)

    anno_files = []
    for root, dirs, files in os.walk(anno_root):
        for f in files:
            if f.endswith(".json"):
                anno_files.append(os.path.join(root, f))

    anno_files.sort()
    if max_videos > 0:
        anno_files = anno_files[:max_videos]

    print(f"Found {len(anno_files)} annotation files to process.")

    stats = {"total": 0, "success": 0, "failed": 0, "skipped": 0}

    for i, anno_path in enumerate(anno_files):
        stats["total"] += 1

        with open(anno_path, "r") as fp:
            anno = json.load(fp)

        video_id = anno.get("video_id", "")
        if not video_id:
            print(f"  [{i+1}/{len(anno_files)}] SKIP (no video_id): {anno_path}")
            stats["skipped"] += 1
            continue

        rel_id = video_id.replace("Training/", "", 1) if video_id.startswith("Training/") else video_id
        npz_path = os.path.join(output_dir, rel_id + ".npz")

        if os.path.exists(npz_path):
            print(f"  [{i+1}/{len(anno_files)}] CACHED: {rel_id}")
            stats["success"] += 1
            continue

        video_path = find_video_path(data_root, video_id, preferred_camera)
        if video_path is None:
            print(f"  [{i+1}/{len(anno_files)}] NOT FOUND: {video_id}")
            stats["failed"] += 1
            continue

        try:
            info = preprocess_single_video(video_path, npz_path)
            print(
                f"  [{i+1}/{len(anno_files)}] OK: {rel_id} "
                f"({info['frame_count']} frames, {info['detection_rate']:.1%} hand detected)"
            )
            stats["success"] += 1
        except Exception as e:
            print(f"  [{i+1}/{len(anno_files)}] ERROR: {rel_id} — {e}")
            import traceback
            traceback.print_exc()
            stats["failed"] += 1

    print(f"\nDone! {stats['success']}/{stats['total']} processed "
          f"({stats['failed']} failed, {stats['skipped']} skipped)")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess: extract MediaPipe hand landmarks")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--anno_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--preferred_camera", type=str, default="cam_832112070255")
    parser.add_argument("--max_videos", type=int, default=-1)
    args = parser.parse_args()

    preprocess_dataset(
        data_root=args.data_root,
        anno_root=args.anno_root,
        output_dir=args.output_dir,
        preferred_camera=args.preferred_camera,
        max_videos=args.max_videos,
    )


if __name__ == "__main__":
    main()
