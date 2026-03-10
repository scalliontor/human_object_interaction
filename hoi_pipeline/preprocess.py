"""
Preprocessing V2: YOLO Pose (full body) + Grounding DINO (object detection).

Extracts:
  - Body pose: 17 COCO keypoints + velocity via YOLO Pose
  - Object bbox: text-guided detection via Grounding DINO
  - Person bbox: from YOLO Pose detection

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
        --output_dir ./preprocessed_v2
"""
import os
import sys
import json
import glob
import argparse
import numpy as np
import cv2
import torch


def load_yolo_pose_model(model_name: str = "yolo11n-pose.pt"):
    """Load YOLO Pose model."""
    from ultralytics import YOLO
    model = YOLO(model_name)
    return model


def load_grounding_dino(model_id: str = "IDEA-Research/grounding-dino-tiny", device: str = "cpu"):
    """Load Grounding DINO model from HuggingFace transformers."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def detect_object_gdino(frame_rgb, processor, model, text_prompt, device="cpu",
                         box_threshold=0.25, text_threshold=0.2):
    """
    Run Grounding DINO on a single frame with a text prompt.
    Returns (bbox [x1,y1,x2,y2] normalized, confidence) or (None, 0.0).
    """
    from PIL import Image

    h, w = frame_rgb.shape[:2]
    pil_image = Image.fromarray(frame_rgb)

    inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(h, w)],
    )[0]

    if len(results["scores"]) == 0:
        return None, 0.0

    # Take highest confidence detection
    best_idx = results["scores"].argmax().item()
    bbox = results["boxes"][best_idx].cpu().numpy()  # [x1, y1, x2, y2] in pixels
    conf = results["scores"][best_idx].item()

    # Normalize
    bbox_norm = np.array([bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h], dtype=np.float32)
    return bbox_norm, conf


def extract_pose_and_object(
    video_path: str,
    yolo_model,
    gdino_processor,
    gdino_model,
    object_prompts: dict,
    gdino_device: str = "cpu",
    sample_rate: int = 1,
):
    """
    Extract body pose (YOLO) and object bbox (Grounding DINO) from video.

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

    # Build combined text prompt for Grounding DINO
    # Join all object descriptions
    combined_prompt = ". ".join(object_prompts.values()) + "."

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

        # ── YOLO Pose ──
        results = yolo_model(frame_rgb, verbose=False, conf=0.3)
        result = results[0]

        person_detected = False
        kpts = np.zeros((17, 3), dtype=np.float32)
        pbbox = np.zeros(4, dtype=np.float32)
        pconf = 0.0

        if result.keypoints is not None and len(result.keypoints) > 0:
            # Take the person with highest confidence
            if result.boxes is not None and len(result.boxes) > 0:
                confs = result.boxes.conf.cpu().numpy()
                best_person_idx = confs.argmax()
                pconf = float(confs[best_person_idx])

                # Keypoints [x, y, conf] normalized
                kpts_data = result.keypoints[best_person_idx].data.cpu().numpy()[0]  # [17, 3]
                # Normalize x, y to [0, 1]
                kpts[:, 0] = kpts_data[:, 0] / w
                kpts[:, 1] = kpts_data[:, 1] / h
                kpts[:, 2] = kpts_data[:, 2]  # confidence

                # Person bbox normalized
                box = result.boxes[best_person_idx].xyxyn.cpu().numpy()[0]  # [x1,y1,x2,y2] normalized
                pbbox = box.astype(np.float32)
                person_detected = True

        pose_list.append(kpts)
        person_bbox_list.append(pbbox)
        person_conf_list.append(pconf)

        # ── Grounding DINO (object detection) ──
        # Run every N frames for speed, interpolate between
        obbox = np.zeros(4, dtype=np.float32)
        oconf = 0.0

        if frame_idx % 5 == 0:  # Run every 5 frames
            # Try each object prompt, keep best
            best_obj_conf = 0.0
            best_obj_bbox = None
            for obj_name, obj_desc in object_prompts.items():
                bbox, conf = detect_object_gdino(
                    frame_rgb, gdino_processor, gdino_model,
                    obj_desc, gdino_device,
                )
                if conf > best_obj_conf:
                    best_obj_conf = conf
                    best_obj_bbox = bbox

            if best_obj_bbox is not None:
                obbox = best_obj_bbox
                oconf = best_obj_conf

        object_bbox_list.append(obbox)
        object_conf_list.append(oconf)

        frame_idx += 1

    cap.release()

    # Stack
    pose_landmarks = np.stack(pose_list)       # [T, 17, 3]
    person_bbox = np.stack(person_bbox_list)    # [T, 4]
    person_conf = np.array(person_conf_list)    # [T]
    object_bbox = np.stack(object_bbox_list)    # [T, 4]
    object_conf = np.array(object_conf_list)    # [T]

    # Fill Grounding DINO gaps (every 5 frames) with nearest
    nonzero_obj = np.where(object_conf > 0)[0]
    if len(nonzero_obj) > 0:
        for i in range(total_frames):
            if i >= len(object_conf):
                break
            if object_conf[i] == 0.0:
                # Find nearest detected frame
                nearest_idx = nonzero_obj[np.abs(nonzero_obj - i).argmin()]
                object_bbox[i] = object_bbox[nearest_idx]
                object_conf[i] = object_conf[nearest_idx] * 0.9  # slight discount

    # Compute velocity
    pose_velocity = np.zeros_like(pose_landmarks)
    pose_velocity[1:] = pose_landmarks[1:] - pose_landmarks[:-1]

    return pose_landmarks, pose_velocity, person_bbox, person_conf, object_bbox, object_conf


def preprocess_dataset(
    data_root: str,
    anno_root: str,
    output_dir: str,
    preferred_camera: str = "cam_832112070255",
):
    """Preprocess all annotated videos."""
    from .config import OBJECT_PROMPTS

    # Load models
    print("Loading YOLO Pose model...")
    yolo_model = load_yolo_pose_model()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Grounding DINO (device={device})...")
    gdino_processor, gdino_model = load_grounding_dino(device=device)

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
                video_path, yolo_model, gdino_processor, gdino_model,
                OBJECT_PROMPTS, device,
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
            obj_rate = (oconf > 0.2).mean() * 100
            print(f"  [{idx+1}/{len(anno_files)}] OK: {rel_id} "
                  f"({len(pconf)} frames, {person_rate:.1f}% person, {obj_rate:.1f}% object)")

        except Exception as e:
            print(f"  [{idx+1}/{len(anno_files)}] ERROR: {rel_id}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess V2: YOLO Pose + Grounding DINO")
    parser.add_argument("--data_root", type=str, required=True, help="Training/ dir")
    parser.add_argument("--anno_root", type=str, required=True, help="annotations/Training/ dir")
    parser.add_argument("--output_dir", type=str, required=True, help="Output preprocessed dir")
    parser.add_argument("--camera", type=str, default="cam_832112070255")
    args = parser.parse_args()

    preprocess_dataset(args.data_root, args.anno_root, args.output_dir, args.camera)


if __name__ == "__main__":
    main()
