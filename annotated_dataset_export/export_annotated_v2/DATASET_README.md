# VidHOI Annotated Dataset Export v2

This archive contains the annotated subset of the VidHOI dataset, specifically extracted and packaged for server-side training.

## 📊 Dataset Statistics

- **Total Videos:** 73
- **Total Frames:** 25,944
- **Total Labeled Frames:** 19,750 (76.1% of all frames)
- **Resolution:** 640x384
- **FPS:** 30
- **Viewpoint:** `cam_832112070255` (Front/side view)

### 🏷️ Action Class Distribution

The dataset focuses on human intention and object interaction, categorized into 4 predicates (actions):

1. **`commit`**: 7,803 frames (39.5%)
2. **`abort`**: 5,067 frames (25.7%)
3. **`waiting`**: 4,406 frames (22.3%)
4. **`hesitate`**: 2,474 frames (12.5%)

### 👥 Subjects

- `Person01`: 62 videos
- `Person02`: 11 videos

### 🎬 Action Scenarios

The dataset captures various scenarios of humans interacting with specific objects:

- `fully_abort_bottle_01` (plastic bottle): 15 videos
- `fully_abort_bottle_02` (metallic bottle): 15 videos
- `fully_abort_box` (cake box): 13 videos
- `hesitate_abort_bottle_01`: 13 videos
- `hesitate_commit_bottle_02`: 5 videos
- `hesitate_abort_box`: 5 videos
- `fully_commit_bottle_01`: 2 videos
- `fully_commit_bottle_02`: 1 video
- `fully_commit_box`: 1 video
- `fully_abort_cup`: 1 video
- `fully_abort_notebook`: 1 video
- `hesitate_commit_bottle_01`: 1 video

## 📂 Directory Structure

The archive mimics the exact directory structure required by the `hoi_pipeline` scripts:

```
annotated_dataset_export/
├── DATASET_README.md                  # This file
├── Training/                          # Raw Video Files (.mp4)
│   ├── Person01/
│   │   ├── fully_abort_bottle_01/
│   │   │   ├── video000/
│   │   │   │   └── cam_832112070255/
│   │   │   │       └── rgb/
│   │   │   │           └── fully_abort_bottle_01_...mp4
│   │   │   ...
│   ...
└── annotations/                       # Annotation Files (.json)
    └── Training/
        ├── Person01/
        │   ├── fully_abort_bottle_01/
        │   │   ├── video000.json
        │   │   ...
        ...
```

## ⚙️ Model Pipeline Configuration

When preprocessing this dataset using `hoi_pipeline.preprocess`, ensure the following configuration is used so that the custom YOLO Object Detection model filters objects correctly based on the scenario names:

- `bottle_01` scenarios -> Class 2 (`plastic bottle`)
- `bottle_02` scenarios -> Class 0 (`bottle`)
- `box` scenarios -> Class 1 (`cake box`)

## 🚀 Usage

1. Extract the ZIP file directly into your training server's project root directory.
2. The `Training/` and `annotations/` folders will merge seamlessly with your existing directory structure.
3. Run the preprocessing script:
   ```bash
   python3 -m hoi_pipeline.preprocess --data_root ./Training --anno_root ./annotations/Training --output_dir ./preprocessed_v2 --pose_model yolo26x-pose.pt --obj_model ./Training/best.pt
   ```
4. Start training:
   ```bash
   python3 -m hoi_pipeline.train --data_root ./Training --anno_root ./annotations/Training --cache_dir ./preprocessed_v2 --ckpt_dir ./checkpoints_v2
   ```
