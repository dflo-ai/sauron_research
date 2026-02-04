# ReID Configuration Parameters Reference

Comprehensive documentation for 46 pipeline parameters across 5 config classes.

## Overview

The ReID pipeline uses YAML configuration with Pydantic validation. Parameters control model selection, inference behavior, gallery matching, output generation, and visualization.

**Config File:** `config.yaml`
**Source:** `src/reid_research/config.py`

```yaml
model:
  reid_variant: "osnet_x1_0"
  device: "cuda"
inference:
  confidence_threshold: 0.75
gallery:
  use_full_reranking: false
output:
  save_video: true
visualization:
  show_gallery_panel: true
```

---

## Pipeline I/O Specification

| Stage | Input | Output | Data Shape | Key Config |
|-------|-------|--------|-----------|------------|
| **1. Detection** | Frame (BGR) | List[Detection] N items | (H,W,3) → N×(bbox,conf,orientation) | `confidence_threshold`≥0.75 |
| **2. Extraction** | Crops list, batch=32 | Features matrix | N×(256,128) → (N,512) float32 | `image_size`, `batch_size` |
| **3. Assignment** | Features (N,512), gallery M | Assignments (N,) | (N,512) × (M,512) → (N,) IDs | `rank_list_size`=20, `similarity_threshold` |
| **4. Motion Valid.** | Assignments, track history | Validated assignments | (N,) → (N,) bool mask | `motion_max_distance`<150px |
| **5. ID Assignment** | Validated, detections | Detection.track_id | (N,) matched IDs | `min_frames_for_id`=5 |
| **6. Gallery Update** | Detections+features | GalleryEntry state | EMA: α×qual×feat | `ema_alpha`=0.7, `quality_min_threshold`>0.3 |

---

## 1. Model Configuration

**Class:** `ModelConfig` | **Source:** `config.py:8-16`

### 1.1 reid_variant

- **Type:** `string`
- **Default:** `"osnet_x1_0"`
- **Valid Values:** `osnet_x1_0`, `osnet_x0_75`, `osnet_x0_5`, `osnet_x0_25`
- **Code Location:** `config.py:11`, `feature_extractor.py:31`
- **Academic Reference:** Zhou et al., "Omni-Scale Feature Learning for Person Re-Identification", ICCV 2019
- **Impact:**
  - `osnet_x1_0`: Best accuracy (mAP 94.8% Market-1501), 2.2M params
  - `osnet_x0_5`: 50% width, faster inference, reduced accuracy
  - `osnet_x0_25`: 4× faster, minimal accuracy
- **Notes:** All variants output 512-dim L2-normalized features

### 1.2 reid_weights

- **Type:** `string | None`
- **Default:** `None` (auto-download pretrained)
- **Code Location:** `config.py:12`, `feature_extractor.py:32`
- **Impact:** Custom weights override pretrained; None uses torchreid defaults
- **Notes:** Path to `.pth` file for custom-trained models

### 1.3 yolo_weights

- **Type:** `string`
- **Default:** `"data/weights/jointbdoe_m.pt"`
- **Code Location:** `config.py:15`, `jointbdoe_detector.py:65`
- **Academic Reference:** Hu et al., "Joint Multi-Person Body Detection and Orientation Estimation", PRCV 2024
- **Impact:** Model variants: S (mobile), M (balanced), L (max accuracy)
- **Notes:** JointBDOE provides bbox + body orientation in single inference

### 1.4 device

- **Type:** `string`
- **Default:** `"cuda"`
- **Valid Values:** `"cuda"`, `"cpu"`, `"cuda:0"`, `"cuda:1"`, etc.
- **Code Location:** `config.py:13`, `feature_extractor.py:34`, `jointbdoe_detector.py:69`
- **Impact:** GPU required for real-time (30+ FPS); CPU ~5 FPS

---

## 2. Inference Configuration

**Class:** `InferenceConfig` | **Source:** `config.py:18-25`

### 2.1 image_size

- **Type:** `tuple[int, int]`
- **Default:** `(256, 128)` (H, W)
- **Code Location:** `config.py:21`, `feature_extractor.py:33`
- **Academic Reference:** Zhou et al., ICCV 2019 (256×128 de facto standard)
- **Impact:**
  - ↑ Size: Better accuracy, slower inference
  - ↓ Size: Faster, reduced accuracy
- **Notes:** Match with reid_variant for optimal results

### 2.2 batch_size

- **Type:** `int`
- **Default:** `32`
- **Valid Range:** `1-128`
- **Code Location:** `config.py:22`, `feature_extractor.py:55`
- **Impact:**
  - ↑ Batch: Better GPU utilization, higher VRAM
  - ↓ Batch: Lower latency per detection, less throughput
- **Notes:** Reduce if OOM errors occur

### 2.3 confidence_threshold

- **Type:** `float`
- **Default:** `0.75`
- **Valid Range:** `0.0-1.0`
- **Code Location:** `config.py:23`, `jointbdoe_detector.py:107`
- **Academic Reference:** JointBDOE recommends 0.5-0.7 for orientation accuracy
- **Impact:**
  - ↑ Threshold: Fewer false positives, may miss occluded persons
  - ↓ Threshold: More detections, more noise
- **Notes:** 0.75 balances precision/recall for crowded scenes

### 2.4 similarity_threshold

- **Type:** `float`
- **Default:** `0.5`
- **Valid Range:** `0.3-0.9`
- **Code Location:** `config.py:24`, `gallery.py:66,571`
- **Impact:**
  - ↑ Threshold: Fewer matches, lower ID switches, may miss same person
  - ↓ Threshold: More matches, risk false ID merges
- **Notes:** Used as baseline before adaptive threshold kicks in

---

## 3. Gallery Configuration

**Class:** `GalleryConfig` | **Source:** `config.py:27-91`

### 3.1 Feature Storage

#### max_features_per_id

- **Type:** `int`
- **Default:** `10`
- **Valid Range:** `5-50`
- **Code Location:** `config.py:30`, `gallery.py:34,610-612`
- **Impact:**
  - ↑ Features: More robust matching, higher memory
  - ↓ Features: Faster updates, less temporal averaging
- **Notes:** Rolling FIFO buffer per track

### 3.2 EMA Feature Fusion

#### ema_alpha

- **Type:** `float`
- **Default:** `0.7`
- **Valid Range:** `0.5-0.95`
- **Code Location:** `config.py:31`, `gallery.py:663-672`
- **Formula:** `EMA_t = α × feature_t + (1-α) × EMA_{t-1}`
- **Impact:**
  - Higher α (0.9): Fast adaptation, sensitive to noise
  - Lower α (0.5): Stable, slower to adapt to appearance changes
- **Notes:** Balance between temporal stability and responsiveness

### 3.3 Rank-List Voting

#### rank_list_size

- **Type:** `int`
- **Default:** `20`
- **Valid Range:** `10-50`
- **Code Location:** `config.py:34`, `gallery.py:504`, `matching.py:856`
- **Impact:**
  - ↑ k: More votes per ID, better for crowded scenes, slower
  - ↓ k: Faster, less robust to feature variance
- **Notes:** Top-k neighbors participate in majority voting

#### rank_distance_threshold

- **Type:** `float | None`
- **Default:** `None` (auto-computed as median)
- **Code Location:** `config.py:35`, `gallery.py:513`, `matching.py:905`
- **Impact:** Filters neighbors beyond threshold from voting
- **Notes:** None = adaptive based on distance distribution

#### rank_min_entries_per_id

- **Type:** `int`
- **Default:** `3`
- **Valid Range:** `2-10`
- **Code Location:** `config.py:36`, `gallery.py:518`, `matching.py:919`
- **Impact:** Minimum gallery entries before voting activates
- **Notes:** Prevents voting with insufficient evidence

#### rank_fallback_threshold

- **Type:** `float`
- **Default:** `1.2`
- **Code Location:** `config.py:37`, `gallery.py:171,176`
- **Impact:** Distance threshold for early matching bypass
- **Notes:** Very close matches skip full voting process

### 3.4 K-Reciprocal Re-ranking

**Academic Reference:** Zhong et al., "Re-Ranking Person Re-Identification with k-Reciprocal Encoding", CVPR 2017

**Algorithm:**
```
D_final = λ × D_original + (1-λ) × D_jaccard
```

#### use_full_reranking

- **Type:** `bool`
- **Default:** `False`
- **Code Location:** `config.py:40`
- **Impact:** +5-15% mAP improvement, 30% slower
- **Notes:** Enable for high-accuracy applications

#### rerank_k1

- **Type:** `int`
- **Default:** `20`
- **Valid Range:** `10-50`
- **Code Location:** `config.py:41`, `matching.py:43,57`
- **Academic Reference:** CVPR 2017 recommends k1=20
- **Impact:** Initial k-nearest neighbors for reciprocal encoding
- **Notes:** Higher = more neighbors, better recall, slower

#### rerank_k2

- **Type:** `int`
- **Default:** `6`
- **Valid Range:** `1-20`
- **Code Location:** `config.py:42`, `matching.py:44,58`
- **Academic Reference:** CVPR 2017 recommends k2=6
- **Impact:** Local query expansion parameter
- **Notes:** k2=1 disables expansion (baseline)

#### rerank_lambda

- **Type:** `float`
- **Default:** `0.3`
- **Valid Range:** `0.0-1.0`
- **Code Location:** `config.py:43`, `matching.py:45,59`
- **Academic Reference:** CVPR 2017 recommends λ=0.3
- **Impact:** Weight for original vs Jaccard distance
- **Notes:** Lower λ = more weight on Jaccard (k-reciprocal)

### 3.5 Quality-Aware Weighting

**Formula:**
```
quality = conf_weight × confidence + geo_weight × geometry_score
geometry_score = (area_score + aspect_ratio_score) / 2
```

#### use_quality_weighting

- **Type:** `bool`
- **Default:** `True`
- **Code Location:** `config.py:46`, `gallery.py:650,663`
- **Impact:** Weights feature updates by detection quality
- **Notes:** Reduces noise from partial/occluded detections

#### quality_min_threshold

- **Type:** `float`
- **Default:** `0.3`
- **Valid Range:** `0.1-0.5`
- **Code Location:** `config.py:47`, `gallery.py:650`
- **Impact:** Skip feature updates below this quality
- **Notes:** Prevents low-quality features from degrading gallery

#### quality_confidence_weight

- **Type:** `float`
- **Default:** `0.6`
- **Code Location:** `config.py:48`, `pipeline.py:177`, `matching.py:419`
- **Impact:** Weight of detection confidence in quality score
- **Notes:** Must sum to 1.0 with geometry_weight

#### quality_geometry_weight

- **Type:** `float`
- **Default:** `0.4`
- **Code Location:** `config.py:49`, `pipeline.py:178`, `matching.py:420`
- **Impact:** Weight of bbox geometry in quality score
- **Notes:** Penalizes non-person-like aspect ratios

#### ideal_aspect_ratio

- **Type:** `float`
- **Default:** `0.4`
- **Code Location:** `config.py:50`, `pipeline.py:176`, `matching.py:418`
- **Impact:** Target W/H ratio for person bboxes
- **Notes:** 0.4 typical for standing persons

#### min_bbox_area

- **Type:** `int`
- **Default:** `2000`
- **Valid Range:** `500-5000`
- **Code Location:** `config.py:51`, `pipeline.py:175`, `matching.py:417`
- **Impact:** Minimum pixels for valid detection
- **Notes:** Filters noise from tiny detections

### 3.6 Velocity-Based Motion Prediction

**Academic Reference:** BoT-SORT, Aharon et al., arXiv:2206.14651, ECCV 2022

**Kalman State:** `[x, y, aspect_ratio, height, vx, vy, va, vh]`

#### use_velocity_prediction

- **Type:** `bool`
- **Default:** `True`
- **Code Location:** `config.py:54`, `gallery.py:621,654,681`
- **Impact:** Enables motion-aware matching
- **Notes:** Improves tracking through occlusions

#### velocity_history_frames

- **Type:** `int`
- **Default:** `5`
- **Valid Range:** `3-10`
- **Code Location:** `config.py:55`, `gallery.py:276`
- **Impact:** Frames to average for velocity estimation
- **Notes:** Higher = smoother, slower to adapt to direction changes

#### velocity_max_speed

- **Type:** `float`
- **Default:** `100.0`
- **Valid Range:** `50-200`
- **Code Location:** `config.py:56`, `gallery.py:395`, `matching.py:471`
- **Impact:** Maximum pixels/frame velocity clamp
- **Notes:** Prevents outliers from corrupting motion model

#### prediction_radius

- **Type:** `float`
- **Default:** `75.0`
- **Valid Range:** `30-150`
- **Code Location:** `config.py:57`, `gallery.py:264,704`
- **Impact:** Match search radius around predicted position
- **Notes:** Adaptive: radius = velocity × frame_gap

#### prediction_boost

- **Type:** `float`
- **Default:** `0.15`
- **Valid Range:** `0.0-0.3`
- **Code Location:** `config.py:58`, `gallery.py:256`, `matching.py:524`
- **Impact:** Similarity bonus for position-matched detections
- **Notes:** Reduces ID switches when appearance similar

### 3.7 Adaptive Threshold

**Academic Reference:** DDTAS, arXiv:2404.19282

**Method:** Percentile-based threshold from match score distribution

#### use_adaptive_threshold

- **Type:** `bool`
- **Default:** `False`
- **Code Location:** `config.py:61`, `gallery.py:731`
- **Impact:** Dynamic threshold adjustment
- **Notes:** Removes need for manual per-scene tuning

#### adaptive_min_threshold

- **Type:** `float`
- **Default:** `0.50`
- **Code Location:** `config.py:62`, `gallery.py:737`, `matching.py:553`
- **Impact:** Floor for adaptive threshold
- **Notes:** Prevents over-adaptation to easy scenes

#### adaptive_max_threshold

- **Type:** `float`
- **Default:** `0.80`
- **Code Location:** `config.py:63`, `gallery.py:738`, `matching.py:554`
- **Impact:** Ceiling for adaptive threshold
- **Notes:** Maintains specificity in difficult scenes

#### adaptive_window_size

- **Type:** `int`
- **Default:** `100`
- **Code Location:** `config.py:64`, `gallery.py:64`
- **Impact:** Match history window for statistics
- **Notes:** Larger = more stable, slower adaptation

#### adaptive_target_percentile

- **Type:** `float`
- **Default:** `0.15`
- **Valid Range:** `0.10-0.40`
- **Code Location:** `config.py:65`, `gallery.py:735`, `matching.py:551`
- **Impact:** Target false positive rate (15%)
- **Notes:** Lower = stricter threshold

#### adaptive_warmup_matches

- **Type:** `int`
- **Default:** `20`
- **Valid Range:** `10-50`
- **Code Location:** `config.py:66`, `gallery.py:738`, `matching.py:555`
- **Impact:** Conservative phase before adapting
- **Notes:** Uses fixed threshold until warmup complete

### 3.8 Crossing Detection

#### use_crossing_detection

- **Type:** `bool`
- **Default:** `True`
- **Code Location:** `config.py:69`, `gallery.py:310`
- **Impact:** Enables crossing-specific handling
- **Notes:** Critical for crowded scenes

#### crossing_detection_radius

- **Type:** `float`
- **Default:** `100.0`
- **Valid Range:** `50-200`
- **Code Location:** `config.py:70`, `gallery.py:335`
- **Impact:** Distance threshold for crossing detection (pixels)
- **Notes:** Tracks within radius considered crossing

#### crossing_iou_threshold

- **Type:** `float`
- **Default:** `0.1`
- **Valid Range:** `0.05-0.3`
- **Code Location:** `config.py:71`, `gallery.py:336`
- **Impact:** Bbox overlap threshold for crossing
- **Notes:** Lower = more sensitive to proximity

#### crossing_threshold_boost

- **Type:** `float`
- **Default:** `0.6`
- **Code Location:** `config.py:72`
- **Impact:** Stricter similarity threshold during crossing
- **Notes:** Effective threshold = base × 0.6

#### crossing_boost_reduction

- **Type:** `float`
- **Default:** `0.3`
- **Code Location:** `config.py:73`, `gallery.py:255`
- **Impact:** Reduce position boost to 30% during crossing
- **Notes:** Relies more on appearance during overlap

#### rerank_on_crossing

- **Type:** `bool`
- **Default:** `True`
- **Code Location:** `config.py:74`
- **Impact:** Apply k-reciprocal during crossing events
- **Notes:** Additional accuracy at cost of latency

### 3.9 Motion Validation

#### use_motion_validation

- **Type:** `bool`
- **Default:** `True`
- **Code Location:** `config.py:84`, `gallery.py:183`
- **Impact:** Validates assignments against motion history
- **Notes:** Rejects impossible teleportation

#### motion_max_distance

- **Type:** `float`
- **Default:** `150.0`
- **Valid Range:** `50-300`
- **Code Location:** `config.py:85`, `gallery.py:198`, `matching.py:708`
- **Impact:** Max distance from prediction (pixels)
- **Notes:** Reject if detection too far from expected position

#### motion_direction_threshold

- **Type:** `float`
- **Default:** `120.0`
- **Valid Range:** `60-180`
- **Code Location:** `config.py:86`, `gallery.py:199`, `matching.py:709`
- **Impact:** Max angle change (degrees)
- **Notes:** 120° allows turns but rejects reversals

### 3.10 Tentative Track Confirmation

**Academic Reference:** DeepSORT (arXiv:1703.07402), ByteTrack (arXiv:2110.06864)

**Track Lifecycle:**
1. New detection → tentative track (negative ID)
2. Probation: monitor for consecutive matches
3. Confirmation: promote to permanent track (positive ID)
4. Deletion: unconfirmed tracks removed after max_age

#### min_frames_for_id

- **Type:** `int`
- **Default:** `5`
- **Valid Range:** `2-10`
- **Code Location:** `config.py:89`, `pipeline.py:73`
- **Impact:** Frames before permanent ID assignment
- **Notes:** Higher = fewer spurious IDs, slower confirmation

#### tentative_max_age

- **Type:** `int`
- **Default:** `10`
- **Valid Range:** `5-30`
- **Code Location:** `config.py:90`, `pipeline.py:74`
- **Impact:** Max frames to keep unconfirmed track
- **Notes:** Delete if not confirmed within window

---

## 4. Output Configuration

**Class:** `OutputConfig` | **Source:** `config.py:112-118`

### 4.1 save_video

- **Type:** `bool`
- **Default:** `True`
- **Code Location:** `config.py:115`
- **Impact:** Outputs annotated video with bboxes and IDs
- **Notes:** H.264 codec, resolution matches input

### 4.2 save_tracks

- **Type:** `bool`
- **Default:** `True`
- **Code Location:** `config.py:116`
- **Impact:** Outputs JSON tracks file
- **Notes:** Contains per-frame ID assignments and bboxes

### 4.3 visualization

- **Type:** `bool`
- **Default:** `True`
- **Code Location:** `config.py:117`
- **Impact:** Enables all visual overlays
- **Notes:** Set False for headless processing

---

## 5. Visualization Configuration

**Class:** `VisualizationConfig` | **Source:** `config.py:93-110`

### 5.1 Layout Parameters

#### gallery_panel_width

- **Type:** `int`
- **Default:** `200`
- **Valid Range:** `100-400`
- **Code Location:** `config.py:97`
- **Impact:** Thumbnail panel width (pixels)
- **Notes:** Wider = larger thumbnails, less video area

#### gallery_panel_position

- **Type:** `string`
- **Default:** `"right"`
- **Valid Values:** `"left"`, `"right"`
- **Code Location:** `config.py:98`
- **Impact:** Panel placement relative to video

#### show_gallery_panel

- **Type:** `bool`
- **Default:** `True`
- **Code Location:** `config.py:99`
- **Impact:** Toggle thumbnail panel visibility

#### show_pipeline_hud

- **Type:** `bool`
- **Default:** `True`
- **Code Location:** `config.py:100`
- **Impact:** Toggle stats overlay (FPS, track count)

#### max_gallery_entries

- **Type:** `int`
- **Default:** `10`
- **Valid Range:** `5-20`
- **Code Location:** `config.py:101`
- **Impact:** Most recent persons shown in panel

#### extended_frame_enabled

- **Type:** `bool`
- **Default:** `True`
- **Code Location:** `config.py:109`
- **Impact:** Analytics layout (video + side panels)
- **Notes:** False = overlay mode

### 5.2 Styling Parameters

#### panel_bg_opacity

- **Type:** `float`
- **Default:** `0.75`
- **Valid Range:** `0.0-1.0`
- **Code Location:** `config.py:104`
- **Impact:** Background transparency (0=clear, 1=solid)

#### bbox_thickness_matched

- **Type:** `int`
- **Default:** `3`
- **Valid Range:** `1-5`
- **Code Location:** `config.py:105`
- **Impact:** Stroke width for confirmed tracks

#### bbox_thickness_unmatched

- **Type:** `int`
- **Default:** `2`
- **Valid Range:** `1-4`
- **Code Location:** `config.py:106`
- **Impact:** Stroke width for tentative tracks

---

## 6. Performance Impact Tables

### 6.1 Accuracy vs Speed Tradeoffs

| Parameter | Change | Accuracy | Speed | ID Switches |
|-----------|--------|----------|-------|-------------|
| `use_full_reranking=true` | Enable | ↑↑ +5-15% mAP | ↓↓ -30% | ↓↓ |
| `rank_list_size: 20→50` | Increase | ↑ | ↓ -10% | ↓ |
| `reid_variant: x1_0→x0_5` | Downgrade | ↓↓ | ↑↑ +100% | ↑ |
| `batch_size: 32→64` | Increase | = | ↑ +15% | = |
| `use_velocity_prediction=true` | Enable | ↑ | ↓ -5% | ↓↓ |
| `use_crossing_detection=true` | Enable | ↑ | ↓ -10% | ↓↓ |
| `use_adaptive_threshold=true` | Enable | ↑ | ↓ -5% | ↓ |

### 6.2 Memory Usage by Parameter

| Parameter | Change | VRAM Impact |
|-----------|--------|-------------|
| `reid_variant: x1_0→x0_25` | Downgrade | ↓ -50MB |
| `batch_size: 32→64` | Increase | ↑ +32MB |
| `max_features_per_id: 10→50` | Increase | ↑ +200KB/track |
| `rank_list_size: 20→50` | Increase | ↑ +10MB temp |
| `adaptive_window_size: 100→500` | Increase | ↑ +2MB |

### 6.3 ID Switch Reduction

| Parameter | Impact on ID Switches |
|-----------|----------------------|
| `min_frames_for_id: 3→7` | ↓↓ Fewer false IDs |
| `use_motion_validation=true` | ↓↓ Reject impossible jumps |
| `crossing_threshold_boost: 0.6→0.5` | ↓ Stricter during crossing |
| `ema_alpha: 0.9→0.6` | ↓ More stable features |
| `prediction_boost: 0.15→0.25` | ↓ Favor predicted positions |

---

## 7. Recommended Configurations

### High Accuracy (Surveillance)

Prioritizes accuracy over speed. Use for forensic analysis.

```yaml
model:
  reid_variant: "osnet_x1_0"
  device: "cuda"

inference:
  confidence_threshold: 0.8
  similarity_threshold: 0.6
  batch_size: 32

gallery:
  use_full_reranking: true
  rank_list_size: 30
  rerank_k1: 25
  min_frames_for_id: 5
  use_velocity_prediction: true
  use_crossing_detection: true
  crossing_threshold_boost: 0.5
```

**Expected:** ~15 FPS, minimal ID switches

### Real-time (30+ FPS)

Prioritizes speed for live monitoring.

```yaml
model:
  reid_variant: "osnet_x0_5"
  device: "cuda"

inference:
  confidence_threshold: 0.7
  similarity_threshold: 0.55
  batch_size: 64

gallery:
  use_full_reranking: false
  rank_list_size: 15
  use_adaptive_threshold: false
  min_frames_for_id: 3
```

**Expected:** 30+ FPS, moderate ID switches

### Crowded Scenes

Optimized for high-density environments with frequent crossings.

```yaml
gallery:
  use_crossing_detection: true
  crossing_detection_radius: 80.0
  crossing_threshold_boost: 0.5
  crossing_boost_reduction: 0.2
  rerank_on_crossing: true

  use_motion_validation: true
  motion_max_distance: 100.0
  motion_direction_threshold: 90.0

  min_frames_for_id: 7
  tentative_max_age: 15

  use_velocity_prediction: true
  prediction_radius: 50.0
```

**Expected:** Robust tracking in crowds, ~20 FPS

---

## 8. Output File Schemas

### 8.1 Tracks JSON (`*_tracks.json`)

Pipeline tracking summary per video.

```json
{
  "video": "data/videos/demo.mp4",
  "config": {
    "reid_variant": "osnet_x1_0",
    "similarity_threshold": 0.8
  },
  "stats": {
    "frames": 347,
    "detections": 2654,
    "unique_persons": 13
  },
  "persons": {
    "0": { "feature_dim": 512 },
    "1": { "feature_dim": 512 }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `video` | string | Input video path |
| `config` | object | Key config params used |
| `stats.frames` | int | Total frames processed |
| `stats.detections` | int | Total detections |
| `stats.unique_persons` | int | Unique track IDs assigned |
| `persons.<id>` | object | Per-person metadata |

### 8.2 Gallery JSON (`*_gallery.json`)

Final gallery state with stored features.

```json
{
  "next_id": 13,
  "frame_idx": 347,
  "entries": {
    "0": {
      "avg_feature": [0.015, 0.035, ...],  // 512-dim float32
      "last_seen": 267
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `next_id` | int | Next available track ID |
| `frame_idx` | int | Last processed frame |
| `entries.<id>.avg_feature` | float[512] | EMA-averaged L2-normalized feature |
| `entries.<id>.last_seen` | int | Frame index last detected |

---

## 9. Academic References

1. **OSNet** — Zhou, K., Yang, Y., Cavallaro, A., & Xiang, T. (2019). Omni-Scale Feature Learning for Person Re-Identification. *ICCV 2019*. [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Omni-Scale_Feature_Learning_for_Person_Re-Identification_ICCV_2019_paper.pdf)

2. **K-Reciprocal** — Zhong, Z., Zheng, L., Cao, D., & Li, S. (2017). Re-ranking Person Re-identification with k-reciprocal Encoding. *CVPR 2017*, pp. 3652-3661. [Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf)

3. **BoT-SORT** — Aharon, N., Orfaig, R., & Bobrovsky, B.-Z. (2022). BoT-SORT: Robust Associations Multi-Pedestrian Tracking. *arXiv:2206.14651*. [Paper](https://arxiv.org/abs/2206.14651)

4. **DeepSORT** — Wojke, N., Bewley, A., & Paulus, D. (2017). Simple Online and Realtime Tracking with a Deep Association Metric. *ICIP 2017*. [Paper](https://arxiv.org/abs/1703.07402)

5. **ByteTrack** — Zhang, Y., et al. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. *ECCV 2022*. [Paper](https://arxiv.org/abs/2110.06864)

6. **JointBDOE** — Hu, Y., et al. (2024). Joint Multi-Person Body Detection and Orientation Estimation via One Unified Embedding. *PRCV 2024*. [GitHub](https://github.com/hnuzhy/JointBDOE)

7. **DDTAS** — Dual Dynamic Threshold Adjustment Strategy for Deep Metric Learning. *arXiv:2404.19282*. [Paper](https://arxiv.org/abs/2404.19282)

---

## 10. Appendix

### 10.1 Default Configuration YAML

```yaml
model:
  reid_variant: "osnet_x1_0"
  reid_weights: null
  device: "cuda"
  yolo_weights: "data/weights/jointbdoe_m.pt"

inference:
  image_size: [256, 128]
  batch_size: 32
  confidence_threshold: 0.75
  similarity_threshold: 0.5

gallery:
  max_features_per_id: 10
  ema_alpha: 0.7

  rank_list_size: 20
  rank_distance_threshold: null
  rank_min_entries_per_id: 3
  rank_fallback_threshold: 1.2

  use_full_reranking: false
  rerank_k1: 20
  rerank_k2: 6
  rerank_lambda: 0.3

  use_quality_weighting: true
  quality_min_threshold: 0.3
  quality_confidence_weight: 0.6
  quality_geometry_weight: 0.4
  ideal_aspect_ratio: 0.4
  min_bbox_area: 2000

  use_velocity_prediction: true
  velocity_history_frames: 5
  velocity_max_speed: 100.0
  prediction_radius: 75.0
  prediction_boost: 0.15

  use_adaptive_threshold: false
  adaptive_min_threshold: 0.50
  adaptive_max_threshold: 0.80
  adaptive_window_size: 100
  adaptive_target_percentile: 0.15
  adaptive_warmup_matches: 20

  use_crossing_detection: true
  crossing_detection_radius: 100.0
  crossing_iou_threshold: 0.1
  crossing_threshold_boost: 0.6
  crossing_boost_reduction: 0.3
  rerank_on_crossing: true

  use_motion_validation: true
  motion_max_distance: 150.0
  motion_direction_threshold: 120.0

  min_frames_for_id: 5
  tentative_max_age: 10

output:
  save_video: true
  save_tracks: true
  visualization: true

visualization:
  gallery_panel_width: 200
  gallery_panel_position: "right"
  show_gallery_panel: true
  show_pipeline_hud: true
  max_gallery_entries: 10
  panel_bg_opacity: 0.75
  bbox_thickness_matched: 3
  bbox_thickness_unmatched: 2
  extended_frame_enabled: true
```

### 10.2 Parameter Interaction Notes

| Interaction | Effect |
|-------------|--------|
| `use_full_reranking` + `rerank_on_crossing` | Reranking only during crossing if both true |
| `ema_alpha` + `max_features_per_id` | Higher features + lower alpha = most stable |
| `confidence_threshold` + `quality_min_threshold` | Both filter low-quality detections |
| `use_velocity_prediction` + `use_motion_validation` | Complementary: predict then validate |
| `adaptive_threshold` + `similarity_threshold` | Adaptive overrides base when active |

---

**Document Version:** 1.0
**Last Updated:** 2026-02-04
**Parameter Count:** 46
