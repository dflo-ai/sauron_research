# Codebase Summary: HAT-ReID Architecture & Key Classes

## Overview

**Total LOC:** 3,444 (src only)
**Main Package:** `src/reid_research/`
**Language:** Python 3.10+
**Architecture:** Modular pipeline with pluggable detectors & extractors

```
src/reid_research/
├── pipeline.py                     # Main orchestration
├── gallery.py                      # Track storage + matching
├── matching.py                     # Feature matching + re-ranking
├── config.py                       # Configuration models
├── jointbdoe_detector.py           # Primary detector wrapper
├── feature_extractor.py            # OSNet extractor wrapper
├── utils.py                        # Utilities
├── models/                         # Neural network architectures (ported)
│   ├── __init__.py
│   └── osnet.py                    # OSNet backbone (~400 LOC)
├── extractors/                     # Feature extraction backends (ported)
│   ├── __init__.py
│   └── torchreid-feature-extractor.py
├── detectors/                      # Detection backends (ported)
│   ├── __init__.py
│   └── jointbdoe/
│       ├── model-loader-attempt-load.py
│       ├── preprocessing-letterbox-resize.py
│       └── postprocessing-nms-scale-coords.py
└── visualization/
    ├── colors.py                   # Okabe-Ito palette
    ├── gallery_panel.py            # Thumbnail renderer
    ├── hud_renderer.py             # Stats/event HUD
    └── extended_frame_renderer.py  # Analytics layout
```

## 1. Core Pipeline: `pipeline.py` (830 LOC)

**Class:** `VideoReIDPipeline`

Main orchestrator for end-to-end video processing.

### Key Methods

| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `__init__()` | config, optional components | self | Initialize with detector, extractor, gallery |
| `process_frame()` | np.ndarray (RGB frame) | List[Detection] | Detect → Extract → Match → Assign IDs |
| `process_video()` | str/Path (video file) | list[dict] | Batch process entire video |
| `write_output()` | results, output_path | None | Save video + JSON tracks |

### Internal State

```python
# Dual-track system (tentative → permanent)
self._tentative_tracks: dict[int, dict]  # Negative IDs (-1, -2, ...)
self._next_tentative_id: int

# Gallery management
self.gallery: PersonGallery  # Permanent tracks

# Animation state
self._match_animations: dict[int, int]   # track_id → frames_since_match

# IoU-based track continuation
self._track_bboxes: dict[int, tuple]     # track_id → (bbox, frame)
self._iou_threshold: float = 0.3

# Visualization
self._gallery_renderer: GalleryPanelRenderer
self._hud_renderer: HUDRenderer
self._extended_renderer: ExtendedFrameRenderer
```

### Processing Flow

```python
def process_frame(self, frame):
    # 1. Detection
    detections = self.detector.detect(frame)

    # 2. Feature Extraction
    features = self.extractor.extract(frame, detections)

    # 3. IoU-based Track Continuation (fast path)
    # Check if existing tracks visible in frame

    # 4. Hungarian Assignment + Rank-List Voting
    # Match new detections to gallery

    # 5. Motion Validation
    # Reject assignments violating velocity constraints

    # 6. Tentative→Permanent Transition
    # Promote confirmed tentative tracks

    # 7. Gallery Update
    # Add/update features, increment counters

    # 8. Visualization
    # Render boxes, IDs, HUD, gallery panel

    return detections_with_ids
```

### Configuration Dependencies

```python
config.gallery.min_frames_for_id   # Frames before permanent ID
config.visualization.*             # Rendering options
```

## 2. Gallery & Assignment: `gallery.py` (904 LOC)

**Classes:** `PersonGallery`, `GalleryEntry`

Manages person tracks and implements Hungarian assignment.

### GalleryEntry (Track Data)

```python
@dataclass
class GalleryEntry:
    features: deque(maxlen=10)        # Recent feature vectors
    quality_scores: deque(maxlen=10)  # Detection quality scores
    avg_feature: np.ndarray | None    # Averaged feature vector
    avg_quality: float                # Average quality score
    last_seen: int                    # Frame index when last detected
```

### PersonGallery (Core Methods)

| Method | Purpose | Returns |
|--------|---------|---------|
| `match_batch()` | Hungarian assignment for batch of detections | np.ndarray (N,2) assignments |
| `update_feature()` | Add new feature, compute EMA average | None |
| `get_distance_matrix()` | Compute cosine similarity distance matrix | np.ndarray (Q, G) distances |
| `get_avg_feature()` | Retrieve averaged feature for ID | np.ndarray or None |

### Key Features

**Rank-List Majority Voting:**
```python
# For each query feature:
# 1. Find top-k similar gallery entries (k=20)
# 2. Apply distance threshold filtering
# 3. Count ID votes among top-k
# 4. Return ID with max votes (or highest similarity if tie)
```

**Quality-Weighted Feature Fusion:**
```python
# Only update features if quality > threshold
quality_score = (confidence * 0.6) + (geometry * 0.4)
if quality_score >= min_threshold:
    # EMA: avg_feature = ema_alpha * new_feature + (1-ema_alpha) * avg_feature
    avg_feature = update_via_ema(new_feature, quality_score)
```

**Velocity-Based Motion Tracking:**
```python
self._track_motion: dict[int, TrackMotion]  # Position history per track
# Validates assignments: rejects if velocity > max_speed or angle > max_direction_change
```

### Configuration Parameters

```yaml
gallery:
  max_features_per_id: 200          # Rolling window size
  ema_alpha: 0.6                    # Feature update weight (0.7 = more recent)
  rank_list_size: 20                # Top-k for voting
  rank_distance_threshold: 1.0      # Distance cutoff for voting
  similarity_threshold: 0.8         # Match acceptance threshold
  min_frames_for_id: 5              # Frames before permanent ID
  tentative_max_age: 10             # Frames to keep unconfirmed
```

## 3. Matching & Re-ranking: `matching.py` (1,009 LOC)

**Key Functions:** All module-level (no classes)

Implements feature matching, re-ranking, and validation algorithms.

### Section 1: Official CVPR2017 Re-ranking

```python
def re_ranking_torchreid(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    """Full k-reciprocal with Jaccard distance (CVPR 2017)

    Args:
        q_g_dist: (Q, G) query-gallery distance matrix
        q_q_dist: (Q, Q) query-query distance matrix
        g_g_dist: (G, G) gallery-gallery distance matrix

    Returns:
        (Q, G) re-ranked distance matrix
    """
    # Compute reciprocal neighbors R(p,k1)
    # Expand with query expansion k2
    # Apply Jaccard distance
    # Final distance: lambda * original + (1-lambda) * reranked
```

### Section 2: Majority Voting

```python
def majority_vote_reidentify(query_dist, gallery_ids, k=20, threshold=1.0):
    """Triplet-loss style voting

    Returns:
        (Q,) array of matched IDs (or -1 for no match)
    """
    # For each query:
    # - Get top-k gallery entries
    # - Filter by distance threshold
    # - Count votes by ID
    # - Return ID with max votes
```

### Section 3: Motion Validation

```python
@dataclass
class TrackMotion:
    """Motion history for a track"""
    positions: deque                  # Last N positions
    velocities: deque                 # Computed velocities
    directions: deque                 # Velocity directions

def predict_position(motion, frame_gap=1):
    """Predict next position based on velocity history"""
    # Average recent velocities
    # Clamp to max_speed
    # Return: (x, y, radius)

def validate_motion_consistency(motion, assignment, max_distance, max_angle):
    """Check if assignment violates motion constraints

    Returns:
        bool: True if assignment is valid, False if rejected
    """
```

### Section 4: Quality & Geometry

```python
def compute_quality_score(confidence, bbox, ideal_aspect_ratio, min_area):
    """Compute detection quality (0-1)

    Quality = confidence * 0.6 + geometry * 0.4
    where geometry considers bbox aspect ratio and area
    """

def compute_position_boost(detection, motion, radius):
    """Boost similarity score if detection near motion prediction"""
    # Returns 0.0-0.15 boost based on proximity
```

### Section 5: Crossing Detection

```python
def detect_crossing_tracks(tracks, radius=100.0, iou_threshold=0.1):
    """Identify pairs of tracks that are crossing

    Returns:
        list of (track_id1, track_id2) pairs
    """
```

### Section 6: Adaptive Threshold

```python
def compute_adaptive_threshold(match_scores, percentile=0.15, min_threshold=0.5, max_threshold=0.8):
    """Dynamically adjust threshold based on match quality distribution

    Returns:
        float: New threshold value
    """
```

## 4. Configuration: `config.py` (159 LOC)

**Classes:** Pydantic BaseModel hierarchy

Defines all configuration parameters with defaults and validation.

### Class Hierarchy

```
ReIDConfig (root)
├── ModelConfig
│   └── device, reid_variant, yolo_weights, use_fastreid, ...
├── InferenceConfig
│   └── image_size, batch_size, confidence_threshold, similarity_threshold
├── GalleryConfig
│   └── max_features_per_id, rank_list_size, motion validation, thresholds, ...
├── VisualizationConfig
│   └── gallery_panel, HUD, split_view, extended_frame
└── OutputConfig
    └── save_video, save_tracks, visualization
```

### Load Configuration

```python
config = load_config("configs/default.yaml")  # Returns ReIDConfig instance
```

## 5. Detectors

### JointBDOEDetector: `jointbdoe_detector.py` (137 LOC)

```python
class JointBDOEDetector:
    """Detects persons with body orientation estimation"""

    def __init__(self, config: ReIDConfig):
        # Load pretrained JointBDOE model
        # Model outputs: boxes (N, 4) + confidence (N,) + orientation (N,)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Returns detections with confidence and orientation"""
```

**Output:** List[Detection] where Detection includes:
- `box`: (x1, y1, x2, y2)
- `conf`: confidence score
- `orientation`: body orientation angle (optional)

## 6. Feature Extractors

### OSNet: `feature_extractor.py` + `models/osnet.py`

```python
class ReIDFeatureExtractor:
    """Extracts 512-dim features using OSNet backbone

    Model: osnet_x1_0 (pretrained on ImageNet, fine-tuned on Market1501)
    Output: 512-dim L2-normalized feature vectors
    """

    def extract(self, crops: list[np.ndarray]) -> np.ndarray:
        """Returns (N, 512) feature matrix"""
```

### Models Module: `models/`

```python
from src.reid_research.models import osnet_x1_0, osnet_ibn_x1_0, build_model

# Build model directly
model = osnet_x1_0(num_classes=1, pretrained=True)

# Or via factory
model = build_model("osnet_x1_0", pretrained=True)
```

**Available models:**
- `osnet_x1_0`: Standard OSNet (512-dim)
- `osnet_ibn_x1_0`: OSNet with Instance Normalization

## 7. Visualization Renderers

### ExtendedFrameRenderer: `extended_frame_renderer.py` (227 LOC)

```python
class ExtendedFrameRenderer:
    """Analytics layout: video + stats on sides + match timeline at bottom"""

    def render_frame(self, frame, detections, gallery_info):
        # Adds padding around video frame
        # Left panel: Pipeline stage HUD
        # Right panel: Recent gallery entries (thumbnails)
        # Bottom: Match timeline + recent assignments
```

### GalleryPanelRenderer: `gallery_panel.py` (237 LOC)

```python
class GalleryPanelRenderer:
    """Renders thumbnail gallery of recent persons

    Shows: person crops with ID, first_seen, detection_count
    """

    def render_panel(self, thumbnails, ids):
        # Returns np.ndarray panel image
```

### HUDRenderer: `hud_renderer.py` (296 LOC)

```python
class HUDRenderer:
    """Renders stats HUD: pipeline stage, FPS, track count"""

    def render_hud(self, frame, stats):
        # Renders text overlay with performance metrics
```

### Color Palette: `colors.py`

```python
# Okabe-Ito colorblind-safe palette (8 colors)
OKABE_ITO_PALETTE = [(230, 25, 75), (60, 180, 75), ...]

def get_id_color(track_id):
    """Returns RGB color for track ID (cyclic)"""
```

## 8. Utilities: `utils.py` (59 LOC)

```python
def compute_iou(box1, box2) -> float:
    """Jaccard intersection over union"""

def bbox_to_tlwh(box) -> tuple:
    """Convert (x1,y1,x2,y2) → (top, left, width, height)"""

def tlwh_to_bbox(tlwh) -> tuple:
    """Convert (top, left, width, height) → (x1,y1,x2,y2)"""
```

## 9. Entry Point: `demo_video_reid_inference.py` (210 LOC)

```python
def main():
    # Parse arguments: --video, --config, --output, --threshold, etc.
    # Load config, initialize pipeline
    # Process video
    # Save output video + JSON tracks
```

**Usage:**
```bash
python scripts/demo_video_reid_inference.py \
  --video input.mp4 \
  --config configs/default.yaml \
  --output outputs/ \
  --threshold 0.8
```

## 10. Data Structures

### Detection

```python
@dataclass
class Detection:
    box: tuple              # (x1, y1, x2, y2)
    conf: float             # Confidence [0, 1]
    feature: np.ndarray     # Feature vector (512,) or (2048,)
    track_id: int | None    # Assigned track ID (after matching)
    orientation: float | None  # Body orientation (JointBDOE only)
```

### TrackMotion

```python
@dataclass
class TrackMotion:
    positions: deque        # Last N center positions
    velocities: deque       # Last N velocity vectors
    directions: deque       # Last N velocity directions (angles)
```

### GalleryEntry

```python
@dataclass
class GalleryEntry:
    features: deque         # Rolling window of feature vectors
    quality_scores: deque   # Corresponding quality scores
    avg_feature: np.ndarray # EMA-averaged feature
    avg_quality: float      # Average quality
    last_seen: int          # Last frame index
```

## 11. Key Algorithms Summary

| Algorithm | File | Purpose | Status |
|-----------|------|---------|--------|
| Hungarian Assignment | gallery.py | Optimal detections-to-tracks matching | ✓ lap library |
| Rank-List Voting | matching.py | Triplet-loss weighted matching | ✓ Implemented |
| k-Reciprocal Re-ranking | matching.py | CVPR2017 Jaccard distance | ✓ from torchreid |
| Motion Validation | matching.py | Temporal consistency checking | ✓ Implemented |
| Quality Weighting | matching.py | Confidence + geometry fusion | ✓ Implemented |
| Crossing Detection | matching.py | Prevent ID theft at crossings | ✓ Implemented |
| Adaptive Threshold | matching.py | Dynamic similarity adjustment | ✓ Implemented |

## 12. Dependencies Map

```
VideoReIDPipeline
├── JointBDOEDetector
│   └── detectors/jointbdoe/ (ported utilities)
│   └── Pre-trained JointBDOE weights (data/weights/)
├── ReIDFeatureExtractor
│   └── models/osnet.py (ported from TorchReID)
│   └── extractors/torchreid-feature-extractor.py
├── PersonGallery
│   ├── matching module (all algorithms)
│   └── lap (Hungarian algorithm)
└── Visualization renderers
    └── colors.py (palette)
```

**Note:** All external dependencies (TorchReID, JointBDOE) have been ported into
the package for self-contained operation. Only model weights need to be downloaded.

## 13. Configuration Parameter Reference

See [system-architecture.md](./system-architecture.md) for complete parameter details.

---

**Document Version:** 1.0
**Last Updated:** 2025-02-03
**Total Source LOC:** 3,444
