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
├── config.py                       # Configuration models (updated: FAISS, rerank, optimize flags)
├── jointbdoe_detector.py           # Primary detector (optimized: torch.compile + FP16)
├── feature_extractor.py            # OSNet extractor (optimized: batch processing + pinned memory)
├── utils.py                        # Utilities (new: safe_compile)
├── faiss-gallery-index-wrapper.py  # FAISS IVF indexing wrapper (new)
├── gallery-knn-cache-for-selective-reranking.py # k-NN cache module (new)
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

## 1. Core Pipeline: `pipeline.py` (825 LOC)

**Class:** `VideoReIDPipeline`

Main orchestrator for end-to-end video processing. Hardened with 25 edge case fixes including IoU div-by-zero guard, off-by-one correction for frame counter, IoU match preference over weak ReID, periodic stale entry cleanup.

### Key Methods

| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `process_frame()` | frame (H,W,3) BGR | List[Detection] with track_id | Full pipeline: detect→extract→match→assign |
| `process_video()` | str/Path video file | list[dict] tracks | Process entire video, populate gallery |
| `write_output()` | results, output_path | None | Render + save video, write JSON |

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

## 2. Gallery & Assignment: `gallery.py` (1120 LOC)

**Classes:** `PersonGallery`, `GalleryEntry`

Manages person tracks and implements Hungarian assignment. Robust edge case handling: schema validation on load, backups before save, EMA re-normalization post-update, k-NN cache eager rebuild on gallery changes (10+ changes), O(1) dict lookups replacing O(n) list.index(), stale motion entry cleanup.

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

| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `match_batch()` | features (N,512), bboxes (N,) | [(id\|None, conf)] (N,) | Rank-list vote + Hungarian assignment |
| `update()` | track_id, feature (512,), quality | None | EMA update feature: α×qual×f + (1−α)×old |
| `get_distance_matrix()` | features (Q,512) | distances (Q, M) | L2 distance to gallery features |
| `add()` | feature (512,) | new_id | Create new gallery entry |

### Key Algorithms

**Rank-List Majority Voting:**
- Top-k=20 neighbors per query by L2 distance
- Filter by threshold: distance < 1.2
- Count ID votes, return ID with max votes
- Confidence = 1.0 - (min_distance / 1.2)

**Quality-Weighted Feature Fusion:**
- Quality = 0.6×confidence + 0.4×geometry (aspect ratio + area)
- Skip update if quality < 0.3
- EMA update: `new_avg = α × quality × feature + (1−α) × old_avg` (α=0.7)
- Feature buffer: deque maxlen=10, maintain rolling average

**Velocity-Based Motion Tracking:**
- Track motion history: positions, velocities, directions
- Position validation: distance from predicted < 150px
- Direction validation: angle change < 120°
- Stationary exemption: speed < 5px/frame bypass motion check

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

## 3. Matching & Re-ranking: `matching.py` (1016 LOC)

**Key Functions:** All module-level (no classes)

Implements feature matching, re-ranking, and validation algorithms. NaN prevention: col_max clamped to 1e-8 in re_ranking, degenerate bbox rejection (w<=0 or h<=0) in quality scoring, deterministic tie-breaking in majority voting.

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

## 8. Utilities: `utils.py` (87 LOC)

```python
def safe_compile(model: torch.nn.Module, **kwargs) -> torch.nn.Module:
    """Compile model with torch.compile if available, else return unchanged.

    Graceful fallback for older PyTorch versions or compilation failures.
    """

def compute_iou(box1, box2) -> float:
    """Jaccard intersection over union. Guard: returns 0.0 if union <= 1e-8"""

def extract_crop(frame, bbox, padding=10) -> np.ndarray:
    """Extract person crop with validation. Returns 1x1 fallback for degenerate bbox"""
```

## 9. Optimization Modules

### FAISS Gallery Index: `faiss-gallery-index-wrapper.py` (165 LOC)

Optional module for accelerated gallery search using FAISS (Facebook AI Similarity Search).

```python
class FAISSGalleryIndexWrapper:
    """O(log n) gallery search via Inverted File (IVF) indexing

    - IVF clustering for large galleries (>100 tracks)
    - Graceful fallback to brute-force L2 distance if FAISS unavailable
    - Configurable: nlist (clusters), nprobe (search width)
    - GPU resource pooling: StandardGpuResources() stored as instance var, reused on rebuild (prevents leaks)
    - Rebuild interval to maintain index freshness
    """

    def build_index(features: np.ndarray) -> None
    def search_knn(query_feature: np.ndarray, k: int) -> (indices, distances)
    def add_features(new_features: np.ndarray) -> None
    def rebuild_if_needed() -> None
```

**Config Keys:**
- `gallery.use_faiss`: Enable/disable (default: True)
- `gallery.faiss_nlist`: IVF clusters (default: 64)
- `gallery.faiss_nprobe`: Search width (default: 8)
- `gallery.faiss_min_train_size`: Min vectors before IVF (default: 100)
- `gallery.faiss_rebuild_interval`: Rebuild frequency (default: 50 frames)

### Selective Re-ranking Cache: `gallery-knn-cache-for-selective-reranking.py` (3.7KB)

Optional module for confidence-triggered k-reciprocal re-ranking with k-NN caching.

```python
class SelectiveRerankingCache:
    """Cache k-NN graph to accelerate selective re-ranking

    - Confidence threshold triggers re-ranking on low-confidence matches
    - Pre-computed k-NN graph eliminates redundant distance calculations
    - Distance skip optimization: bypass re-ranking if already confident
    - Lazy rebuild on gallery updates
    """

    def compute_knn_graph(gallery_features: np.ndarray) -> None
    def should_rerank(confidence: float) -> bool
    def get_rerank_candidates(query_idx: int, k: int) -> list
    def invalidate_cache() -> None
```

**Config Keys:**
- `gallery.rerank_confidence_threshold`: Trigger threshold (default: 0.7)
- `gallery.rerank_distance_skip`: Skip if distance < this (default: 0.5)
- `gallery.rerank_cache_knn`: Enable caching (default: True)
- `gallery.rerank_knn_rebuild_interval`: Cache rebuild (default: 100 frames)

## 10. Performance Optimization Details

### Detector Optimization: `jointbdoe_detector.py`

```python
# torch.compile + FP16 + inference_mode for 2-3× speedup
model = safe_compile(model, mode="reduce-overhead")
with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
    outputs = model(x)  # Lower memory, faster inference
```

### Extractor Optimization: `feature_extractor.py`

```python
# Batch processing with pre-allocated tensors and pinned memory
self.feature_buffer = torch.empty(batch_size, 512, pin_memory=True)  # Pre-alloc
crops_tensor = torch.from_numpy(crops).to(self.device, non_blocking=True)  # Pinned copy

# Warmup batch on first call to initialize GPU kernels
with torch.inference_mode(), torch.autocast(...):
    _ = model(warmup_batch)  # ~50ms one-time cost
    features = model(crops_tensor)
```

### Benchmark & Validation Scripts

**`scripts/benchmark-reid-pipeline-performance.py`**
- Measures latency per pipeline stage (detect, extract, assign, etc.)
- Tracks FPS, memory usage, throughput
- Compares with/without FAISS, reranking, torch.compile
- Output: JSON report with per-frame and per-stage metrics

**`scripts/validate-reid-pipeline-accuracy.py`**
- Validates re-ranking correctness (CVPR2017 algorithm)
- Checks FAISS results match brute-force baseline
- Tests k-NN cache invalidation on gallery updates
- Verifies confidence-triggered selective re-ranking
- Output: Test report with pass/fail per validation check

## 11. Entry Point: `demo_video_reid_inference.py` (210 LOC)

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

### Detection (JointBDOE Output)

```python
@dataclass
class Detection:
    bbox: tuple[float, 4]           # (x1, y1, x2, y2) pixel coords
    confidence: float               # [0, 1] detection score
    crop: np.ndarray               # (H', W', 3) BGR with 10px padding
    orientation: float | None       # Degrees [0, 360] or None
    track_id: int | None           # +id (permanent) or -id (tentative) after assignment
    features: np.ndarray | None    # (512,) float32 after extraction
    is_matched: bool               # True if ReID matched gallery entry
    match_similarity: float        # Similarity score [0, 1]
    top_similar: list[tuple]       # [(id, similarity), ...] top 3 matches
```

### TrackMotion (Velocity Tracking)

```python
@dataclass
class TrackMotion:
    positions: deque[tuple]         # maxlen=5, center (x, y)
    velocities: deque[tuple]        # maxlen=5, (vx, vy) px/frame
    directions: deque[float]        # maxlen=5, angle degrees
```

### GalleryEntry (Track Storage)

```python
@dataclass
class GalleryEntry:
    features: deque[np.ndarray]     # maxlen=10, (512,) each
    quality_scores: deque[float]    # maxlen=10, [0, 1]
    avg_feature: np.ndarray | None  # (512,) EMA-averaged
    avg_quality: float              # Rolling average quality
    last_seen: int                  # Frame index of last detection
```

## 12. Key Algorithms Summary

| Algorithm | File | Purpose | Optimization |
|-----------|------|---------|---------------|
| Hungarian Assignment | gallery.py | Optimal detections-to-tracks matching | lap library |
| Rank-List Voting | matching.py | Triplet-loss weighted matching | Implemented |
| k-Reciprocal Re-ranking | matching.py | CVPR2017 Jaccard distance | Selective + cached k-NN |
| FAISS Gallery Search | faiss-gallery-index-wrapper.py | O(log n) similarity search | IVF indexing (optional) |
| Motion Validation | matching.py | Temporal consistency checking | Implemented |
| Quality Weighting | matching.py | Confidence + geometry fusion | Implemented |
| Crossing Detection | matching.py | Prevent ID theft at crossings | Implemented |
| Adaptive Threshold | matching.py | Dynamic similarity adjustment | Implemented |
| torch.compile | utils.py | Model graph compilation | FP16 + inference_mode |
| Batch Processing | feature_extractor.py | Vectorized extraction | Pinned memory + pre-alloc tensors |

## 13. Dependencies Map

```
VideoReIDPipeline (torch.compile optimized)
├── JointBDOEDetector (FP16 + inference_mode)
│   └── detectors/jointbdoe/ (ported utilities)
│   └── Pre-trained JointBDOE weights (data/weights/)
├── ReIDFeatureExtractor (batch processing + pinned memory)
│   └── models/osnet.py (ported from TorchReID)
│   └── extractors/torchreid-feature-extractor.py
├── PersonGallery
│   ├── matching module (all algorithms)
│   ├── faiss-gallery-index-wrapper (optional: IVF acceleration)
│   ├── gallery-knn-cache-for-selective-reranking (optional: confidence-triggered)
│   └── lap (Hungarian algorithm)
└── Visualization renderers
    └── colors.py (palette)
```

**Optional Dependencies (graceful fallback):**
- `faiss-cpu` or `faiss-gpu`: Enable `gallery.use_faiss=true` for O(log n) search
- Gracefully falls back to brute-force L2 distance if unavailable

## 14. Configuration Parameter Reference: Optimization Options

**Inference Optimizations:**

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `inference.batch_size` | int | 32 | Feature extraction batch size |
| Model device | str | cuda | GPU/CPU inference |

**FAISS Acceleration:**

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `gallery.use_faiss` | bool | true | Enable IVF indexing (O(log n)) |
| `gallery.faiss_nlist` | int | 64 | Number of IVF clusters |
| `gallery.faiss_nprobe` | int | 8 | Clusters to search per query |
| `gallery.faiss_min_train_size` | int | 100 | Threshold for IVF activation |
| `gallery.faiss_rebuild_interval` | int | 50 | Frame frequency for index rebuild |

**Selective Re-ranking:**

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `gallery.rerank_confidence_threshold` | float | 0.7 | Trigger re-ranking below this |
| `gallery.rerank_distance_skip` | float | 0.5 | Skip if distance already low |
| `gallery.rerank_cache_knn` | bool | true | Cache k-NN graph for speed |
| `gallery.rerank_knn_rebuild_interval` | int | 100 | Cache rebuild frequency |

See [system-architecture.md](./system-architecture.md) for complete architecture details.

---

**Document Version:** 1.2
**Last Updated:** 2026-02-10
**Total Source LOC:** 3,700+ (edge case hardening added 205 LOC, net +49 after deletions)
**Optimizations:** torch.compile, FP16, FAISS indexing, selective reranking, batch processing
**Robustness:** 25 edge case fixes: IoU guards, NaN prevention, degenerate bbox handling, GPU resource pooling, stale entry cleanup, O(1) lookups, deterministic tie-breaking
