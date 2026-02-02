# Configuration Parameters

Complete reference for all configurable parameters in HAT-ReID. Configuration is loaded from `configs/default.yaml` and controls behavior of the HAT transformation, tracking, and embedding extraction systems.

## Table of Contents

- [HAT (History-Aware Transformation)](#hat-parameters)
- [Tracker Configuration](#tracker-parameters)
- [Embedding Extraction](#embedding-parameters)
- [Tuning Guide](#tuning-guide)
- [Configuration Loading](#configuration-loading)

---

## HAT Parameters

History-Aware Transformation parameters control the LDA-based feature transformation that improves track discrimination as history accumulates.

### `use_shrinkage`

**Type:** Boolean
**Default:** `true`
**Valid values:** `true`, `false`

Enables Ledoit-Wolf shrinkage covariance estimation in LDA. Shrinkage improves numerical stability when fitting the projection matrix with limited track history samples.

**Impact on behavior:**
- `true`: More stable LDA fitting with small sample counts (early in video). Recommended for most scenarios.
- `false`: Standard empirical covariance. May be unstable with <100 samples per class.

**Tuning recommendation:** Keep enabled unless you have >1000 samples per track class.

---

### `use_weighted_class_mean`

**Type:** Boolean
**Default:** `true`
**Valid values:** `true`, `false`

When computing LDA class means (one per track identity), weight features by detection confidence scores. Higher-confidence detections have stronger influence on the class mean.

**Impact on behavior:**
- `true`: Class means emphasize high-confidence detections. Reduces noisy/low-quality sample influence.
- `false`: All samples weighted equally. May degrade discrimination with low-confidence detections.

**Tuning recommendation:** Enable for real-world data with varying detection quality. Disable only if detection scores are unreliable.

---

### `weighted_class_mean_alpha`

**Type:** Float
**Default:** `1.0`
**Valid range:** `0.0` to `2.0` (higher = stronger weighting effect)

Exponent for score weighting when `use_weighted_class_mean=true`. Controls how strongly detection confidence influences class means.

**Mathematical effect:**
Each feature is weighted as `score^alpha` when computing class means.

**Examples:**
- `alpha=0.0`: Uniform weighting (ignores scores)
- `alpha=1.0`: Linear weighting by score
- `alpha=2.0`: Quadratic weighting (high-confidence detections dominate)

**Impact on behavior:**
- Low values (0.3-0.7): Gentle emphasis on high-confidence samples
- Medium values (0.8-1.2): Balanced weighting (default behavior)
- High values (1.5-2.0): Strong emphasis on high-confidence samples only

**Tuning recommendation:** Start with `1.0`. Increase if many low-confidence detections are present; decrease if all detections are high-quality.

---

### `history_queue_type`

**Type:** String
**Default:** `"fifo"`
**Valid values:** `"fifo"`, `"score"`

Determines which samples are kept in the track history queue when capacity is exceeded.

**Queue types:**

| Mode | Behavior | Use case |
|------|----------|----------|
| `"fifo"` | Keeps most recent samples. Older samples exponentially decay in weight. | Stable appearance with gradual changes (default) |
| `"score"` | Keeps top-k samples by detection confidence. Low-confidence samples dropped. | High-variability scenes with detection quality outliers |

**FIFO details:**
- All samples kept in order of addition
- Weights decay exponentially: `weight(t) = 1.0 * decay_ratio^(age_in_frames)`
- Newer samples have higher influence on LDA

**Score details:**
- Only highest-confidence samples retained when full
- Removed when: `new_sample_score > min_score_in_queue`
- Older high-confidence samples preferred over recent low-confidence

**Impact on behavior:**
- FIFO: Sensitive to appearance changes; better for dynamic clothing/lighting
- Score: Robust to outliers; better for static appearance

**Tuning recommendation:** Use FIFO for video sequences with appearance drift. Use Score for noisy detection scenarios.

---

### `history_max_len`

**Type:** Integer
**Default:** `100`
**Valid range:** `20` to `500+` (depends on memory budget)

Maximum number of samples stored per track history. When exceeded, oldest (FIFO) or lowest-scoring (Score) samples are dropped.

**Memory impact:** Each sample stores one embedding vector of dimension `embedding.dim` (default 256 floats = ~1KB per sample).
Total memory: `num_tracks * history_max_len * 1KB`

**Impact on behavior:**
- Too low (<30): Insufficient history for stable LDA fitting. Discrimination weak in early frames.
- Optimal (60-150): Balance between memory and LDA stability. Most scenarios use 60-100.
- Too high (>300): May degrade performance due to stale samples. Increases memory usage.

**Performance vs. accuracy trade-off:**
- Lower values = faster LDA fitting + lower memory
- Higher values = more stable LDA + better long-term discrimination

**Tuning recommendation:**
- Start with `60` for real-time systems (memory-constrained)
- Use `100` for typical scenarios
- Increase to `150+` only if memory available and tracking long videos

---

### `history_weight_decay`

**Type:** Float
**Default:** `0.7`
**Valid range:** `0.5` to `1.0`

Exponential decay factor applied to FIFO queue sample weights over time. Controls how quickly old samples lose influence.

**Mathematical effect:**
Each frame, sample weight becomes: `weight *= history_weight_decay`

**Examples:**
- `decay=0.5`: Aggressive decay. Weight halves every frame. (Focus on very recent samples)
- `decay=0.7`: Moderate decay. Weight to 70% every frame. (Default)
- `decay=0.9`: Gradual decay. Weight to 90% every frame. (Retain old samples)
- `decay=0.99`: Very gradual. Samples stay influential long-term.

**Impact on behavior:**
- Low decay (0.5-0.6): Strong recency bias. Latest samples dominate LDA. Sensitive to recent appearance changes.
- Medium decay (0.7-0.8): Balanced. Recent samples matter more but older samples still influential.
- High decay (0.9-0.99): Weak recency bias. All samples equally weighted. Smooth LDA across video.

**Only affects FIFO queue.** ScoreQueue does not use decay (uses confidence scores instead).

**Tuning recommendation:**
- Use `0.7` for videos with moderate appearance drift (lighting/pose changes)
- Use `0.5-0.6` for fast appearance changes (clothing occlusions)
- Use `0.85-0.95` for very stable appearance (fixed camera angle/lighting)

---

### `transfer_factor_threshold`

**Type:** Float
**Default:** `3.0`
**Valid range:** `1.0` to `10.0+`

Activation threshold for HAT transformation. HAT activates when:
`(total_history_samples) > (transfer_factor_threshold * num_active_tracks)`

**Rationale:** LDA requires sufficient samples per class. Threshold prevents fitting with too-few samples per track.

**Examples:**
- 5 active tracks, 20 total history samples: `20 / 5 = 4.0`
  - If threshold is `3.0`: HAT activates (4.0 > 3.0) ✓
  - If threshold is `4.5`: HAT inactive (4.0 < 4.5) ✗

**Impact on behavior:**
- Low threshold (1.0-1.5): HAT activates early with few samples per track. May produce unstable projections.
- Medium threshold (2.5-3.5): Balanced. Activates once system has reasonable history. (Default)
- High threshold (5.0+): HAT activates late with abundant history. Very stable but delayed discrimination boost.

**When HAT is inactive:** Feature transformation not applied; tracker uses raw embedding similarity only.

**Tuning recommendation:**
- Keep at `3.0` for balanced behavior
- Reduce to `2.0` for short videos (limited frames to accumulate history)
- Increase to `4.0+` for long videos to ensure sample-rich LDA fits

---

## Tracker Parameters

Multi-object tracker parameters control track lifecycle, similarity matching, and embedding updates.

### `match_score_threshold`

**Type:** Float
**Default:** `0.4`
**Valid range:** `0.0` to `1.0`

Minimum similarity score (cosine/bisoftmax/masa) required to match a detection to an existing track. Below this threshold, detection is treated as a new track.

**Impact on behavior:**
- Too low (0.1-0.2): Every detection matches to some track. ID fragmentation reduced but identity mixing increased.
- Medium (0.4-0.6): Balanced. Typical tracking scenarios use 0.4-0.5.
- Too high (0.8-0.95): Only very similar detections match. Many ID switches. Suitable only for perfect embeddings.

**Trade-off:**
- Lower threshold → fewer ID switches, more fragmentation
- Higher threshold → fewer fragmented tracks, more identity mixing

**Tuning recommendation:**
- Use `0.4` for typical multi-object tracking
- Increase to `0.5-0.6` for crowded scenes with identity confusion
- Decrease to `0.3` for sparse scenes where IDs are stable

---

### `init_score_threshold`

**Type:** Float
**Default:** `0.8`
**Valid range:** `0.0` to `1.0`

Minimum detection confidence required to create a new track. Detections below this are ignored (not tracked).

**Impact on behavior:**
- Low (0.5-0.6): Initializes tracks for low-confidence detections. May create noisy tracks.
- Medium (0.7-0.8): Balanced. Most detections become tracks. (Default)
- High (0.9+): Only very confident detections tracked. Cleaner tracks, fewer false positives.

**Trade-off:**
- Lower threshold → better coverage, more noise
- Higher threshold → fewer false tracks, may miss detections

**Tuning recommendation:**
- Use `0.8` for most scenarios
- Decrease to `0.7` if detector misses many valid detections
- Increase to `0.9` if detector outputs many false positives

---

### `max_lost_frames`

**Type:** Integer
**Default:** `1000`
**Valid range:** `5` to `10000+`

Maximum number of frames a track can remain unmatched before being removed. Allows temporary occlusions and detector failures.

**Impact on behavior:**
- Too low (5-10): Track dies on minor occlusions. Many ID switches on reappearance.
- Medium (30-100): Tracks survive brief occlusions. Typical for controlled environments.
- High (500+): Tracks survive long-term occlusions. Better for outdoor/crowded scenes.

**Trade-off:**
- Lower values → less memory, ID switches on re-detection
- Higher values → longer track lifespan, potential ghost tracks

**Tuning recommendation:**
- Use `1000` to allow very long occlusions (safe default)
- Reduce to `30-100` for real-time systems with memory constraints
- Increase to `2000+` only for archival/offline analysis

---

### `memo_momentum`

**Type:** Float
**Default:** `0.6`
**Valid range:** `0.0` to `1.0`

Exponential Moving Average (EMA) factor for updating track embedding when a new detection is matched.

**Mathematical effect:**
`track_embed = (1 - momentum) * track_embed + momentum * new_embed`

**Examples:**
- `momentum=0.0`: Never update. Track embedding frozen at initialization.
- `momentum=0.5`: Equal weight to old and new detections.
- `momentum=0.8`: Strongly weight new detections. Track adapts quickly to appearance changes.
- `momentum=1.0`: Replace embedding entirely. Only latest detection matters.

**Impact on behavior:**
- Low (0.3-0.4): Conservative updates. Stable tracks but slow to adapt to appearance changes.
- Medium (0.5-0.7): Balanced. (Default 0.6)
- High (0.8-0.95): Aggressive updates. Tracks adapt quickly but become unstable with appearance drift.

**Tuning recommendation:**
- Use `0.6` for typical scenarios
- Decrease to `0.4` if appearance is stable (fixed pose/lighting)
- Increase to `0.8` if appearance changes rapidly (pose/occlusion variations)

---

### `similarity_mode`

**Type:** String
**Default:** `"masa"`
**Valid values:** `"cosine"`, `"bisoftmax"`, `"masa"`

Metric used to compute similarity between detections and track embeddings.

**Modes:**

| Mode | Formula | Range | Use case |
|------|---------|-------|----------|
| `"cosine"` | `X · Y / (‖X‖ ‖Y‖)` | [0, 1] | Standard ReID. Fast, robust. |
| `"bisoftmax"` | `(1 + tanh(2*cosine - 1)) / 2` | [0, 1] | Emphasizes medium similarities. Smoother gradients. |
| `"masa"` | HAT-MASA metric. Combines embedding and transformation scores. | [0, 1] | State-of-the-art when HAT active. |

**Impact on behavior:**
- Cosine: Linear similarity. Straightforward but may be too lenient/strict with threshold.
- Bisoftmax: Non-linear. Highlights confusion boundary. Better discrimination at decision boundary.
- MASA: Adaptive. Leverages HAT transformation. Best accuracy when history available.

**Tuning recommendation:**
- Use `"masa"` (HAT project default) for best accuracy with HAT
- Use `"cosine"` if MASA unavailable or for baseline comparison
- Use `"bisoftmax"` for debug/analysis of similarity distributions

---

## Embedding Parameters

Feature extraction configuration for ReID embeddings used in matching.

### `model`

**Type:** String
**Default:** `"resnet50"`
**Valid values:** `"resnet50"`, `"resnet18"`, `"efficientnet_b0"`

Backbone CNN architecture for embedding extraction. All models use ImageNet pretraining.

**Models:**

| Model | Parameters | Speed | Accuracy | Memory |
|-------|------------|-------|----------|--------|
| ResNet50 | ~23M | Medium | High | ~400MB |
| ResNet18 | ~11M | Fast | Good | ~200MB |
| EfficientNet-B0 | ~5M | Very Fast | Good | ~100MB |

**Impact on behavior:**
- ResNet50: Most discriminative. Best for challenging scenarios. Higher latency.
- ResNet18: Balanced. Good for real-time systems.
- EfficientNet-B0: Fastest. Best for embedded systems. Slightly lower accuracy.

**Tuning recommendation:**
- Use `"resnet50"` for offline/high-accuracy requirements
- Use `"resnet18"` for real-time systems (good balance)
- Use `"efficientnet_b0"` for edge devices (memory/compute constrained)

---

### `dim`

**Type:** Integer
**Default:** `256`
**Valid range:** `64` to `2048`

Output embedding dimension after projection layer. Controls feature space dimensionality for matching.

**Impact on behavior:**
- Too low (32-64): Fast but limited discriminative power. May confuse similar-looking people.
- Medium (128-256): Balanced. Standard in ReID literature. (Default)
- Too high (512+): Highly discriminative but slower matching and higher memory.

**Memory per sample:** `dim * 4 bytes` (float32)
Example: 256D embedding = 1KB per sample

**Tuning recommendation:**
- Use `256` for typical scenarios (standard practice)
- Reduce to `128` for speed-critical applications
- Increase to `512` only if accuracy is critical and resources available

---

### `normalize`

**Type:** Boolean
**Default:** `true`
**Valid values:** `true`, `false`

L2-normalize embeddings after extraction. Ensures embeddings lie on unit hypersphere (norm = 1).

**Impact on behavior:**
- `true`: Cosine similarity equals dot product. Embeddings magnitude-invariant. Recommended.
- `false`: Embeddings not normalized. Magnitude encodes information but makes similarity computation less stable.

**Tuning recommendation:** Keep enabled. Normalization is standard in modern ReID and improves similarity stability.

---

### `input_size`

**Type:** List of integers
**Default:** `[256, 128]`
**Valid range:** Height 64-512, Width 32-256

Input crop dimensions for CNN backbone: `[height, width]` in pixels. Detections are resized to this size before extraction.

**Aspect ratio:** Default 256×128 = 2:1 (tall/narrow) matches typical person bounding box proportions.

**Impact on behavior:**
- Smaller size (e.g., 128×64): Faster inference but loses detail. Good for small objects.
- Default (256×128): Balanced. Recommended for person ReID.
- Larger size (e.g., 384×192): More detail but slower. Only if high-resolution required.

**Trade-off:**
- Smaller → faster extraction but lower accuracy
- Larger → higher accuracy but slower extraction

**Tuning recommendation:**
- Use `[256, 128]` (default)
- Use `[128, 64]` for real-time systems
- Use `[384, 192]` only if extracting high-resolution crops (>500px tall)

---

## Tuning Guide

### Scenario: Real-Time Tracking (Low Latency)

Optimize for speed with acceptable accuracy:

```yaml
hat:
  use_shrinkage: true
  history_max_len: 30              # Fewer samples = faster LDA
  transfer_factor_threshold: 3.0

tracker:
  match_score_threshold: 0.4
  max_lost_frames: 100             # Lower for memory efficiency
  memo_momentum: 0.6

embedding:
  model: "resnet18"                # Smaller backbone
  dim: 128                         # Lower dimension
  input_size: [128, 64]            # Smaller crops
```

### Scenario: High-Accuracy Offline (Highest Accuracy)

Optimize for accuracy with computation budget available:

```yaml
hat:
  use_shrinkage: true
  use_weighted_class_mean: true
  weighted_class_mean_alpha: 1.2   # Emphasize high-confidence samples
  history_max_len: 150             # More samples for stable LDA
  history_weight_decay: 0.85       # Retain older samples
  transfer_factor_threshold: 2.5   # Activate HAT earlier

tracker:
  match_score_threshold: 0.5       # More selective
  max_lost_frames: 1000            # Allow long occlusions
  memo_momentum: 0.5               # Conservative updates
  similarity_mode: "masa"          # Best metric

embedding:
  model: "resnet50"                # Largest backbone
  dim: 256                         # Standard dimension
  input_size: [256, 128]           # Standard crops
```

### Scenario: Crowded Scene (Identity Confusion)

Improve discrimination in dense crowds:

```yaml
hat:
  use_weighted_class_mean: true
  weighted_class_mean_alpha: 1.5   # Strong confidence emphasis
  history_queue_type: "score"      # Keep high-quality samples
  history_max_len: 100
  transfer_factor_threshold: 2.0   # Activate HAT quickly

tracker:
  match_score_threshold: 0.55      # Higher threshold
  init_score_threshold: 0.85       # Only confident detections
  memo_momentum: 0.7               # Adapt to appearance changes
  similarity_mode: "masa"

embedding:
  model: "resnet50"
  dim: 256
```

### Scenario: Long-Term Tracking (Hours of Video)

Handle long sequences with appearance drift:

```yaml
hat:
  history_queue_type: "fifo"       # Time-based, not score-based
  history_max_len: 200             # Large history
  history_weight_decay: 0.9        # Retain older samples
  transfer_factor_threshold: 2.5

tracker:
  match_score_threshold: 0.4       # Lower to retain IDs
  max_lost_frames: 2000            # Extended occlusion tolerance
  memo_momentum: 0.7               # Adapt to changes
  similarity_mode: "masa"

embedding:
  model: "resnet50"
  dim: 256
```

---

## Configuration Loading

### Loading from YAML

```python
import yaml
from src.tracker import HATTracker
from src.embeddings import EmbeddingExtractor

# Load config
with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

# Initialize tracker
tracker = HATTracker(
    **config["tracker"],
    use_shrinkage=config["hat"]["use_shrinkage"],
    use_weighted_class_mean=config["hat"]["use_weighted_class_mean"],
)

# Initialize extractor
extractor = EmbeddingExtractor(
    model_name=config["embedding"]["model"],
    embedding_dim=config["embedding"]["dim"],
    input_size=tuple(config["embedding"]["input_size"]),
    normalize=config["embedding"]["normalize"],
)
```

### Programmatic Override

```python
# Create tracker with custom parameters
tracker = HATTracker(
    match_score_threshold=0.5,    # Override default
    history_max_len=80,
    device="cuda",
)
```

---

## Quick Reference Table

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| **HAT** | | | |
| `use_shrinkage` | true | bool | Stability |
| `weighted_class_mean_alpha` | 1.0 | 0.0-2.0 | Weighting strength |
| `history_max_len` | 100 | 20-500 | Memory/stability |
| `history_weight_decay` | 0.7 | 0.5-1.0 | Recency bias |
| `transfer_factor_threshold` | 3.0 | 1.0-10.0 | HAT activation |
| **Tracker** | | | |
| `match_score_threshold` | 0.4 | 0.0-1.0 | Matching sensitivity |
| `init_score_threshold` | 0.8 | 0.0-1.0 | Track creation |
| `max_lost_frames` | 1000 | 5-10000 | Occlusion tolerance |
| `memo_momentum` | 0.6 | 0.0-1.0 | Update speed |
| **Embedding** | | | |
| `model` | resnet50 | see table | Speed/accuracy |
| `dim` | 256 | 64-2048 | Discriminative power |
| `input_size` | [256,128] | varies | Detail/speed |
