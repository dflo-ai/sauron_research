# Sauron Services Integration Guide

## Overview

This document describes how to integrate HAT-ReID into sauron-services tracking.

## Target Files

| Sauron File | Changes |
|-------------|---------|
| `sfsort_tracker.py` | Add embedding support, HAT integration |
| `person_detector.py` | Add embedding extraction call |
| `detector_manager.py` | Pass embeddings through pipeline |

## Step 1: Add HAT-ReID Package

Copy or install the research package:

```bash
# Option A: Copy source
cp -r src/hat_reid sauron-services/table_status/src/
cp -r src/embeddings sauron-services/table_status/src/

# Option B: Install as package
pip install -e /path/to/sauron_research
```

## Step 2: Modify Track Class

In `sfsort_tracker.py`, add embedding storage to Track:

```python
from hat_reid.queues import FIFOQueue

@dataclass
class Track:
    id: int
    bbox: np.ndarray
    score: float
    last_frame: int
    # NEW: ReID fields
    embedding: np.ndarray | None = None
    history: FIFOQueue | None = None
```

## Step 3: Add Embedding Support to SFSORT.update()

```python
def update(
    self,
    boxes: np.ndarray,
    scores: np.ndarray,
    embeddings: np.ndarray | None = None,  # NEW
    frame_id: int = 0,
) -> np.ndarray:
    """
    Args:
        embeddings: Optional (N, D) ReID features
    """
    # Existing IoU matching...

    # NEW: If embeddings provided, use appearance-only cost
    if embeddings is not None and self.use_appearance:
        appearance_cost = self._compute_appearance_cost(embeddings)
        cost = appearance_cost  # Pure ReID matching
    else:
        cost = spatial_cost
```

## Step 4: Integrate HAT

```python
from hat_reid import LDA

class SFSORTTracker:
    def __init__(self, ..., use_hat: bool = False):
        self.use_hat = use_hat
        self.hat_factor_thr = 4.0

    def _compute_appearance_cost(self, det_embeds):
        track_embeds, hist_embeds, hist_ids = self._get_embedding_memo()

        # HAT activation check
        if self.use_hat and len(hist_embeds) > self.hat_factor_thr * len(self.tracks):
            lda = LDA(device="cuda")
            lda.fit(hist_embeds, hist_ids)
            det_embeds = lda.transform(det_embeds)
            track_embeds = lda.transform(track_embeds)

        # Cosine similarity -> cost
        det_norm = det_embeds / det_embeds.norm(dim=1, keepdim=True)
        track_norm = track_embeds / track_embeds.norm(dim=1, keepdim=True)
        similarity = det_norm @ track_norm.T
        return 1 - similarity  # Cost = 1 - similarity
```

## Step 5: Add Embedding Extraction in PersonDetector

```python
from embeddings import EmbeddingExtractor

class PersonDetector:
    def __init__(self, ..., use_reid: bool = False):
        self.use_reid = use_reid
        if use_reid:
            self.extractor = EmbeddingExtractor(device=self.device)

    def detect(self, frame):
        detections = self.model(frame)

        embeddings = None
        if self.use_reid and len(detections) > 0:
            embeddings = self.extractor.extract_from_frame(
                frame, detections.xyxy
            )

        track_ids = self.tracker.update(
            detections.xyxy,
            detections.confidence,
            embeddings=embeddings,  # NEW
        )
```

## API Contract

### EmbeddingExtractor

```python
class EmbeddingExtractor:
    def __init__(
        self,
        model_name: str = "resnet50",
        embedding_dim: int = 256,
        device: str = "cuda",
    ): ...

    def extract_from_frame(
        self,
        frame: Tensor,      # (H, W, 3) or (3, H, W)
        boxes: Tensor,      # (N, 4) xyxy
    ) -> Tensor:            # (N, embedding_dim)
        ...
```

### LDA

```python
class LDA:
    def __init__(
        self,
        use_shrinkage: bool = True,
        use_weighted_class_mean: bool = True,
        device: str = "cuda",
    ): ...

    def fit(
        self,
        X: Tensor,             # (N, D) features
        y: Tensor,             # (N,) track IDs
        scores: Tensor = None, # (N,) weights
    ) -> LDA: ...

    def transform(self, X: Tensor) -> Tensor: ...
```

## Configuration Flags

Add to sauron config:

```yaml
tracking:
  use_reid: true           # Enable ReID embeddings
  use_hat: true            # Enable HAT transformation
  hat_factor_threshold: 4.0
  reid_model: "resnet50"
  embedding_dim: 256
```

For comprehensive parameter tuning and configuration examples, see [`configuration-parameters.md`](./configuration-parameters.md).

## Backward Compatibility

- `embeddings` parameter is optional - falls back to existing behavior
- Feature flag `use_reid` gates all ReID code paths
- No changes to existing API signatures (additive only)

## Performance Impact

| Config | Added Latency |
|--------|---------------|
| ReID only | +15-25ms |
| ReID + HAT | +25-35ms |

Acceptable for 25-30 FPS restaurant use case.

## Testing Checklist

- [ ] Unit tests pass with `embeddings=None`
- [ ] Track IDs stable with ReID enabled
- [ ] HAT activates after history builds up
- [ ] No memory leak from history queues
- [ ] Performance within latency budget
