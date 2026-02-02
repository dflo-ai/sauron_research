# HAT-ReID API Reference

## Core Modules

### `src.hat_reid`

#### `get_device() -> torch.device`
Get CUDA device. Raises `RuntimeError` if GPU unavailable.

#### `LDA`
Linear Discriminant Analysis for feature transformation.

```python
class LDA:
    def __init__(
        self,
        use_shrinkage: bool = True,
        use_weighted_class_mean: bool = True,
        weighted_class_mean_alpha: float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
    ): ...

    def fit(self, X: Tensor, y: Tensor, scores: Tensor | None = None) -> LDA:
        """Fit projection matrix from track history.

        Args:
            X: Features (N, D) from all tracks
            y: Track IDs (N,) - class labels
            scores: Detection confidence (N,) - optional weights
        """

    def transform(self, X: Tensor) -> Tensor:
        """Project features to discriminative subspace.

        Returns: (N, K) where K = num_classes - 1
        """

    def is_fitted(self) -> bool: ...
    def clear(self) -> None: ...
```

#### `FIFOQueue`
Time-based history queue with exponential weight decay.

```python
class FIFOQueue:
    def __init__(
        self,
        max_len: int = 60,
        decay_ratio: float = 0.9,
        use_decay_as_weight: bool = True,
    ): ...

    def add(self, feature: Tensor, score: float) -> None: ...
    def get_features_and_weights(self) -> tuple[Tensor, Tensor]: ...
    def clear(self) -> None: ...
```

#### `ScoreQueue`
Score-based history queue keeping top-k by confidence.

```python
class ScoreQueue:
    def __init__(self, max_len: int = 60, decay_ratio: float = 0.9): ...
    def add(self, feature: Tensor, score: float) -> None: ...
    def get_features_and_weights(self) -> tuple[Tensor, Tensor]: ...
    def clear(self) -> None: ...
```

---

### `src.tracker`

#### `HATTracker`
Multi-object tracker with HAT integration.

```python
class HATTracker:
    def __init__(
        self,
        # Track management
        init_score_thr: float = 0.8,
        match_score_thr: float = 0.5,
        max_lost_frames: int = 10,
        memo_momentum: float = 0.8,
        # HAT config
        use_hat: bool = True,
        hat_factor_thr: float = 4.0,
        history_max_len: int = 60,
        history_decay: float = 0.9,
        queue_type: str = "fifo",  # "fifo" or "score"
        # Matching
        similarity_mode: str = "cosine",  # "cosine", "bisoftmax", "masa"
        device: str = "cuda",
    ): ...

    def update(
        self,
        boxes: Tensor,       # (N, 4) xyxy
        scores: Tensor,      # (N,)
        embeddings: Tensor,  # (N, D)
        frame_id: int,
    ) -> Tensor:
        """Process frame and return track IDs.

        Returns: (N,) track IDs, -1 for unmatched
        """

    def reset(self) -> None: ...

    @property
    def num_tracks(self) -> int: ...

    @property
    def hat_active(self) -> bool: ...
```

#### `Track`
Single track state dataclass.

```python
@dataclass
class Track:
    id: int
    bbox: Tensor           # (4,) xyxy
    embed: Tensor          # (D,) averaged embedding
    score: float
    last_frame: int
    history: FIFOQueue | ScoreQueue
```

---

### `src.embeddings`

#### `EmbeddingExtractor`
ReID feature extraction from person crops.

```python
class EmbeddingExtractor:
    def __init__(
        self,
        model_name: str = "resnet50",  # resnet50, resnet18, efficientnet_b0
        embedding_dim: int = 256,
        input_size: tuple[int, int] = (256, 128),  # H, W
        normalize: bool = True,
        device: str = "cuda",
    ): ...

    def extract(self, crops: list[Tensor] | Tensor) -> Tensor:
        """Extract embeddings from crops.

        Args:
            crops: List of (C,H,W) or batch (N,C,H,W)

        Returns: (N, embedding_dim) L2-normalized
        """

    def extract_from_frame(
        self,
        frame: Tensor,      # (H,W,C) or (C,H,W)
        boxes: Tensor,      # (N, 4) xyxy
        min_size: int = 20,
    ) -> Tensor:
        """Extract embeddings from frame regions.

        Returns: (N, embedding_dim), zeros for skipped boxes
        """
```

#### `create_extractor(model, dim, device) -> EmbeddingExtractor`
Factory function with defaults.

---

## Configuration Schema

```yaml
hat:
  use_shrinkage: true
  use_weighted_class_mean: true
  weighted_class_mean_alpha: 1.0
  history_queue_type: "fifo"      # or "score"
  history_max_len: 60
  history_weight_decay: 0.9
  transfer_factor_threshold: 4.0  # HAT activates when history > 4x tracks

tracker:
  match_score_threshold: 0.5
  init_score_threshold: 0.8
  max_lost_frames: 10
  memo_momentum: 0.8
  similarity_mode: "cosine"       # cosine, bisoftmax, masa

embedding:
  model: "resnet50"
  dim: 256
  normalize: true
  input_size: [256, 128]
```

For detailed parameter documentation, tuning guides, and examples, see [`configuration-parameters.md`](./configuration-parameters.md).
