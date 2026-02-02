"""History queues for HAT feature accumulation.

Two queue types supported:
- FIFOQueue: Time-based with exponential weight decay (older samples decay)
- ScoreQueue: Score-based keeping top-k by confidence
"""

from collections import deque
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class HistoryItem:
    """Single item in history queue."""
    feature: Tensor
    score: float
    weight: float = 1.0


class FIFOQueue:
    """Time-based queue with exponential weight decay.

    Older samples have decayed weights, newer samples have higher weights.
    Based on HAT-MASA fifo_queue.py implementation.
    """

    def __init__(self, max_len: int = 60, decay_ratio: float = 0.9, use_decay_as_weight: bool = True):
        """Initialize FIFO queue.

        Args:
            max_len: Maximum number of samples to store
            decay_ratio: Weight decay factor applied each time new sample added
            use_decay_as_weight: If True, use decayed weights; else use original scores
        """
        self.max_len = max_len
        self.decay_ratio = decay_ratio
        self.use_decay_as_weight = use_decay_as_weight
        self._features: deque[Tensor] = deque(maxlen=max_len)
        self._scores: deque[float] = deque(maxlen=max_len)
        self._weights: deque[float] = deque(maxlen=max_len)

    def add(self, feature: Tensor, score: float) -> None:
        """Add new feature to queue with weight decay on existing items.

        Args:
            feature: Feature tensor (D,)
            score: Detection confidence score
        """
        self._features.append(feature.detach().clone())
        self._scores.append(score)
        self._weights.append(1.0)

        # Decay all weights (including the just-added one will be decayed on next add)
        for i in range(len(self._weights)):
            self._weights[i] *= self.decay_ratio

    def get_features_and_weights(self) -> tuple[Tensor, Tensor]:
        """Get all features and their weights.

        Returns:
            features: (N, D) tensor of features
            weights: (N,) tensor of weights (decayed or scores based on config)
        """
        if not self._features:
            return torch.empty(0), torch.empty(0)

        features = torch.stack(list(self._features))
        if self.use_decay_as_weight:
            weights = torch.tensor(list(self._weights), device=features.device)
        else:
            weights = torch.tensor(list(self._scores), device=features.device)
        return features, weights

    def __len__(self) -> int:
        return len(self._features)

    def clear(self) -> None:
        """Clear all items from queue."""
        self._features.clear()
        self._scores.clear()
        self._weights.clear()


class ScoreQueue:
    """Score-based queue keeping top-k samples by confidence.

    When queue exceeds max_len, removes lowest-scoring samples.
    Scores decay over time so newer samples are preferred at equal confidence.
    Based on HAT-MASA score_queue.py implementation.
    """

    def __init__(self, max_len: int = 60, decay_ratio: float = 0.9):
        """Initialize score queue.

        Args:
            max_len: Maximum number of samples to keep
            decay_ratio: Score decay factor applied to existing scores on each add
        """
        self.max_len = max_len
        self.decay_ratio = decay_ratio
        self.features: Tensor | None = None
        self.scores: Tensor | None = None

    def add(self, feature: Tensor, score: float) -> None:
        """Add new feature, keeping only top-k by score.

        Args:
            feature: Feature tensor (D,)
            score: Detection confidence score
        """
        feat = feature.detach().clone().unsqueeze(0)  # (1, D)
        sc = torch.tensor([score], device=feature.device, dtype=feature.dtype)

        if self.features is None:
            self.features = feat
            self.scores = sc
        else:
            # Decay existing scores
            self.scores = self.scores * self.decay_ratio
            # Concatenate new
            self.features = torch.cat([self.features, feat], dim=0)
            self.scores = torch.cat([self.scores, sc], dim=0)

        # Keep top-k by score
        if len(self) > self.max_len:
            top_idx = torch.argsort(self.scores, descending=True)[:self.max_len]
            self.features = self.features[top_idx]
            self.scores = self.scores[top_idx]

    def get_features_and_weights(self) -> tuple[Tensor, Tensor]:
        """Get all features and scores.

        Returns:
            features: (N, D) tensor
            scores: (N,) tensor used as weights
        """
        if self.features is None:
            return torch.empty(0), torch.empty(0)
        return self.features, self.scores

    def __len__(self) -> int:
        return 0 if self.features is None else len(self.features)

    def clear(self) -> None:
        """Clear all items from queue."""
        self.features = None
        self.scores = None
