"""Cached k-NN graph for selective re-ranking optimization.

Pre-computes gallery-gallery nearest neighbor relationships to accelerate
k-reciprocal re-ranking. Only rebuild when gallery changes significantly.
"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class GalleryKNNCache:
    """Cached k-NN graph for fast re-ranking lookups.

    Maintains a pre-computed k-NN graph over gallery features. Used by
    selective re-ranking to quickly determine reciprocal neighbor sets
    without recomputing all-pairs distances on every query.

    Args:
        k: Number of nearest neighbors to store per gallery entry
    """

    def __init__(self, k: int = 20):
        self.k = k
        self._graph: np.ndarray | None = None  # (G, k) neighbor indices
        self._distances: np.ndarray | None = None  # (G, k) distances
        self._gallery_ids: list[int] = []
        self._needs_rebuild = True

    def update(self, gallery_features: dict[int, np.ndarray]) -> None:
        """Rebuild k-NN graph from current gallery features.

        Args:
            gallery_features: Dict mapping track_id -> avg_feature vector
        """
        if not gallery_features:
            self._graph = None
            self._distances = None
            self._gallery_ids = []
            return

        self._gallery_ids = list(gallery_features.keys())
        features = np.vstack([gallery_features[tid] for tid in self._gallery_ids])

        # All-pairs euclidean distances
        dist_matrix = euclidean_distances(features, features)

        # Get top-k neighbors for each gallery item (exclude self at index 0)
        k = min(self.k, len(self._gallery_ids) - 1)
        if k < 1:
            self._graph = None
            self._distances = None
            return

        self._graph = np.argsort(dist_matrix, axis=1)[:, 1:k + 1]
        self._distances = np.sort(dist_matrix, axis=1)[:, 1:k + 1]
        self._needs_rebuild = False

    def get_neighbors(self, track_id: int) -> tuple[list[int], list[float]]:
        """Get k-NN for a gallery track.

        Args:
            track_id: Gallery track ID

        Returns:
            Tuple of (neighbor_track_ids, neighbor_distances)
        """
        if self._graph is None or track_id not in self._gallery_ids:
            return [], []

        idx = self._gallery_ids.index(track_id)
        neighbor_indices = self._graph[idx]
        neighbor_ids = [self._gallery_ids[i] for i in neighbor_indices]
        neighbor_dists = self._distances[idx].tolist()

        return neighbor_ids, neighbor_dists

    def get_reciprocal_set(self, track_id: int) -> set[int]:
        """Get k-reciprocal neighbors (bidirectional nearest neighbors).

        A neighbor is reciprocal if track_id is in neighbor's top-k AND
        neighbor is in track_id's top-k.

        Args:
            track_id: Gallery track ID

        Returns:
            Set of track IDs that are reciprocal neighbors
        """
        if self._graph is None:
            return set()

        neighbors, _ = self.get_neighbors(track_id)
        reciprocal = set()

        for neighbor_id in neighbors:
            reverse_neighbors, _ = self.get_neighbors(neighbor_id)
            if track_id in reverse_neighbors:
                reciprocal.add(neighbor_id)

        return reciprocal

    @property
    def size(self) -> int:
        """Number of entries in the cache."""
        return len(self._gallery_ids)

    @property
    def is_stale(self) -> bool:
        """Whether cache needs rebuilding."""
        return self._needs_rebuild

    def invalidate(self) -> None:
        """Mark cache as needing rebuild."""
        self._needs_rebuild = True
