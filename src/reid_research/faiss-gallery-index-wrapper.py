"""FAISS-based gallery index for fast approximate nearest neighbor search.

Provides O(log n) search via IVF indexing with automatic fallback to flat index
for small galleries. Supports optional GPU acceleration.
"""
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FAISSGalleryIndex:
    """Wrapper for FAISS index with dynamic updates and auto IVF/flat selection.

    Uses flat (brute-force) index for small galleries and IVF for larger ones.
    Rebuilds index lazily when data changes, on the next search call.

    Args:
        dimension: Feature vector dimension (512 for OSNet)
        nlist: Number of IVF clusters (tune to sqrt of expected gallery size)
        nprobe: Clusters to search per query (accuracy vs speed tradeoff)
        use_gpu: Whether to use GPU acceleration if available
        min_train_size: Minimum vectors before switching from flat to IVF
    """

    def __init__(
        self,
        dimension: int = 512,
        nlist: int = 64,
        nprobe: int = 8,
        use_gpu: bool = True,
        min_train_size: int = 100,
    ):
        self.dimension = dimension
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu and FAISS_AVAILABLE and faiss.get_num_gpus() > 0
        self.min_train_size = min_train_size

        self._index = None
        self._id_map: list[int] = []  # FAISS internal idx -> track_id
        self._features: list[np.ndarray] = []  # Raw features for retraining
        self._needs_rebuild = True

    def add(self, track_id: int, feature: np.ndarray) -> None:
        """Add feature to index (deferred rebuild until next search)."""
        self._id_map.append(track_id)
        self._features.append(feature.astype(np.float32))
        self._needs_rebuild = True

    def remove(self, track_id: int) -> None:
        """Remove track from index (applied on next rebuild)."""
        if track_id in self._id_map:
            idx = self._id_map.index(track_id)
            self._id_map.pop(idx)
            self._features.pop(idx)
            self._needs_rebuild = True

    def update(self, track_id: int, feature: np.ndarray) -> None:
        """Update feature for existing track, or add if new."""
        if track_id in self._id_map:
            idx = self._id_map.index(track_id)
            self._features[idx] = feature.astype(np.float32)
            self._needs_rebuild = True
        else:
            self.add(track_id, feature)

    def search(self, query: np.ndarray, k: int = 50) -> list[tuple[int, float]]:
        """Search for k nearest neighbors.

        Args:
            query: (D,) or (1, D) query feature vector
            k: Number of neighbors to return

        Returns:
            List of (track_id, L2_distance) sorted ascending by distance
        """
        if not self._features:
            return []

        self._ensure_built()

        query = query.reshape(1, -1).astype(np.float32)
        k = min(k, len(self._features))
        distances, indices = self._index.search(query, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self._id_map):
                results.append((self._id_map[idx], float(dist)))
        return results

    def search_batch(self, queries: np.ndarray, k: int = 50) -> list[list[tuple[int, float]]]:
        """Batch search for multiple queries.

        Args:
            queries: (Q, D) query feature matrix
            k: Number of neighbors per query

        Returns:
            List of Q result lists, each containing (track_id, distance) tuples
        """
        if not self._features:
            return [[] for _ in range(len(queries))]

        self._ensure_built()

        queries = queries.astype(np.float32)
        k = min(k, len(self._features))
        distances, indices = self._index.search(queries, k)

        results = []
        for q_dists, q_indices in zip(distances, indices):
            q_results = []
            for dist, idx in zip(q_dists, q_indices):
                if 0 <= idx < len(self._id_map):
                    q_results.append((self._id_map[idx], float(dist)))
            results.append(q_results)
        return results

    def _ensure_built(self) -> None:
        """Build or rebuild index if data changed."""
        if not self._needs_rebuild and self._index is not None:
            return

        features = np.vstack(self._features).astype(np.float32)
        n_features = len(features)

        # Use flat index for small galleries (exact search)
        if n_features < self.min_train_size:
            self._index = faiss.IndexFlatL2(self.dimension)
        else:
            # IVF index for larger galleries (approximate search)
            nlist = min(self.nlist, n_features // 4)  # At least 4 vectors per cluster
            quantizer = faiss.IndexFlatL2(self.dimension)
            self._index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self._index.train(features)
            self._index.nprobe = self.nprobe

        self._index.add(features)

        # Move to GPU if available
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self._index = faiss.index_cpu_to_gpu(res, 0, self._index)

        self._needs_rebuild = False

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return len(self._id_map)

    def clear(self) -> None:
        """Clear all data and reset index."""
        self._index = None
        self._id_map.clear()
        self._features.clear()
        self._needs_rebuild = True
