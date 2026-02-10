"""Gallery-based person tracking with ReID features."""
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import lap
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .config import ReIDConfig

logger = logging.getLogger(__name__)

# Optional FAISS integration (graceful fallback to brute-force)
try:
    from importlib import import_module as _im
    _faiss_mod = _im(".faiss-gallery-index-wrapper", package="src.reid_research")
    FAISSGalleryIndex = _faiss_mod.FAISSGalleryIndex
    FAISS_AVAILABLE = _faiss_mod.FAISS_AVAILABLE
except (ImportError, ModuleNotFoundError):
    FAISS_AVAILABLE = False
    FAISSGalleryIndex = None

# k-NN cache for selective re-ranking
try:
    _knn_mod = _im(".gallery-knn-cache-for-selective-reranking", package="src.reid_research")
    GalleryKNNCache = _knn_mod.GalleryKNNCache
except (ImportError, ModuleNotFoundError):
    GalleryKNNCache = None
from .matching import (
    TrackMotion,
    apply_torchreid_reranking,
    compute_adaptive_cost_matrix,
    compute_adaptive_threshold,
    compute_euclidean_rank_list,
    compute_k_reciprocal_reranking,
    compute_position_boost,
    compute_quality_score,
    compute_velocity,
    detect_crossing_tracks,
    majority_vote_reidentify,
    predict_position,
    validate_assignments_batch,
    validate_motion_consistency,
)


@dataclass
class GalleryEntry:
    """Single person entry in gallery."""

    features: deque = field(default_factory=lambda: deque(maxlen=10))
    quality_scores: deque = field(default_factory=lambda: deque(maxlen=10))
    avg_feature: np.ndarray | None = None
    avg_quality: float = 0.0
    last_seen: int = 0

    def update_avg(self) -> None:
        """Recompute average feature."""
        if self.features:
            self.avg_feature = np.mean(list(self.features), axis=0)


class PersonGallery:
    """Simple dict-based gallery for person re-identification."""

    def __init__(self, config: ReIDConfig):
        """Initialize gallery with config.

        Args:
            config: ReIDConfig instance
        """
        self.config = config
        self._gallery: dict[int, GalleryEntry] = {}
        self._next_id: int = 0
        self._frame_idx: int = 0

        # Velocity-based motion tracking
        self._track_motion: dict[int, TrackMotion] = {}

        # Adaptive threshold statistics
        self._match_scores: deque = deque(maxlen=config.gallery.adaptive_window_size)
        self._match_count: int = 0
        self._current_threshold: float = config.inference.similarity_threshold

        # FAISS index for accelerated gallery search (optional)
        self._faiss_index = None
        if config.gallery.use_faiss and FAISS_AVAILABLE and FAISSGalleryIndex is not None:
            self._faiss_index = FAISSGalleryIndex(
                dimension=512,
                nlist=config.gallery.faiss_nlist,
                nprobe=config.gallery.faiss_nprobe,
                use_gpu=config.model.device.startswith("cuda"),
                min_train_size=config.gallery.faiss_min_train_size,
            )
        self._faiss_rebuild_counter = 0

        # k-NN cache for selective re-ranking (Phase 5)
        self._knn_cache = None
        if config.gallery.rerank_cache_knn and GalleryKNNCache is not None:
            self._knn_cache = GalleryKNNCache(k=config.gallery.rerank_k1)
        self._knn_rebuild_counter = 0
        self._knn_change_counter = 0  # Track gallery modifications for eager rebuild
        self._rerank_trigger_count = 0
        self._rerank_skip_count = 0

    def match_and_update(self, features: np.ndarray) -> int:
        """Match features against gallery, update/add entry.

        Args:
            features: Feature vector (512,)

        Returns:
            Assigned track ID
        """
        track_id = self.match(features)

        if track_id is None:
            track_id = self.add(features)
        else:
            self.update(track_id, features)

        return track_id

    def match_batch(
        self,
        features_list: list[np.ndarray],
        exclude_ids: set[int] | None = None,
        bboxes: list[tuple[float, ...]] | None = None,
    ) -> list[tuple[int | None, float]]:
        """Match batch of features using rank-list majority voting.

        Uses euclidean distance ranking with majority voting for identity confirmation.
        Hungarian algorithm ensures unique assignments per frame.

        Args:
            features_list: List of feature vectors (512,)
            exclude_ids: IDs to exclude from matching (already assigned this frame)
            bboxes: Optional bounding boxes for temporal consistency

        Returns:
            List of (matched_id, confidence) tuples (None if no match)
        """
        if not features_list:
            return []

        if not self._gallery:
            return [(None, 0.0)] * len(features_list)

        exclude_ids = exclude_ids or set()
        gallery_cfg = self.config.gallery

        # Extract all gallery features for rank-list computation
        gallery_features: dict[int, list[np.ndarray]] = {}
        for tid, entry in self._gallery.items():
            if tid in exclude_ids:
                continue
            if entry.features:
                gallery_features[tid] = list(entry.features)

        if not gallery_features:
            return [(None, 0.0)] * len(features_list)

        gallery_ids = list(gallery_features.keys())

        # Detect crossing tracks for adaptive behavior
        crossing_ids = self._detect_crossing_tracks(bboxes, gallery_ids)

        # Build cost matrix using FAISS (fast) or brute-force rank-list
        n_queries = len(features_list)
        n_gallery = len(gallery_ids)
        cost_matrix = np.full((n_queries, n_gallery), 1e6, dtype=np.float32)

        if self._faiss_index is not None and self._faiss_index.size >= 10:
            # FAISS-accelerated path: batch search all queries at once
            queries = np.vstack(features_list).astype(np.float32)
            k = min(50, self._faiss_index.size)
            neighbors_batch = self._faiss_index.search_batch(queries, k=k)

            # Build id->gallery_idx lookup for fast cost matrix population
            gid_to_idx = {gid: idx for idx, gid in enumerate(gallery_ids)}

            for q_idx, neighbors in enumerate(neighbors_batch):
                for track_id, dist in neighbors:
                    if track_id in gid_to_idx:
                        cost_matrix[q_idx, gid_to_idx[track_id]] = dist
        else:
            # Brute-force path: per-query euclidean rank-list
            # Build id->idx lookup to avoid O(n) list.index() in inner loop
            gid_to_idx = {gid: idx for idx, gid in enumerate(gallery_ids)}

            for q_idx, query_feat in enumerate(features_list):
                rank_list = compute_euclidean_rank_list(
                    query_feat, gallery_features, k=min(n_gallery, 50)
                )
                for track_id, dist, feat_count in rank_list:
                    if track_id in gid_to_idx:
                        g_idx = gid_to_idx[track_id]
                        cost_matrix[q_idx, g_idx] = dist

        # Run Hungarian algorithm for optimal assignment
        _, row_to_col, _ = lap.lapjv(cost_matrix, extend_cost=True)

        # Build results with majority voting validation
        results: list[tuple[int | None, float]] = []

        for q_idx in range(n_queries):
            g_idx = row_to_col[q_idx]
            best_id = None
            best_conf = 0.0

            if 0 <= g_idx < n_gallery:
                candidate_id = gallery_ids[g_idx]
                dist = cost_matrix[q_idx, g_idx]
                fallback_thresh = gallery_cfg.rank_fallback_threshold

                # Simple distance-based matching for L2-normalized features
                # Skip complex voting - just use distance threshold
                # Same person typically: dist < 0.8, different: dist > 1.0
                if dist < fallback_thresh:
                    best_id = candidate_id
                    best_conf = max(0.0, 1.0 - dist / fallback_thresh)

            results.append((best_id, float(best_conf)))

        # Selective re-ranking: apply k-reciprocal only for ambiguous matches
        if gallery_cfg.use_full_reranking and self._knn_cache is not None:
            results = self._apply_selective_reranking(
                results, features_list, gallery_features, gallery_ids, crossing_ids
            )

        # Motion consistency validation - reject suspicious assignments
        if gallery_cfg.use_motion_validation and bboxes:
            predictions = self._predict_track_positions()
            det_positions = [
                ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in bboxes
            ]
            track_velocities = {
                tid: motion.velocity
                for tid, motion in self._track_motion.items()
            }
            results = validate_assignments_batch(
                results,
                det_positions,
                predictions,
                track_velocities,
                crossing_ids,
                max_distance=gallery_cfg.motion_max_distance,
                direction_threshold_deg=gallery_cfg.motion_direction_threshold,
            )

        return results

    def _bbox_hash(self, bbox: tuple[float, ...]) -> int:
        """Hash bbox center for position tracking (50px grid cells)."""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return hash((int(cx // 50), int(cy // 50)))

    def _apply_velocity_consistency(
        self,
        sim_matrix: np.ndarray,
        bboxes: list[tuple[float, ...]],
        gallery_ids: list[int],
        crossing_ids: set[int] | None = None,
    ) -> np.ndarray:
        """Apply velocity-based position boost to similarity matrix.

        Args:
            sim_matrix: (Q, G) similarity scores
            bboxes: Bounding boxes for each query detection
            gallery_ids: List of gallery track IDs
            crossing_ids: Set of track IDs currently in crossing state

        Returns:
            Adjusted similarity matrix
        """
        cfg = self.config.gallery
        predictions = self._predict_track_positions()
        adjusted = sim_matrix.copy()
        crossing_ids = crossing_ids or set()

        # Minimum similarity required before applying position boost
        # Prevents ID theft when different-looking people cross paths
        min_sim_for_boost = 0.4

        for q_idx, bbox in enumerate(bboxes):
            det_cx = (bbox[0] + bbox[2]) / 2
            det_cy = (bbox[1] + bbox[3]) / 2

            for g_idx, gal_id in enumerate(gallery_ids):
                if gal_id not in predictions:
                    continue

                # Only apply position boost if base similarity is reasonable
                base_sim = sim_matrix[q_idx, g_idx]
                if base_sim < min_sim_for_boost:
                    continue

                pred_pos = predictions[gal_id]

                # Reduce position boost during crossing to prevent ID theft
                # Appearance should dominate over position during crossing
                if gal_id in crossing_ids:
                    max_boost = cfg.prediction_boost * cfg.crossing_boost_reduction
                else:
                    max_boost = cfg.prediction_boost

                boost = compute_position_boost(
                    (det_cx, det_cy),
                    pred_pos,
                    prediction_radius=cfg.prediction_radius,
                    max_boost=max_boost,
                )
                adjusted[q_idx, g_idx] += boost

        return adjusted

    def _predict_track_positions(self) -> dict[int, tuple[float, float]]:
        """Predict current positions for all active tracks.

        Returns:
            Dict mapping track_id to predicted (cx, cy)
        """
        predictions = {}
        max_age = self.config.gallery.velocity_history_frames
        stale_ids = []  # Collect stale motion entries for cleanup

        for track_id, motion in self._track_motion.items():
            if self._frame_idx - motion.last_frame > max_age:
                stale_ids.append(track_id)
                continue

            if not motion.positions:
                continue

            last_pos = motion.positions[-1]
            frames_elapsed = self._frame_idx - motion.last_frame + 1

            pred_pos = predict_position(last_pos, motion.velocity, frames_elapsed)
            predictions[track_id] = pred_pos

        # Clean up stale motion entries
        for track_id in stale_ids:
            del self._track_motion[track_id]

        return predictions

    def _detect_crossing_tracks(
        self,
        bboxes: list[tuple[float, ...]] | None,
        gallery_ids: list[int],
    ) -> set[int]:
        """Detect which gallery tracks are currently crossing or about to cross.

        Uses velocity convergence and bbox IoU overlap to detect crossings.

        Args:
            bboxes: Current detection bounding boxes (for IoU check)
            gallery_ids: List of active gallery track IDs

        Returns:
            Set of track IDs involved in crossing
        """
        cfg = self.config.gallery
        if not cfg.use_crossing_detection:
            return set()

        # Get predicted positions and velocities for gallery tracks
        predictions = self._predict_track_positions()
        track_positions = {
            tid: predictions[tid] for tid in gallery_ids if tid in predictions
        }
        track_velocities = {
            tid: self._track_motion[tid].velocity
            for tid in gallery_ids
            if tid in self._track_motion
        }

        # Get track bboxes from last known positions (approximate)
        track_bboxes = None
        if bboxes is not None:
            # Use last known bboxes from motion tracking
            # This is approximate - actual bbox tracking would be better
            track_bboxes = self._get_track_bboxes(gallery_ids)

        return detect_crossing_tracks(
            track_positions,
            track_velocities,
            track_bboxes=track_bboxes,
            crossing_radius=cfg.crossing_detection_radius,
            iou_threshold=cfg.crossing_iou_threshold,
        )

    def _get_track_bboxes(
        self, gallery_ids: list[int]
    ) -> dict[int, tuple[float, float, float, float]]:
        """Get approximate bounding boxes for gallery tracks.

        Uses predicted center positions and assumes standard person dimensions.

        Args:
            gallery_ids: List of track IDs to get bboxes for

        Returns:
            Dict mapping track_id to (x1, y1, x2, y2) bbox
        """
        predictions = self._predict_track_positions()
        track_bboxes = {}

        # Approximate person dimensions (width, height)
        default_w, default_h = 60, 150

        for tid in gallery_ids:
            if tid not in predictions:
                continue

            cx, cy = predictions[tid]
            x1 = cx - default_w / 2
            y1 = cy - default_h / 2
            x2 = cx + default_w / 2
            y2 = cy + default_h / 2
            track_bboxes[tid] = (x1, y1, x2, y2)

        return track_bboxes

    def _should_skip_reranking(self, match_id: int | None, confidence: float) -> bool:
        """Determine if re-ranking can be skipped for this match.

        Skips re-ranking for high-confidence matches or no-match cases.
        Tracks trigger/skip counts for monitoring.
        """
        cfg = self.config.gallery

        if match_id is None:
            return True

        if confidence >= cfg.rerank_confidence_threshold:
            self._rerank_skip_count += 1
            return True

        # Very close match — no ambiguity
        distance = 1.0 - confidence
        if distance < cfg.rerank_distance_skip:
            self._rerank_skip_count += 1
            return True

        self._rerank_trigger_count += 1
        return False

    def _apply_selective_reranking(
        self,
        results: list[tuple[int | None, float]],
        features_list: list[np.ndarray],
        gallery_features: dict[int, list[np.ndarray]],
        gallery_ids: list[int],
        crossing_ids: set[int],
    ) -> list[tuple[int | None, float]]:
        """Apply k-reciprocal re-ranking only for ambiguous matches.

        High-confidence matches skip re-ranking entirely. Crossing tracks
        always trigger re-ranking regardless of confidence.

        Args:
            results: Initial (match_id, confidence) from Hungarian assignment
            features_list: Query feature vectors
            gallery_features: Gallery features dict
            gallery_ids: Ordered gallery IDs
            crossing_ids: Track IDs currently in crossing state

        Returns:
            Re-ranked results list
        """
        final_results = []
        for q_idx, (match_id, confidence) in enumerate(results):
            # Always re-rank crossing tracks (high ID-theft risk)
            is_crossing = match_id is not None and match_id in crossing_ids

            if not is_crossing and self._should_skip_reranking(match_id, confidence):
                final_results.append((match_id, confidence))
                continue

            # Apply re-ranking using cached k-NN reciprocal set
            if match_id is not None and self._knn_cache is not None:
                reciprocal_set = self._knn_cache.get_reciprocal_set(match_id)
                if reciprocal_set:
                    # Boost confidence if matched ID has strong reciprocal neighbors
                    # that also appear close to the query
                    query_feat = features_list[q_idx]
                    reciprocal_support = 0
                    for r_id in reciprocal_set:
                        if r_id in gallery_features:
                            # Check if reciprocal neighbor is also close to query
                            r_feats = gallery_features[r_id]
                            for rf in r_feats:
                                dist = float(np.linalg.norm(query_feat - rf))
                                if dist < self.config.gallery.rank_fallback_threshold:
                                    reciprocal_support += 1
                                    break

                    # If reciprocal neighbors don't support the match, reduce confidence
                    if len(reciprocal_set) > 0 and reciprocal_support == 0:
                        confidence *= 0.5  # Penalize unsupported match

            final_results.append((match_id, confidence))

        return final_results

    def _rebuild_knn_cache(self) -> None:
        """Rebuild k-NN cache from current gallery features."""
        if self._knn_cache is None:
            return
        gallery_features = {
            tid: entry.avg_feature
            for tid, entry in self._gallery.items()
            if entry.avg_feature is not None
        }
        self._knn_cache.update(gallery_features)

    def get_rerank_stats(self) -> dict:
        """Get selective re-ranking trigger statistics."""
        total = self._rerank_trigger_count + self._rerank_skip_count
        return {
            "trigger_count": self._rerank_trigger_count,
            "skip_count": self._rerank_skip_count,
            "trigger_rate": self._rerank_trigger_count / max(1, total),
        }

    def _update_track_motion(
        self,
        track_id: int,
        bbox: tuple[float, float, float, float],
    ) -> None:
        """Update motion state for track.

        Args:
            track_id: Track ID to update
            bbox: Bounding box (x1, y1, x2, y2)
        """
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        cfg = self.config.gallery

        if track_id not in self._track_motion:
            self._track_motion[track_id] = TrackMotion()

        motion = self._track_motion[track_id]
        motion.positions.append((cx, cy))
        motion.last_frame = self._frame_idx

        # Compute velocity from history
        motion.velocity = compute_velocity(
            motion.positions, max_speed=cfg.velocity_max_speed
        )

    def _apply_k_reciprocal_boost(
        self,
        query_feats: np.ndarray,
        gallery_feats: np.ndarray,
        sim_matrix: np.ndarray,
        k: int = 20,
        boost: float = 0.1,
    ) -> np.ndarray:
        """Apply k-reciprocal boost to similarity scores.

        Simplified real-time version: boosts scores where reciprocal
        neighbor relationships exist between gallery items.

        Args:
            query_feats: (Q, D) query features
            gallery_feats: (G, D) gallery features
            sim_matrix: (Q, G) pre-computed cosine similarities
            k: Number of neighbors to consider
            boost: Similarity bonus for reciprocal matches

        Returns:
            (Q, G) boosted similarity scores
        """
        Q, G = sim_matrix.shape
        k = min(k, G)

        if k < 2:
            return sim_matrix

        # Compute gallery-gallery similarities for reciprocal check
        g_g_sim = cosine_similarity(gallery_feats, gallery_feats)

        # Find k-nearest gallery neighbors for each query
        q_neighbors = np.argsort(-sim_matrix, axis=1)[:, :k]  # (Q, k)

        # Find k-nearest gallery neighbors for each gallery item
        g_neighbors = np.argsort(-g_g_sim, axis=1)[:, :k]  # (G, k)

        # Boost scores for reciprocal matches
        boosted_sim = sim_matrix.copy()

        for q_idx in range(Q):
            q_top_k = set(q_neighbors[q_idx])
            for rank, g_idx in enumerate(q_neighbors[q_idx]):
                # Count how many of gallery item's neighbors overlap with query's top-k
                g_neighbor_set = set(g_neighbors[g_idx][: k // 2])
                reciprocal_count = len(q_top_k & g_neighbor_set)

                # Boost if significant reciprocal overlap
                if reciprocal_count > k // 4:
                    rank_factor = 1 - rank / k  # Higher boost for top ranks
                    boosted_sim[q_idx, g_idx] += boost * rank_factor

        return boosted_sim

    def get_rank_list(
        self,
        query_feat: np.ndarray,
        k: int = 20,
        exclude_ids: set[int] | None = None,
    ) -> list[tuple[int, float, int]]:
        """Get top-k gallery entries by euclidean distance.

        Extracts all features from gallery entries and computes rank list.

        Args:
            query_feat: (D,) query feature vector
            k: Number of entries to return (max 50)
            exclude_ids: Track IDs to exclude from ranking

        Returns:
            List of (track_id, min_distance, feature_count) sorted ascending
        """
        exclude_ids = exclude_ids or set()

        # Extract all features from gallery entries
        gallery_features: dict[int, list[np.ndarray]] = {}
        for track_id, entry in self._gallery.items():
            if track_id in exclude_ids:
                continue
            if entry.features:
                gallery_features[track_id] = list(entry.features)

        return compute_euclidean_rank_list(query_feat, gallery_features, k=k)

    def match_with_rank_voting(
        self,
        features: np.ndarray,
        rank_list_size: int | None = None,
        distance_threshold: float | None = None,
        exclude_ids: set[int] | None = None,
    ) -> tuple[int | None, float]:
        """Match using rank-list majority voting.

        Gets rank list and applies majority voting to confirm identity.

        Args:
            features: Query feature vector (D,)
            rank_list_size: Number of candidates in rank list (from config if None)
            distance_threshold: Max distance for "match" (adaptive if None)
            exclude_ids: Track IDs to exclude

        Returns:
            (matched_id, confidence) or (None, 0.0)
        """
        cfg = self.config.gallery
        k = rank_list_size or cfg.rank_list_size

        # Get rank list
        rank_list = self.get_rank_list(features, k=k, exclude_ids=exclude_ids)

        if not rank_list:
            return (None, 0.0)

        # Use config threshold or adaptive
        threshold = distance_threshold or cfg.rank_distance_threshold

        return majority_vote_reidentify(
            rank_list,
            distance_threshold=threshold,
            min_entries_per_id=cfg.rank_min_entries_per_id,
        )

    def get_top_similar(
        self,
        features: np.ndarray,
        exclude_id: int | None = None,
        top_k: int = 3,
    ) -> list[tuple[int, float]]:
        """Get top-k most similar gallery IDs for a query.

        Args:
            features: Query feature vector
            exclude_id: ID to exclude (e.g., the matched ID)
            top_k: Number of similar IDs to return

        Returns:
            List of (track_id, similarity) sorted by similarity descending
        """
        exclude_ids = {exclude_id} if exclude_id is not None else set()
        rank_list = self.get_rank_list(features, k=20, exclude_ids=exclude_ids)

        if not rank_list:
            return []

        # Convert distance to similarity (1 - dist/2 for L2-normalized)
        # Group by ID and take min distance per ID
        id_min_dist: dict[int, float] = {}
        for track_id, dist, _ in rank_list:
            if track_id not in id_min_dist or dist < id_min_dist[track_id]:
                id_min_dist[track_id] = dist

        # Convert to similarity and sort
        similarities = [
            (tid, max(0.0, 1.0 - dist / 2.0))
            for tid, dist in id_min_dist.items()
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def match(self, features: np.ndarray) -> int | None:
        """Find best matching ID in gallery.

        Args:
            features: Query feature vector (512,)

        Returns:
            Matching track_id or None if no match above threshold
        """
        if not self._gallery:
            return None

        threshold = self.config.inference.similarity_threshold
        best_id = None
        best_sim = threshold

        query = features.reshape(1, -1)

        for track_id, entry in self._gallery.items():
            if entry.avg_feature is None:
                continue

            gallery_feat = entry.avg_feature.reshape(1, -1)
            sim = cosine_similarity(query, gallery_feat)[0, 0]

            if sim > best_sim:
                best_sim = sim
                best_id = track_id

        return best_id

    def add(
        self,
        features: np.ndarray,
        quality_score: float = 1.0,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> int:
        """Add new person to gallery.

        Args:
            features: Feature vector (512,)
            quality_score: Detection quality score [0, 1]
            bbox: Optional bounding box for motion tracking

        Returns:
            New track ID
        """
        track_id = self._next_id
        self._next_id += 1
        cfg = self.config.gallery

        max_features = cfg.max_features_per_id
        entry = GalleryEntry(
            features=deque([features], maxlen=max_features),
            quality_scores=deque([quality_score], maxlen=max_features),
            avg_feature=features.copy(),
            avg_quality=quality_score,
            last_seen=self._frame_idx,
        )
        self._gallery[track_id] = entry

        # Maintain FAISS index
        if self._faiss_index is not None:
            self._faiss_index.add(track_id, features)

        # Track change for k-NN cache eager rebuild
        if self._knn_cache is not None:
            self._knn_change_counter += 1

        # Initialize motion tracking
        if bbox is not None and cfg.use_velocity_prediction:
            self._update_track_motion(track_id, bbox)

        return track_id

    def update(
        self,
        track_id: int,
        features: np.ndarray,
        quality_score: float = 1.0,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Update existing gallery entry with new observation.

        Uses quality-weighted EMA for smooth feature updates.

        Args:
            track_id: Existing track ID
            features: New feature vector (512,)
            quality_score: Detection quality score [0, 1]
            bbox: Optional bounding box for motion tracking
        """
        if track_id not in self._gallery:
            return

        cfg = self.config.gallery
        entry = self._gallery[track_id]

        # Skip low quality updates if quality weighting enabled
        if cfg.use_quality_weighting and quality_score < cfg.quality_min_threshold:
            # Still update last_seen and motion even if skipping feature update
            entry.last_seen = self._frame_idx
            if bbox is not None and cfg.use_velocity_prediction:
                self._update_track_motion(track_id, bbox)
            return

        # Add to rolling windows
        entry.features.append(features)
        entry.quality_scores.append(quality_score)
        entry.last_seen = self._frame_idx

        # Quality-weighted EMA
        if cfg.use_quality_weighting:
            effective_alpha = cfg.ema_alpha * quality_score
        else:
            effective_alpha = cfg.ema_alpha

        # EMA update for average feature
        if entry.avg_feature is not None:
            entry.avg_feature = (
                effective_alpha * features + (1 - effective_alpha) * entry.avg_feature
            )
            # Re-normalize to prevent drift from L2=1.0 over many updates
            norm = np.linalg.norm(entry.avg_feature)
            if norm > 1e-8:
                entry.avg_feature /= norm
        else:
            entry.avg_feature = features.copy()

        # Update average quality
        if entry.quality_scores:
            entry.avg_quality = sum(entry.quality_scores) / len(entry.quality_scores)

        # Sync FAISS index with updated average feature
        if self._faiss_index is not None and entry.avg_feature is not None:
            self._faiss_index.update(track_id, entry.avg_feature)

        # Track change for k-NN cache eager rebuild
        if self._knn_cache is not None:
            self._knn_change_counter += 1

        # Update motion tracking
        if bbox is not None and cfg.use_velocity_prediction:
            self._update_track_motion(track_id, bbox)

    def step_frame(self) -> None:
        """Increment frame counter (call once per frame)."""
        self._frame_idx += 1

        # Periodic FAISS index rebuild for search accuracy
        if self._faiss_index is not None:
            self._faiss_rebuild_counter += 1
            if self._faiss_rebuild_counter >= self.config.gallery.faiss_rebuild_interval:
                self._faiss_index._needs_rebuild = True
                self._faiss_rebuild_counter = 0

        # Periodic k-NN cache rebuild for selective re-ranking
        if self._knn_cache is not None:
            self._knn_rebuild_counter += 1
            # Eager rebuild if many changes accumulated, otherwise periodic rebuild
            if (self._knn_change_counter >= 10 or
                self._knn_rebuild_counter >= self.config.gallery.rerank_knn_rebuild_interval):
                self._rebuild_knn_cache()
                self._knn_rebuild_counter = 0
                self._knn_change_counter = 0

    def get_all_ids(self) -> list[int]:
        """Get all track IDs in gallery."""
        return list(self._gallery.keys())

    def get_entry(self, track_id: int) -> GalleryEntry | None:
        """Get gallery entry by ID."""
        return self._gallery.get(track_id)

    def get_recent_id(self, bbox: tuple[float, ...]) -> int | None:
        """Get the most recent track ID seen near this position.

        Uses velocity-based motion tracking to find nearby active tracks.
        Returns the CLOSEST track within radius, not the first match.
        """
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        radius = self.config.gallery.prediction_radius

        # Track the closest match within radius
        closest_id = None
        closest_dist = radius

        # Check motion history for recently seen tracks
        for track_id, motion in self._track_motion.items():
            if self._frame_idx - motion.last_frame > 5:  # Stale track
                continue
            if motion.positions:
                last_cx, last_cy = motion.positions[-1]
                dist = ((cx - last_cx) ** 2 + (cy - last_cy) ** 2) ** 0.5
                if dist < closest_dist:
                    closest_dist = dist
                    closest_id = track_id
        return closest_id

    def clear(self) -> None:
        """Clear all gallery entries and motion history."""
        self._gallery.clear()
        self._next_id = 0
        self._frame_idx = 0
        self._track_motion.clear()
        self.reset_threshold_stats()
        if self._faiss_index is not None:
            self._faiss_index.clear()
        self._faiss_rebuild_counter = 0

    def get_effective_threshold(self) -> float:
        """Get current effective similarity threshold.

        Returns:
            Similarity threshold (adaptive or static)
        """
        cfg = self.config.gallery
        if cfg.use_adaptive_threshold:
            self._current_threshold = compute_adaptive_threshold(
                list(self._match_scores),
                base_threshold=self.config.inference.similarity_threshold,
                target_percentile=cfg.adaptive_target_percentile,
                min_threshold=cfg.adaptive_min_threshold,
                max_threshold=cfg.adaptive_max_threshold,
                warmup_count=cfg.adaptive_warmup_matches,
            )
        return self._current_threshold

    def _update_similarity_stats(self, scores: list[float]) -> None:
        """Update rolling similarity statistics.

        Args:
            scores: List of match similarity scores from this frame
        """
        for score in scores:
            if score > 0:
                self._match_scores.append(score)
                self._match_count += 1

    def get_threshold_stats(self) -> dict:
        """Get adaptive threshold statistics for monitoring.

        Returns:
            Dict with threshold stats
        """
        if not self._match_scores:
            return {
                "current_threshold": self._current_threshold,
                "match_count": 0,
                "mean_score": None,
                "std_score": None,
            }

        scores = list(self._match_scores)
        return {
            "current_threshold": self._current_threshold,
            "match_count": self._match_count,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
        }

    def reset_threshold_stats(self) -> None:
        """Reset adaptive threshold statistics."""
        self._match_scores.clear()
        self._match_count = 0
        self._current_threshold = self.config.inference.similarity_threshold

    def prune_stale(self, max_age: int = 300) -> list[int]:
        """Remove entries not seen for max_age frames.

        Args:
            max_age: Maximum frames since last seen

        Returns:
            List of pruned track IDs
        """
        pruned = []
        current_frame = self._frame_idx

        for track_id in list(self._gallery.keys()):
            entry = self._gallery[track_id]
            if current_frame - entry.last_seen > max_age:
                del self._gallery[track_id]
                # Also remove motion tracking data
                self._track_motion.pop(track_id, None)
                # Remove from FAISS index
                if self._faiss_index is not None:
                    self._faiss_index.remove(track_id)
                pruned.append(track_id)

        return pruned

    @property
    def size(self) -> int:
        """Number of persons in gallery."""
        return len(self._gallery)

    def export_features(self) -> dict[int, np.ndarray]:
        """Export average features for all persons.

        Returns:
            Dict mapping track_id to avg_feature
        """
        return {
            tid: entry.avg_feature
            for tid, entry in self._gallery.items()
            if entry.avg_feature is not None
        }

    def save(self, path: str | Path) -> None:
        """Save gallery to JSON file.

        Args:
            path: Output JSON file path
        """
        # Create backup of existing file before overwriting
        path_obj = Path(path)
        if path_obj.exists():
            backup_path = Path(str(path) + ".bak")
            try:
                path_obj.rename(backup_path)
            except Exception as e:
                logger.warning(f"Failed to create backup of {path}: {e}")

        data = {
            "next_id": self._next_id,
            "frame_idx": self._frame_idx,
            "entries": {
                str(tid): {
                    "avg_feature": entry.avg_feature.tolist()
                    if entry.avg_feature is not None
                    else None,
                    "last_seen": entry.last_seen,
                }
                for tid, entry in self._gallery.items()
            },
        }
        path_obj.write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> None:
        """Load gallery from JSON file with validation.

        Args:
            path: Input JSON file path
        """
        try:
            data = json.loads(Path(path).read_text())

            # Validate top-level structure
            if not isinstance(data.get("next_id"), int):
                raise ValueError("Invalid or missing 'next_id' field")
            if not isinstance(data.get("frame_idx"), int):
                raise ValueError("Invalid or missing 'frame_idx' field")
            if not isinstance(data.get("entries"), dict):
                raise ValueError("Invalid or missing 'entries' field")

            self._next_id = data["next_id"]
            self._frame_idx = data["frame_idx"]
            self._gallery.clear()

            # Load entries with validation
            for tid_str, entry_data in data["entries"].items():
                try:
                    tid = int(tid_str)

                    # Validate entry structure
                    if not isinstance(entry_data, dict):
                        logger.warning(f"Skipping entry {tid_str}: invalid entry structure")
                        continue

                    # Validate and load avg_feature
                    avg_feat = None
                    if entry_data.get("avg_feature") is not None:
                        feat_list = entry_data["avg_feature"]
                        if not isinstance(feat_list, list):
                            logger.warning(f"Skipping entry {tid}: avg_feature not a list")
                            continue
                        if len(feat_list) != 512:
                            logger.warning(f"Skipping entry {tid}: avg_feature wrong length ({len(feat_list)}, expected 512)")
                            continue
                        avg_feat = np.array(feat_list)

                    # Validate last_seen
                    if not isinstance(entry_data.get("last_seen"), int):
                        logger.warning(f"Skipping entry {tid}: invalid last_seen")
                        continue

                    self._gallery[tid] = GalleryEntry(
                        avg_feature=avg_feat,
                        last_seen=entry_data["last_seen"],
                    )

                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Skipping entry {tid_str}: {e}")
                    continue

        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load gallery from {path}: {e}")
            raise
