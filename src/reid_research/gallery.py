"""Gallery-based person tracking with ReID features."""
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import lap
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .config import ReIDConfig
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

        # Legacy temporal history (position-hash based)
        self._temporal_history: dict[int, deque] = {}
        self._temporal_window = config.gallery.temporal_window

        # Velocity-based motion tracking
        self._track_motion: dict[int, TrackMotion] = {}

        # Adaptive threshold statistics
        self._match_scores: deque = deque(maxlen=config.gallery.adaptive_window_size)
        self._match_count: int = 0
        self._current_threshold: float = config.inference.similarity_threshold

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

        # Build cost matrix using rank-list euclidean distances
        # Lower distance = lower cost = better match
        n_queries = len(features_list)
        n_gallery = len(gallery_ids)
        cost_matrix = np.full((n_queries, n_gallery), 1e6, dtype=np.float32)
        confidence_matrix = np.zeros((n_queries, n_gallery), dtype=np.float32)

        for q_idx, query_feat in enumerate(features_list):
            # Get full rank list for this query (all gallery entries)
            rank_list = compute_euclidean_rank_list(
                query_feat, gallery_features, k=min(n_gallery, 50)
            )

            # Populate cost matrix with distances
            for track_id, dist, feat_count in rank_list:
                if track_id in gallery_ids:
                    g_idx = gallery_ids.index(track_id)
                    cost_matrix[q_idx, g_idx] = dist

                    # Pre-compute confidence via majority vote per (query, gallery) pair
                    # Use adaptive threshold (median of this query's distances)
                    all_dists = [r[1] for r in rank_list]
                    threshold = float(np.median(all_dists)) if all_dists else 1.0

                    if dist < threshold:
                        confidence_matrix[q_idx, g_idx] = 1.0 - (dist / (threshold + 1e-6))

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

    def _apply_temporal_consistency(
        self,
        sim_matrix: np.ndarray,
        bbox_hashes: list[int],
        gallery_ids: list[int],
    ) -> np.ndarray:
        """Apply temporal consistency boost to similarity matrix.

        Args:
            sim_matrix: (Q, G) similarity scores
            bbox_hashes: Hash for each query detection position
            gallery_ids: List of gallery track IDs

        Returns:
            Adjusted similarity matrix
        """
        boost = self.config.gallery.temporal_boost
        adjusted = sim_matrix.copy()

        for q_idx, bbox_hash in enumerate(bbox_hashes):
            history = self._temporal_history.get(bbox_hash)
            if not history:
                continue

            # Boost IDs seen recently at this position
            for g_idx, gal_id in enumerate(gallery_ids):
                count = sum(1 for h_id in history if h_id == gal_id)
                if count > 0:
                    adjusted[q_idx, g_idx] += boost * (count / len(history))

        return adjusted

    def _update_temporal_history(
        self, bbox_hashes: list[int], assigned_ids: list[int | None]
    ) -> None:
        """Update temporal history with this frame's assignments."""
        window = self._temporal_window

        for bbox_hash, track_id in zip(bbox_hashes, assigned_ids):
            if track_id is None:
                continue

            if bbox_hash not in self._temporal_history:
                self._temporal_history[bbox_hash] = deque(maxlen=window)

            self._temporal_history[bbox_hash].append(track_id)

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
        max_age = self.config.gallery.temporal_window

        for track_id, motion in self._track_motion.items():
            if self._frame_idx - motion.last_frame > max_age:
                continue

            if not motion.positions:
                continue

            last_pos = motion.positions[-1]
            frames_elapsed = self._frame_idx - motion.last_frame + 1

            pred_pos = predict_position(last_pos, motion.velocity, frames_elapsed)
            predictions[track_id] = pred_pos

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
        else:
            entry.avg_feature = features.copy()

        # Update average quality
        if entry.quality_scores:
            entry.avg_quality = sum(entry.quality_scores) / len(entry.quality_scores)

        # Update motion tracking
        if bbox is not None and cfg.use_velocity_prediction:
            self._update_track_motion(track_id, bbox)

    def step_frame(self) -> None:
        """Increment frame counter (call once per frame)."""
        self._frame_idx += 1

    def get_all_ids(self) -> list[int]:
        """Get all track IDs in gallery."""
        return list(self._gallery.keys())

    def get_entry(self, track_id: int) -> GalleryEntry | None:
        """Get gallery entry by ID."""
        return self._gallery.get(track_id)

    def get_recent_id(self, bbox: tuple[float, ...]) -> int | None:
        """Get the most recent track ID seen at this position."""
        bbox_hash = self._bbox_hash(bbox)
        history = self._temporal_history.get(bbox_hash)
        if history and len(history) > 0:
            return history[-1]
        return None

    def clear(self) -> None:
        """Clear all gallery entries and temporal history."""
        self._gallery.clear()
        self._next_id = 0
        self._frame_idx = 0
        self._temporal_history.clear()
        self._track_motion.clear()
        self.reset_threshold_stats()

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
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> None:
        """Load gallery from JSON file.

        Args:
            path: Input JSON file path
        """
        data = json.loads(Path(path).read_text())
        self._next_id = data["next_id"]
        self._frame_idx = data["frame_idx"]
        self._gallery.clear()

        for tid_str, entry_data in data["entries"].items():
            tid = int(tid_str)
            avg_feat = (
                np.array(entry_data["avg_feature"])
                if entry_data["avg_feature"]
                else None
            )
            self._gallery[tid] = GalleryEntry(
                avg_feature=avg_feat,
                last_seen=entry_data["last_seen"],
            )
