"""Tests for matching module: k-reciprocal, quality scoring, velocity, adaptive threshold."""
import numpy as np
import pytest
from collections import deque

from src.reid_research.matching import (
    TrackMotion,
    compute_adaptive_threshold,
    compute_k_reciprocal_reranking,
    compute_position_boost,
    compute_quality_score,
    compute_velocity,
    predict_position,
    _jaccard_distance,
)


class TestKReciprocalReranking:
    """Tests for k-reciprocal reranking algorithm."""

    def test_small_gallery(self):
        """Test reranking with small gallery returns valid similarities."""
        query = np.random.randn(2, 512).astype(np.float32)
        gallery = np.random.randn(5, 512).astype(np.float32)

        # Normalize
        query /= np.linalg.norm(query, axis=1, keepdims=True)
        gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)

        result = compute_k_reciprocal_reranking(query, gallery, k1=3, k2=2)

        assert result.shape == (2, 5)
        assert np.all(result >= -1) and np.all(result <= 1)

    def test_identical_features_high_similarity(self):
        """Test identical features produce high similarity."""
        feat = np.random.randn(512).astype(np.float32)
        feat /= np.linalg.norm(feat)

        query = feat.reshape(1, -1)
        gallery = np.vstack([feat, np.random.randn(512).astype(np.float32)])
        gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)

        result = compute_k_reciprocal_reranking(query, gallery, k1=2, k2=1)

        # First gallery item (identical) should have highest similarity
        assert result[0, 0] > result[0, 1]

    def test_single_gallery_item(self):
        """Test with single gallery item."""
        query = np.random.randn(1, 512).astype(np.float32)
        gallery = np.random.randn(1, 512).astype(np.float32)

        query /= np.linalg.norm(query, axis=1, keepdims=True)
        gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)

        result = compute_k_reciprocal_reranking(query, gallery)
        assert result.shape == (1, 1)


class TestJaccardDistance:
    """Tests for Jaccard distance computation."""

    def test_identical_sets(self):
        """Identical sets have distance 0."""
        s = {1, 2, 3}
        assert _jaccard_distance(s, s) == 0.0

    def test_disjoint_sets(self):
        """Disjoint sets have distance 1."""
        s1 = {1, 2, 3}
        s2 = {4, 5, 6}
        assert _jaccard_distance(s1, s2) == 1.0

    def test_partial_overlap(self):
        """Partial overlap gives distance between 0 and 1."""
        s1 = {1, 2, 3}
        s2 = {2, 3, 4}
        # intersection = {2, 3}, union = {1, 2, 3, 4}
        # jaccard = 1 - 2/4 = 0.5
        assert _jaccard_distance(s1, s2) == 0.5

    def test_empty_sets(self):
        """Empty sets have distance 0."""
        assert _jaccard_distance(set(), set()) == 0.0


class TestQualityScore:
    """Tests for quality-aware feature scoring."""

    def test_high_confidence_high_quality(self):
        """High confidence detection gets high quality score."""
        score = compute_quality_score(
            confidence=0.95,
            bbox=(100, 100, 200, 400),  # Good aspect ratio ~0.33
            min_bbox_area=2000,
        )
        assert score > 0.7

    def test_low_confidence_lower_quality(self):
        """Low confidence detection gets lower score."""
        high_conf = compute_quality_score(
            confidence=0.95,
            bbox=(100, 100, 200, 400),
        )
        low_conf = compute_quality_score(
            confidence=0.3,
            bbox=(100, 100, 200, 400),
        )
        assert low_conf < high_conf

    def test_small_bbox_penalty(self):
        """Small bounding box gets lower score."""
        large = compute_quality_score(
            confidence=0.8,
            bbox=(0, 0, 100, 300),  # 30000 px
            min_bbox_area=2000,
        )
        small = compute_quality_score(
            confidence=0.8,
            bbox=(0, 0, 20, 50),  # 1000 px
            min_bbox_area=2000,
        )
        assert small < large

    def test_bad_aspect_ratio_penalty(self):
        """Non-person aspect ratio gets lower score."""
        good_aspect = compute_quality_score(
            confidence=0.8,
            bbox=(0, 0, 80, 200),  # 0.4 W/H
            ideal_aspect_ratio=0.4,
        )
        bad_aspect = compute_quality_score(
            confidence=0.8,
            bbox=(0, 0, 200, 100),  # 2.0 W/H (too wide)
            ideal_aspect_ratio=0.4,
        )
        assert bad_aspect < good_aspect

    def test_score_bounds(self):
        """Score always in [0, 1]."""
        # Edge cases
        scores = [
            compute_quality_score(0.0, (0, 0, 1, 1)),
            compute_quality_score(1.0, (0, 0, 1000, 2000)),
            compute_quality_score(0.5, (0, 0, 50, 125)),
        ]
        for score in scores:
            assert 0 <= score <= 1


class TestVelocityPrediction:
    """Tests for velocity-based temporal consistency."""

    def test_compute_velocity_stationary(self):
        """Stationary object has zero velocity."""
        positions = deque([(100, 100), (100, 100), (100, 100)])
        vx, vy = compute_velocity(positions)
        assert vx == 0.0 and vy == 0.0

    def test_compute_velocity_moving(self):
        """Moving object has non-zero velocity."""
        positions = deque([(0, 0), (10, 20), (20, 40)])
        vx, vy = compute_velocity(positions)
        # Average: (20-0)/3 = 6.67, (40-0)/3 = 13.33
        assert abs(vx - 6.67) < 0.1
        assert abs(vy - 13.33) < 0.1

    def test_compute_velocity_clamped(self):
        """Velocity clamped to max speed."""
        positions = deque([(0, 0), (1000, 0)])  # Very fast
        vx, vy = compute_velocity(positions, max_speed=50.0)
        speed = (vx**2 + vy**2) ** 0.5
        assert speed <= 50.0 + 0.01

    def test_compute_velocity_insufficient_history(self):
        """Single position returns zero velocity."""
        positions = deque([(100, 100)])
        vx, vy = compute_velocity(positions)
        assert vx == 0.0 and vy == 0.0

    def test_predict_position(self):
        """Position prediction with velocity."""
        last_pos = (100, 100)
        velocity = (10, -5)
        pred = predict_position(last_pos, velocity, frames_elapsed=3)
        assert pred == (130, 85)

    def test_position_boost_within_radius(self):
        """Boost applied when within prediction radius."""
        boost = compute_position_boost(
            detection_center=(100, 100),
            predicted_center=(110, 100),  # 10px away
            prediction_radius=50.0,
            max_boost=0.15,
        )
        assert 0 < boost < 0.15

    def test_position_boost_outside_radius(self):
        """No boost when outside prediction radius."""
        boost = compute_position_boost(
            detection_center=(100, 100),
            predicted_center=(200, 200),  # ~141px away
            prediction_radius=50.0,
            max_boost=0.15,
        )
        assert boost == 0.0

    def test_position_boost_exact_match(self):
        """Max boost when at predicted position."""
        boost = compute_position_boost(
            detection_center=(100, 100),
            predicted_center=(100, 100),
            prediction_radius=50.0,
            max_boost=0.15,
        )
        assert boost == 0.15


class TestAdaptiveThreshold:
    """Tests for adaptive similarity threshold."""

    def test_warmup_returns_base(self):
        """During warmup, return base threshold."""
        scores = [0.7, 0.75, 0.8]  # Only 3 scores
        threshold = compute_adaptive_threshold(
            scores,
            base_threshold=0.65,
            warmup_count=20,
        )
        assert threshold == 0.65

    def test_adapts_after_warmup(self):
        """After warmup, threshold adapts to distribution."""
        # Generate scores with clear distribution
        scores = [0.6 + i * 0.01 for i in range(50)]  # 0.60 to 1.09
        threshold = compute_adaptive_threshold(
            scores,
            base_threshold=0.65,
            target_percentile=0.15,
            warmup_count=20,
        )
        # 15th percentile of [0.6, ..., 1.09] ~ 0.67
        assert 0.60 < threshold < 0.80

    def test_threshold_min_clamp(self):
        """Threshold doesn't go below minimum."""
        scores = [0.3] * 50  # All low scores
        threshold = compute_adaptive_threshold(
            scores,
            base_threshold=0.65,
            min_threshold=0.50,
            warmup_count=20,
        )
        assert threshold >= 0.50

    def test_threshold_max_clamp(self):
        """Threshold doesn't go above maximum."""
        scores = [0.95] * 50  # All high scores
        threshold = compute_adaptive_threshold(
            scores,
            base_threshold=0.65,
            max_threshold=0.80,
            warmup_count=20,
        )
        assert threshold <= 0.80

    def test_empty_scores_returns_base(self):
        """Empty scores return base threshold."""
        threshold = compute_adaptive_threshold(
            [],
            base_threshold=0.65,
            warmup_count=20,
        )
        assert threshold == 0.65


class TestTrackMotion:
    """Tests for TrackMotion dataclass."""

    def test_default_values(self):
        """TrackMotion has sensible defaults."""
        motion = TrackMotion()
        assert len(motion.positions) == 0
        assert motion.velocity == (0.0, 0.0)
        assert motion.last_frame == 0

    def test_position_history_maxlen(self):
        """Position deque respects maxlen."""
        motion = TrackMotion()
        for i in range(20):
            motion.positions.append((i, i))
        # Default maxlen is 10
        assert len(motion.positions) == 10
        assert motion.positions[0] == (10, 10)
