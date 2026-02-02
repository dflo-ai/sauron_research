"""Tests for history queue modules."""

import pytest
import torch
from src.hat_reid.queues import FIFOQueue, ScoreQueue


class TestFIFOQueue:
    """Test suite for FIFOQueue."""

    def test_add_single(self):
        """Test adding single item."""
        q = FIFOQueue(max_len=10, decay_ratio=0.9)
        feat = torch.randn(256)
        q.add(feat, 0.8)

        assert len(q) == 1
        features, weights = q.get_features_and_weights()
        assert features.shape == (1, 256)
        assert len(weights) == 1

    def test_weight_decay(self):
        """Test that older items have decayed weights."""
        q = FIFOQueue(max_len=10, decay_ratio=0.9, use_decay_as_weight=True)
        q.add(torch.randn(256), 0.8)
        q.add(torch.randn(256), 0.9)

        _, weights = q.get_features_and_weights()

        # First item decayed twice (once on add, once when second added)
        # Second item decayed once
        assert weights[0] < weights[1]

    def test_max_len_enforced(self):
        """Test that queue respects max_len."""
        q = FIFOQueue(max_len=5, decay_ratio=0.9)

        for i in range(10):
            q.add(torch.randn(64), 0.5 + i * 0.05)

        assert len(q) == 5

    def test_use_scores_as_weights(self):
        """Test using original scores instead of decay weights."""
        q = FIFOQueue(max_len=10, decay_ratio=0.9, use_decay_as_weight=False)
        q.add(torch.randn(64), 0.7)
        q.add(torch.randn(64), 0.9)

        _, weights = q.get_features_and_weights()

        assert weights[0].item() == pytest.approx(0.7)
        assert weights[1].item() == pytest.approx(0.9)

    def test_empty_queue(self):
        """Test getting from empty queue."""
        q = FIFOQueue(max_len=10, decay_ratio=0.9)
        features, weights = q.get_features_and_weights()

        assert len(features) == 0
        assert len(weights) == 0

    def test_clear(self):
        """Test clear method."""
        q = FIFOQueue(max_len=10, decay_ratio=0.9)
        q.add(torch.randn(64), 0.8)
        q.add(torch.randn(64), 0.9)

        assert len(q) == 2
        q.clear()
        assert len(q) == 0


class TestScoreQueue:
    """Test suite for ScoreQueue."""

    def test_add_single(self):
        """Test adding single item."""
        q = ScoreQueue(max_len=10, decay_ratio=0.9)
        feat = torch.randn(256)
        q.add(feat, 0.8)

        assert len(q) == 1
        features, scores = q.get_features_and_weights()
        assert features.shape == (1, 256)
        assert len(scores) == 1

    def test_keeps_top_k_by_score(self):
        """Test that queue keeps highest scoring samples."""
        q = ScoreQueue(max_len=3, decay_ratio=1.0)  # No decay for clarity

        # Add 5 samples with different scores
        for i, score in enumerate([0.5, 0.9, 0.3, 0.7, 0.8]):
            q.add(torch.full((64,), float(i)), score)

        assert len(q) == 3
        features, scores = q.get_features_and_weights()

        # Should keep indices 1 (0.9), 4 (0.8), 3 (0.7)
        # Features are filled with their index
        kept_indices = set(features[:, 0].int().tolist())
        assert 1 in kept_indices  # score 0.9
        assert 4 in kept_indices  # score 0.8

    def test_score_decay(self):
        """Test that existing scores decay when new item added."""
        q = ScoreQueue(max_len=10, decay_ratio=0.5)

        q.add(torch.randn(64), 1.0)
        _, scores1 = q.get_features_and_weights()
        initial_score = scores1[0].item()

        q.add(torch.randn(64), 1.0)
        _, scores2 = q.get_features_and_weights()

        # First score should have decayed
        assert scores2[0].item() == pytest.approx(initial_score * 0.5)
        # New score should be 1.0
        assert scores2[1].item() == pytest.approx(1.0)

    def test_empty_queue(self):
        """Test getting from empty queue."""
        q = ScoreQueue(max_len=10, decay_ratio=0.9)
        features, scores = q.get_features_and_weights()

        assert len(features) == 0
        assert len(scores) == 0

    def test_clear(self):
        """Test clear method."""
        q = ScoreQueue(max_len=10, decay_ratio=0.9)
        q.add(torch.randn(64), 0.8)
        q.add(torch.randn(64), 0.9)

        assert len(q) == 2
        q.clear()
        assert len(q) == 0

    def test_device_consistency(self):
        """Test that output stays on same device as input."""
        q = ScoreQueue(max_len=10, decay_ratio=0.9)
        feat = torch.randn(64)  # CPU tensor
        q.add(feat, 0.8)

        features, scores = q.get_features_and_weights()
        assert features.device == feat.device
        assert scores.device == feat.device
