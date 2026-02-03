"""Unit tests for rank-list majority voting implementation."""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from reid_research.config import ReIDConfig
from reid_research.gallery import PersonGallery
from reid_research.matching import compute_euclidean_rank_list, majority_vote_reidentify


class TestComputeEuclideanRankList:
    """Tests for compute_euclidean_rank_list function."""

    def test_basic_ranking(self):
        """Test basic rank list computation with simple gallery."""
        np.random.seed(42)
        query = np.random.randn(512).astype(np.float32)

        # Create gallery with known distances
        gallery = {
            0: [query + 0.1 * np.random.randn(512).astype(np.float32)],  # Close
            1: [query + 0.5 * np.random.randn(512).astype(np.float32)],  # Medium
            2: [query + 1.0 * np.random.randn(512).astype(np.float32)],  # Far
        }

        result = compute_euclidean_rank_list(query, gallery, k=5)

        assert len(result) == 3
        assert all(isinstance(r, tuple) and len(r) == 3 for r in result)
        # Verify sorted ascending by distance
        distances = [r[1] for r in result]
        assert distances == sorted(distances)

    def test_empty_gallery(self):
        """Test with empty gallery returns empty list."""
        query = np.random.randn(512).astype(np.float32)
        result = compute_euclidean_rank_list(query, {}, k=10)
        assert result == []

    def test_k_larger_than_gallery(self):
        """Test k larger than available entries."""
        np.random.seed(42)
        query = np.random.randn(512).astype(np.float32)
        gallery = {0: [np.random.randn(512).astype(np.float32)]}

        result = compute_euclidean_rank_list(query, gallery, k=100)
        assert len(result) == 1

    def test_multiple_features_per_id(self):
        """Test that multiple features per ID are handled correctly."""
        np.random.seed(42)
        query = np.random.randn(512).astype(np.float32)

        # ID 0 has 3 features, ID 1 has 2 features
        gallery = {
            0: [np.random.randn(512).astype(np.float32) for _ in range(3)],
            1: [np.random.randn(512).astype(np.float32) for _ in range(2)],
        }

        result = compute_euclidean_rank_list(query, gallery, k=10)

        # Now returns all features (3 + 2 = 5 entries)
        assert len(result) == 5
        # Feature counts in each entry should match the ID's total features
        id0_entries = [r for r in result if r[0] == 0]
        id1_entries = [r for r in result if r[0] == 1]
        assert len(id0_entries) == 3  # 3 features for ID 0
        assert len(id1_entries) == 2  # 2 features for ID 1
        # Each entry should have correct feature_count for its ID
        assert all(r[2] == 3 for r in id0_entries)
        assert all(r[2] == 2 for r in id1_entries)

    def test_k_capped_at_50(self):
        """Test that k is capped at 50 per specification."""
        np.random.seed(42)
        query = np.random.randn(512).astype(np.float32)

        # Create gallery with 60 entries
        gallery = {
            i: [np.random.randn(512).astype(np.float32)] for i in range(60)
        }

        result = compute_euclidean_rank_list(query, gallery, k=100)
        assert len(result) == 50  # Capped at 50


class TestMajorityVoteReidentify:
    """Tests for majority_vote_reidentify function."""

    def test_clear_winner(self):
        """Test clear majority case."""
        # ID 0 has 4 close entries, ID 1 has 1 far entry
        rank_list = [
            (0, 0.5, 1),
            (0, 0.6, 1),
            (0, 0.7, 1),
            (0, 0.8, 1),
            (1, 1.5, 1),
        ]
        best_id, conf = majority_vote_reidentify(rank_list, distance_threshold=1.0)
        assert best_id == 0
        assert conf > 0.5

    def test_no_majority(self):
        """Test no clear majority returns None."""
        # Each ID has 1 match and 1 non-match
        rank_list = [
            (0, 0.5, 1),
            (0, 1.5, 1),  # 1 match, 1 non-match
            (1, 0.6, 1),
            (1, 1.6, 1),  # 1 match, 1 non-match
        ]
        best_id, conf = majority_vote_reidentify(rank_list, distance_threshold=1.0)
        assert best_id is None

    def test_empty_list(self):
        """Test empty rank list."""
        best_id, conf = majority_vote_reidentify([], distance_threshold=1.0)
        assert best_id is None
        assert conf == 0.0

    def test_adaptive_threshold(self):
        """Test auto-computed threshold (median)."""
        # Distances: 0.2, 0.4, 0.6, 0.8, 1.0 -> median = 0.6
        rank_list = [
            (0, 0.2, 1),
            (0, 0.4, 1),
            (1, 0.6, 1),
            (1, 0.8, 1),
            (2, 1.0, 1),
        ]
        # With median threshold 0.6, ID 0 has 2 matches (0.2, 0.4), 0 non-matches
        best_id, conf = majority_vote_reidentify(rank_list, distance_threshold=None)
        assert best_id == 0
        assert conf > 0.5

    def test_min_entries_threshold(self):
        """Test minimum entries per ID requirement."""
        # ID 0 has only 1 entry (below threshold of 2)
        rank_list = [
            (0, 0.3, 1),
            (1, 0.4, 1),
            (1, 0.5, 1),
        ]
        best_id, conf = majority_vote_reidentify(
            rank_list, distance_threshold=1.0, min_entries_per_id=2
        )
        # ID 0 should be skipped, ID 1 has 2 entries both matching
        assert best_id == 1

    def test_tie_breaking(self):
        """Test tie-breaking with most matches wins."""
        # ID 0: 3 matches, ID 1: 2 matches
        rank_list = [
            (0, 0.3, 1),
            (0, 0.4, 1),
            (0, 0.5, 1),
            (1, 0.3, 1),
            (1, 0.4, 1),
        ]
        best_id, conf = majority_vote_reidentify(
            rank_list, distance_threshold=1.0, min_entries_per_id=2
        )
        assert best_id == 0  # More matches wins


class TestGalleryRankVotingIntegration:
    """Integration tests with PersonGallery."""

    def test_get_rank_list_method(self):
        """Test PersonGallery.get_rank_list() method."""
        config = ReIDConfig()
        gallery = PersonGallery(config)

        # Add entries with known features
        np.random.seed(42)
        feat1 = np.random.randn(512).astype(np.float32)
        feat1 /= np.linalg.norm(feat1)

        feat2 = np.random.randn(512).astype(np.float32)
        feat2 /= np.linalg.norm(feat2)

        id1 = gallery.add(feat1)
        id2 = gallery.add(feat2)

        # Query with feature similar to feat1
        query = feat1 + 0.01 * np.random.randn(512).astype(np.float32)
        query /= np.linalg.norm(query)

        rank_list = gallery.get_rank_list(query, k=10)

        assert len(rank_list) == 2
        # First entry should be id1 (closest to query)
        assert rank_list[0][0] == id1

    def test_match_with_rank_voting(self):
        """Test PersonGallery.match_with_rank_voting() method."""
        config = ReIDConfig()
        gallery = PersonGallery(config)

        # Add multiple features for reliable voting
        np.random.seed(42)
        base_feat = np.random.randn(512).astype(np.float32)
        base_feat /= np.linalg.norm(base_feat)

        id1 = gallery.add(base_feat.copy())
        # Add more features to same ID for voting
        for _ in range(4):
            f = base_feat + 0.05 * np.random.randn(512).astype(np.float32)
            f /= np.linalg.norm(f)
            gallery.update(id1, f)

        # Add another ID
        other_feat = np.random.randn(512).astype(np.float32)
        other_feat /= np.linalg.norm(other_feat)
        id2 = gallery.add(other_feat)
        for _ in range(4):
            f = other_feat + 0.05 * np.random.randn(512).astype(np.float32)
            f /= np.linalg.norm(f)
            gallery.update(id2, f)

        # Query similar to id1
        query = base_feat + 0.02 * np.random.randn(512).astype(np.float32)
        query /= np.linalg.norm(query)

        result_id, confidence = gallery.match_with_rank_voting(query)

        # Should match id1 with reasonable confidence
        assert result_id == id1
        assert confidence > 0

    def test_match_batch_uses_rank_voting(self):
        """Test that match_batch uses rank voting internally."""
        config = ReIDConfig()
        gallery = PersonGallery(config)

        np.random.seed(42)
        # Add two distinct persons with multiple features each
        feat1 = np.random.randn(512).astype(np.float32)
        feat1 /= np.linalg.norm(feat1)
        id1 = gallery.add(feat1)
        for _ in range(3):
            f = feat1 + 0.05 * np.random.randn(512).astype(np.float32)
            f /= np.linalg.norm(f)
            gallery.update(id1, f)

        feat2 = np.random.randn(512).astype(np.float32)
        feat2 /= np.linalg.norm(feat2)
        id2 = gallery.add(feat2)
        for _ in range(3):
            f = feat2 + 0.05 * np.random.randn(512).astype(np.float32)
            f /= np.linalg.norm(f)
            gallery.update(id2, f)

        # Query batch: one similar to id1, one similar to id2
        q1 = feat1 + 0.02 * np.random.randn(512).astype(np.float32)
        q1 /= np.linalg.norm(q1)
        q2 = feat2 + 0.02 * np.random.randn(512).astype(np.float32)
        q2 /= np.linalg.norm(q2)

        results = gallery.match_batch([q1, q2])

        assert len(results) == 2
        # Each result is (id, confidence)
        assert results[0][0] == id1
        assert results[1][0] == id2

    def test_small_gallery_fallback(self):
        """Test behavior with very small gallery."""
        config = ReIDConfig()
        gallery = PersonGallery(config)

        np.random.seed(42)
        # Add only one entry
        feat = np.random.randn(512).astype(np.float32)
        feat /= np.linalg.norm(feat)
        gallery.add(feat)

        query = np.random.randn(512).astype(np.float32)
        query /= np.linalg.norm(query)

        # Should handle gracefully
        results = gallery.match_batch([query])
        assert len(results) == 1
        # May not match due to min_entries_per_id requirement
        assert isinstance(results[0], tuple)

    def test_exclude_ids_respected(self):
        """Test that excluded IDs are not matched."""
        config = ReIDConfig()
        gallery = PersonGallery(config)

        np.random.seed(42)
        feat = np.random.randn(512).astype(np.float32)
        feat /= np.linalg.norm(feat)

        id1 = gallery.add(feat)
        for _ in range(4):
            f = feat + 0.05 * np.random.randn(512).astype(np.float32)
            f /= np.linalg.norm(f)
            gallery.update(id1, f)

        # Query same feature but exclude id1
        results = gallery.match_batch([feat], exclude_ids={id1})

        assert len(results) == 1
        assert results[0][0] is None  # Should not match excluded ID


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
