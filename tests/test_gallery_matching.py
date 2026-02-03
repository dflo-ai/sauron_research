"""Tests for gallery matching logic."""
import numpy as np
import pytest

from src.reid_research import PersonGallery, ReIDConfig


@pytest.fixture
def gallery(config):
    """Create gallery with test config."""
    return PersonGallery(config)


def test_add_new_person(gallery, dummy_features):
    """Test adding new person to gallery."""
    track_id = gallery.add(dummy_features)
    assert track_id == 0
    assert gallery.size == 1


def test_match_same_person(gallery, dummy_features):
    """Test matching same features returns same ID."""
    track_id1 = gallery.match_and_update(dummy_features)
    track_id2 = gallery.match_and_update(dummy_features)
    assert track_id1 == track_id2


def test_match_different_persons(gallery):
    """Test different features get different IDs."""
    feat1 = np.random.randn(512).astype(np.float32)
    feat1 /= np.linalg.norm(feat1)

    feat2 = np.random.randn(512).astype(np.float32)
    feat2 /= np.linalg.norm(feat2)

    # Set high threshold to ensure different IDs for random features
    gallery.config.inference.similarity_threshold = 0.95

    track_id1 = gallery.match_and_update(feat1)
    track_id2 = gallery.match_and_update(feat2)

    # Different random features should get different IDs with high threshold
    assert track_id1 != track_id2
    assert gallery.size == 2


def test_prune_stale_entries(gallery, dummy_features):
    """Test stale entry pruning."""
    gallery.add(dummy_features)

    # Simulate many frames passing
    for _ in range(400):
        gallery.step_frame()

    pruned = gallery.prune_stale(max_age=300)
    assert len(pruned) == 1
    assert gallery.size == 0


def test_export_features(gallery, dummy_features):
    """Test feature export."""
    gallery.add(dummy_features)

    feat2 = np.random.randn(512).astype(np.float32)
    gallery.add(feat2)

    exported = gallery.export_features()
    assert len(exported) == 2
    assert all(len(f) == 512 for f in exported.values())


def test_save_load_gallery(gallery, dummy_features, tmp_path):
    """Test gallery save and load."""
    gallery.add(dummy_features)
    gallery.step_frame()
    gallery.step_frame()

    save_path = tmp_path / "gallery.json"
    gallery.save(save_path)

    # Create new gallery and load
    new_gallery = PersonGallery(gallery.config)
    new_gallery.load(save_path)

    assert new_gallery.size == 1
    assert new_gallery._next_id == 1
    assert new_gallery._frame_idx == 2


def test_ema_feature_update(gallery):
    """Test EMA feature update."""
    # Create two similar but different features
    feat1 = np.ones(512, dtype=np.float32)
    feat1 /= np.linalg.norm(feat1)

    feat2 = np.ones(512, dtype=np.float32) * 0.9
    feat2[0] = 1.1  # Slightly different
    feat2 /= np.linalg.norm(feat2)

    # Low threshold to match
    gallery.config.inference.similarity_threshold = 0.5

    track_id = gallery.add(feat1)
    original_avg = gallery.get_entry(track_id).avg_feature.copy()

    gallery.update(track_id, feat2)
    updated_avg = gallery.get_entry(track_id).avg_feature

    # EMA should have updated the average
    assert not np.allclose(original_avg, updated_avg)


def test_gallery_clear(gallery, dummy_features):
    """Test gallery clear."""
    gallery.add(dummy_features)
    gallery.add(dummy_features)
    gallery.step_frame()

    gallery.clear()

    assert gallery.size == 0
    assert gallery._next_id == 0
    assert gallery._frame_idx == 0


def test_batch_match_unique_ids(gallery):
    """Test batch matching ensures unique IDs per frame."""
    # Add two persons to gallery
    feat1 = np.array([1.0] * 512, dtype=np.float32)
    feat1 /= np.linalg.norm(feat1)
    feat2 = np.array([0.5] * 256 + [1.0] * 256, dtype=np.float32)
    feat2 /= np.linalg.norm(feat2)

    id1 = gallery.add(feat1)
    id2 = gallery.add(feat2)

    # Query with features similar to both
    query1 = feat1.copy()
    query2 = feat2.copy()

    # Batch match should assign different IDs
    matched = gallery.match_batch([query1, query2])

    assert matched[0] == id1
    assert matched[1] == id2
    assert matched[0] != matched[1]


def test_batch_match_no_duplicates(gallery):
    """Test same gallery ID not assigned to multiple queries."""
    # Add one person
    feat = np.ones(512, dtype=np.float32)
    feat /= np.linalg.norm(feat)
    gallery_id = gallery.add(feat)

    # Two similar queries (both would match the same gallery entry)
    query1 = feat.copy()
    query2 = feat * 0.99  # Very similar
    query2 /= np.linalg.norm(query2)

    gallery.config.inference.similarity_threshold = 0.5

    matched = gallery.match_batch([query1, query2])

    # First query gets the match, second gets None (no duplicate assignment)
    assert matched[0] == gallery_id
    assert matched[1] is None  # Cannot match same ID twice
