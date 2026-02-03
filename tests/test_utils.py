"""Tests for utility functions."""
import numpy as np

from src.reid_research.utils import compute_iou, extract_crop


def test_extract_crop_basic():
    """Test basic crop extraction."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:40, 30:50] = 255  # White rectangle

    crop = extract_crop(frame, (30, 20, 50, 40), padding=0)
    assert crop.shape == (20, 20, 3)
    assert np.all(crop == 255)


def test_extract_crop_with_padding():
    """Test crop extraction with padding."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = extract_crop(frame, (30, 20, 50, 40), padding=5)
    assert crop.shape == (30, 30, 3)  # 20+5+5, 20+5+5 (padding on each side)


def test_extract_crop_boundary_clipping():
    """Test crop respects frame boundaries."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = extract_crop(frame, (0, 0, 20, 20), padding=10)
    # Should clip to frame boundaries
    assert crop.shape[0] <= 30
    assert crop.shape[1] <= 30


def test_compute_iou_identical():
    """Test IoU of identical boxes is 1.0."""
    box = (10, 10, 50, 50)
    assert compute_iou(box, box) == 1.0


def test_compute_iou_no_overlap():
    """Test IoU of non-overlapping boxes is 0.0."""
    box1 = (0, 0, 10, 10)
    box2 = (20, 20, 30, 30)
    assert compute_iou(box1, box2) == 0.0


def test_compute_iou_partial_overlap():
    """Test IoU of partially overlapping boxes."""
    box1 = (0, 0, 20, 20)
    box2 = (10, 10, 30, 30)
    # Intersection: 10x10 = 100
    # Union: 400 + 400 - 100 = 700
    iou = compute_iou(box1, box2)
    assert 0.14 < iou < 0.15  # ~100/700 ≈ 0.143


def test_compute_iou_contained():
    """Test IoU when one box contains another."""
    outer = (0, 0, 100, 100)
    inner = (25, 25, 75, 75)
    # Inner area: 50x50 = 2500
    # Outer area: 100x100 = 10000
    # Union: 10000 (outer contains inner)
    iou = compute_iou(outer, inner)
    assert iou == 2500 / 10000  # 0.25
