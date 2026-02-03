"""Pytest fixtures for ReID research module tests."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reid_research import ReIDConfig


@pytest.fixture
def config() -> ReIDConfig:
    """Default test config (CPU for CI)."""
    cfg = ReIDConfig()
    cfg.model.device = "cpu"
    return cfg


@pytest.fixture
def dummy_crop() -> np.ndarray:
    """Dummy person crop (256x128 BGR)."""
    return np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)


@pytest.fixture
def dummy_frame() -> np.ndarray:
    """Dummy frame (1080p BGR)."""
    return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def dummy_features() -> np.ndarray:
    """Dummy ReID features (512-dim, normalized)."""
    feat = np.random.randn(512).astype(np.float32)
    return feat / np.linalg.norm(feat)
