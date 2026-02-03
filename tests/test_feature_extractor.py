"""Tests for feature extractor wrapper (lightweight, no model loading)."""
import numpy as np

from src.reid_research import ReIDConfig, ReIDFeatureExtractor


def test_extractor_initialization(config):
    """Test extractor initializes without loading model."""
    extractor = ReIDFeatureExtractor(config)
    assert extractor._extractor is None  # Lazy loading


def test_feature_dim(config):
    """Test feature dimension property."""
    extractor = ReIDFeatureExtractor(config)
    assert extractor.feature_dim == 512


def test_extract_empty_list(config):
    """Test extracting from empty list returns empty array."""
    extractor = ReIDFeatureExtractor(config)
    # Don't load model, just test empty case
    result = extractor.extract([])
    assert result.shape == (0, 512)


def test_extractor_config_access(config):
    """Test extractor has access to config."""
    extractor = ReIDFeatureExtractor(config)
    assert extractor.config.model.device == "cpu"
    assert extractor._device == "cpu"
