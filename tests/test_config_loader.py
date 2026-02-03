"""Tests for config loading and validation."""
import tempfile

from src.reid_research import ReIDConfig, load_config


def test_default_config():
    """Test default config values."""
    config = ReIDConfig()
    assert config.model.reid_variant == "osnet_x1_0"
    assert config.model.device == "cuda"
    assert config.inference.batch_size == 32


def test_load_config_from_yaml():
    """Test loading config from YAML file."""
    yaml_content = """
model:
  reid_variant: osnet_x0_75
  device: cpu
inference:
  similarity_threshold: 0.7
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()

        config = load_config(f.name)
        assert config.model.reid_variant == "osnet_x0_75"
        assert config.model.device == "cpu"
        assert config.inference.similarity_threshold == 0.7


def test_config_validation():
    """Test config validates types."""
    config = ReIDConfig()
    config.inference.batch_size = 16
    assert config.inference.batch_size == 16


def test_config_nested_defaults():
    """Test nested config defaults are applied."""
    config = ReIDConfig()
    assert config.gallery.max_features_per_id == 10
    assert config.gallery.ema_alpha == 0.7
    assert config.output.save_video is True
    assert config.output.save_tracks is True
