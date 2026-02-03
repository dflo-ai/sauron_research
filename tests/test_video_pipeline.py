"""Integration tests for video pipeline."""
import numpy as np
import pytest

from src.reid_research import ReIDConfig, VideoReIDPipeline


@pytest.fixture
def pipeline(config):
    """Create pipeline with CPU config."""
    return VideoReIDPipeline(config)


def test_pipeline_components_initialized(pipeline):
    """Test pipeline initializes all components."""
    assert pipeline.detector is not None
    assert pipeline.extractor is not None
    assert pipeline.gallery is not None


def test_pipeline_config_propagation(config):
    """Test config propagates to all components."""
    pipeline = VideoReIDPipeline(config)
    assert pipeline.detector.config == config
    assert pipeline.extractor.config == config
    assert pipeline.gallery.config == config


def test_pipeline_custom_components(config):
    """Test pipeline accepts custom components."""
    from src.reid_research import PersonDetector, PersonGallery, ReIDFeatureExtractor

    detector = PersonDetector(config)
    extractor = ReIDFeatureExtractor(config)
    gallery = PersonGallery(config)

    pipeline = VideoReIDPipeline(
        config, detector=detector, extractor=extractor, gallery=gallery
    )

    assert pipeline.detector is detector
    assert pipeline.extractor is extractor
    assert pipeline.gallery is gallery


@pytest.mark.skip(reason="Requires model download - run manually")
def test_process_frame(pipeline, dummy_frame):
    """Test single frame processing."""
    detections = pipeline.process_frame(dummy_frame)
    # May have 0 detections on random noise
    assert isinstance(detections, list)


def test_visualize_empty_detections(pipeline, dummy_frame):
    """Test visualization with no detections."""
    vis = pipeline._visualize(dummy_frame, [])
    assert vis.shape == dummy_frame.shape


def test_visualize_with_detections(pipeline, dummy_frame):
    """Test visualization with mock detections."""
    from src.reid_research import Detection

    detections = [
        Detection(
            bbox=(100, 100, 200, 300),
            confidence=0.9,
            crop=np.zeros((200, 100, 3), dtype=np.uint8),
            track_id=0,
        ),
        Detection(
            bbox=(300, 100, 400, 300),
            confidence=0.85,
            crop=np.zeros((200, 100, 3), dtype=np.uint8),
            track_id=1,
        ),
    ]

    vis = pipeline._visualize(dummy_frame, detections)
    assert vis.shape == dummy_frame.shape
    # Visual verification would require manual inspection
