"""Integration tests for visualization components."""
import numpy as np
import pytest

from src.reid_research.config import ReIDConfig, VisualizationConfig, load_config
from src.reid_research.jointbdoe_detector import Detection
from src.reid_research.pipeline import VideoReIDPipeline
from src.reid_research.visualization import (
    GalleryPanelEntry,
    GalleryPanelRenderer,
    HUDRenderer,
    get_id_color,
)


class TestColorSystem:
    """Tests for the color palette system."""

    def test_get_id_color_deterministic(self):
        """Same ID always returns same color."""
        color1 = get_id_color(5)
        color2 = get_id_color(5)
        assert color1 == color2

    def test_get_id_color_different_ids(self):
        """Different IDs get different colors (within palette size)."""
        colors = [get_id_color(i) for i in range(10)]
        # At least 8 unique colors in first 10
        unique_colors = set(colors)
        assert len(unique_colors) >= 8

    def test_get_id_color_negative_id(self):
        """Negative ID returns gray."""
        color = get_id_color(-1)
        assert color == (128, 128, 128)


class TestGalleryPanelRenderer:
    """Tests for gallery panel rendering."""

    @pytest.fixture
    def config(self):
        return VisualizationConfig()

    @pytest.fixture
    def renderer(self, config):
        return GalleryPanelRenderer(config)

    def test_render_empty_panel(self, renderer):
        """Renders panel with no entries."""
        panel = renderer.render(720, [], fps=30)
        assert panel.shape == (720, 200, 3)

    def test_render_with_entries(self, renderer):
        """Renders panel with gallery entries."""
        entry = GalleryPanelEntry(
            track_id=1,
            thumbnail=np.zeros((80, 60, 3), dtype=np.uint8),
            first_seen=0,
            last_seen=100,
            detection_count=10,
            is_active=True,
        )
        panel = renderer.render(720, [entry], fps=30)
        assert panel.shape == (720, 200, 3)

    def test_composite_on_frame(self, renderer):
        """Composites panel onto main frame."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        panel = renderer.render(720, [], fps=30)
        result = renderer.composite_on_frame(frame, panel)
        assert result.shape == frame.shape


class TestHUDRenderer:
    """Tests for HUD rendering."""

    @pytest.fixture
    def config(self):
        return VisualizationConfig()

    @pytest.fixture
    def renderer(self, config):
        return HUDRenderer(config)

    def test_render_pipeline_bar(self, renderer):
        """Renders pipeline stage bar."""
        bar = renderer.render_pipeline_bar(1280, active_stage=2)
        assert bar.shape == (45, 1280, 3)

    def test_render_stats_panel(self, renderer):
        """Renders stats panel."""
        panel = renderer.render_stats_panel(
            fps=30,
            frame_idx=100,
            total_frames=1000,
            detections=5,
            reid_matches=3,
            unique_ids=12,
            threshold=0.65,
        )
        assert panel.shape == (110, 180, 3)

    def test_event_ticker(self, renderer):
        """Event ticker renders events."""
        renderer.add_event("ID#1 matched", 50, "match")
        ticker = renderer.render_event_ticker(1280, 60)
        assert ticker.shape == (30, 1280, 3)

    def test_composite_hud(self, renderer):
        """Composites all HUD elements."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        bar = renderer.render_pipeline_bar(1280, 0)
        stats = renderer.render_stats_panel(30, 0, 100, 0, 0, 0, 0.5)
        ticker = renderer.render_event_ticker(1280, 0)

        result = renderer.composite_hud(frame, bar, stats, ticker)
        assert result.shape == frame.shape


class TestBackwardCompatibility:
    """Tests for backward compatibility with old configs."""

    def test_config_without_visualization_section(self):
        """Config without visualization section uses defaults."""
        # Create config without visualization (simulates old YAML)
        config = ReIDConfig()
        # Should have defaults
        assert config.visualization.show_gallery_panel is True
        assert config.visualization.show_pipeline_hud is True
        assert config.visualization.gallery_panel_width == 200

    def test_load_default_config(self):
        """Default config loads with visualization section."""
        config = load_config("configs/default.yaml")
        assert hasattr(config, "visualization")
        assert config.visualization.show_gallery_panel is True


class TestPipelineVisualization:
    """Tests for pipeline visualization integration."""

    @pytest.fixture
    def config(self):
        return ReIDConfig()

    def test_visualize_empty_detections(self, config):
        """Pipeline visualizes empty detection list."""

        class MockDetector:
            def detect(self, frame):
                return []

        class MockExtractor:
            def extract(self, crops):
                return []

        pipeline = VideoReIDPipeline(
            config, detector=MockDetector(), extractor=MockExtractor()
        )

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        vis = pipeline._visualize(frame, [])
        assert vis.shape == frame.shape

    def test_visualize_with_detections(self, config):
        """Pipeline visualizes detections with all components."""

        class MockDetector:
            def detect(self, frame):
                return []

        class MockExtractor:
            def extract(self, crops):
                return []

        pipeline = VideoReIDPipeline(
            config, detector=MockDetector(), extractor=MockExtractor()
        )

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        det = Detection(
            bbox=(100, 100, 200, 300),
            confidence=0.9,
            crop=np.zeros((200, 100, 3), dtype=np.uint8),
        )
        det.track_id = 1
        det.is_matched = True

        vis = pipeline._visualize(frame, [det])
        assert vis.shape == frame.shape

    def test_draw_hud(self, config):
        """Pipeline draws HUD correctly."""

        class MockDetector:
            def detect(self, frame):
                return []

        class MockExtractor:
            def extract(self, crops):
                return []

        pipeline = VideoReIDPipeline(
            config, detector=MockDetector(), extractor=MockExtractor()
        )

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = pipeline._draw_hud(frame, 50, 1000, 5, 3, 10)
        assert result.shape == frame.shape

