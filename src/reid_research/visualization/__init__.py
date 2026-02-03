"""Visualization subpackage for ReID pipeline."""
from .colors import get_id_color, get_bbox_style, OKABE_ITO_PALETTE
from .gallery_panel import GalleryPanelRenderer, GalleryPanelEntry
from .hud_renderer import HUDRenderer, EventTicker
from .split_view_renderer import SplitViewRenderer

__all__ = [
    "get_id_color",
    "get_bbox_style",
    "OKABE_ITO_PALETTE",
    "GalleryPanelRenderer",
    "GalleryPanelEntry",
    "HUDRenderer",
    "EventTicker",
    "SplitViewRenderer",
]
