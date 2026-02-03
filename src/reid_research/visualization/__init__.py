"""Visualization subpackage for ReID pipeline."""
from .colors import get_id_color, get_bbox_style, OKABE_ITO_PALETTE
from .gallery_panel import GalleryPanelRenderer, GalleryPanelEntry
from .hud_renderer import HUDRenderer, EventTicker
from .extended_frame_renderer import ExtendedFrameRenderer

__all__ = [
    "get_id_color",
    "get_bbox_style",
    "OKABE_ITO_PALETTE",
    "GalleryPanelRenderer",
    "GalleryPanelEntry",
    "HUDRenderer",
    "EventTicker",
    "ExtendedFrameRenderer",
]
