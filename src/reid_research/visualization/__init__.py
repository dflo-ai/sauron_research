"""Visualization subpackage for ReID pipeline."""
from .colors import get_id_color, get_bbox_style, OKABE_ITO_PALETTE
from .gallery_panel import GalleryPanelRenderer, GalleryPanelEntry
from .hud_renderer import HUDRenderer, EventTicker
from .extended_frame_renderer import ExtendedFrameRenderer

# Import with importlib due to kebab-case filename
import importlib
_id_switch_module = importlib.import_module(
    ".id-switch-frame-capturer", package=__name__
)
IDSwitchCapturer = _id_switch_module.IDSwitchCapturer
FrameData = _id_switch_module.FrameData
PendingCapture = _id_switch_module.PendingCapture

__all__ = [
    "get_id_color",
    "get_bbox_style",
    "OKABE_ITO_PALETTE",
    "GalleryPanelRenderer",
    "GalleryPanelEntry",
    "HUDRenderer",
    "EventTicker",
    "ExtendedFrameRenderer",
    "IDSwitchCapturer",
    "FrameData",
    "PendingCapture",
]
