"""ReID Research Module - TorchReID + YOLO11 video inference."""
__version__ = "0.1.0"

from .config import ReIDConfig, load_config
from .detector import Detection, PersonDetector
from .feature_extractor import ReIDFeatureExtractor
from .gallery import GalleryEntry, PersonGallery
from .jointbdoe_detector import JointBDOEDetector
from .pipeline import VideoReIDPipeline

# Lazy import FastReIDExtractor to avoid fastreid dependency at import time
def __getattr__(name):
    if name == "FastReIDExtractor":
        from .fastreid_extractor import FastReIDExtractor
        return FastReIDExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ReIDConfig",
    "load_config",
    "ReIDFeatureExtractor",
    "FastReIDExtractor",
    "PersonDetector",
    "JointBDOEDetector",
    "Detection",
    "PersonGallery",
    "GalleryEntry",
    "VideoReIDPipeline",
    "__version__",
]
