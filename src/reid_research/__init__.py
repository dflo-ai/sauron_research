"""ReID Research Module - JointBDOE + TorchReID video inference."""
__version__ = "0.1.0"

from .config import ReIDConfig, load_config
from .feature_extractor import ReIDFeatureExtractor
from .gallery import GalleryEntry, PersonGallery
from .jointbdoe_detector import JointBDOEDetector, Detection
from .pipeline import VideoReIDPipeline

__all__ = [
    "ReIDConfig",
    "load_config",
    "ReIDFeatureExtractor",
    "JointBDOEDetector",
    "Detection",
    "PersonGallery",
    "GalleryEntry",
    "VideoReIDPipeline",
    "__version__",
]
