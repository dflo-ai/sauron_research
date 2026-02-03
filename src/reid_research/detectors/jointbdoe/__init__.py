"""JointBDOE detector utilities ported for standalone use."""
from importlib import import_module

# Import from kebab-case modules
_loader = import_module(".model-loader-attempt-load", package=__name__)
_preprocess = import_module(".preprocessing-letterbox-resize", package=__name__)
_postprocess = import_module(".postprocessing-nms-scale-coords", package=__name__)

attempt_load = _loader.attempt_load
Ensemble = _loader.Ensemble

letterbox = _preprocess.letterbox
check_img_size = _preprocess.check_img_size
make_divisible = _preprocess.make_divisible

non_max_suppression = _postprocess.non_max_suppression
scale_coords = _postprocess.scale_coords
clip_coords = _postprocess.clip_coords
xywh2xyxy = _postprocess.xywh2xyxy

__all__ = [
    "attempt_load",
    "Ensemble",
    "letterbox",
    "check_img_size",
    "make_divisible",
    "non_max_suppression",
    "scale_coords",
    "clip_coords",
    "xywh2xyxy",
]
