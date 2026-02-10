"""Shared utilities for ReID research module."""
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def safe_compile(model: torch.nn.Module, **kwargs) -> torch.nn.Module:
    """Compile model with torch.compile if available, else return unchanged.

    Args:
        model: PyTorch model to compile
        **kwargs: Arguments passed to torch.compile

    Returns:
        Compiled model or original model if compilation unavailable/fails
    """
    if not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model, **kwargs)
    except Exception as e:
        print(f"Warning: torch.compile failed ({e}), using eager mode")
        return model


def extract_crop(frame: np.ndarray, bbox: tuple, padding: int = 10) -> np.ndarray:
    """Extract person crop from frame with optional padding.

    Args:
        frame: BGR numpy array (H, W, 3)
        bbox: Bounding box (x1, y1, x2, y2)
        padding: Pixels to add around box

    Returns:
        Cropped BGR image
    """
    x1, y1, x2, y2 = map(int, bbox[:4])
    h, w = frame.shape[:2]

    # Apply padding with bounds checking
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    # Validate crop dimensions after bounds checking
    if x2 <= x1 or y2 <= y1:
        logger.debug(f"Degenerate bbox after clamping: ({x1},{y1},{x2},{y2}), returning fallback")
        return np.zeros((1, 1, 3), dtype=frame.dtype)  # 1x1 black pixel fallback

    return frame[y1:y2, x1:x2]


def compute_iou(box1: tuple, box2: tuple) -> float:
    """Compute IoU between two bounding boxes.

    Args:
        box1: First box (x1, y1, x2, y2)
        box2: Second box (x1, y1, x2, y2)

    Returns:
        IoU value in [0, 1]
    """
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]

    # Intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area
