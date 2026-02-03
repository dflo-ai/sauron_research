"""JointBDOE preprocessing utilities ported from jointbdoe/utils.

Provides letterbox resize and image size checking for YOLO-style inference.
"""
import math

import cv2
import numpy as np


def make_divisible(x: int | float, divisor: int) -> int:
    """Returns x evenly divisible by divisor."""
    return math.ceil(x / divisor) * divisor


def check_img_size(imgsz: int | list, s: int = 32, floor: int = 0) -> int | list:
    """Verify image size is a multiple of stride s in each dimension.

    Args:
        imgsz: Image size (int or [h, w])
        s: Stride to align to
        floor: Minimum size

    Returns:
        Aligned image size
    """
    if isinstance(imgsz, int):
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]

    if new_size != imgsz:
        print(f"WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}")

    return new_size


def letterbox(
    im: np.ndarray,
    new_shape: tuple[int, int] | int = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scaleFill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Resize and pad image while meeting stride-multiple constraints.

    Args:
        im: Input image (H, W, C)
        new_shape: Target shape (H, W) or single int
        color: Padding color (B, G, R)
        auto: Minimum rectangle (pad to stride multiple)
        scaleFill: Stretch to fill (no padding)
        scaleup: Allow scaling up (set False for val to avoid upscaling)
        stride: Stride for auto padding alignment

    Returns:
        Tuple of (padded_image, ratio, padding):
            - padded_image: Resized and padded image
            - ratio: (width_ratio, height_ratio)
            - padding: (dw, dh) padding amounts
    """
    shape = im.shape[:2]  # Current shape [height, width]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # Only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # Minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # Stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # Divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # Resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, ratio, (dw, dh)
