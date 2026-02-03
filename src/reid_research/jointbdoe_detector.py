"""JointBDOE person detection wrapper - optimized for human detection."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import ReIDConfig
from .detectors.jointbdoe import (
    attempt_load,
    letterbox,
    check_img_size,
    non_max_suppression,
    scale_coords,
)
from .utils import extract_crop


@dataclass
class Detection:
    """Single person detection with orientation."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    crop: np.ndarray  # BGR image
    orientation: float | None = None  # Body orientation in degrees (0-360)
    track_id: int | None = None  # Assigned after ReID matching
    features: np.ndarray | None = None
    is_matched: bool = False  # Whether ReID matched existing gallery entry
    previous_id: int | None = None  # Original ID before ReID rematch
    previous_id_timestamp: int | None = None  # Last seen frame of previous ID
    match_similarity: float = 0.0  # Similarity score of the ReID match
    is_recovery: bool = False  # True if ReID match occurs after spatial track loss
    top_similar: list = None  # Top 3 similar IDs: [(id, similarity), ...]


class JointBDOEDetector:
    """JointBDOE person detector - finetuned on human class."""

    NUM_ANGLES = 1  # Single orientation angle output

    def __init__(self, config: ReIDConfig):
        """Initialize detector with config.

        Args:
            config: ReIDConfig instance
        """
        self.config = config
        self._model = None
        self._device = None
        self._stride = 32
        self._imgsz = 1024  # JointBDOE default

    def _ensure_loaded(self) -> None:
        """Lazy load JointBDOE model."""
        if self._model is not None:
            return

        weights_path = Path(self.config.model.yolo_weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"JointBDOE weights not found: {weights_path}")

        self._device = torch.device(self.config.model.device)
        self._model = attempt_load(str(weights_path), map_location=self._device)
        self._stride = int(self._model.stride.max())
        self._imgsz = check_img_size(self._imgsz, s=self._stride)

        # Warmup
        if self._device.type != "cpu":
            self._model(
                torch.zeros(1, 3, self._imgsz, self._imgsz)
                .to(self._device)
                .type_as(next(self._model.parameters()))
            )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect persons in frame.

        Args:
            frame: BGR numpy array (H, W, 3)

        Returns:
            List of Detection objects with crops and orientations
        """
        self._ensure_loaded()

        # Preprocess
        img = letterbox(frame, self._imgsz, stride=self._stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self._device)
        img = img.float() / 255.0
        if len(img.shape) == 3:
            img = img[None]  # Add batch dimension

        # Inference
        with torch.no_grad():
            out = self._model(img, augment=False)[0]
            out = non_max_suppression(
                out,
                self.config.inference.confidence_threshold,
                iou_thres=0.45,
                num_angles=self.NUM_ANGLES,
            )

        detections = []
        if len(out[0]) == 0:
            return detections

        # Scale boxes to original frame size
        # Output format: x1, y1, x2, y2, conf, class, orientation
        boxes = scale_coords(img.shape[2:], out[0][:, :4], frame.shape[:2])
        boxes = boxes.cpu().numpy()
        confs = out[0][:, 4].cpu().numpy()
        orientations = out[0][:, 6:].cpu().numpy() * 360  # (0,1) -> (0,360)

        for bbox, conf, orient in zip(boxes, confs, orientations):
            crop = extract_crop(frame, bbox, padding=10)

            # Skip tiny crops
            if crop.shape[0] < 32 or crop.shape[1] < 16:
                continue

            detections.append(
                Detection(
                    bbox=tuple(bbox),
                    confidence=float(conf),
                    crop=crop,
                    orientation=float(orient[0]) if len(orient) > 0 else None,
                )
            )

        return detections
