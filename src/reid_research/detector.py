"""YOLO11 person detection wrapper."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .config import ReIDConfig
from .utils import extract_crop


@dataclass
class Detection:
    """Single person detection."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    crop: np.ndarray  # BGR image
    track_id: int | None = None  # Assigned after ReID matching
    features: np.ndarray | None = None
    is_matched: bool = False  # True if ReID matched existing gallery entry
    previous_id: int | None = None  # Original ID before ReID rematch
    previous_id_timestamp: int | None = None  # Last seen frame of previous ID
    match_similarity: float = 0.0  # Similarity score of the ReID match
    is_recovery: bool = False  # True if ReID match occurs after spatial track loss


class PersonDetector:
    """YOLO11 person detector."""

    PERSON_CLASS = 0  # COCO class ID for person

    def __init__(self, config: ReIDConfig):
        """Initialize detector with config.

        Args:
            config: ReIDConfig instance
        """
        self.config = config
        self._model: YOLO | None = None

    def _ensure_loaded(self) -> None:
        """Lazy load YOLO model."""
        if self._model is not None:
            return

        weights_path = Path(self.config.model.yolo_weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

        self._model = YOLO(str(weights_path))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect persons in frame.

        Args:
            frame: BGR numpy array (H, W, 3)

        Returns:
            List of Detection objects with crops
        """
        self._ensure_loaded()

        conf_thresh = self.config.inference.confidence_threshold
        results = self._model(frame, conf=conf_thresh, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id != self.PERSON_CLASS:
                continue

            bbox = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
            conf = float(box.conf[0])
            crop = extract_crop(frame, bbox, padding=10)

            # Skip tiny crops
            if crop.shape[0] < 32 or crop.shape[1] < 16:
                continue

            detections.append(
                Detection(
                    bbox=tuple(bbox),
                    confidence=conf,
                    crop=crop,
                )
            )

        return detections
