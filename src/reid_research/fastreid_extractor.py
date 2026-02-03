"""FastReID feature extractor wrapper with config-driven interface."""
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from .config import ReIDConfig

# Lazy import fastreid to avoid dependency errors when not using FastReID
if TYPE_CHECKING:
    from fastreid.config import CfgNode
    from fastreid.engine.defaults import DefaultPredictor

# Path to fastreid module
FASTREID_PATH = Path(__file__).parent.parent.parent / "fast-reid"


class FastReIDExtractor:
    """FastReID wrapper matching ReIDFeatureExtractor interface.

    Uses SBS(R50-ibn) model by default - 95.7% Rank@1, 89.3% mAP on Market1501.
    Outputs 2048-dim features (ResNet50 backbone with IBN).
    """

    # ImageNet normalization (FastReID default)
    PIXEL_MEAN = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255])
    PIXEL_STD = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])

    def __init__(self, config: ReIDConfig):
        """Initialize with config. Model loaded lazily on first extract call.

        Args:
            config: ReIDConfig instance with fastreid_config and fastreid_weights
        """
        self.config = config
        self._predictor: DefaultPredictor | None = None
        self._device = config.model.device
        self._feature_dim: int = 2048  # SBS R50-ibn default

    def _ensure_loaded(self) -> None:
        """Lazy load the FastReID model on first use."""
        if self._predictor is not None:
            return

        # Lazy import fastreid (adds to sys.path if needed)
        if str(FASTREID_PATH) not in sys.path:
            sys.path.insert(0, str(FASTREID_PATH))

        from fastreid.config import get_cfg
        from fastreid.engine.defaults import DefaultPredictor

        model_cfg = self.config.model

        # Build FastReID config
        cfg = get_cfg()
        cfg.merge_from_file(model_cfg.fastreid_config)
        cfg.MODEL.DEVICE = self._device
        cfg.MODEL.WEIGHTS = model_cfg.fastreid_weights
        cfg.MODEL.BACKBONE.PRETRAIN = False  # Use loaded weights only

        # Create predictor (loads model)
        self._predictor = DefaultPredictor(cfg)

        # Determine actual feature dimension from config
        embed_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        backbone_dim = cfg.MODEL.BACKBONE.FEAT_DIM
        self._feature_dim = embed_dim if embed_dim > 0 else backbone_dim

    def extract(self, crops: list[np.ndarray]) -> np.ndarray:
        """Extract features from batch of crops.

        Args:
            crops: List of BGR numpy arrays (H, W, 3)

        Returns:
            Features array (N, 2048) normalized
        """
        self._ensure_loaded()

        if not crops:
            return np.empty((0, self._feature_dim), dtype=np.float32)

        # Get input size from config (H, W)
        input_size = self.config.model.fastreid_input_size

        # Preprocess all crops: BGR->RGB, resize, normalize, to tensor
        batch = []
        for crop in crops:
            # BGR to RGB
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            # Resize to model input size (H, W)
            resized = cv2.resize(
                rgb, (input_size[1], input_size[0]), interpolation=cv2.INTER_CUBIC
            )
            # Normalize with ImageNet mean/std
            img = resized.astype(np.float32)
            img = (img - self.PIXEL_MEAN) / self.PIXEL_STD
            # HWC -> CHW
            tensor = torch.as_tensor(img.transpose(2, 0, 1))
            batch.append(tensor)

        # Stack into batch tensor (B, C, H, W)
        batch_tensor = torch.stack(batch).to(self._device)

        # Run inference
        with torch.no_grad():
            features = self._predictor(batch_tensor)

        return features.cpu().numpy().astype(np.float32)

    def extract_single(self, crop: np.ndarray) -> np.ndarray:
        """Extract features from single crop.

        Args:
            crop: BGR numpy array (H, W, 3)

        Returns:
            Feature vector (2048,)
        """
        return self.extract([crop])[0]

    @property
    def feature_dim(self) -> int:
        """Feature dimension (2048 for SBS R50-ibn)."""
        self._ensure_loaded()
        return self._feature_dim
