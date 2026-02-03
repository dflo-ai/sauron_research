"""TorchReID FeatureExtractor wrapper with config-driven interface."""
import sys
from pathlib import Path

import numpy as np
import torch

# Add torchreid to path
TORCHREID_PATH = Path(__file__).parent.parent.parent / "deep-person-reid"
if str(TORCHREID_PATH) not in sys.path:
    sys.path.insert(0, str(TORCHREID_PATH))

from torchreid.utils import FeatureExtractor

from .config import ReIDConfig


class ReIDFeatureExtractor:
    """Wrapper around torchreid FeatureExtractor with config support."""

    def __init__(self, config: ReIDConfig):
        """Initialize with config. Model loaded lazily on first extract call.

        Args:
            config: ReIDConfig instance
        """
        self.config = config
        self._extractor: FeatureExtractor | None = None
        self._device = config.model.device

    def _ensure_loaded(self) -> None:
        """Lazy load the model on first use."""
        if self._extractor is not None:
            return

        model_cfg = self.config.model
        inf_cfg = self.config.inference

        self._extractor = FeatureExtractor(
            model_name=model_cfg.reid_variant,
            model_path=model_cfg.reid_weights,  # None = auto-download
            image_size=inf_cfg.image_size,
            device=self._device,
        )

    def extract(self, crops: list[np.ndarray]) -> np.ndarray:
        """Extract features from batch of crops.

        Args:
            crops: List of BGR numpy arrays (H, W, 3)

        Returns:
            Features array (N, 512)
        """
        self._ensure_loaded()

        if not crops:
            return np.empty((0, 512), dtype=np.float32)

        # Convert BGR to RGB (torchreid expects RGB)
        rgb_crops = [crop[:, :, ::-1] for crop in crops]

        # Process in batches to avoid OOM
        batch_size = self.config.inference.batch_size
        all_features = []

        for i in range(0, len(rgb_crops), batch_size):
            batch = rgb_crops[i : i + batch_size]
            features = self._extractor(batch)

            # Convert to numpy if tensor
            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy()

            all_features.append(features)

        return np.vstack(all_features)

    def extract_single(self, crop: np.ndarray) -> np.ndarray:
        """Extract features from single crop.

        Args:
            crop: BGR numpy array (H, W, 3)

        Returns:
            Feature vector (512,)
        """
        features = self.extract([crop])
        return features[0]

    @property
    def feature_dim(self) -> int:
        """Feature dimension (always 512 for OSNet)."""
        return 512
