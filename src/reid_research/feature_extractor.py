"""TorchReID FeatureExtractor wrapper with config-driven interface."""
import numpy as np
import torch

from .config import ReIDConfig
from .extractors import FeatureExtractor
from .utils import safe_compile


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
        """Lazy load the model on first use. Applies torch.compile if available."""
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

        # Apply torch.compile for inference optimization
        self._extractor.model = safe_compile(
            self._extractor.model,
            mode="reduce-overhead",
            fullgraph=False,
        )

        # Warmup batch: trigger torch.compile graph capture + CUDA kernel caching
        use_cuda = self._device.startswith("cuda") if isinstance(self._device, str) else str(self._device).startswith("cuda")
        if use_cuda:
            dummy = torch.randn(
                inf_cfg.batch_size, 3, inf_cfg.image_size[0], inf_cfg.image_size[1],
                device=self._device, dtype=torch.float16,
            )
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = self._extractor.model(dummy)
            torch.cuda.synchronize()

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

        # Pre-allocate output array to avoid repeated vstack allocations
        n_crops = len(rgb_crops)
        batch_size = self.config.inference.batch_size
        features = np.empty((n_crops, 512), dtype=np.float32)

        # Use inference_mode + FP16 autocast for GPU acceleration
        use_cuda = self._device.startswith("cuda") if isinstance(self._device, str) else str(self._device).startswith("cuda")
        ctx_autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if use_cuda else torch.autocast(device_type="cpu", enabled=False)

        with torch.inference_mode(), ctx_autocast:
            for i in range(0, n_crops, batch_size):
                batch = rgb_crops[i : i + batch_size]
                batch_features = self._extractor(batch)

                # Convert to numpy (cast back to float32 from FP16)
                if isinstance(batch_features, torch.Tensor):
                    batch_features = batch_features.float().cpu().numpy()

                # Write directly into pre-allocated array
                end = min(i + batch_size, n_crops)
                features[i:end] = batch_features

        # L2 normalize in-place
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        np.maximum(norms, 1e-8, out=norms)
        features /= norms

        return features

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
