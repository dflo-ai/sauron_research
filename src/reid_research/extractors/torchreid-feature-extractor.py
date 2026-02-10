"""TorchReID FeatureExtractor ported for standalone use.

A simple API for feature extraction that accepts:
    - a list of numpy.ndarray each with shape (H, W, C)
    - a single numpy.ndarray with shape (H, W, C)
    - a torch.Tensor with shape (B, C, H, W) or (C, H, W)

Returns a torch tensor with shape (B, D) where D is the feature dimension (512 for OSNet).
"""
from __future__ import absolute_import
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from ..models import build_model


def check_isfile(fpath: str | Path) -> bool:
    """Check if file exists and is a file."""
    fpath = Path(fpath)
    return fpath.is_file()


def load_pretrained_weights(model: torch.nn.Module, weight_path: str | Path) -> None:
    """Load pretrained weights into model.

    Args:
        model: PyTorch model
        weight_path: Path to weights file (.pth or .pth.tar)
    """
    weight_path = Path(weight_path)
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded pretrained weights from {weight_path}")


class FeatureExtractor:
    """Simple API for ReID feature extraction.

    Args:
        model_name: Model name (osnet_x1_0, osnet_ibn_x1_0)
        model_path: Path to custom model weights (optional)
        image_size: Image height and width as (H, W)
        pixel_mean: Pixel mean for normalization
        pixel_std: Pixel std for normalization
        pixel_norm: Whether to normalize pixels
        device: Device string ('cpu' or 'cuda')
        verbose: Print model details

    Example:
        extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            device='cuda'
        )
        features = extractor(image_list)  # (N, 512)
    """

    def __init__(
        self,
        model_name: str = "",
        model_path: str = "",
        image_size: tuple[int, int] = (256, 128),
        pixel_mean: list[float] = None,
        pixel_std: list[float] = None,
        pixel_norm: bool = True,
        device: str = "cuda",
        verbose: bool = True,
    ):
        if pixel_mean is None:
            pixel_mean = [0.485, 0.456, 0.406]
        if pixel_std is None:
            pixel_std = [0.229, 0.224, 0.225]

        # Build model - load pretrained if no custom weights provided
        has_custom_weights = model_path and check_isfile(model_path)
        model = build_model(
            model_name,
            num_classes=1,
            pretrained=not has_custom_weights,
            use_gpu=device.startswith("cuda"),
        )
        model.eval()

        if verbose:
            num_params = sum(p.numel() for p in model.parameters())
            print(f"Model: {model_name}")
            print(f"- params: {num_params:,}")

        # Load custom weights if provided
        if has_custom_weights:
            load_pretrained_weights(model, model_path)

        # Build transform pipeline
        transforms = [T.Resize(image_size), T.ToTensor()]
        if pixel_norm:
            transforms.append(T.Normalize(mean=pixel_mean, std=pixel_std))
        preprocess = T.Compose(transforms)

        to_pil = T.ToPILImage()

        device = torch.device(device)
        model.to(device)

        # Class attributes
        self.model = model
        self.preprocess = preprocess
        self.to_pil = to_pil
        self.device = device

    def __call__(self, input) -> torch.Tensor:
        """Extract features from input images.

        Args:
            input: List of numpy arrays, single numpy array, or torch tensor

        Returns:
            Feature tensor with shape (B, D)
        """
        if isinstance(input, list):
            images = []
            for element in input:
                if isinstance(element, str):
                    image = Image.open(element).convert("RGB")
                elif isinstance(element, np.ndarray):
                    image = self.to_pil(element)
                else:
                    raise TypeError(
                        "Type of each element must belong to [str | numpy.ndarray]"
                    )
                image = self.preprocess(image)
                images.append(image)
            images = torch.stack(images, dim=0)
            # Use pinned memory + non-blocking transfer for GPU
            if self.device.type == "cuda":
                images = images.pin_memory().to(self.device, non_blocking=True)
            else:
                images = images.to(self.device)

        elif isinstance(input, str):
            image = Image.open(input).convert("RGB")
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, np.ndarray):
            image = self.to_pil(input)
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, torch.Tensor):
            if input.dim() == 3:
                input = input.unsqueeze(0)
            images = input.to(self.device)

        else:
            raise NotImplementedError(f"Unsupported input type: {type(input)}")

        with torch.inference_mode():
            features = self.model(images)

        return features
