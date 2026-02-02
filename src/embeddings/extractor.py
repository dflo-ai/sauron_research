"""ReID embedding extraction from detection crops.

Uses pretrained CNN backbone with global pooling to generate
L2-normalized feature vectors for person re-identification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models
from typing import Literal

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class EmbeddingExtractor:
    """Extract ReID embeddings from person crops.

    Uses pretrained CNN backbone with projection head.
    Output: L2-normalized feature vectors.
    """

    def __init__(
        self,
        model_name: Literal["resnet50", "resnet18", "efficientnet_b0"] = "resnet50",
        embedding_dim: int = 256,
        input_size: tuple[int, int] = (256, 128),  # H, W (person aspect ratio)
        normalize: bool = True,
        device: str | torch.device = "cuda",
    ):
        """Initialize embedding extractor.

        Args:
            model_name: Backbone model name
            embedding_dim: Output embedding dimension
            input_size: Input crop size (height, width)
            normalize: Whether to L2-normalize outputs
            device: Computation device (must be cuda per requirements)
        """
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.normalize = normalize
        self.device = torch.device(device) if isinstance(device, str) else device

        # Build and setup model
        self.model = self._build_model(model_name, embedding_dim)
        self.model = self.model.to(self.device).eval()

        # Precompute normalization tensors
        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    def _build_model(self, name: str, out_dim: int) -> nn.Module:
        """Build backbone with projection head."""
        if name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Linear(in_features, out_dim)
        elif name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Linear(in_features, out_dim)
        elif name == "efficientnet_b0":
            backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Linear(in_features, out_dim)
        else:
            raise ValueError(f"Unsupported model: {name}. Use resnet50, resnet18, or efficientnet_b0")

        return backbone

    @torch.no_grad()
    def extract(self, crops: list[Tensor] | Tensor) -> Tensor:
        """Extract embeddings from image crops.

        Args:
            crops: List of (C, H, W) tensors or (N, C, H, W) batch.
                   Values should be in [0, 1] or [0, 255] range.

        Returns:
            (N, embedding_dim) normalized embeddings
        """
        if isinstance(crops, list):
            if len(crops) == 0:
                return torch.empty(0, self.embedding_dim, device=self.device)
            # Resize each crop to input_size before stacking (handles variable sizes)
            resized = []
            for c in crops:
                c = c.to(self.device)
                if c.shape[-2:] != tuple(self.input_size):
                    c = F.interpolate(
                        c.unsqueeze(0), size=self.input_size,
                        mode="bilinear", align_corners=False
                    ).squeeze(0)
                resized.append(c)
            batch = torch.stack(resized)
        else:
            batch = crops

        if batch.numel() == 0:
            return torch.empty(0, self.embedding_dim, device=self.device)

        batch = batch.to(self.device)

        # Ensure correct spatial size
        if batch.shape[-2:] != tuple(self.input_size):
            batch = F.interpolate(
                batch, size=self.input_size, mode="bilinear", align_corners=False
            )

        # Normalize to [0, 1] if needed
        if batch.max() > 1.0:
            batch = batch / 255.0

        # Apply ImageNet normalization
        batch = (batch - self._mean) / self._std

        # Extract features
        embeddings = self.model(batch)

        # L2 normalize
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    @torch.no_grad()
    def extract_from_frame(
        self,
        frame: Tensor,         # (C, H, W) or (H, W, C) BGR/RGB
        boxes: Tensor,         # (N, 4) xyxy format
        min_size: int = 20,    # Skip tiny boxes
    ) -> Tensor:
        """Extract embeddings from frame given bounding boxes.

        Args:
            frame: Full frame tensor
            boxes: Bounding boxes in xyxy format
            min_size: Minimum box dimension to process

        Returns:
            (N, embedding_dim) embeddings, zeros for skipped boxes
        """
        if len(boxes) == 0:
            return torch.empty(0, self.embedding_dim, device=self.device)

        # Ensure CHW format
        if frame.dim() == 3 and frame.shape[-1] == 3:  # HWC
            frame = frame.permute(2, 0, 1)

        frame = frame.to(self.device).float()
        if frame.max() > 1.0:
            frame = frame / 255.0

        crops = []
        valid_idx = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.int().tolist()

            # Clamp to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[2], x2)
            y2 = min(frame.shape[1], y2)

            w, h = x2 - x1, y2 - y1

            if w < min_size or h < min_size:
                continue

            crop = frame[:, y1:y2, x1:x2]
            crops.append(crop)
            valid_idx.append(i)

        # Initialize output with zeros
        embeddings = torch.zeros(len(boxes), self.embedding_dim, device=self.device)

        if crops:
            # Resize all crops to same size
            resized = []
            for c in crops:
                c_resized = F.interpolate(
                    c.unsqueeze(0), size=self.input_size,
                    mode="bilinear", align_corners=False
                ).squeeze(0)
                resized.append(c_resized)

            batch_embeds = self.extract(resized)
            for j, idx in enumerate(valid_idx):
                embeddings[idx] = batch_embeds[j]

        return embeddings


def create_extractor(
    model: str = "resnet50",
    dim: int = 256,
    device: str = "cuda",
) -> EmbeddingExtractor:
    """Factory function to create embedding extractor with defaults.

    Args:
        model: Backbone model name
        dim: Embedding dimension
        device: Computation device

    Returns:
        Configured EmbeddingExtractor instance
    """
    return EmbeddingExtractor(
        model_name=model,
        embedding_dim=dim,
        device=device,
    )
