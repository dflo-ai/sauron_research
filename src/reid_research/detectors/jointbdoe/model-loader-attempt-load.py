"""JointBDOE model loading utilities ported from jointbdoe/models/experimental.py.

Provides attempt_load function for loading JointBDOE/YOLO model weights.
"""
from pathlib import Path

import torch
import torch.nn as nn


class Ensemble(nn.ModuleList):
    """Ensemble of models for inference."""

    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            y.append(module(x, augment, profile, visualize)[0])
        y = torch.cat(y, 1)  # NMS ensemble
        return y, None  # inference, train output


def attempt_download(weights: str | Path) -> Path:
    """Check if weights file exists, return path.

    Note: Unlike original, this does not download - weights must exist locally.
    """
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")
    return weights


def attempt_load(
    weights: str | Path | list,
    map_location=None,
    inplace: bool = True,
    fuse: bool = True,
) -> nn.Module:
    """Load JointBDOE/YOLO model weights.

    Args:
        weights: Path to weights file or list of paths for ensemble
        map_location: Device to load weights to
        inplace: Enable inplace operations for activations
        fuse: Whether to fuse Conv+BN layers

    Returns:
        Loaded model (or Ensemble if multiple weights)
    """
    model = Ensemble()

    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(
            attempt_download(w), map_location=map_location, weights_only=False
        )

        # Extract model from checkpoint
        if fuse:
            ema_or_model = ckpt["ema"] if ckpt.get("ema") else ckpt["model"]
            model.append(ema_or_model.float().fuse().eval())
        else:
            ema_or_model = ckpt["ema"] if ckpt.get("ema") else ckpt["model"]
            model.append(ema_or_model.float().eval())

    # Compatibility updates for various PyTorch versions
    for m in model.modules():
        t = type(m)
        if t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = inplace
        elif t.__name__ == "Conv":
            # Handle custom Conv class from model checkpoint
            if hasattr(m, "_non_persistent_buffers_set"):
                m._non_persistent_buffers_set = set()

    if len(model) == 1:
        return model[-1]  # Return single model
    else:
        # Return ensemble
        print(f"Ensemble created with {weights}")
        for k in ["names"]:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[
            torch.argmax(torch.tensor([m.stride.max() for m in model])).int()
        ].stride
        return model
