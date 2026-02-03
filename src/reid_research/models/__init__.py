"""Neural network model architectures for person re-identification."""
from .osnet import OSNet, osnet_x1_0, osnet_ibn_x1_0

__all__ = ["OSNet", "osnet_x1_0", "osnet_ibn_x1_0"]

# Model factory for building models by name
_MODEL_FACTORY = {
    "osnet_x1_0": osnet_x1_0,
    "osnet_ibn_x1_0": osnet_ibn_x1_0,
}


def build_model(
    name: str,
    num_classes: int = 1,
    pretrained: bool = True,
    use_gpu: bool = True,
):
    """Build a model by name.

    Args:
        name: Model name (osnet_x1_0, osnet_ibn_x1_0)
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights
        use_gpu: Whether to use GPU (unused, kept for API compatibility)

    Returns:
        Model instance
    """
    if name not in _MODEL_FACTORY:
        raise KeyError(f"Unknown model: {name}. Available: {list(_MODEL_FACTORY.keys())}")
    return _MODEL_FACTORY[name](num_classes=num_classes, pretrained=pretrained)
