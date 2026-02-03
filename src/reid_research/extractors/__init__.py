"""Feature extraction wrappers for person re-identification."""
# Note: Module uses kebab-case filename, import with underscore
from importlib import import_module

# Import from kebab-case module
_module = import_module(".torchreid-feature-extractor", package=__name__)
FeatureExtractor = _module.FeatureExtractor

__all__ = ["FeatureExtractor"]
