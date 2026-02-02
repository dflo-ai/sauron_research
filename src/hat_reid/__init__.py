"""HAT-ReID core module - LDA transformation and history queues."""

import torch

def get_device() -> torch.device:
    """Get CUDA device. Raises error if GPU unavailable (per project requirements)."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required but not available. "
            "This research module requires GPU for efficient LDA computation."
        )
    return torch.device("cuda")


from .lda import LDA
from .queues import FIFOQueue, ScoreQueue

__all__ = ["LDA", "FIFOQueue", "ScoreQueue", "get_device"]
