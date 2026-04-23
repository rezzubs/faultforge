"""Common items shared across other modules."""

from pathlib import Path

import torch

type DeviceLike = torch.device | str | int

DEFAULT_DEVICE = torch.device("cpu")
DEFAULT_BATCH_SIZE: int = 256

CACHE_DIRECTORY = Path("~/.cache/faultforge/")
