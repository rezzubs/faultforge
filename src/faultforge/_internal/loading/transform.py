"""Transform steps shared by the bundled image datasets."""

from collections.abc import Callable

import torch
from torch import Tensor
from torchvision.transforms import v2

type TensorTransform = Callable[[Tensor], Tensor]


def dtype_transforms(dtype: torch.dtype | None) -> list[TensorTransform]:
    """Transform steps casting an image tensor to `dtype`, or none if `dtype` is `None`.

    Meant to be spliced onto the end of a `Compose` pipeline. Only casts; the
    values are not rescaled.
    """
    if dtype is None:
        return []
    return [v2.ToDtype(dtype, scale=False)]
