"""Transform steps shared by the bundled image datasets."""

lazy from collections.abc import Callable

lazy import torch
lazy from torch import Tensor
lazy from torchvision.transforms import v2

type _TensorTransform = Callable[[Tensor], Tensor]


def dtype_transforms(dtype: torch.dtype | None) -> list[_TensorTransform]:
    """Transform steps casting an image tensor to `dtype`, or none if `dtype` is `None`.

    Meant to be spliced onto the end of a `Compose` pipeline. Only casts; the
    values are not rescaled.
    """
    if dtype is None:
        return []
    return [v2.ToDtype(dtype, scale=False)]
