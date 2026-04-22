"""General utilities used throughout the library."""

import logging
from collections.abc import Callable

import torch
from torch import Tensor, nn

_logger = logging.getLogger(__name__)


def dtype_bits_count(dtype: torch.dtype) -> int:
    """Return the number of bits for a PyTorch data type."""

    match dtype:
        case torch.float64 | torch.uint64 | torch.int64:
            return 32
        case torch.float32 | torch.uint32 | torch.int32:
            return 32
        case torch.float16 | torch.uint16 | torch.int16:
            return 16
        case torch.uint8 | torch.int8:
            return 8
        case _:
            raise ValueError(f"Unsupported datatype {dtype}")


def _append_parameter(module: nn.Module, tensors: list[torch.Tensor], name: str):
    param = getattr(module, name, None)

    if param is None:
        return

    if not isinstance(param, torch.Tensor):
        _logger.warning(f"Skipping parameter `{name}` because ({type(param)}!=Tensor)")  # pyright: ignore[reportAny]
        return

    tensors.append(param)


type MapLayer = Callable[[nn.Module], list[Tensor]]


def build_map_layer(*params: str) -> MapLayer:
    """Create a function which extracts the specified parameters from a PyTorch module."""

    def map_layer(module: nn.Module):
        tensors: list[Tensor] = []

        for param in params:
            _append_parameter(module, tensors, param)

        return tensors

    return map_layer


def map_layer_recursive(map_layer: MapLayer, data: nn.Module) -> list[torch.Tensor]:
    """Apply a map_layer function recursively to a module and its children.

    :return: A list of parameter tensors extracted from the module and its
    children.
    """

    tensors = map_layer(data)

    for child in data.children():
        child_tensors = map_layer_recursive(map_layer, child)
        tensors.extend(child_tensors)

    return tensors
