"""A :class:`System` implementation for ImageNet DNN models."""

from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor, nn

from faultforge.dtype import DnnDtype
from faultforge.imagenet.dataset import ImageNet
from faultforge.imagenet.model import Model
from faultforge.system import System
from faultforge.utils import build_map_layer, map_layer_recursive

_map_layer = build_map_layer("weight", "bias")


@dataclass
class ImagenetSystem(System[nn.Module]):
    """The :class:`System` implementation for ImageNet DNN models."""

    batch_size: int
    device: torch.device
    dtype: DnnDtype
    model: Model
    dataset: ImageNet

    @override
    def system_data(self) -> nn.Module:
        return self.model.get_root_module().to(self.dtype.to_torch())

    @override
    def system_accuracy(self, data: nn.Module) -> float:
        model = data.to(self.device)

        _ = model.eval()
        total_count = torch.tensor(0).to(self.device)
        correct_count = torch.tensor(0).to(self.device)

        for batch in self.dataset.batches(
            self.batch_size, self.dtype.to_torch(), self.device
        ):
            inputs, labels = batch

            logits = model(inputs)  # pyright: ignore[reportAny]
            assert isinstance(logits, Tensor)

            predictions = logits.data.argmax(dim=1)

            batch_size = labels.size(0)
            assert batch_size <= self.batch_size
            assert batch_size == logits.size(0)

            total_count += batch_size

            correct_count += (predictions == labels).sum()

        return (correct_count / total_count * 100).item()

    @override
    def system_data_tensors(self, data: nn.Module) -> list[torch.Tensor]:
        return map_layer_recursive(_map_layer, data)

    @override
    def system_metadata(self) -> dict[str, str]:
        return {
            "dtype": self.dtype.name,
            "model": self.model.name,
            "dataset": "ImageNet",
        }
