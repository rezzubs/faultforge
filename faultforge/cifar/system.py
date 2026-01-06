import typing
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from typing_extensions import override

from faultforge.cifar.dataset import Cifar
from faultforge.cifar.model import CachedModel
from faultforge.dtype import DnnDtype
from faultforge.system import System
from faultforge.utils import build_map_layer, map_layer_recursive

_map_layer = build_map_layer("weight", "bias", "running_mean", "running_var")


@dataclass
class CifarSystem(System[nn.Module]):
    dataset: Cifar
    model: CachedModel
    dtype: DnnDtype
    device: torch.device
    batch_size: int
    dataset_cache: Path | None

    @override
    def system_data(self) -> nn.Module:
        return self.model.root_module(self.dataset).to(self.dtype.to_torch())

    @override
    def system_accuracy(
        self,
        data: nn.Module,
    ) -> float:
        data = data.to(self.device)

        _ = data.eval()
        samples_count = torch.tensor(0).to(self.device)
        correct_count = torch.tensor(0).to(self.device)

        for data_unit in self.dataset.batches(
            self.batch_size,
            self.dtype.to_torch(),
            self.device,
        ):
            inputs, targets = data_unit[0], data_unit[1]
            inputs = inputs.to(self.device).to(self.dtype.to_torch())
            targets = targets.to(self.device).to(self.dtype.to_torch())

            outputs = typing.cast(torch.Tensor, data(inputs))
            assert isinstance(outputs, torch.Tensor)

            outputs = outputs.argmax(dim=1)

            samples_count += targets.size(0)
            correct_count += (outputs == targets).sum()

        return (correct_count / samples_count * 100).item()

    @override
    def system_data_tensors(self, data: nn.Module) -> list[torch.Tensor]:
        return map_layer_recursive(_map_layer, data)

    @override
    def system_metadata(self) -> dict[str, str]:
        return {
            "dtype": self.dtype.name,
            "model": self.model.name,
            "dataset": self.dataset.kind.name,
        }
