from __future__ import annotations

import enum
import logging
import typing
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_logger = logging.getLogger(__name__)

Dataset = datasets.CIFAR10 | datasets.CIFAR100

_CACHE: dict[Cifar.Kind, Dataset] = dict()


@dataclass
class Cifar:
    """A CIFAR dataset with a global cache."""

    kind: Cifar.Kind
    on_disk_cache: Path
    _batches_cache: dict[
        tuple[int, torch.dtype, torch.device],
        list[tuple[torch.Tensor, torch.Tensor]],
    ] = field(default_factory=dict, init=False, repr=False)

    class Kind(enum.Enum):
        CIFAR10 = "cifar10"
        CIFAR100 = "cifar100"

    def load_data(self) -> Dataset:
        """Load the CIFAR dataset."""

        _logger.info(f"Loading dataset `{self.kind.name}`")

        try:
            match self.kind:
                case Cifar.Kind.CIFAR10:
                    mean = (0.49139968, 0.48215827, 0.44653124)
                    std = (0.24703233, 0.24348505, 0.26158768)
                    transform = transforms.Compose(
                        [transforms.ToTensor(), transforms.Normalize(mean, std)]
                    )
                    return datasets.CIFAR10(
                        root=self.on_disk_cache,
                        train=False,
                        download=True,
                        transform=transform,
                    )
                case Cifar.Kind.CIFAR100:
                    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
                    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
                    transform = transforms.Compose(
                        [
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(15),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std),
                        ]
                    )

                    return datasets.CIFAR100(
                        root=self.on_disk_cache,
                        train=False,
                        download=True,
                        transform=transform,
                    )
        finally:
            _logger.debug("Dataset loading finished")

    def _dataset(self) -> Dataset:
        dataset = _CACHE.get(self.kind)

        if dataset is None:
            _logger.debug(f"Dataset {self.kind.name} not yet loaded")
            dataset = self.load_data()
            _CACHE[self.kind] = dataset
        else:
            _logger.debug(f"Got dataset {self.kind.name} from in memory cache")

        return dataset

    def batches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Compute batches for the CIFAR dataset.

        :param batch_size: The number of elements in a batch.
        """
        cache_key = (batch_size, dtype, device)

        if cache_key in self._batches_cache:
            _logger.debug(
                f"Using cached batch for batch_size={batch_size}, dtype={dtype}, device={device}"
            )
            return self._batches_cache[cache_key]

        _logger.info(
            f"Computing CIFAR batches for batch_size={batch_size}, \
dtype={dtype}, device={device}"
        )

        dataloader = typing.cast(
            DataLoader[list[torch.Tensor]],
            DataLoader(
                self._dataset(),
                batch_size=batch_size,
                shuffle=False,
            ),
        )

        batches: list[tuple[torch.Tensor, torch.Tensor]] = []
        # NOTE: The following is necessary because pytorch doesn't provide a
        # type safe API for `DataLoader`.
        iterator = typing.cast(Iterator[list[torch.Tensor]], iter(dataloader))
        for batch in iterator:
            assert isinstance(batch, list)
            assert len(batch) == 2
            assert all(isinstance(x, torch.Tensor) for x in batch)

            batches.append(
                (batch[0].to(dtype).to(device), batch[1].to(dtype).to(device))
            )

        self._batches_cache[cache_key] = batches
        return batches
