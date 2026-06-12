"""Types related to datasets."""

import abc
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Self, final, override

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from faultforge._internal.common import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DeviceLike,
)


@dataclass(slots=True)
class DataBatch:
    """A single batch of data containing input tensors and targets."""

    inputs: Tensor
    """A batch of input images."""
    targets: Tensor
    """A batch of expected outputs."""

    def to(self, device: DeviceLike) -> Self:
        """Moves the batch to the specified device.

        This is done in-place.
        """
        if isinstance(device, str):
            device = torch.device(device)

        self.inputs = self.inputs.to(device)
        self.targets = self.targets.to(device)

        return self


class BatchedDataset(abc.ABC):
    """An iterator over batches of image data.

    Provides a type safe API over `torch.utils.data.DataLoader`.

    Use `from_dataset` to create a `BatchedDataset` from any existing dataset.
    """

    @abc.abstractmethod
    def __next__(self) -> DataBatch:
        """Returns the next batch of data."""

    @abc.abstractmethod
    def batch_size(self) -> int:
        """Return the batch size of this dataset."""

    @abc.abstractmethod
    def to(self, device: DeviceLike) -> Self:
        """Maps all future batches to the specified device."""

    def __iter__(self) -> Self:
        return self

    @staticmethod
    def from_dataset(
        dataset: Dataset[Any],
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: DeviceLike = DEFAULT_DEVICE,
    ) -> BatchedDataset:
        return _BatchedDataset(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
            ),
            torch.device(device),
            batch_size,
        )

    def precompute(self, limit: int | None = None) -> CachedDataset:
        """Precompute all batches.

        See `CachedDataset` for details.
        """
        return CachedDataset(self, limit)


@final
@dataclass(slots=True)
class _BatchedDataset(BatchedDataset):
    _loader: DataLoader[Any]
    _device: torch.device
    _batch_size: int

    @override
    def __next__(self) -> DataBatch:
        batch = next(iter(self._loader))

        if not isinstance(batch, Iterable):
            raise TypeError(
                f"Expected next(dataloader) to return an instance of `Iterable`, got {type(batch)}"
            )

        batch = list(batch)
        if len(batch) != 2:
            raise ValueError(
                f"Expected the dataset to yield two items, got {len(batch)}"
            )

        [inputs, labels] = batch

        if not isinstance(inputs, Tensor):
            raise TypeError(f"expected inputs to be a Tensor, got {type(inputs)}")
        if not isinstance(labels, Tensor):
            raise TypeError(f"expected labels to be a Tensor, got {type(labels)}")

        return DataBatch(inputs, labels)

    @override
    def to(self, device: DeviceLike) -> Self:
        self.device = torch.device(device)
        return self

    @override
    def batch_size(self) -> int:
        return self._batch_size


@final
@dataclass(slots=True)
class CachedDataset(BatchedDataset):
    """A dataset that precomputes all batches and keeps them in memory.

    This is useful when the dataset needs to be used multiple times and the
    overhead of computing the batches is significant. As a rule of thumb, you
    should almost definitely use this over a normal `BatchedDataset` if it's
    feasible to store the full data in memory.
    """

    cursor: int
    _items: list[DataBatch]
    _batch_size: int

    def __init__(self, dataset: BatchedDataset, limit: int | None = None) -> None:
        self._batch_size = dataset.batch_size()
        self._items = []
        for i, batch in enumerate(dataset):
            if limit is not None and i >= limit:
                break
            self._items.append(batch)
        self.cursor = 0

    def reset(self) -> None:
        """Enables iteration from the beginning"""
        self.cursor = 0

    @override
    def batch_size(self) -> int:
        return self._batch_size

    @override
    def to(self, device: DeviceLike) -> Self:
        for item in self._items:
            item.to(device)
        return self

    @override
    def __next__(self) -> DataBatch:
        if self.cursor >= len(self._items):
            raise StopIteration
        batch = self._items[self.cursor]
        self.cursor += 1
        return batch
