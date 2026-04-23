"""Types related to datasets."""

import abc
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Self, override

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from faultforge._internal_new.common import (
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

    def precompute(self) -> CachedDataset:
        """Precompute all batches.

        See `CachedDataset` for details.
        """
        return CachedDataset(self)


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


@dataclass(slots=True)
class CachedDataset(Dataset[DataBatch]):
    """A dataset that precomputes all batches and keeps them in memory.

    This is useful when the dataset needs to be used multiple times and the
    overhead of computing the batches is significant. As a rule of thumb, you
    should almost definitely use this over a normal `BatchedDataset` if it's
    feasible to store the full data in memory.
    """

    _items: list[DataBatch]
    _batch_size: int

    def __init__(self, dataset: BatchedDataset) -> None:
        self._batch_size = dataset.batch_size()
        self._items = list(dataset)

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset[Any],
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: DeviceLike = DEFAULT_DEVICE,
    ) -> CachedDataset:
        """Create a new batched dataset from an existing dataset.

        The dataset is expected to contain an iterable which yields two tensors.
        This is checked at runtime.
        """
        return cls(BatchedDataset.from_dataset(dataset, batch_size, device))

    @override
    def __getitem__(self, index: int) -> DataBatch:
        return self._items[index]
