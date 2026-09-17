"""Abstract interface for loading a model together with its dataset."""

import abc

from torch import nn

from faultforge._internal.dataset import BatchedDataset, DeviceLike
from faultforge._internal.fingerprint import Fingerprint
from faultforge._internal.progress import Progress


class ModelBundle(abc.ABC):
    """A type which can load a model and its associated dataset.

    Anything that changes what gets loaded, such as the parameter dtype, is
    configuration of the concrete bundle and part of its `fingerprint`.
    """

    @abc.abstractmethod
    def load_model(
        self,
        device: DeviceLike,
        *,
        progress: Progress | None = None,
    ) -> nn.Module:
        """Load the model."""

    @abc.abstractmethod
    def load_dataset(
        self,
        batch_size: int,
        device: DeviceLike,
        *,
        shuffle: bool = False,
        seed: int | None = None,
        progress: Progress | None = None,
    ) -> BatchedDataset:
        """Load the dataset.

        `shuffle`/`seed` control random-order iteration (e.g. for uniform
        dataset subsampling); the default preserves the dataset's on-disk
        order.
        """

    @abc.abstractmethod
    def fingerprint(self) -> Fingerprint:
        """Return a structural identity for this model and dataset.

        Only the parts that change which model and dataset are loaded belong in
        the fingerprint; environmental details like the filesystem cache or
        download location are left out.
        """
