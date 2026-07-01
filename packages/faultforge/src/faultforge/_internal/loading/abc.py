"""Abstract interface for loading a model together with its dataset."""

import abc

from torch import nn

from faultforge._internal.common import DeviceLike
from faultforge._internal.dataset import BatchedDataset
from faultforge._internal.fingerprint import Fingerprint


class ModelBundle(abc.ABC):
    """A type which can load a model and its associated dataset"""

    @abc.abstractmethod
    def load_model(self, device: DeviceLike) -> nn.Module:
        """Load the model"""

    @abc.abstractmethod
    def load_dataset(self, batch_size: int, device: DeviceLike) -> BatchedDataset:
        """Load the dataset."""

    @abc.abstractmethod
    def fingerprint(self) -> Fingerprint:
        """Return a structural identity for this model and dataset.

        Only the parts that change which model and dataset are loaded belong in
        the fingerprint; environmental details like the filesystem cache or
        download location are left out.
        """
