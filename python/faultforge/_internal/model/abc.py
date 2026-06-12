import abc

from torch import nn

from faultforge._internal.common import DeviceLike
from faultforge._internal.dataset import BatchedDataset


class ModelBundle(abc.ABC):
    """A type which can load a model and its associated dataset"""

    @abc.abstractmethod
    def load_model(self) -> nn.Module:
        """Load the model"""

    @abc.abstractmethod
    def load_dataset(self, batch_size: int, device: DeviceLike) -> BatchedDataset:
        """Load the dataset."""
