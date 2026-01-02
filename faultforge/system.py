"""The definition of the `system` base class."""

from __future__ import annotations

import abc
import copy

import torch

from .tensor_ops import (
    tensor_list_bits_count,
    tensor_list_fault_injection,
)


class BaseSystem[T](abc.ABC):
    """Base class for systems.

    All of the methods of this class should be examined carefully before
    subclassing and are safe to override.

    A system is the fundamental building block of the faultforge library. It
    represents a component that can be affected by fault injection like a neural
    network.

    Anything that has the following properties can be used as a system:
    - Have core data that can be represented as or converted to a list of pytorch tensors.
    - Be able to generate an accuracy metric from that data.
    - Perform fault injection on its data.

    A custom system implementation should implement at least the following
    methods:
    - :func:`BaseSystem.system_data`
    - :func:`BaseSystem.system_accuracy`
    - :func:`BaseSystem.system_data_tensors`
    - :func:`BaseSystem.system_fault_injection` (A default implementation is
      provided for cases where the system data maps 1:1 to tensors. Note that
      :func:`BaseSystem.system_data_tensors` **must** return a reference to the
      underlying data or the default implementation will not work.)
    """

    @abc.abstractmethod
    def system_data(self) -> T:
        """Return the data that will be used as input to other methods with a ``data`` parameter.

        The data is expected to contain all the components which are affected by
        fault injection that used to determine the accuracy of the system. In
        the case of a DNN it would most likely be the root module but it can
        really be anything.

        Other methods may receive this instance directly or copies of the data.
        See :func:`BaseSystem.system_clone_data`.
        """

    @abc.abstractmethod
    def system_accuracy(self, data: T) -> float:
        """Get the accuracy of the system for the given ``data``."""

    @abc.abstractmethod
    def system_data_tensors(self, data: T) -> list[torch.Tensor]:
        """Return the tensor representation of the underlying data.

        These tensors are only expected to actually reference the data if the
        default implementation of :func:`system_inject_n_faults` is expected to
        be used.
        """

    def system_inject_n_faults(self, data: T, n: int):
        """Inject `n` faults uniformly into the system `data`.

        The default implementation uses :func:`tensor_list_fault_injection`
        which will only work if updating the return value of
        :func:`system_data_tensors` actually changes the data.
        """

        tensors = self.system_data_tensors(data)
        tensor_list_fault_injection(tensors, n)

    def system_metadata(self) -> dict[str, str]:
        """Return metadata about the system.

        This is used to uniquely identify the system.
        """

        return dict()

    def system_clone_data(self, data: T) -> T:
        """Clone the data.

        The default implementation uses :func:`copy.deepcopy`.
        """

        return copy.deepcopy(data)

    def system_total_bits_count(self) -> int:
        """Get the total number of bits in the system."""

        return tensor_list_bits_count(self.system_data_tensors(self.system_data()))
