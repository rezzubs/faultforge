"""Abstract base class for evaluating matrix multiplications on an array."""

lazy import abc

lazy from torch import Tensor

lazy from systolic._rust import Fault
lazy from systolic.array import Array, ArrayMatmul


class Evaluator(abc.ABC):
    """A way to compute the output of a matrix multiplication on an array."""

    def __init__(self, array: Array) -> None:
        self._array = array

    @property
    def array(self) -> Array:
        """The array that the multiplications run on."""
        return self._array

    @abc.abstractmethod
    def evaluate(self, matmul: ArrayMatmul, fault: Fault | None) -> Tensor:
        """Compute `matmul` on the array with `fault` present.

        `None` means a fault-free array. Returns
        `([stack,] out_features, batch)` on the same device as the operands.
        The operands are left unchanged.

        Raises:
            TypeError: If the kind of `fault` isn't supported.
            ValueError: If `fault` is outside the array.
        """
        ...
