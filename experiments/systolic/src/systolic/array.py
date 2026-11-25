"""A systolic array and the matrix multiplications that run on it."""

lazy from collections.abc import Iterator
lazy from dataclasses import dataclass
lazy from typing import final

lazy import torch
lazy from torch import Tensor

lazy from systolic._rust import Fault, LiftedFault, Mapping

_REGISTER_BITS = 32
"""Every register of a processing element holds a float32."""


# No generated `__eq__`: comparing tensors with `==` doesn't give a bool.
@final
@dataclass(frozen=True, slots=True, eq=False)
class ArrayMatmul:
    """A `weights @ activations` matrix multiplication to run on an `Array`.

    Either one multiplication of matrices or a stack of independent ones of the
    same shape. A stack adds one leading dimension to both operands. Its
    multiplications run one after another on the same array.

    Raises:
        ValueError: If the operands aren't float32, are on different devices,
            are empty, or their shapes don't match.
    """

    weights: Tensor
    """The stationary operand, `([stack,] out_features, in_features)`."""
    activations: Tensor
    """The streamed operand, `([stack,] in_features, batch)`."""

    def __post_init__(self) -> None:
        for name, operand in (
            ("weights", self.weights),
            ("activations", self.activations),
        ):
            if operand.dtype != torch.float32:
                raise ValueError(f"{name} must be float32, got {operand.dtype}")
            if operand.dim() not in (2, 3):
                raise ValueError(
                    f"{name} must have 2 or 3 dimensions, got shape {tuple(operand.shape)}"
                )
            if operand.numel() == 0:
                raise ValueError(
                    f"{name} must not be empty, got shape {tuple(operand.shape)}"
                )

        if self.weights.device != self.activations.device:
            raise ValueError(
                "weights and activations must be on the same device, got "
                f"{self.weights.device} and {self.activations.device}"
            )

        # Without this check torch would broadcast mismatched stacks.
        if (
            self.weights.shape[:-2] != self.activations.shape[:-2]
            or self.weights.shape[-1] != self.activations.shape[-2]
        ):
            raise ValueError(
                f"cannot multiply weights {tuple(self.weights.shape)} with "
                f"activations {tuple(self.activations.shape)}"
            )

    @property
    def out_features(self) -> int:
        """The number of rows in the output."""
        return self.weights.shape[-2]

    @property
    def in_features(self) -> int:
        """The number of rows in the activations."""
        return self.weights.shape[-1]

    def unstack(self) -> Iterator[ArrayMatmul]:
        """The individual matrix multiplications, in the order they run."""
        if self.weights.dim() == 2:
            yield self
            return

        for weights, activations in zip(self.weights, self.activations, strict=True):
            yield ArrayMatmul(weights, activations)


@final
class Array:
    """A weight-stationary systolic array of a fixed size.

    Remembers how it maps matrix multiplications, so reuse one instance instead
    of creating a new one for every multiplication.

    Raises:
        ValueError: If `nrows` or `ncols` isn't positive.
    """

    def __init__(self, nrows: int, ncols: int) -> None:
        if nrows <= 0 or ncols <= 0:
            raise ValueError(f"array size must be positive, got {nrows}x{ncols}")
        self._nrows = nrows
        self._ncols = ncols
        # Both are keyed by `(out_features, in_features)`.
        self._mappings: dict[tuple[int, int], Mapping] = {}
        self._lifts: dict[tuple[int, int], LiftedFault] = {}
        # The fault that `_lifts` belongs to. A campaign goes through many
        # faults but uses each for many consecutive matmuls, so keeping the
        # lifts of only one bounds the memory use without losing reuse.
        self._lifts_fault: Fault | None = None

    @property
    def nrows(self) -> int:
        """The number of rows of processing elements."""
        return self._nrows

    @property
    def ncols(self) -> int:
        """The number of columns of processing elements."""
        return self._ncols

    def mapping(self, matmul: ArrayMatmul) -> Mapping:
        """How `matmul` is laid out on the array.

        Applies to each multiplication in a stack.
        """
        shape = (matmul.out_features, matmul.in_features)
        mapping = self._mappings.get(shape)
        if mapping is None:
            mapping = Mapping.auto(
                matmul.in_features, matmul.out_features, self._nrows, self._ncols
            )
            self._mappings[shape] = mapping
        return mapping

    def lift(self, matmul: ArrayMatmul, fault: Fault) -> LiftedFault:
        """The effect of `fault` on the operands and output of `matmul`.

        Applies to each multiplication in a stack.

        Raises:
            TypeError: If `fault` isn't a register fault.
            ValueError: If `fault` is outside the array or its registers.
        """
        if fault != self._lifts_fault:
            self.check_fault(fault)
            self._lifts.clear()
            self._lifts_fault = fault

        shape = (matmul.out_features, matmul.in_features)
        lifted = self._lifts.get(shape)
        if lifted is None:
            lifted = self.mapping(matmul).lift(fault)
            self._lifts[shape] = lifted
        return lifted

    def check_fault(self, fault: Fault) -> None:
        """Check that `fault` is in a register of this array.

        Raises:
            TypeError: If `fault` isn't a register fault.
            ValueError: If `fault` is outside the array or its registers.
        """
        if not isinstance(fault, Fault.Register):
            raise TypeError(f"unsupported fault kind: {type(fault)!r}")
        if fault.target.y >= self._nrows or fault.target.x >= self._ncols:
            raise ValueError(
                f"fault target {fault.target} is outside the "
                f"{self._nrows}x{self._ncols} array"
            )
        if fault.bit_index >= _REGISTER_BITS:
            raise ValueError(
                f"fault bit {fault.bit_index} is outside a {_REGISTER_BITS}-bit register"
            )
