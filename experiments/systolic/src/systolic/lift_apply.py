"""Torch implementation of `LiftedRegisterFault::matmul`
(`crates/systolic/src/fault/register_lift.rs`). See `docs/fault-lifting.md`
for the theory behind each recipe.

Each arm follows the same recipe as the Rust one. Parts that share a range
of activation rows are applied together, since every torch call has a fixed
overhead that adds up when a lift has one part per output row.
"""

lazy from collections.abc import Iterable
lazy from dataclasses import dataclass
lazy from typing import Protocol

lazy import torch
lazy from torch import Tensor

lazy from systolic._rust import Fault, LiftedFault, StuckAtKind


def _corrupt(tensor: Tensor, stuck_at: StuckAtKind, bit_index: int) -> Tensor:
    """Force `bit_index` of every element to `stuck_at`, via a same-width int view."""
    if tensor.dtype != torch.float32:
        raise ValueError(f"corrupt() only supports float32, got {tensor.dtype}")

    # An unsigned view means `1 << bit_index` (up to bit 31) is always
    # representable directly, unlike a signed int32 view where it would
    # overflow the sign bit.
    view = tensor.view(torch.uint32)
    mask = 1 << bit_index
    match stuck_at:
        case StuckAtKind.One:
            corrupted = view | mask
        case StuckAtKind.Zero:
            corrupted = view & ~mask
    return corrupted.view(tensor.dtype)


def matmul_in_torch_layout(weights: Tensor, activations: Tensor) -> Tensor:
    """`weights @ activations`, stored so that its transpose is contiguous.

    So the result in torch's `([stack,] batch, out_features)` layout is
    contiguous without a copy.
    """
    # `(weights @ activations)^T = activations^T @ weights^T`. The operands
    # are usually transposed views of contiguous tensors, so this matmul runs
    # on them without copying.
    return (activations.mT @ weights.mT).mT


class _LiftPart(Protocol):
    """A part of a weight or accumulator lift."""

    @property
    def affected_output_row(self) -> int: ...

    @property
    def for_activations(self) -> tuple[int, int]: ...


@dataclass(frozen=True, slots=True)
class _RowGroup:
    """Consecutive parts of a lift that share a range of activation rows."""

    output_rows: list[int]
    """Never contains a row twice."""
    activation_start: int
    activation_end: int


def _group_rows(
    parts: Iterable[_LiftPart],
) -> list[_RowGroup]:
    """Group `parts` so that each group can be applied with one operation.

    Parts stay in their order. This keeps the order of the operations on each
    output row, so the rounding is the same as applying the parts one by one.
    """
    groups: list[_RowGroup] = []
    rows_in_last_group: set[int] = set()
    for part in parts:
        start, end = part.for_activations
        row = part.affected_output_row
        last = groups[-1] if groups else None
        # A row that is already in the group would be written twice by one
        # indexed assignment, which keeps only one of the writes.
        if (
            last is None
            or (last.activation_start, last.activation_end) != (start, end)
            or row in rows_in_last_group
        ):
            groups.append(_RowGroup([], start, end))
            rows_in_last_group = set()
        groups[-1].output_rows.append(row)
        rows_in_last_group.add(row)
    return groups


def _index(rows: Iterable[int], like: Tensor) -> Tensor:
    """`rows` as an index tensor on the device of `like`."""
    return torch.tensor(list(rows), dtype=torch.long, device=like.device)


def _weight_fault_matmul(
    weights: Tensor,
    activations: Tensor,
    fault: Fault.Register,
    lifted: LiftedFault.Weight,
) -> Tensor:
    weights = weights.clone()
    for group in _group_rows(lifted.parts):
        rows = _index(group.output_rows, weights)
        columns = slice(group.activation_start, group.activation_end)
        weights[..., rows, columns] = _corrupt(
            weights[..., rows, columns], fault.stuck_at, fault.bit_index
        )
    return matmul_in_torch_layout(weights, activations)


def _activation_fault_matmul(
    weights: Tensor,
    activations: Tensor,
    fault: Fault.Register,
    lifted: LiftedFault.Activation,
) -> Tensor:
    output = matmul_in_torch_layout(weights, activations)

    activations_faulty = activations.clone()
    activation_rows = _index(lifted.affected_activation_rows, activations)
    activations_faulty[..., activation_rows, :] = _corrupt(
        activations_faulty[..., activation_rows, :], fault.stuck_at, fault.bit_index
    )
    output_faulty = matmul_in_torch_layout(weights, activations_faulty)

    output_rows = _index(lifted.affected_output_rows, output)
    output[..., output_rows, :] = output_faulty[..., output_rows, :]

    return output


def _accumulator_fault_matmul(
    weights: Tensor,
    activations: Tensor,
    fault: Fault.Register,
    lifted: LiftedFault.Accumulator,
) -> Tensor:
    output = matmul_in_torch_layout(weights, activations)
    groups = _group_rows(lifted.parts)

    # A row of `output` is strided in torch's layout, and the groups update
    # rows many times over. So the groups update a contiguous copy of the
    # affected rows, which is written back once at the end.
    affected_rows = sorted({row for group in groups for row in group.output_rows})
    position_of_row = {row: position for position, row in enumerate(affected_rows)}
    affected_index = _index(affected_rows, output)
    affected = output[..., affected_index, :].contiguous()

    for group in groups:
        rows = _index(group.output_rows, output)
        positions = _index((position_of_row[row] for row in group.output_rows), output)
        start, end = group.activation_start, group.activation_end

        # An empty range means no products had reached the faulty PE yet
        # (a fault above the pass's used band): the partial sum is zero, a
        # "faulty zero" rather than a separate case. See
        # `docs/fault-lifting.md`'s accumulator section.
        if start == end:
            partial_sums = torch.zeros_like(affected[..., positions, :])
        else:
            partial_sum_weights = weights[..., rows, start:end]
            partial_sum_activations = activations[..., start:end, :]
            partial_sums = partial_sum_weights @ partial_sum_activations

        corrupted = _corrupt(partial_sums, fault.stuck_at, fault.bit_index)
        affected[..., positions, :] = (
            affected[..., positions, :] - partial_sums + corrupted
        )

    output[..., affected_index, :] = affected
    return output


def apply_lifted_register_fault(
    lifted: LiftedFault, fault: Fault, weights: Tensor, activations: Tensor
) -> Tensor:
    """Apply a lifted register fault to a matmul.

    The matmul may be a stack of matmuls, with one leading dimension on both
    operands. The fault applies to each of them.

    Produces a result equivalent to running the same fault through the
    literal systolic array (`SimulatedEvaluator`), without simulating the
    array.
    """
    if not isinstance(fault, Fault.Register):
        raise TypeError(f"unsupported fault kind: {type(fault)!r}")

    if isinstance(lifted, LiftedFault.Weight):
        return _weight_fault_matmul(weights, activations, fault, lifted)
    elif isinstance(lifted, LiftedFault.Activation):
        return _activation_fault_matmul(weights, activations, fault, lifted)
    elif isinstance(lifted, LiftedFault.Accumulator):
        return _accumulator_fault_matmul(weights, activations, fault, lifted)
    else:
        raise TypeError(f"unsupported lifted fault kind: {type(lifted)!r}")
