"""Tests for `LiftedEvaluator`."""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from systolic import (
    Array,
    ArrayConfig,
    ArrayMatmul,
    Fault,
    Index2,
    LiftedEvaluator,
    PeRegisterKind,
    RegisterFaults,
    StuckAtKind,
)

_ARRAY = Array(2, 3)
_ARRAY_CONFIG = ArrayConfig(nrows=2, ncols=3, dtype_bits=32)
_FAULTS = st.integers(
    min_value=0, max_value=RegisterFaults().radix(_ARRAY_CONFIG) - 1
).map(lambda fault_id: RegisterFaults().fault_from_id(fault_id, _ARRAY_CONFIG))


@st.composite
def _stacked_matmuls(draw: st.DrawFn) -> ArrayMatmul:
    dimension = st.integers(min_value=1, max_value=6)
    stack, out_features, in_features, batch = (draw(dimension) for _ in range(4))
    generator = torch.Generator().manual_seed(draw(st.integers(0, 1000)))

    # Small non-negative integers. Their sums are exact, and a corrupted bit
    # other than the sign can't cause cancellation between large values, so the
    # result barely depends on the summation order.
    def operand(*shape: int) -> torch.Tensor:
        return torch.randint(0, 5, shape, generator=generator).float()

    return ArrayMatmul(
        operand(stack, out_features, in_features),
        operand(stack, in_features, batch),
    )


@given(matmul=_stacked_matmuls())
def test_without_a_fault_matches_torch(matmul: ArrayMatmul) -> None:
    output = LiftedEvaluator(_ARRAY).evaluate(matmul, None)
    assert torch.equal(output, matmul.weights @ matmul.activations)


@given(matmul=_stacked_matmuls(), fault=_FAULTS)
def test_stack_matches_individual_matmuls(matmul: ArrayMatmul, fault: Fault) -> None:
    evaluator = LiftedEvaluator(_ARRAY)

    stacked = evaluator.evaluate(matmul, fault)
    individual = torch.stack(
        [evaluator.evaluate(single, fault) for single in matmul.unstack()]
    )

    torch.testing.assert_close(
        stacked, individual, rtol=1e-5, atol=1e-3, equal_nan=True
    )


@given(matmul=_stacked_matmuls(), fault=_FAULTS)
def test_operands_are_unchanged(matmul: ArrayMatmul, fault: Fault) -> None:
    weights = matmul.weights.clone()
    activations = matmul.activations.clone()

    LiftedEvaluator(_ARRAY).evaluate(matmul, fault)

    assert torch.equal(matmul.weights, weights)
    assert torch.equal(matmul.activations, activations)


def test_rejects_fault_outside_the_array() -> None:
    fault = Fault.Register(
        target=Index2(x=3, y=0),
        register=PeRegisterKind.Weight,
        stuck_at=StuckAtKind.One,
        bit_index=0,
    )
    matmul = ArrayMatmul(torch.ones(4, 4), torch.ones(4, 4))
    with pytest.raises(ValueError, match="outside the 2x3 array"):
        LiftedEvaluator(_ARRAY).evaluate(matmul, fault)
