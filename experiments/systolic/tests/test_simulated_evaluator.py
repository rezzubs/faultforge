"""Tests for `SimulatedEvaluator`, and that `LiftedEvaluator` agrees with it."""

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
    SimulatedEvaluator,
    StuckAtKind,
)

_SIGN_BIT = 31


@st.composite
def _matmuls(draw: st.DrawFn) -> ArrayMatmul:
    """Single or stacked matmuls, often larger than the arrays below."""
    dimension = st.integers(min_value=1, max_value=8)
    out_features, in_features, batch = (draw(dimension) for _ in range(3))
    stack = draw(st.one_of(st.just(()), st.tuples(st.integers(1, 3))))
    generator = torch.Generator().manual_seed(draw(st.integers(0, 1000)))

    # Small non-negative integers. Their sums are exact, and a corrupted bit
    # other than the sign can't cause cancellation between large values, so the
    # result barely depends on the summation order.
    def operand(*shape: int) -> torch.Tensor:
        return torch.randint(0, 5, shape, generator=generator).float()

    return ArrayMatmul(
        operand(*stack, out_features, in_features),
        operand(*stack, in_features, batch),
    )


@st.composite
def _arrays(draw: st.DrawFn) -> Array:
    size = st.integers(min_value=1, max_value=5)
    return Array(draw(size), draw(size))


@st.composite
def _arrays_with_faults(draw: st.DrawFn) -> tuple[Array, Fault]:
    array = draw(_arrays())
    config = ArrayConfig(array.nrows, array.ncols, dtype_bits=32)
    faults = RegisterFaults()
    fault_id = draw(st.integers(min_value=0, max_value=faults.radix(config) - 1))
    return array, faults.fault_from_id(fault_id, config)


@given(array=_arrays(), matmul=_matmuls())
def test_without_a_fault_matches_torch(array: Array, matmul: ArrayMatmul) -> None:
    output = SimulatedEvaluator(array).evaluate(matmul, None)
    assert torch.equal(output, matmul.weights @ matmul.activations)


@given(array_and_fault=_arrays_with_faults(), matmul=_matmuls())
def test_lifted_agrees_with_simulated(
    array_and_fault: tuple[Array, Fault], matmul: ArrayMatmul
) -> None:
    array, fault = array_and_fault

    simulated = SimulatedEvaluator(array).evaluate(matmul, fault)
    lifted = LiftedEvaluator(array).evaluate(matmul, fault)

    torch.testing.assert_close(lifted, simulated, rtol=1e-5, atol=1e-3, equal_nan=True)


@given(
    array_and_fault=_arrays_with_faults(),
    matmul=_matmuls(),
    stuck_at=st.sampled_from([StuckAtKind.Zero, StuckAtKind.One]),
)
def test_lifted_agrees_exactly_for_sign_faults(
    array_and_fault: tuple[Array, Fault], matmul: ArrayMatmul, stuck_at: StuckAtKind
) -> None:
    """A sign fault keeps integers as integers, so nothing is rounded."""
    array, fault = array_and_fault
    assert isinstance(fault, Fault.Register)
    fault = Fault.Register(fault.target, fault.register, stuck_at, _SIGN_BIT)
    # Operands of both signs, so that both stuck-at values have an effect.
    matmul = ArrayMatmul(matmul.weights - 2, matmul.activations - 2)

    simulated = SimulatedEvaluator(array).evaluate(matmul, fault)
    lifted = LiftedEvaluator(array).evaluate(matmul, fault)

    assert torch.equal(lifted, simulated)


def test_faults_have_an_effect() -> None:
    """Guards against the agreement tests passing because nothing is faulted."""
    matmul = ArrayMatmul(torch.ones(2, 2, 2), torch.ones(2, 2, 2))
    evaluator = SimulatedEvaluator(Array(2, 2))
    for register in (
        PeRegisterKind.Activation,
        PeRegisterKind.Weight,
        PeRegisterKind.Accumulator,
    ):
        fault = Fault.Register(Index2(x=0, y=0), register, StuckAtKind.One, _SIGN_BIT)
        output = evaluator.evaluate(matmul, fault)
        # Every multiplication in the stack sees the fault.
        for single in output:
            assert not torch.equal(single, torch.full((2, 2), 2.0))


def test_rejects_fault_outside_the_array() -> None:
    fault = Fault.Register(
        target=Index2(x=3, y=0),
        register=PeRegisterKind.Weight,
        stuck_at=StuckAtKind.One,
        bit_index=0,
    )
    matmul = ArrayMatmul(torch.ones(4, 4), torch.ones(4, 4))
    with pytest.raises(ValueError, match="outside the 2x3 array"):
        SimulatedEvaluator(Array(2, 3)).evaluate(matmul, fault)
