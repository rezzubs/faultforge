"""Tests for `ProfilingEvaluator`."""

import numpy as np
import pytest
import torch
from systolic import (
    Array,
    ArrayMatmul,
    Fault,
    Index2,
    PeRegisterKind,
    ProfilingEvaluator,
    StuckAtKind,
)
from systolic._rust import ProfilingArtifact


def _integers(*shape: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randint(-4, 5, shape, generator=generator).float()


def _evaluator() -> ProfilingEvaluator:
    return ProfilingEvaluator(Array(4, 4), capacity=1000, seed=0)


def _assert_artifacts_equal(a: ProfilingArtifact, b: ProfilingArtifact) -> None:
    for name in (
        "first_triples",
        "first_fill",
        "active_triples",
        "active_fill",
        "drain_partial_sums",
        "drain_fill",
    ):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


@pytest.mark.parametrize("stack", [(), (2,)])
def test_output_matches_torch(stack: tuple[int, ...]) -> None:
    matmul = ArrayMatmul(_integers(*stack, 5, 6), _integers(*stack, 6, 3))
    output = _evaluator().evaluate(matmul, None)
    assert torch.equal(output, matmul.weights @ matmul.activations)


def test_accumulates_across_calls() -> None:
    matmul = ArrayMatmul(_integers(3, 4), _integers(4, 2))
    evaluator = _evaluator()

    evaluator.evaluate(matmul, None)
    fill_after_one = evaluator.artifact().active_fill
    evaluator.evaluate(matmul, None)
    fill_after_two = evaluator.artifact().active_fill

    assert fill_after_one.sum() > 0
    np.testing.assert_array_equal(fill_after_two, 2 * fill_after_one)


def test_stack_records_the_same_as_individual_matmuls() -> None:
    matmul = ArrayMatmul(_integers(3, 5, 6), _integers(3, 6, 2))

    stacked = _evaluator()
    stacked.evaluate(matmul, None)
    individual = _evaluator()
    for single in matmul.unstack():
        individual.evaluate(single, None)

    _assert_artifacts_equal(stacked.artifact(), individual.artifact())


def test_rejects_faults() -> None:
    fault = Fault.Register(
        target=Index2(x=0, y=0),
        register=PeRegisterKind.Weight,
        stuck_at=StuckAtKind.One,
        bit_index=0,
    )
    matmul = ArrayMatmul(_integers(3, 4), _integers(4, 2))
    with pytest.raises(ValueError, match="with a fault"):
        _evaluator().evaluate(matmul, fault)
