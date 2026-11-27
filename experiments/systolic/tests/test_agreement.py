import math

import torch
from systolic.agreement import (
    Agreement,
    Evaluation,
    _compare,
    compare_matmul,
    summarize,
)
from systolic.array import (
    Array,
    ArrayMatmul,
)

_UNUSED_OUTPUT = torch.zeros(1)
"""`Evaluation.output` isn't read by `summarize`, so a placeholder is fine."""


def _agreements(
    abs_error: float, relative_error: float, flip: float
) -> list[Agreement]:
    return [
        Agreement(
            baseline="simulated",
            other="lifted",
            max_abs_error=abs_error,
            max_relative_error=relative_error,
            top1_flip_fraction=flip,
        )
    ]


def test_summarize_aggregates_mean_and_max() -> None:
    samples = [
        _agreements(abs_error=1.0, relative_error=0.1, flip=0.0),
        _agreements(abs_error=3.0, relative_error=0.3, flip=0.5),
    ]

    [summary] = summarize(samples)

    assert summary.baseline == "simulated"
    assert summary.other == "lifted"
    assert summary.samples == 2
    assert summary.mean_max_abs_error == 2.0
    assert summary.max_max_abs_error == 3.0
    assert summary.mean_max_relative_error == 0.2
    assert summary.max_max_relative_error == 0.3
    assert summary.mean_top1_flip_fraction == 0.25
    assert summary.max_top1_flip_fraction == 0.5


def test_compare_treats_matching_infinities_as_agreement() -> None:
    baseline = Evaluation(
        name="simulated", output=torch.tensor([[1.0], [float("inf")]])
    )
    other = Evaluation(name="lifted", output=torch.tensor([[1.0], [float("inf")]]))

    agreement = _compare(baseline, other)

    assert agreement.max_abs_error == 0.0
    assert agreement.max_relative_error == 0.0


def test_compare_treats_matching_nan_as_agreement() -> None:
    baseline = Evaluation(
        name="simulated", output=torch.tensor([[1.0], [float("nan")]])
    )
    other = Evaluation(name="lifted", output=torch.tensor([[1.0], [float("nan")]]))

    agreement = _compare(baseline, other)

    assert agreement.max_abs_error == 0.0
    assert agreement.max_relative_error == 0.0


def test_compare_still_reports_real_divergence_at_infinity() -> None:
    """One evaluator saturating while the other doesn't is a real disagreement,
    not something to zero out - it must stay visible (as a large or infinite
    value), not be swallowed the way matching infinities/NaNs are."""
    baseline = Evaluation(
        name="simulated", output=torch.tensor([[1.0], [float("inf")]])
    )
    other = Evaluation(name="lifted", output=torch.tensor([[1.0], [2.0]]))

    agreement = _compare(baseline, other)

    assert math.isinf(agreement.max_abs_error)
    assert math.isinf(agreement.max_relative_error)


def test_compare_reports_overflow_on_one_side_as_infinitely_far() -> None:
    baseline = Evaluation(
        name="simulated", output=torch.tensor([[1.0], [float("-inf")]])
    )
    other = Evaluation(name="lifted", output=torch.tensor([[1.0], [3e37]]))

    agreement = _compare(baseline, other)

    assert agreement.max_abs_error == math.inf
    assert agreement.max_relative_error == math.inf


def test_compare_reports_nan_on_one_side_as_infinitely_far() -> None:
    baseline = Evaluation(name="simulated", output=torch.tensor([[1.0], [2.0]]))
    other = Evaluation(name="lifted", output=torch.tensor([[1.0], [float("nan")]]))

    agreement = _compare(baseline, other)

    assert agreement.max_abs_error == math.inf
    assert agreement.max_relative_error == math.inf


def test_summarize_keeps_a_divergence_at_infinity() -> None:
    # The divergence comes after an agreeing sample. Python's `max` skips a
    # NaN in that position.
    baseline = Evaluation(
        name="simulated", output=torch.tensor([[1.0], [float("-inf")]])
    )
    diverged = Evaluation(name="lifted", output=torch.tensor([[1.0], [3e37]]))
    agreed = Evaluation(name="lifted", output=torch.tensor([[1.0], [float("-inf")]]))

    [summary] = summarize(
        [[_compare(baseline, agreed)], [_compare(baseline, diverged)]]
    )

    assert summary.max_max_abs_error == math.inf
    assert summary.max_max_relative_error == math.inf


def test_compare_matmul_agrees_without_a_fault_on_a_stack() -> None:
    generator = torch.Generator().manual_seed(0)
    matmul = ArrayMatmul(
        weights=torch.randn(2, 5, 6, generator=generator),
        activations=torch.randn(2, 6, 3, generator=generator),
    )

    agreements = compare_matmul(Array(4, 4), matmul, None)

    assert [agreement.other for agreement in agreements] == ["torch", "lifted"]
    for agreement in agreements:
        assert agreement.max_abs_error < 1e-5
        assert agreement.top1_flip_fraction == 0.0
