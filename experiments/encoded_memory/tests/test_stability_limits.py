"""Tests for `StabilityLimits`."""

import math

import pytest
import typer
from encoded_memory.commands import (
    EstimateChoice,
    _resolve_stability_limits,
)
from encoded_memory.stability import StabilityLimits

from faultforge.experiment import (
    AbsoluteThreshold,
    FailureRate,
    Mean,
    Stability,
)
from faultforge.metric import (
    Accuracy,
    Perplexity,
    ScoreDirection,
)


def test_requires_a_margin():
    with pytest.raises(ValueError, match="max_absolute_margin"):
        _ = StabilityLimits(min_runs=5)


@pytest.mark.parametrize("margin", [math.inf, math.nan, -1.0])
def test_rejects_invalid_margins(margin: float):
    with pytest.raises(ValueError, match="finite"):
        _ = StabilityLimits(max_relative_margin=margin)


def test_cli_reports_invalid_margins():
    with pytest.raises(typer.BadParameter, match="finite"):
        _ = _resolve_stability_limits(
            estimate=EstimateChoice.Mean,
            relative_failure_threshold=None,
            absolute_failure_threshold=None,
            max_relative_margin=math.inf,
            max_absolute_margin=None,
            min_runs=None,
        )


def test_mean_without_failure_threshold():
    limits = StabilityLimits(min_runs=3, max_absolute_margin=0.5)

    assert limits.stability(Accuracy()) == Stability(
        estimator=Mean(), min_samples=3, max_absolute_margin=0.5
    )


def test_failure_rate_direction_follows_the_metric():
    limits = StabilityLimits(
        failure_threshold=AbsoluteThreshold(5.0), max_relative_margin=1.0
    )

    assert limits.stability(Accuracy()).estimator == FailureRate(
        AbsoluteThreshold(5.0), ScoreDirection.HigherIsBetter
    )
    assert limits.stability(Perplexity()).estimator == FailureRate(
        AbsoluteThreshold(5.0), ScoreDirection.LowerIsBetter
    )
