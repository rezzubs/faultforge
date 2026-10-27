"""Tests for the Stability stop condition."""

import math

import pytest

from faultforge.experiment import (
    AbsoluteThreshold,
    FailureRate,
    Mean,
    RelativeThreshold,
    Stability,
)
from faultforge.metric import ScoreDirection

from .conftest import make


def test_stability_requires_a_limit():
    with pytest.raises(ValueError):
        _ = Stability(estimator=Mean())


@pytest.mark.parametrize("limit", [-1.0, math.inf, math.nan])
def test_stability_rejects_invalid_limits(limit: float):
    with pytest.raises(ValueError):
        _ = Stability(estimator=Mean(), max_absolute_margin=limit)
    with pytest.raises(ValueError):
        _ = Stability(estimator=Mean(), max_relative_margin=limit)


def test_stability_doesnt_stop_without_an_estimate():
    stability = Stability(estimator=Mean(), max_absolute_margin=1e9)
    assert stability(make([1.0])) is None


def test_stability_reports_the_reached_limit():
    stability = Stability(estimator=Mean(), max_absolute_margin=1.0)

    reason = stability(make([1.0, 1.0]))

    assert reason is not None
    assert "absolute margin" in reason


def test_stability_relative_limit_unreachable_at_zero():
    # No failures -> value 0 with a nonzero upper bound -> infinite relative
    # margin.
    stability = Stability(
        estimator=FailureRate(AbsoluteThreshold(1.0), ScoreDirection.LowerIsBetter),
        max_relative_margin=100.0,
    )
    assert stability(make([0.0] * 1000)) is None


def test_stability_requests_golden_only_when_required():
    absolute = make([1.0, 2.0], golden_score=1.0)
    _ = Stability(
        estimator=FailureRate(AbsoluteThreshold(1.0), ScoreDirection.LowerIsBetter),
        max_absolute_margin=1.0,
    )(absolute)
    assert absolute.golden_requests == 0

    relative = make([1.0, 2.0], golden_score=1.0)
    _ = Stability(
        estimator=FailureRate(RelativeThreshold(1.5), ScoreDirection.LowerIsBetter),
        max_absolute_margin=1.0,
    )(relative)
    assert relative.golden_requests == 1


def test_stability_status_shows_only_configured_limits():
    exp = make([1.0, 2.0])
    score_display = exp.display().score_display()

    absolute = Stability(estimator=Mean(), max_absolute_margin=0.5)
    relative = Stability(estimator=Mean(), max_relative_margin=1.0)
    both = Stability(estimator=Mean(), max_absolute_margin=0.5, max_relative_margin=1.0)

    absolute_status = absolute.status(exp, score_display)
    relative_status = relative.status(exp, score_display)
    both_status = both.status(exp, score_display)

    assert absolute_status is not None
    assert "(limit 0.50)" in absolute_status
    assert "of value" not in absolute_status

    assert relative_status is not None
    assert "of value (limit 1.00%)" in relative_status
    assert "(limit 0.50)" not in relative_status

    assert both_status is not None
    assert "(limit 0.50)" in both_status
    assert "of value (limit 1.00%)" in both_status


def test_stability_status_shows_the_estimate():
    exp = make([1.0, 2.0])
    stability = Stability(estimator=Mean(), max_absolute_margin=0.5)

    status = stability.status(exp, exp.display().score_display())

    assert status is not None
    assert status.startswith("mean 1.50 [")
    assert "(95% CI)" in status


def test_stability_status_without_estimate():
    exp = make([1.0])
    stability = Stability(estimator=Mean(), max_absolute_margin=0.5)

    assert stability.status(exp, exp.display().score_display()) is None
