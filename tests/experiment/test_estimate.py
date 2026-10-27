"""Tests for the Estimate and Estimator types (faultforge.experiment)."""

import math

import hypothesis.strategies as st
import pytest
from hypothesis import given

from faultforge.experiment import (
    AbsoluteThreshold,
    Estimate,
    FailureRate,
    Mean,
    RelativeThreshold,
    ScoreDisplay,
)
from faultforge.metric import ScoreDirection

finite_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
any_floats = st.floats(allow_nan=True, allow_infinity=True)
directions = st.sampled_from(ScoreDirection)


class _Display(ScoreDisplay):
    pass


@pytest.mark.parametrize(
    ("value", "lower", "upper"),
    [
        (1.0, 2.0, 3.0),
        (3.0, 1.0, 2.0),
        (math.nan, 0.0, 1.0),
        (0.5, 0.0, math.inf),
        (0.5, -math.inf, 1.0),
    ],
)
def test_estimate_rejects_invalid_fields(value: float, lower: float, upper: float):
    with pytest.raises(ValueError):
        _ = Estimate(value=value, lower=lower, upper=upper)


def test_estimate_margin_is_the_larger_distance():
    estimate = Estimate(value=10.0, lower=9.0, upper=13.0)

    assert estimate.margin() == 3.0
    assert estimate.relative_margin() == 30.0


def test_estimate_relative_margin_at_zero():
    assert Estimate(value=0.0, lower=0.0, upper=0.0).relative_margin() == 0.0
    assert Estimate(value=0.0, lower=0.0, upper=1.0).relative_margin() == math.inf


def test_mean_needs_two_scores():
    assert Mean().estimate([], None) is None
    assert Mean().estimate([1.0], None) is None


def test_mean_known_interval():
    estimate = Mean().estimate([1.0, 2.0, 3.0, 4.0, 5.0], None)

    assert estimate is not None
    assert estimate.value == 3.0
    # t(0.975, df=4) * stdev / sqrt(5) = 2.7764451 * 1.5811388 / 2.2360680
    assert estimate.margin() == pytest.approx(1.9632432, rel=1e-6)


@given(st.lists(finite_floats, min_size=2))
def test_mean_is_inside_its_interval(scores: list[float]):
    estimate = Mean().estimate(scores, None)

    assert estimate is not None
    assert estimate.lower <= estimate.value <= estimate.upper


@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
def test_mean_rejects_non_finite_scores(non_finite: float):
    # NOTE: also with a single score, so the error isn't delayed until there
    # are enough scores for an estimate.
    with pytest.raises(ValueError, match="FailureRate"):
        _ = Mean().estimate([non_finite], None)
    with pytest.raises(ValueError, match="FailureRate"):
        _ = Mean().estimate([1.0, non_finite, 2.0], None)


def test_failure_rate_needs_one_score():
    failure_rate = FailureRate(AbsoluteThreshold(1.0), ScoreDirection.LowerIsBetter)

    assert failure_rate.estimate([], None) is None
    assert failure_rate.estimate([0.0], None) is not None


def test_failure_rate_known_interval():
    failure_rate = FailureRate(AbsoluteThreshold(1.0), ScoreDirection.LowerIsBetter)

    estimate = failure_rate.estimate([0.0] * 10, None)

    assert estimate is not None
    assert estimate.value == 0.0
    assert estimate.lower == 0.0
    assert estimate.upper == pytest.approx(27.753280, rel=1e-6)


@given(st.lists(any_floats, min_size=1), finite_floats, directions)
def test_failure_rate_counts_failures(
    scores: list[float], threshold: float, direction: ScoreDirection
):
    estimate = FailureRate(AbsoluteThreshold(threshold), direction).estimate(
        scores, None
    )

    def is_failure(score: float) -> bool:
        if not math.isfinite(score):
            return True
        if direction == ScoreDirection.LowerIsBetter:
            return score > threshold
        return score < threshold

    expected = sum(1 for score in scores if is_failure(score)) / len(scores) * 100
    assert estimate is not None
    assert estimate.value == pytest.approx(expected)
    assert 0 <= estimate.lower <= estimate.value <= estimate.upper <= 100


@pytest.mark.parametrize("direction", list(ScoreDirection))
@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
def test_failure_rate_non_finite_scores_always_fail(
    direction: ScoreDirection, non_finite: float
):
    failure_rate = FailureRate(AbsoluteThreshold(1.0), direction)

    estimate = failure_rate.estimate([non_finite], None)

    assert estimate is not None
    assert estimate.value == 100.0


@pytest.mark.parametrize("direction", list(ScoreDirection))
def test_failure_rate_threshold_itself_isnt_a_failure(direction: ScoreDirection):
    failure_rate = FailureRate(AbsoluteThreshold(1.0), direction)

    estimate = failure_rate.estimate([1.0], None)

    assert estimate is not None
    assert estimate.value == 0.0


@pytest.mark.parametrize(
    ("direction", "scores", "expected"),
    [
        # golden 10, factor 2: fail above 20
        (ScoreDirection.LowerIsBetter, [19.0, 20.0, 21.0, 40.0], 50.0),
        # golden 10, factor 0.5: fail below 5
        (ScoreDirection.HigherIsBetter, [4.0, 5.0, 6.0, 10.0], 25.0),
    ],
)
def test_failure_rate_relative_threshold(
    direction: ScoreDirection, scores: list[float], expected: float
):
    factor = 2.0 if direction == ScoreDirection.LowerIsBetter else 0.5
    failure_rate = FailureRate(RelativeThreshold(factor), direction)

    estimate = failure_rate.estimate(scores, 10.0)

    assert estimate is not None
    assert estimate.value == expected


@pytest.mark.parametrize("golden_score", [None, 0.0, -1.0, math.inf, math.nan])
def test_failure_rate_relative_threshold_rejects_unusable_golden(
    golden_score: float | None,
):
    failure_rate = FailureRate(RelativeThreshold(2.0), ScoreDirection.LowerIsBetter)

    # NOTE: even without scores, so the problem shows up before any runs.
    with pytest.raises(ValueError):
        _ = failure_rate.estimate([], golden_score)


def test_failure_rate_requires_golden_only_when_relative():
    relative = FailureRate(RelativeThreshold(2.0), ScoreDirection.LowerIsBetter)
    absolute = FailureRate(AbsoluteThreshold(2.0), ScoreDirection.LowerIsBetter)

    assert relative.requires_golden()
    assert not absolute.requires_golden()


@pytest.mark.parametrize("factor", [0.0, -1.0, math.inf, math.nan])
def test_relative_threshold_rejects_invalid_factor(factor: float):
    with pytest.raises(ValueError):
        _ = RelativeThreshold(factor)


@pytest.mark.parametrize("score", [math.inf, -math.inf, math.nan])
def test_absolute_threshold_rejects_non_finite_score(score: float):
    with pytest.raises(ValueError):
        _ = AbsoluteThreshold(score)


def test_failure_rate_name():
    assert (
        FailureRate(RelativeThreshold(2.0), ScoreDirection.LowerIsBetter).name()
        == "failure rate (>2x golden)"
    )
    assert (
        FailureRate(AbsoluteThreshold(50.0), ScoreDirection.HigherIsBetter).name()
        == "failure rate (<50)"
    )
