"""Tests for the `Estimator`s and `Estimate`."""

import math

import pytest
import scipy.stats
from hypothesis import given
from hypothesis import strategies as st

from faultforge.experiment import Estimate, FailureRate, Mean

# Scores as a run would report them, including the non-finite ones a broken
# run can produce.
_scores = st.lists(
    st.one_of(
        st.floats(min_value=0, max_value=1e6),
        st.just(math.inf),
        st.just(math.nan),
    ),
    min_size=1,
    max_size=200,
)


def test_mean_needs_two_scores():
    assert Mean().estimate([], None) is None
    assert Mean().estimate([1.0], None) is None


def test_mean_identical_values_have_no_margin():
    estimate = Mean().estimate([3.0, 3.0], None)
    assert estimate == Estimate(value=3.0, lower=3.0, upper=3.0)


def test_mean_matches_t_interval():
    scores = [1.0, 2.0, 4.0, 8.0]
    estimate = Mean().estimate(scores, None)
    assert estimate is not None

    low, high = scipy.stats.t.interval(
        0.95, df=len(scores) - 1, loc=3.75, scale=scipy.stats.sem(scores)
    )
    assert estimate.value == pytest.approx(3.75)
    assert estimate.lower == pytest.approx(low)
    assert estimate.upper == pytest.approx(high)


def test_failure_rate_counts_scores_above_threshold():
    # Golden 10, threshold 2 -> anything above 20 fails. Exactly 20 doesn't.
    estimate = FailureRate(2.0).estimate([10.0, 20.0, 21.0, 100.0], 10.0)
    assert estimate is not None
    assert estimate.value == pytest.approx(50.0)


def test_failure_rate_non_finite_scores_fail():
    estimate = FailureRate(2.0).estimate([1.0, math.inf, math.nan, 1.0], 1.0)
    assert estimate is not None
    assert estimate.value == pytest.approx(50.0)


def test_failure_rate_matches_wilson_interval():
    scores = [1.0] * 7 + [100.0] * 3
    estimate = FailureRate(2.0).estimate(scores, 1.0)
    assert estimate is not None

    interval = scipy.stats.binomtest(3, 10).proportion_ci(method="wilson")
    assert estimate.lower == pytest.approx(interval.low * 100)
    assert estimate.upper == pytest.approx(interval.high * 100)


def test_failure_rate_without_failures_still_has_an_upper_bound():
    # 0 failures doesn't mean a 0% failure rate, just a small one.
    estimate = FailureRate(2.0).estimate([1.0] * 100, 1.0)
    assert estimate is not None
    assert estimate.value == 0.0
    assert estimate.lower == 0.0
    assert estimate.upper == pytest.approx(3.7, abs=0.1)
    assert estimate.relative_margin() == math.inf


def test_failure_rate_needs_scores():
    assert FailureRate(2.0).estimate([], 1.0) is None


@pytest.mark.parametrize("golden", [None, 0.0, -1.0, math.inf, math.nan])
def test_failure_rate_rejects_invalid_golden(golden: float | None):
    with pytest.raises(ValueError, match="golden"):
        _ = FailureRate(2.0).estimate([1.0], golden)


@pytest.mark.parametrize("threshold", [0.0, -1.0, math.nan])
def test_failure_rate_rejects_invalid_threshold(threshold: float):
    with pytest.raises(ValueError, match="threshold"):
        _ = FailureRate(threshold)


def test_failure_rate_requires_golden():
    assert FailureRate(2.0).requires_golden()
    assert not Mean().requires_golden()


def test_margin_uses_the_larger_side():
    estimate = Estimate(value=1.0, lower=0.5, upper=3.0)
    assert estimate.margin() == 2.0


def test_relative_margin_uses_absolute_value():
    # A negative value must not produce a negative relative margin.
    estimate = Estimate(value=-2.0, lower=-3.0, upper=-1.0)
    assert estimate.relative_margin() == pytest.approx(50.0)


def test_relative_margin_at_zero():
    assert Estimate(value=0.0, lower=0.0, upper=0.0).relative_margin() == 0.0
    assert Estimate(value=0.0, lower=0.0, upper=1.0).relative_margin() == math.inf


@given(
    scores=_scores,
    golden=st.floats(min_value=1e-3, max_value=1e3),
    threshold=st.floats(min_value=1e-3, max_value=1e3),
)
def test_failure_rate_bounds_contain_value(
    scores: list[float], golden: float, threshold: float
):
    estimate = FailureRate(threshold).estimate(scores, golden)
    assert estimate is not None
    assert 0.0 <= estimate.lower <= estimate.value <= estimate.upper <= 100.0


@given(
    scores=_scores,
    golden=st.floats(min_value=1e-3, max_value=1e3),
    thresholds=st.lists(st.floats(min_value=1e-3, max_value=1e3), min_size=2),
)
def test_failure_rate_never_increases_with_threshold(
    scores: list[float], golden: float, thresholds: list[float]
):
    rates = []
    for threshold in sorted(thresholds):
        estimate = FailureRate(threshold).estimate(scores, golden)
        assert estimate is not None
        rates.append(estimate.value)
    assert rates == sorted(rates, reverse=True)
