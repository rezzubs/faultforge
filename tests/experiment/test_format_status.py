"""Tests for Experiment.format_status."""

from faultforge.experiment import AdditionalRuns, FailureRate, Stability

from .conftest import make


def test_format_status_no_results_is_none():
    assert make().format_status() is None


def test_format_status_shows_mean_once_two_scores():
    # The estimate shows up as soon as there's enough data, regardless of
    # whether any stop condition is configured at all.
    exp = make([1.0, 2.0])
    status = exp.format_status()
    assert status is not None
    assert "mean" in status
    assert "(95% CI)" in status


def test_format_status_omits_estimate_with_one_score():
    exp = make([1.0])
    status = exp.format_status()
    assert status is not None
    assert "(95% CI)" not in status


def test_format_status_omits_margin_without_stability():
    exp = make([1.0, 2.0])
    status = exp.format_status(stop_conditions=[AdditionalRuns(5)])
    assert status is not None
    assert "margin" not in status


def test_format_status_shows_margin_with_stability():
    exp = make([1.0, 2.0])
    status = exp.format_status(
        stop_conditions=[Stability(min_samples=0, max_relative_margin=1.0)]
    )
    assert status is not None
    assert "margin" in status


def test_format_status_uses_given_estimator():
    exp = make([1.0, 3.0], golden=1.0)
    status = exp.format_status(FailureRate(2.0))
    assert status is not None
    assert "failure rate (>2x golden) 50.00%" in status
