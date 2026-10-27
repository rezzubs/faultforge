"""Tests for Experiment.format_status."""

from faultforge.experiment import (
    AdditionalRuns,
    Mean,
    Stability,
)

from .conftest import make


def test_format_status_no_results_is_none():
    assert make().format_status() is None


def test_format_status_without_conditions_shows_only_the_score():
    exp = make([1.0, 2.0])
    status = exp.format_status()
    assert status == "[Run 2]:   2.00"


def test_format_status_ignores_conditions_without_status():
    exp = make([1.0, 2.0])
    status = exp.format_status([AdditionalRuns(5)])
    assert status == "[Run 2]:   2.00"


def test_format_status_shows_stability():
    exp = make([1.0, 2.0])
    stability = Stability(estimator=Mean(), max_relative_margin=1.0)

    status = exp.format_status([stability])

    expected = stability.status(exp, exp.display().score_display())
    assert expected is not None
    assert status == f"[Run 2]:   2.00 | {expected}"


def test_format_status_skips_stability_without_estimate():
    exp = make([1.0])
    status = exp.format_status([Stability(estimator=Mean(), max_relative_margin=1.0)])
    assert status == "[Run 1]:   1.00"
