"""Tests for Experiment.run_loop."""

import pytest

from faultforge.experiment import (
    AbsoluteThreshold,
    AdditionalRuns,
    FailureRate,
    MaxRuns,
    Mean,
    RelativeThreshold,
    Stability,
)
from faultforge.metric import ScoreDirection

from .conftest import make


def test_run_loop_stops_at_additional_runs():
    exp = make()
    exp.run_loop(stop_conditions=[AdditionalRuns(5)])
    assert exp.run_count() == 5


def test_run_loop_stops_when_stable():
    # Identical values -> margin of error = 0, below any positive limit.
    # The experiment already has min_samples+1 results so stability is
    # checked immediately.
    min_samples = 5
    exp = make([1.0] * (min_samples + 1))
    initial = exp.run_count()
    exp.run_loop(
        stop_conditions=[
            Stability(
                estimator=Mean(), min_samples=min_samples, max_relative_margin=0.01
            )
        ]
    )
    assert exp.run_count() == initial


def test_run_loop_does_not_stop_before_min_samples():
    # Even with a low margin of error, stability is skipped until min_samples
    # is reached; AdditionalRuns is what actually stops this run. It counts
    # runs from when it starts tracking, so 3 more on top of the 3 pre-loaded
    # results reaches a total of 6.
    exp = make([1.0] * 3)
    exp.run_loop(
        stop_conditions=[
            Stability(estimator=Mean(), min_samples=10, max_relative_margin=999.0),
            AdditionalRuns(3),
        ]
    )
    assert exp.run_count() == 6


def test_run_loop_continues_while_unstable():
    # Incrementing values → margin of error never reaches 0 → runs to the
    # run limit instead of stopping via stability.
    exp = make()
    exp.run_loop(
        stop_conditions=[
            Stability(estimator=Mean(), min_samples=2, max_relative_margin=0.0),
            AdditionalRuns(20),
        ]
    )
    assert exp.run_count() == 20


def test_run_loop_stability_check_with_single_sample_does_not_crash():
    # min_samples=1 means the stability check runs while the Mean estimate
    # is still None (fewer than 2 results), which used to raise
    # UnboundLocalError instead of just continuing.
    exp = make()
    exp.run_loop(
        stop_conditions=[
            Stability(estimator=Mean(), min_samples=1, max_relative_margin=0.01),
            AdditionalRuns(3),
        ]
    )
    assert exp.run_count() == 3


def test_run_loop_merges_intrinsic_and_caller_stop_conditions():
    # An experiment's own `stop_conditions()` override is checked alongside
    # whatever the caller passes to `run_loop` - whichever fires first wins.
    exp = make()
    exp.add_stop_condition(AdditionalRuns(4))
    exp.run_loop(stop_conditions=[AdditionalRuns(10)])
    assert exp.run_count() == 4


def test_run_loop_stops_at_max_runs():
    # Unlike AdditionalRuns, MaxRuns counts existing results toward the total.
    exp = make([1.0, 2.0])
    exp.run_loop(stop_conditions=[MaxRuns(5)])
    assert exp.run_count() == 5


def test_run_loop_max_runs_already_reached_does_nothing():
    exp = make([1.0] * 5)
    exp.run_loop(stop_conditions=[MaxRuns(5)])
    assert exp.run_count() == 5


def test_run_loop_additional_runs_and_max_runs_combined_max_wins():
    # MaxRuns(4) is reached before AdditionalRuns(10) would allow.
    exp = make([1.0, 2.0])
    exp.run_loop(stop_conditions=[AdditionalRuns(10), MaxRuns(4)])
    assert exp.run_count() == 4


def test_run_loop_stops_at_absolute_margin():
    # Incrementing values 1, 2, 3, ... -> the Mean margin is 2.0 or less from
    # 5 runs on.
    exp = make()
    exp.run_loop(
        stop_conditions=[
            Stability(estimator=Mean(), max_absolute_margin=2.0),
            AdditionalRuns(100),
        ]
    )
    assert exp.run_count() == 5


def test_run_loop_stops_at_first_reached_margin_limit():
    # The relative limit is unreachable, the absolute one stops the loop.
    exp = make()
    exp.run_loop(
        stop_conditions=[
            Stability(
                estimator=Mean(), max_absolute_margin=2.0, max_relative_margin=0.0
            ),
            AdditionalRuns(100),
        ]
    )
    assert exp.run_count() == 5


def test_run_loop_with_failure_rate():
    # Every run's score is above the threshold, so the failure rate is 100%.
    # The Wilson margin is at most 30 percentage points from 9 runs on.
    exp = make()
    exp.run_loop(
        stop_conditions=[
            Stability(
                estimator=FailureRate(
                    AbsoluteThreshold(0.0), ScoreDirection.LowerIsBetter
                ),
                max_absolute_margin=30.0,
            ),
            AdditionalRuns(100),
        ]
    )
    assert exp.run_count() == 9


def test_run_loop_rejects_missing_golden_before_any_runs():
    exp = make()
    stability = Stability(
        estimator=FailureRate(RelativeThreshold(2.0), ScoreDirection.LowerIsBetter),
        min_samples=10,
        max_absolute_margin=1.0,
    )
    with pytest.raises(ValueError, match="golden"):
        exp.run_loop(stop_conditions=[stability])
    assert exp.run_count() == 0


def test_run_loop_rejects_non_finite_golden_before_any_runs():
    exp = make(golden_score=float("inf"))
    stability = Stability(
        estimator=FailureRate(RelativeThreshold(2.0), ScoreDirection.LowerIsBetter),
        min_samples=10,
        max_absolute_margin=1.0,
    )
    with pytest.raises(ValueError, match="golden"):
        exp.run_loop(stop_conditions=[stability])
    assert exp.run_count() == 0
