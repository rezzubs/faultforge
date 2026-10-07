"""Tests for Experiment.run_loop."""

from pathlib import Path

import pytest

from faultforge.experiment import (
    AdditionalRuns,
    FailureRate,
    MaxRuns,
    SaveConfig,
    Stability,
)

from .conftest import make


def test_run_loop_stops_at_additional_runs():
    exp = make()
    exp.run_loop(stop_conditions=[AdditionalRuns(5)])
    assert exp.run_count() == 5


def test_run_loop_stops_when_stable():
    # Identical values -> margin of error = 0, below any positive threshold.
    # The experiment already has min_samples+1 results so stability is
    # checked immediately.
    min_samples = 5
    exp = make([1.0] * (min_samples + 1))
    initial = exp.run_count()
    exp.run_loop(
        stop_conditions=[Stability(min_samples=min_samples, max_relative_margin=0.01)]
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
            Stability(min_samples=10, max_relative_margin=999.0),
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
            Stability(min_samples=2, max_relative_margin=0.0),
            AdditionalRuns(20),
        ]
    )
    assert exp.run_count() == 20


def test_run_loop_stability_check_with_single_sample_does_not_crash():
    # min_samples=1 means the stability check runs while there's no
    # estimate yet (fewer than 2 results), which must just continue.
    exp = make()
    exp.run_loop(
        stop_conditions=[
            Stability(min_samples=1, max_relative_margin=0.01),
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


def test_stability_requires_a_limit():
    with pytest.raises(ValueError):
        _ = Stability(min_samples=0)


def test_run_loop_stops_on_absolute_margin():
    # Scores 1, 2, 3, ... have a growing absolute margin of error but a
    # loose enough limit is reached right away.
    exp = make()
    exp.run_loop(
        stop_conditions=[
            Stability(min_samples=2, max_absolute_margin=100.0),
            AdditionalRuns(20),
        ]
    )
    assert exp.run_count() == 2


def test_run_loop_stops_on_either_margin():
    # The relative limit can never be reached (0%), the absolute one can.
    exp = make()
    exp.run_loop(
        stop_conditions=[
            Stability(
                min_samples=2, max_absolute_margin=100.0, max_relative_margin=0.0
            ),
            AdditionalRuns(20),
        ]
    )
    assert exp.run_count() == 2


def test_run_loop_negative_mean_does_not_stop_early():
    # A negative mean used to produce a negative relative margin, which is
    # always below the limit.
    exp = make([-1.0, -3.0])
    exp.run_loop(
        stop_conditions=[
            Stability(min_samples=0, max_relative_margin=1.0),
            AdditionalRuns(3),
        ]
    )
    assert exp.run_count() == 5


def test_run_loop_mean_does_not_request_golden():
    exp = make([1.0, 2.0], golden=1.0)
    exp.run_loop(stop_conditions=[AdditionalRuns(2)])
    assert exp.golden_requests == 0


def test_run_loop_failure_rate_uses_golden():
    # Scores 1, 2, 3, ...: with a golden score of 1 and a threshold of 1.5
    # every run but the first fails.
    exp = make(golden=1.0)
    exp.run_loop(
        estimator=FailureRate(1.5),
        stop_conditions=[AdditionalRuns(4)],
    )
    assert exp.golden_requests > 0
    estimate = exp.estimate(FailureRate(1.5))
    assert estimate is not None
    assert estimate.value == pytest.approx(75.0)


def test_run_loop_failure_rate_without_golden_raises():
    exp = make([1.0, 2.0])
    with pytest.raises(ValueError, match="golden"):
        exp.run_loop(estimator=FailureRate(2.0), stop_conditions=[AdditionalRuns(1)])


def test_run_loop_saves_without_new_runs_when_golden_is_required(tmp_path: Path):
    # Resuming a finished experiment with an estimator that needs the golden
    # score must still persist the golden score, which may have just been
    # computed.
    path = tmp_path / "result.json"
    exp = make([1.0, 2.0], golden=1.0)
    exp.run_loop(
        estimator=FailureRate(2.0),
        stop_conditions=[AdditionalRuns(0)],
        save_config=SaveConfig(path=path, interval_seconds=None),
    )
    assert exp.run_count() == 2
    assert path.exists()


def test_run_loop_does_not_save_without_new_runs_for_mean(tmp_path: Path):
    path = tmp_path / "result.json"
    exp = make([1.0, 2.0])
    exp.run_loop(
        stop_conditions=[AdditionalRuns(0)],
        save_config=SaveConfig(path=path, interval_seconds=None),
    )
    assert not path.exists()
