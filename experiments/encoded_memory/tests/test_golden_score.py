"""Tests for EncodedFaultInjection.golden_score and its persistence."""

import json
import math

import pytest
from encoded_memory import (
    SavedResult,
    discard_bitmasks_in_file,
)

from faultforge.experiment import (
    FailureRate,
    MaxRuns,
    RelativeThreshold,
    SaveConfig,
    Stability,
)
from faultforge.metric import ScoreDirection

from .conftest import (
    _FixedScore,
    _make_experiment,
)


def test_golden_score_is_computed_once():
    metric = _FixedScore(3.0)
    experiment = _make_experiment(compare_bitwise=False, reliability_metric=metric)

    assert experiment.golden_score() == 3.0
    assert experiment.golden_score() == 3.0
    assert metric.score_calls == 1


def test_golden_score_isnt_saved_unless_computed():
    experiment = _make_experiment(compare_bitwise=False)
    experiment.run()

    assert json.loads(experiment.serialize())["golden_score"] is None


@pytest.mark.parametrize("golden_score", [3.0, math.inf, -math.inf])
def test_golden_score_is_loaded_instead_of_recomputed(golden_score: float):
    experiment = _make_experiment(
        compare_bitwise=False, reliability_metric=_FixedScore(golden_score)
    )
    _ = experiment.golden_score()
    serialized = experiment.serialize()

    metric = _FixedScore(golden_score)
    reloaded = _make_experiment(compare_bitwise=False, reliability_metric=metric)
    reloaded.deserialize(serialized)

    assert reloaded.golden_score() == golden_score
    assert metric.score_calls == 0


def test_golden_score_nan_round_trips():
    experiment = _make_experiment(
        compare_bitwise=False, reliability_metric=_FixedScore(math.nan)
    )
    _ = experiment.golden_score()

    loaded = SavedResult.model_validate_json(experiment.serialize())

    assert loaded.golden_score is not None
    assert math.isnan(loaded.golden_score)


def test_loading_a_file_without_golden_score_keeps_a_computed_one():
    without_golden = _make_experiment(
        compare_bitwise=False, reliability_metric=_FixedScore(3.0)
    ).serialize()

    metric = _FixedScore(3.0)
    experiment = _make_experiment(compare_bitwise=False, reliability_metric=metric)
    _ = experiment.golden_score()
    experiment.deserialize(without_golden)

    assert experiment.golden_score() == 3.0
    assert metric.score_calls == 1


def test_discard_bitmasks_in_file_keeps_golden_score(tmp_path):
    path = tmp_path / "result.json"
    experiment = _make_experiment(
        compare_bitwise=True, reliability_metric=_FixedScore(3.0)
    )
    _ = experiment.golden_score()
    experiment.save(path)

    discard_bitmasks_in_file(path)

    assert SavedResult.load(path).golden_score == 3.0


def test_run_loop_with_relative_failure_rate_saves_golden_score(tmp_path):
    path = tmp_path / "result.json"
    experiment = _make_experiment(
        compare_bitwise=False, reliability_metric=_FixedScore(3.0)
    )

    experiment.run_loop(
        stop_conditions=[
            Stability(
                estimator=FailureRate(
                    RelativeThreshold(2.0), ScoreDirection.LowerIsBetter
                ),
                max_absolute_margin=0.0,
            ),
            MaxRuns(3),
        ],
        save_config=SaveConfig(path=path, interval_seconds=None),
    )

    loaded = SavedResult.load(path)
    assert loaded.scores() == [3.0, 3.0, 3.0]
    assert loaded.golden_score == 3.0


def test_run_loop_rejects_non_finite_golden_score_before_any_runs():
    experiment = _make_experiment(
        compare_bitwise=False, reliability_metric=_FixedScore(math.inf)
    )
    stability = Stability(
        estimator=FailureRate(RelativeThreshold(2.0), ScoreDirection.LowerIsBetter),
        max_absolute_margin=1.0,
    )

    with pytest.raises(ValueError, match="golden"):
        experiment.run_loop(stop_conditions=[stability])
    assert experiment.run_count() == 0
