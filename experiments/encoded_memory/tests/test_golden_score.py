"""Tests for EncodedFaultInjection.golden_score."""

import json
from pathlib import Path
from typing import override

from encoded_memory import SavedResult, discard_bitmasks_in_file
from torch import Tensor

from faultforge import Fingerprint
from faultforge.experiment import FailureRate, SaveConfig, Stability
from faultforge.metric import Metric

from .conftest import _make_experiment

# `_` prefixed to not interpret it as a Test class.


class _ConstantMetric(Metric[int]):
    """Scores every run 1.0, so a `FailureRate` always has a valid golden score."""

    @override
    def evaluate_batch(
        self, batch_model_output: Tensor, batch_golden: Tensor, batch_targets: Tensor
    ) -> int:
        _ = batch_model_output, batch_golden, batch_targets
        return 1

    @override
    def requires_golden(self) -> bool:
        return False

    @override
    def accumulate(self, existing: int, new: int) -> int:
        return existing + new

    @override
    def score(self, result: int) -> float:
        _ = result
        return 1.0

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="constant")


def test_golden_score_matches_a_fault_free_run():
    experiment = _make_experiment(compare_bitwise=False, faults=0)
    experiment.run()

    assert experiment.golden_score() == experiment.scores()[0]


def test_golden_score_is_computed_once(monkeypatch):
    experiment = _make_experiment(compare_bitwise=False)
    calls = 0
    infer = experiment._infer

    def counting_infer(model):
        nonlocal calls
        calls += 1
        return infer(model)

    monkeypatch.setattr(experiment, "_infer", counting_infer)

    first = experiment.golden_score()
    second = experiment.golden_score()

    assert first == second
    assert calls == 1


def test_golden_score_round_trips_without_recomputing(monkeypatch):
    original = _make_experiment(compare_bitwise=False)
    golden = original.golden_score()
    serialized = original.serialize()

    loaded = _make_experiment(compare_bitwise=False)
    loaded.deserialize(serialized)

    def failing_infer(model):
        raise AssertionError("golden score should have been loaded, not recomputed")

    monkeypatch.setattr(loaded, "_infer", failing_infer)

    assert loaded.golden_score() == golden


def test_golden_score_is_not_saved_until_requested():
    experiment = _make_experiment(compare_bitwise=False)
    experiment.run()

    assert json.loads(experiment.serialize())["golden_score"] is None


def test_file_without_golden_score_loads_and_computes_it():
    # Files recorded before the golden score existed have no such field.
    original = _make_experiment(compare_bitwise=False)
    original.run()
    content = json.loads(original.serialize())
    del content["golden_score"]

    loaded = _make_experiment(compare_bitwise=False)
    loaded.deserialize(json.dumps(content))

    assert loaded.run_count() == 1
    assert isinstance(loaded.golden_score(), float)


def test_discard_bitmasks_in_file_keeps_golden_score(tmp_path: Path):
    experiment = _make_experiment(compare_bitwise=True)
    experiment.run()
    golden = experiment.golden_score()
    path = tmp_path / "result.json"
    experiment.save(path)

    discard_bitmasks_in_file(path)

    assert SavedResult.load(path).golden_score == golden


def test_resuming_with_failure_rate_saves_golden_score_without_new_runs(
    tmp_path: Path,
):
    path = tmp_path / "result.json"
    recorded = _make_experiment(
        compare_bitwise=False, reliability_metric=_ConstantMetric()
    )
    recorded.run()
    recorded.run()
    recorded.save(path)
    assert SavedResult.load(path).golden_score is None

    resumed = _make_experiment(
        compare_bitwise=False, reliability_metric=_ConstantMetric()
    )
    resumed.load_from(path)
    resumed.run_loop(
        estimator=FailureRate(2.0),
        stop_conditions=[Stability(min_samples=0, max_absolute_margin=100.0)],
        save_config=SaveConfig(path=path, interval_seconds=None),
    )

    saved = SavedResult.load(path)
    assert len(saved.scores()) == 2
    assert saved.golden_score == 1.0
