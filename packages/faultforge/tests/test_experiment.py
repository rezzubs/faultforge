"""Tests for the public Experiment API (faultforge.experiment)."""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import override

import pytest
from faultforge import Fingerprint
from faultforge.experiment import (
    AdditionalRuns,
    Experiment,
    MaxRuns,
    SaveConfig,
    Stability,
    StopCondition,
)
from pydantic import BaseModel

# The following classes are `_` prefixed to not interpret them as Test classes.


def _fingerprint(name: str = "test") -> Fingerprint:
    return Fingerprint(kind="test", scalars={"name": name})


@dataclass(slots=True)
class _TestResult:
    value: float


class _SavedData(BaseModel):
    fingerprint: Fingerprint
    results: dict[int, float]


class _TestExperiment(Experiment):
    _fingerprint: Fingerprint
    _results: dict[int, _TestResult]
    _intrinsic_stop_conditions: list[StopCondition]

    def __init__(
        self,
        results: dict[int, _TestResult] | None = None,
        name: str = "test",
    ) -> None:
        self._fingerprint = _fingerprint(name)
        self._results = results or {}
        self._intrinsic_stop_conditions = []

    def add_stop_condition(self, condition: StopCondition) -> None:
        """Contribute an additional intrinsic condition, as if a subclass had
        overridden `stop_conditions` itself."""
        self._intrinsic_stop_conditions.append(condition)

    @override
    def stop_conditions(self) -> Sequence[StopCondition]:
        return self._intrinsic_stop_conditions

    @override
    def scores(self) -> Sequence[float]:
        return [result.value for result in self._results.values()]

    @override
    def run(self) -> None:
        key = len(self._results)
        self._results[key] = _TestResult(value=float(key + 1))

    @override
    def serialize(self) -> str:
        return _SavedData(
            fingerprint=self._fingerprint,
            results={key: result.value for key, result in self._results.items()},
        ).model_dump_json()

    @override
    def deserialize(self, content: str) -> None:
        loaded = _SavedData.model_validate_json(content)
        self._fingerprint.raise_if_differs(loaded.fingerprint)
        self._results = {
            key: _TestResult(value=value) for key, value in loaded.results.items()
        }


def make(values: list[float] | None = None, name: str = "test") -> _TestExperiment:
    """Create a test experiment with existing results"""
    results = {key: _TestResult(value=value) for key, value in enumerate(values or [])}
    return _TestExperiment(results=results, name=name)


# SECTION margin_of_error


def test_margin_of_error_no_results():
    assert make().margin_of_error() is None


def test_margin_of_error_one_result():
    assert make([1.0]).margin_of_error() is None


def test_margin_of_error_identical_values():
    assert make([3.0, 3.0]).margin_of_error() == 0.0


# SECTION format_status


def test_format_status_no_results_is_none():
    assert make().format_status() is None


def test_format_status_shows_mean_once_two_scores():
    # Mean/margin-of-error display shows up as soon as there's enough data,
    # regardless of whether any stop condition is configured at all.
    exp = make([1.0, 2.0])
    status = exp.format_status()
    assert status is not None
    assert "mean" in status
    assert "±" in status


def test_format_status_omits_margin_with_one_score():
    exp = make([1.0])
    status = exp.format_status()
    assert status is not None
    assert "±" not in status


def test_format_status_omits_relative_moe_without_stability():
    exp = make([1.0, 2.0])
    status = exp.format_status([AdditionalRuns(5)])
    assert status is not None
    assert "Relative MoE" not in status


def test_format_status_shows_relative_moe_with_stability():
    exp = make([1.0, 2.0])
    status = exp.format_status([Stability(min_samples=0, threshold=1.0)])
    assert status is not None
    assert "Relative MoE" in status


# SECTION run_loop


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
    exp.run_loop(stop_conditions=[Stability(min_samples=min_samples, threshold=0.01)])
    assert exp.run_count() == initial


def test_run_loop_does_not_stop_before_min_samples():
    # Even with a low margin of error, stability is skipped until min_samples
    # is reached; AdditionalRuns is what actually stops this run. It counts
    # runs from when it starts tracking, so 3 more on top of the 3 pre-loaded
    # results reaches a total of 6.
    exp = make([1.0] * 3)
    exp.run_loop(
        stop_conditions=[
            Stability(min_samples=10, threshold=999.0),
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
            Stability(min_samples=2, threshold=0.0),
            AdditionalRuns(20),
        ]
    )
    assert exp.run_count() == 20


def test_run_loop_stability_check_with_single_sample_does_not_crash():
    # min_samples=1 means the stability check runs while margin_of_error()
    # still returns None (fewer than 2 results), which used to raise
    # UnboundLocalError instead of just continuing.
    exp = make()
    exp.run_loop(
        stop_conditions=[
            Stability(min_samples=1, threshold=0.01),
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


# SECTION save/load round-trip


def test_save_creates_file(tmp_path: Path):
    exp = make([1.0])
    path = tmp_path / "out.json"
    exp.save(path)
    assert path.exists()


def test_save_writes_correct_json(tmp_path: Path):
    exp = make([1.0, 2.0, 3.0])
    path = tmp_path / "experiment.json"
    exp.save(path)
    assert path.read_text() == exp.serialize()


def test_save_atomic_writes_correct_json(tmp_path: Path):
    exp = make([10.0, 20.0])
    path = tmp_path / "experiment.json"
    exp.save_atomic(path)
    assert path.read_text() == exp.serialize()


def test_load_preserves_results(tmp_path: Path):
    exp = make([1.0, 2.0, 3.0])
    path = tmp_path / "experiment.json"
    exp.save(path)
    loaded = make()
    loaded.load_from(path)
    assert loaded.run_count() == exp.run_count()
    assert all(isinstance(result, _TestResult) for result in loaded._results.values())
    assert {key: result.value for key, result in loaded._results.items()} == {
        0: 1.0,
        1: 2.0,
        2: 3.0,
    }


def test_save_load_via_file_object(tmp_path: Path):
    exp = make([7.0, 8.0])
    path = tmp_path / "experiment.json"
    with open(path, "w") as f:
        exp.save_file(f)
    assert path.read_text() == exp.serialize()

    loaded = make()
    with open(path, "r") as f:
        loaded.load_from_file(f)
    assert {key: result.value for key, result in loaded._results.items()} == {
        0: 7.0,
        1: 8.0,
    }


def test_save_atomic_uses_tempfile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Verify os.replace is called with a source different from the destination,
    # confirming the write-to-temp-then-rename pattern.
    exp = make([1.0])
    path = tmp_path / "out.json"
    replaced: list[tuple[str, str]] = []
    real_replace = os.replace

    def recording_replace(src: str | PathLike[str], dst: str | PathLike[str]) -> None:
        replaced.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", recording_replace)
    exp.save_atomic(path)

    assert len(replaced) == 1
    src, dst = replaced[0]
    assert src != dst
    assert dst == str(path.expanduser())


def test_run_loop_saves_at_end(tmp_path: Path):
    exp = make()
    path = tmp_path / "experiment.json"
    exp.run_loop(
        stop_conditions=[AdditionalRuns(3)],
        save_config=SaveConfig(path=path, interval_seconds=None),
    )
    assert path.exists()
    loaded = make()
    loaded.load_from(path)
    assert loaded.run_count() == 3


def test_parameter_mismatch(tmp_path: Path):
    exp = make([1.0])
    path = tmp_path / "experiment.json"
    exp.save(path)
    mismatched = make(name="mismatch")
    with pytest.raises(ValueError):
        mismatched.load_from(path)
