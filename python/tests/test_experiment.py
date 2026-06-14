"""Tests for the public Experiment API (faultforge.experiment)."""

import os
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Self, override

import pytest
from faultforge.experiment import (
    Data,
    Experiment,
    SaveConfig,
    StabilityConfig,
)
from pydantic import BaseModel

# The following classes are `_` prefixed to not interpret them as Test classes.


class _TestParams(BaseModel):
    name: str = "test"


@dataclass(slots=True)
class _TestResult:
    value: float


@dataclass
class _TestExperiment(Experiment[_TestParams, _TestResult, None]):
    _max_runs: int | None = None
    _latest: int | None = None

    def set_max_runs(self, max_runs: int | None) -> None:
        self._max_runs = max_runs

    @override
    def latest_result(self) -> int | None:
        return self._latest

    @override
    def max_runs(self) -> int | None:
        return self._max_runs

    @override
    def result_score(self, result: _TestResult) -> float:
        return result.value

    @classmethod
    @override
    def from_parameters(cls, parameters: _TestParams) -> Self:
        return cls(
            data=Data[_TestParams, _TestResult, None](
                parameters=parameters, context=None, results=dict()
            )
        )

    @override
    def run(self) -> None:
        key = len(self.data.results)
        self.data.results[key] = _TestResult(value=float(key + 1))
        self._latest = key


def make(values: list[float] | None = None) -> _TestExperiment:
    """Create a test experiment with existing results"""
    data = Data[_TestParams, _TestResult, None](
        parameters=_TestParams(),
        context=None,
        results={k: _TestResult(value=v) for k, v in enumerate(values or [])},
    )
    return _TestExperiment(data=data)


# SECTION ci_half_width


def test_ci_half_width_no_results():
    assert make().ci_half_width() is None


def test_ci_half_width_one_result():
    assert make([1.0]).ci_half_width() is None


def test_ci_half_width_identical_values():
    assert make([3.0, 3.0]).ci_half_width() == 0.0


# SECTION run_loop


def test_run_loop_stops_at_max_runs():
    exp = make()
    exp.set_max_runs(5)
    exp.run_loop()
    assert exp.run_count() == 5


def test_run_loop_stops_when_stable():
    # Identical values -> CI = 0, below any positive threshold. The experiment
    # already has min_samples+1 results so stability is checked immediately.
    min_samples = 5
    exp = make([1.0] * (min_samples + 1))
    exp.stability_config = StabilityConfig(min_samples=min_samples, threshold=0.01)
    initial = exp.run_count()
    exp.run_loop()
    assert exp.run_count() == initial


def test_run_loop_does_not_stop_before_min_samples():
    # Even with low CI, stability is skipped until min_samples is reached.
    exp = make([1.0] * 3)
    exp.set_max_runs(6)
    exp.stability_config = StabilityConfig(min_samples=10, threshold=999.0)
    exp.run_loop()
    assert exp.run_count() == 6


def test_run_loop_continues_while_unstable():
    # Incrementing values → CI never reaches 0 → runs to max_runs.
    exp = make()
    exp.set_max_runs(20)
    exp.stability_config = StabilityConfig(min_samples=2, threshold=0.0)
    exp.run_loop()
    assert exp.run_count() == 20


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
    assert path.read_text() == exp.data.model_dump_json()


def test_save_atomic_writes_correct_json(tmp_path: Path):
    exp = make([10.0, 20.0])
    path = tmp_path / "experiment.json"
    exp.save_atomic(path)
    assert path.read_text() == exp.data.model_dump_json()


def test_load_preserves_results(tmp_path: Path):
    exp = make([1.0, 2.0, 3.0])
    path = tmp_path / "experiment.json"
    exp.save(path)
    loaded = _TestExperiment.load(path, _TestParams())
    assert loaded.run_count() == exp.run_count()
    assert all(isinstance(r, _TestResult) for r in loaded.data.results.values())
    assert {k: r.value for k, r in loaded.data.results.items()} == {
        0: 1.0,
        1: 2.0,
        2: 3.0,
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
    exp.set_max_runs(3)
    path = tmp_path / "experiment.json"
    exp.save_config = SaveConfig(path=path, interval_seconds=None)
    exp.run_loop()
    assert path.exists()
    loaded = _TestExperiment.load(path, _TestParams())
    assert loaded.run_count() == 3


def test_save_load_via_file_object(tmp_path: Path):
    exp = make([7.0, 8.0])
    path = tmp_path / "experiment.json"
    with open(path, "w") as f:
        exp.save_file(f)
    assert path.read_text() == exp.data.model_dump_json()


def test_parameter_mismatch(tmp_path: Path):
    exp = make([1.0])
    path = tmp_path / "experiment.json"
    exp.save(path)
    with pytest.raises(ValueError):
        _TestExperiment.load(path, _TestParams(name="mismatch"))
