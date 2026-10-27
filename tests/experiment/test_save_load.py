"""Tests for Experiment.save/save_atomic/load_from round-trip behavior."""

import os
from os import PathLike
from pathlib import Path

import pytest

from faultforge.experiment import (
    AdditionalRuns,
    Experiment,
    SaveConfig,
)

from .conftest import _TestResult, make


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


def test_run_loop_saves_when_interrupted_by_an_exception(tmp_path: Path):
    def fail_after_three_runs(experiment: Experiment) -> str | None:
        if experiment.run_count() == 3:
            raise RuntimeError("stop")
        return None

    exp = make()
    path = tmp_path / "experiment.json"
    with pytest.raises(RuntimeError, match="stop"):
        exp.run_loop(
            stop_conditions=[fail_after_three_runs],
            save_config=SaveConfig(path=path, interval_seconds=None),
        )
    loaded = make()
    loaded.load_from(path)
    assert loaded.run_count() == 3


def test_run_loop_does_not_save_without_new_runs(tmp_path: Path):
    def fail_immediately(experiment: Experiment) -> str | None:
        _ = experiment
        raise RuntimeError("stop")

    exp = make()
    path = tmp_path / "experiment.json"
    with pytest.raises(RuntimeError, match="stop"):
        exp.run_loop(
            stop_conditions=[fail_immediately],
            save_config=SaveConfig(path=path, interval_seconds=None),
        )
    assert not path.exists()


def test_parameter_mismatch(tmp_path: Path):
    exp = make([1.0])
    path = tmp_path / "experiment.json"
    exp.save(path)
    mismatched = make(name="mismatch")
    with pytest.raises(ValueError):
        mismatched.load_from(path)
