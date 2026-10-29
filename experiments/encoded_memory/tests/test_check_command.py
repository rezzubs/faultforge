"""Tests for the `check` command."""

import json
from pathlib import Path

import click
from encoded_memory.commands import app
from typer.testing import CliRunner

from faultforge.metric import Perplexity

from .conftest import (
    _saved_result,
    _write_saved,
)


def invoke_check(*arguments: str | Path):
    return CliRunner().invoke(app, ["check", *map(str, arguments)])


def test_json_shape(tmp_path: Path):
    stable = _write_saved(tmp_path, _saved_result([2.0] * 5), "stable.json")
    missing = tmp_path / "missing.json"

    result = invoke_check(stable, missing, "--max-absolute-margin", "0.5")

    assert json.loads(result.stdout) == [
        {
            "path": str(stable),
            "status": "stable",
            "reason": "Reached absolute margin of error 0.50% (0.00%)",
            "run_count": 5,
            "estimate": {"value": 2.0, "lower": 2.0, "upper": 2.0},
        },
        {
            "path": str(missing),
            "status": "missing",
            "reason": None,
            "run_count": 0,
            "estimate": None,
        },
    ]


def test_exit_code_without_runs_needed(tmp_path: Path):
    path = _write_saved(tmp_path, _saved_result([2.0] * 3))

    result = invoke_check(path, "--max-runs", "3")

    assert result.exit_code == 0
    assert json.loads(result.stdout)[0]["status"] == "max-runs"


def test_exit_code_with_runs_needed(tmp_path: Path):
    done = _write_saved(tmp_path, _saved_result([2.0] * 3), "done.json")
    incomplete = _write_saved(tmp_path, _saved_result([2.0]), "incomplete.json")

    result = invoke_check(done, incomplete, "--max-runs", "3")

    assert result.exit_code == 1


def test_error_continues_the_batch(tmp_path: Path):
    broken = tmp_path / "broken.json"
    _ = broken.write_text("not json")
    missing = tmp_path / "missing.json"

    result = invoke_check(broken, missing, "--max-runs", "3")

    # The error is worse than the missing file.
    assert result.exit_code == 3
    broken_entry, missing_entry = json.loads(result.stdout)
    assert broken_entry["path"] == str(broken)
    assert broken_entry["status"] == "error"
    assert isinstance(broken_entry["error"], str)
    assert missing_entry["status"] == "missing"


def test_corrupt_compressed_file_is_an_error(tmp_path: Path):
    corrupt = tmp_path / "corrupt.json.zst"
    # The zstd magic number followed by garbage.
    _ = corrupt.write_bytes(b"\x28\xb5\x2f\xfdgarbage")
    done = _write_saved(tmp_path, _saved_result([2.0] * 3), "done.json")

    result = invoke_check(corrupt, done, "--max-runs", "3")

    assert result.exit_code == 3
    corrupt_entry, done_entry = json.loads(result.stdout)
    assert corrupt_entry["status"] == "error"
    assert done_entry["status"] == "max-runs"


def test_missing_golden_score(tmp_path: Path):
    path = _write_saved(tmp_path, _saved_result([2.0] * 3, metric=Perplexity()))

    result = invoke_check(
        path,
        "--estimate",
        "failure-rate",
        "--relative-failure-threshold",
        "2.0",
        "--max-absolute-margin",
        "100",
    )

    assert result.exit_code == 1
    assert json.loads(result.stdout)[0]["reason"] == "no golden score recorded"


def test_requires_a_limit(tmp_path: Path):
    result = invoke_check(tmp_path / "result.json")

    assert result.exit_code == 2
    assert "--max-runs" in click.unstyle(result.output)


def test_rejects_runs(tmp_path: Path):
    result = invoke_check(tmp_path / "result.json", "--runs", "3")

    assert result.exit_code == 2
