"""Tests for `check_saved`."""

import json
import math
import tempfile
from pathlib import Path

import pytest
from encoded_memory import (
    CheckStatus,
    StabilityLimits,
    check_saved,
)
from encoded_memory.check import _SavedRuns
from hypothesis import given
from hypothesis import strategies as st

from faultforge import Fingerprint
from faultforge.experiment import (
    MaxRuns,
    RelativeThreshold,
    SaveConfig,
)
from faultforge.io import open_text
from faultforge.metric import (
    Perplexity,
    Sdc,
)

from .conftest import (
    _make_experiment,
    _saved_result,
    _write_saved,
)

TIGHT = StabilityLimits(max_absolute_margin=0.5)
"""Reached by a few equal scores, but not by a single one."""
FAILURE_RATE = StabilityLimits(
    failure_threshold=RelativeThreshold(2.0), max_absolute_margin=100.0
)
"""Needs a golden score. Any failure rate estimate is within the margin."""


def test_missing(tmp_path: Path):
    result = check_saved(tmp_path / "result.json", max_runs=10)

    assert result.status == CheckStatus.Missing
    assert result.run_count == 0
    assert result.should_run()


def test_incomplete(tmp_path: Path):
    path = _write_saved(tmp_path, _saved_result([1.0, 5.0, 9.0]))

    result = check_saved(path, stability=TIGHT, max_runs=10)

    assert result.status == CheckStatus.Incomplete
    assert result.reason is None
    assert result.run_count == 3
    assert result.estimate is not None
    assert result.should_run()


def test_stable(tmp_path: Path):
    path = _write_saved(tmp_path, _saved_result([2.0] * 5))

    result = check_saved(path, stability=TIGHT)

    assert result.status == CheckStatus.Stable
    assert result.reason is not None
    assert "absolute margin" in result.reason
    assert result.estimate is not None
    assert result.estimate.value == pytest.approx(2.0)
    assert not result.should_run()


def test_max_runs(tmp_path: Path):
    path = _write_saved(tmp_path, _saved_result([1.0, 5.0, 9.0]))

    result = check_saved(path, stability=TIGHT, max_runs=3)

    assert result.status == CheckStatus.MaxRuns
    assert result.reason is not None
    assert not result.should_run()


def test_stable_wins_over_max_runs(tmp_path: Path):
    path = _write_saved(tmp_path, _saved_result([2.0] * 5))

    result = check_saved(path, stability=TIGHT, max_runs=5)

    assert result.status == CheckStatus.Stable


def test_min_runs_delays_stability(tmp_path: Path):
    path = _write_saved(tmp_path, _saved_result([2.0] * 5))
    limits = StabilityLimits(min_runs=6, max_absolute_margin=0.5)

    assert check_saved(path, stability=limits).status == CheckStatus.Incomplete


def test_missing_golden_score_is_incomplete(tmp_path: Path):
    path = _write_saved(tmp_path, _saved_result([2.0] * 5, metric=Perplexity()))

    result = check_saved(path, stability=FAILURE_RATE)

    assert result.status == CheckStatus.Incomplete
    assert result.reason == "no golden score recorded"
    assert result.estimate is None


def test_missing_golden_score_with_max_runs_reached(tmp_path: Path):
    path = _write_saved(tmp_path, _saved_result([2.0] * 5, metric=Perplexity()))

    result = check_saved(path, stability=FAILURE_RATE, max_runs=5)

    assert result.status == CheckStatus.MaxRuns


def test_record_saves_the_missing_golden_score(tmp_path: Path):
    path = tmp_path / "result.json"
    experiment = _make_experiment(
        compare_bitwise=False, reliability_metric=Perplexity()
    )
    for _ in range(3):
        experiment.run()
    experiment.save(path)
    assert check_saved(path, stability=FAILURE_RATE).status == CheckStatus.Incomplete

    # What `record` does with the same limits. The runs already reach the
    # margin, so only the golden score is added.
    experiment.run_loop(
        stop_conditions=[FAILURE_RATE.stability(Perplexity())],
        save_config=SaveConfig(path=path, interval_seconds=None),
    )

    result = check_saved(path, stability=FAILURE_RATE)
    assert result.status == CheckStatus.Stable
    assert result.run_count == 3


def test_recorded_golden_score_is_used(tmp_path: Path):
    path = _write_saved(
        tmp_path, _saved_result([2.0] * 5, metric=Perplexity(), golden_score=2.0)
    )

    result = check_saved(path, stability=FAILURE_RATE)

    assert result.status == CheckStatus.Stable
    assert result.estimate is not None
    assert result.estimate.value == 0.0


def test_reads_compressed_files(tmp_path: Path):
    path = tmp_path / "result.json.zst"
    with open_text(path, "wt", compressed=True) as file:
        _ = file.write(_saved_result([2.0] * 5).model_dump_json())

    assert check_saved(path, stability=TIGHT).status == CheckStatus.Stable


def test_to_dict(tmp_path: Path):
    path = _write_saved(tmp_path, _saved_result([2.0] * 5))

    result = check_saved(path, stability=TIGHT)

    assert json.loads(json.dumps(result.to_dict())) == {
        "path": str(path),
        "status": "stable",
        "reason": "Reached absolute margin of error 0.50% (0.00%)",
        "run_count": 5,
        "estimate": {"value": 2.0, "lower": 2.0, "upper": 2.0},
    }


def test_to_dict_without_estimate(tmp_path: Path):
    result = check_saved(tmp_path / "result.json", max_runs=10)

    assert result.to_dict()["estimate"] is None


def test_requires_a_limit(tmp_path: Path):
    with pytest.raises(ValueError, match="requires"):
        _ = check_saved(tmp_path / "result.json")


def test_mean_rejects_non_finite_scores(tmp_path: Path):
    path = _write_saved(tmp_path, _saved_result([2.0, math.inf, 2.0]))

    with pytest.raises(ValueError):
        _ = check_saved(path, stability=TIGHT)


def test_unreadable_file_raises(tmp_path: Path):
    path = tmp_path / "result.json"
    _ = path.write_text("not json")

    with pytest.raises(ValueError):
        _ = check_saved(path, max_runs=10)


def test_unknown_metric_raises(tmp_path: Path):
    saved = _saved_result([2.0])
    saved.fingerprint.children["reliability_metric"] = [Fingerprint(kind="unknown")]
    path = _write_saved(tmp_path, saved)

    with pytest.raises(ValueError, match="unknown"):
        _ = check_saved(path, max_runs=10)


def test_missing_metric_raises(tmp_path: Path):
    saved = _saved_result([2.0])
    del saved.fingerprint.children["reliability_metric"]
    path = _write_saved(tmp_path, saved)

    with pytest.raises(ValueError, match="reliability metric"):
        _ = check_saved(path, max_runs=10)


@given(
    scores=st.lists(st.floats(min_value=0.0, max_value=100.0), max_size=20),
    max_runs=st.integers(min_value=1, max_value=25),
)
def test_should_run_agrees_with_the_record_stop_conditions(
    scores: list[float], max_runs: int
):
    saved = _saved_result(scores)
    conditions = [TIGHT.stability(Sdc()), MaxRuns(max_runs)]
    runs = _SavedRuns(saved, Sdc())
    record_would_stop = any(condition(runs) is not None for condition in conditions)

    # NOTE: hypothesis doesn't allow function scoped fixtures like `tmp_path`.
    with tempfile.TemporaryDirectory() as directory:
        path = _write_saved(Path(directory), saved)
        result = check_saved(path, stability=TIGHT, max_runs=max_runs)

    assert result.should_run() != record_would_stop
