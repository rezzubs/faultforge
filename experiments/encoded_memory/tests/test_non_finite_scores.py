"""Tests for saving and loading NaN and ±inf scores."""

import json
import math
from typing import override

import pytest
from encoded_memory import (
    DetailedResults,
    DetailedRunResult,
    SavedResult,
    SimpleResults,
    discard_bitmasks_in_file,
)
from hypothesis import given
from hypothesis import strategies as st
from torch import Tensor

from faultforge import Fingerprint
from faultforge.metric import Metric

from .conftest import _make_experiment

NON_FINITE_SCORES = [math.nan, math.inf, -math.inf]

any_float = st.floats(allow_nan=True, allow_infinity=True)


class _FixedScore(Metric[int]):
    """A metric which ignores the model output and always gives the same score."""

    def __init__(self, fixed_score: float) -> None:
        self._fixed_score = fixed_score

    @override
    def evaluate_batch(
        self,
        batch_model_output: Tensor,
        batch_golden: Tensor,
        batch_targets: Tensor,
    ) -> int:
        _ = batch_model_output, batch_golden, batch_targets
        return 0

    @override
    def requires_golden(self) -> bool:
        return False

    @override
    def accumulate(self, existing: int, new: int) -> int:
        return existing + new

    @override
    def score(self, result: int) -> float:
        _ = result
        return self._fixed_score

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="fixed_score")


def same_scores(left: list[float], right: list[float]) -> bool:
    """Whether the scores are equal, treating NaN as equal to itself."""
    return len(left) == len(right) and all(
        (math.isnan(left_score) and math.isnan(right_score))
        or left_score == right_score
        for left_score, right_score in zip(left, right, strict=True)
    )


def saved_result(result: SimpleResults | DetailedResults) -> SavedResult:
    return SavedResult(
        fingerprint=Fingerprint(kind="test"),
        total_bits=8,
        result=result,
        metric_display_name="",
    )


def test_non_finite_scores_are_written_as_json_strings():
    serialized = saved_result(SimpleResults(results=NON_FINITE_SCORES))
    assert json.loads(serialized.model_dump_json())["result"]["results"] == [
        "NaN",
        "Infinity",
        "-Infinity",
    ]


@given(st.lists(any_float))
def test_simple_results_round_trip(scores: list[float]):
    original = saved_result(SimpleResults(results=scores))

    loaded = SavedResult.model_validate_json(original.model_dump_json())

    assert same_scores(loaded.scores(), scores)


@given(st.lists(any_float))
def test_detailed_results_round_trip(scores: list[float]):
    original = saved_result(
        DetailedResults(
            results=[DetailedRunResult(score=score, bitmask=[1]) for score in scores]
        )
    )

    loaded = SavedResult.model_validate_json(original.model_dump_json())

    assert same_scores(loaded.scores(), scores)


@pytest.mark.parametrize("compare_bitwise", [False, True])
def test_experiment_round_trip(compare_bitwise: bool):
    for score in NON_FINITE_SCORES:
        experiment = _make_experiment(
            compare_bitwise=compare_bitwise, reliability_metric=_FixedScore(score)
        )
        experiment.run()
        serialized = experiment.serialize()

        reloaded = _make_experiment(
            compare_bitwise=compare_bitwise, reliability_metric=_FixedScore(score)
        )
        reloaded.deserialize(serialized)

        assert same_scores(list(reloaded.scores()), [score])
        assert reloaded.serialize() == serialized


@pytest.mark.parametrize("compressed", [False, True])
def test_load_file_with_non_finite_scores(tmp_path, compressed: bool):
    path = tmp_path / "result.json"
    for score in NON_FINITE_SCORES:
        experiment = _make_experiment(
            compare_bitwise=False, reliability_metric=_FixedScore(score)
        )
        experiment.run()
        experiment.save_atomic(path, compressed=compressed)

        assert same_scores(SavedResult.load(path).scores(), [score])


def test_discard_bitmasks_in_file_keeps_non_finite_scores(tmp_path):
    path = tmp_path / "result.json"
    for score in NON_FINITE_SCORES:
        experiment = _make_experiment(
            compare_bitwise=True, reliability_metric=_FixedScore(score)
        )
        experiment.run()
        experiment.save(path)

        discard_bitmasks_in_file(path)

        assert same_scores(SavedResult.load(path).scores(), [score])
