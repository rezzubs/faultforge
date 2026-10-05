"""Tests for SavedResult standalone loading."""

import math

import pytest
from encoded_memory import SavedResult
from encoded_memory.experiment import DetailedResults, DetailedRunResult, SimpleResults

from faultforge import Fingerprint

from .conftest import _make_experiment


def test_saved_result_round_trip_scores_and_bit_error_rate(tmp_path):
    experiment = _make_experiment(compare_bitwise=True, faults=3)
    experiment.run()
    experiment.run()

    path = tmp_path / "result.json"
    experiment.save(path)

    loaded = SavedResult.load(path)
    assert loaded.scores() == list(experiment.scores())
    assert loaded.bit_error_rate() == pytest.approx(3 / loaded.total_bits)


@pytest.mark.parametrize(
    "result",
    [
        SimpleResults(results=[1.0, math.inf, math.nan]),
        DetailedResults(
            results=[
                DetailedRunResult(score=math.inf, bitmask=[]),
                DetailedRunResult(score=math.nan, bitmask=[]),
            ]
        ),
    ],
)
def test_non_finite_scores_survive_round_trip(result):
    # E.g. a faulty LLM producing an infinite perplexity.
    saved = SavedResult(
        fingerprint=Fingerprint(kind="test"),
        total_bits=1,
        result=result,
        metric_display_name="",
    )

    loaded = SavedResult.model_validate_json(saved.model_dump_json())

    assert [str(s) for s in loaded.scores()] == [str(s) for s in result.scores()]
