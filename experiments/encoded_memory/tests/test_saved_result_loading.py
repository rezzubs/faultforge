"""Tests for SavedResult standalone loading."""

import pytest
from encoded_memory import SavedResult

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
