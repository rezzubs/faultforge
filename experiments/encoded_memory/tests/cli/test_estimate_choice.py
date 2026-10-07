"""Tests for EstimateChoice.into_estimator."""

import pytest
import typer
from encoded_memory.commands import EstimateChoice

from faultforge.experiment import FailureRate, Mean


def test_mean():
    assert EstimateChoice.Mean.into_estimator(None) == Mean()


def test_mean_rejects_failure_threshold():
    with pytest.raises(typer.BadParameter, match="--failure-threshold"):
        _ = EstimateChoice.Mean.into_estimator(2.0)


def test_failure_rate():
    assert EstimateChoice.FailureRate.into_estimator(2.0) == FailureRate(2.0)


def test_failure_rate_requires_failure_threshold():
    with pytest.raises(typer.BadParameter, match="--failure-threshold"):
        _ = EstimateChoice.FailureRate.into_estimator(None)
