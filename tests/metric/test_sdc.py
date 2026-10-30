"""Tests for `faultforge.metric.Sdc`."""

import pytest
import torch

from faultforge.metric import (
    Sdc,
    SdcResult,
)


def test_sdc_evaluate_batch_nothing_changed():
    output = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    result = Sdc().evaluate_batch(output, output.clone(), torch.empty(0))

    assert result == SdcResult(non_matching_count=0, total_count=4)


def test_sdc_evaluate_batch_everything_changed():
    output = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    golden = torch.tensor([[9.0, 9.0], [9.0, 9.0]])

    result = Sdc().evaluate_batch(output, golden, torch.empty(0))

    assert result == SdcResult(non_matching_count=4, total_count=4)


def test_sdc_evaluate_batch_shape_mismatch_raises():
    output = torch.tensor([[1.0, 2.0]])
    golden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(ValueError):
        Sdc().evaluate_batch(output, golden, torch.empty(0))


def test_sdc_accumulate():
    first = SdcResult(non_matching_count=1, total_count=4)
    second = SdcResult(non_matching_count=2, total_count=4)

    result = Sdc().accumulate(first, second)

    assert result == SdcResult(non_matching_count=3, total_count=8)


def test_sdc_score():
    assert Sdc().score(SdcResult(non_matching_count=1, total_count=4)) == 25.0
