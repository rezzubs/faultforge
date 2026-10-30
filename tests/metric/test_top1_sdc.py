"""Tests for `faultforge.metric.Top1Sdc`."""

import torch

from faultforge.metric import (
    SdcResult,
    Top1Sdc,
)


def test_top1_sdc_preprocess_golden():
    golden_logits = torch.tensor([[1.0, 5.0, 2.0], [9.0, 1.0, 1.0]])

    result = Top1Sdc().preprocess_golden(golden_logits)

    assert torch.equal(result, torch.tensor([1, 0]))


def test_top1_sdc_evaluate_batch_prediction_unchanged():
    logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    # Preprocessed top-1 golden predictions
    golden = torch.tensor([1, 0])
    result = Top1Sdc().evaluate_batch(logits, golden, torch.empty(0))

    assert result == SdcResult(non_matching_count=0, total_count=2)


def test_top1_sdc_evaluate_batch_prediction_changed():
    logits = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    golden = torch.tensor([1, 0])

    result = Top1Sdc().evaluate_batch(logits, golden, torch.empty(0))

    assert result == SdcResult(non_matching_count=1, total_count=2)


def test_top1_sdc_score():
    assert Top1Sdc().score(SdcResult(non_matching_count=1, total_count=4)) == 25.0
