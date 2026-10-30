"""Tests for `faultforge.metric.AccuracyDegradation`."""

import torch

from faultforge.metric import (
    AccuracyDegradation,
    AccuracyDegradationResult,
)


def test_accuracy_degradation_preprocess_golden():
    golden_logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    result = AccuracyDegradation().preprocess_golden(golden_logits)

    assert torch.equal(result, torch.tensor([0, 1]))


def test_accuracy_degradation_evaluate_batch_no_change():
    logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    # Preprocessed golden predictions
    golden = torch.tensor([1, 0])
    targets = torch.tensor([1, 0])

    result = AccuracyDegradation().evaluate_batch(logits, golden, targets)

    assert result == AccuracyDegradationResult(
        correct_count_faulty=2, correct_count_golden=2, total_count=2
    )


def test_accuracy_degradation_evaluate_batch_degraded():
    logits = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    golden = torch.tensor([1, 0])
    targets = torch.tensor([1, 0])

    result = AccuracyDegradation().evaluate_batch(logits, golden, targets)

    assert result == AccuracyDegradationResult(
        correct_count_faulty=1, correct_count_golden=2, total_count=2
    )


def test_accuracy_degradation_accumulate():
    first = AccuracyDegradationResult(
        correct_count_faulty=1, correct_count_golden=2, total_count=2
    )
    second = AccuracyDegradationResult(
        correct_count_faulty=2, correct_count_golden=2, total_count=2
    )

    result = AccuracyDegradation().accumulate(first, second)

    assert result == AccuracyDegradationResult(
        correct_count_faulty=3, correct_count_golden=4, total_count=4
    )


def test_accuracy_degradation_score_drop():
    result = AccuracyDegradationResult(
        correct_count_faulty=1, correct_count_golden=2, total_count=4
    )
    assert AccuracyDegradation().score(result) == 25.0
