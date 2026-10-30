"""Tests for `faultforge.metric.Accuracy`."""

import torch

from faultforge.metric import (
    Accuracy,
    AccuracyResult,
)


def test_accuracy_evaluate_batch_all_correct():
    logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    targets = torch.tensor([1, 0])

    result = Accuracy().evaluate_batch(logits, torch.empty(0), targets)

    assert result == AccuracyResult(correct_count=2, total_count=2)


def test_accuracy_evaluate_batch_partial():
    logits = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([1, 1, 1])

    result = Accuracy().evaluate_batch(logits, torch.empty(0), targets)

    assert result == AccuracyResult(correct_count=2, total_count=3)


def test_accuracy_accumulate():
    first = AccuracyResult(correct_count=2, total_count=3)
    second = AccuracyResult(correct_count=1, total_count=2)

    result = Accuracy().accumulate(first, second)

    assert result == AccuracyResult(correct_count=3, total_count=5)


def test_accuracy_score():
    assert Accuracy().score(AccuracyResult(correct_count=3, total_count=4)) == 75.0
