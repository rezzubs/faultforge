"""Tests for the Metric implementations (faultforge.metric)."""

import pytest
import torch

from faultforge.metric import (
    Accuracy,
    AccuracyDegradation,
    AccuracyDegradationResult,
    AccuracyResult,
    Sdc,
    SdcResult,
    Top1Sdc,
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
