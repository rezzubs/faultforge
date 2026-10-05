"""Tests for the Metric implementations (faultforge.metric)."""

import math

import pytest
import torch

from faultforge.metric import (
    Accuracy,
    AccuracyDegradation,
    AccuracyDegradationResult,
    AccuracyResult,
    Perplexity,
    PerplexityResult,
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


def test_perplexity_evaluate_batch_uniform_logits():
    # Uniform logits give every class equal probability (1/vocab), so the
    # cross-entropy of any valid position is exactly -log(1/vocab) = log(vocab).
    vocab = 3
    logits = torch.zeros(1, 3, vocab)
    targets = torch.tensor([[0, 1, -100]])  # last position masked out

    result = Perplexity().evaluate_batch(logits, torch.empty(0), targets)

    assert result.token_count == 2
    assert result.cross_entropy_sums == pytest.approx(2 * math.log(vocab))


def test_perplexity_evaluate_batch_all_masked():
    logits = torch.zeros(1, 2, 3)
    targets = torch.tensor([[-100, -100]])

    result = Perplexity().evaluate_batch(logits, torch.empty(0), targets)

    assert result.token_count == 0
    assert result.cross_entropy_sums == pytest.approx(0.0)


def test_perplexity_accumulate():
    first = PerplexityResult(cross_entropy_sums=1.0, token_count=2)
    second = PerplexityResult(cross_entropy_sums=2.0, token_count=3)

    result = Perplexity().accumulate(first, second)

    assert result == PerplexityResult(cross_entropy_sums=3.0, token_count=5)


def test_perplexity_score():
    result = PerplexityResult(cross_entropy_sums=2 * math.log(2), token_count=2)

    assert Perplexity().score(result) == pytest.approx(2.0)


def test_perplexity_score_perfect_prediction():
    # log(1) == 0 total surprise -> perplexity of 1 (best possible).
    result = PerplexityResult(cross_entropy_sums=0.0, token_count=5)

    assert Perplexity().score(result) == pytest.approx(1.0)


def test_perplexity_score_no_tokens_raises():
    result = PerplexityResult(cross_entropy_sums=0.0, token_count=0)

    with pytest.raises(ValueError):
        Perplexity().score(result)


def test_perplexity_score_overflow_is_inf():
    # exp(1000) overflows a float; a fault can make the model this wrong.
    result = PerplexityResult(cross_entropy_sums=1000.0, token_count=1)

    assert Perplexity().score(result) == math.inf


def test_perplexity_end_to_end_uniform_logits_gives_vocab_size():
    vocab = 5
    logits = torch.zeros(1, 4, vocab)
    targets = torch.tensor([[0, 1, 2, 3]])

    result = Perplexity().evaluate_batch(logits, torch.empty(0), targets)
    perplexity = Perplexity().score(result)

    assert perplexity == pytest.approx(float(vocab))


def test_perplexity_score_nan_is_inf():
    # Non-finite logits from a fault make the summed cross-entropy NaN.
    result = PerplexityResult(cross_entropy_sums=math.nan, token_count=3)

    assert Perplexity().score(result) == math.inf


def test_perplexity_end_to_end_non_finite_logits_gives_inf():
    logits = torch.zeros(1, 2, 3)
    logits[0, 0, 1] = math.inf
    targets = torch.tensor([[0, 1]])

    result = Perplexity().evaluate_batch(logits, torch.empty(0), targets)

    assert Perplexity().score(result) == math.inf
