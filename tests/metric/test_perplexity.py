"""Tests for `faultforge.metric.Perplexity`."""

import math

import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given

from faultforge.metric import (
    Perplexity,
    PerplexityResult,
)


def test_perplexity_evaluate_batch_uniform_logits():
    # Uniform logits give every class equal probability (1/vocab), so the
    # cross-entropy of any scored position is exactly -log(1/vocab) = log(vocab)
    # and the perplexity is the vocabulary size.
    vocab = 3
    logits = torch.zeros(1, 3, vocab)
    # Last position masked out.
    targets = torch.tensor([[0, 1, Perplexity.IGNORED_TARGET]])

    result = Perplexity().evaluate_batch(logits, torch.empty(0), targets)

    assert result.token_count == 2
    assert result.cross_entropy_sum == pytest.approx(2 * math.log(vocab))
    assert Perplexity().score(result) == pytest.approx(float(vocab))


def test_perplexity_evaluate_batch_all_masked():
    logits = torch.zeros(1, 2, 3)
    targets = torch.tensor([[Perplexity.IGNORED_TARGET, Perplexity.IGNORED_TARGET]])

    result = Perplexity().evaluate_batch(logits, torch.empty(0), targets)

    assert result.token_count == 0
    assert result.cross_entropy_sum == pytest.approx(0.0)


def _random_lm_batch(
    seed: int, batch: int, seq: int, vocab: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, seq, vocab, generator=generator) * 3
    targets = torch.randint(0, vocab, (batch, seq), generator=generator)
    masked = torch.rand(batch, seq, generator=generator) < 0.3
    return logits, targets.masked_fill(masked, Perplexity.IGNORED_TARGET)


def test_perplexity_matches_reference_on_random_logits():
    logits, targets = _random_lm_batch(seed=0, batch=3, seq=7, vocab=11)

    result = Perplexity().evaluate_batch(logits, torch.empty(0), targets)

    # Reference: compare position i with targets[i] (already shifted), mean
    # over unmasked positions only.
    scored = targets != Perplexity.IGNORED_TARGET
    log_probs = torch.log_softmax(logits.double(), dim=-1)
    target_log_probs = log_probs[scored].gather(-1, targets[scored].unsqueeze(-1))
    expected = math.exp(-target_log_probs.mean().item())

    assert result.token_count == int(scored.sum())
    assert Perplexity().score(result) == pytest.approx(expected, rel=1e-5)


@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    batch=st.integers(min_value=2, max_value=6),
    seq=st.integers(min_value=1, max_value=8),
    vocab=st.integers(min_value=2, max_value=16),
    data=st.data(),
)
def test_perplexity_accumulated_batches_equal_single_batch(
    seed: int, batch: int, seq: int, vocab: int, data: st.DataObject
):
    # Perplexity is token-weighted: splitting the same text into batches must
    # not change the score, even if the batches have different token counts.
    logits, targets = _random_lm_batch(seed, batch, seq, vocab)
    targets[0, 0] = 0  # ensure at least one scored token
    split = data.draw(st.integers(min_value=1, max_value=batch - 1))
    metric = Perplexity()

    whole = metric.evaluate_batch(logits, torch.empty(0), targets)
    accumulated = metric.accumulate(
        metric.evaluate_batch(logits[:split], torch.empty(0), targets[:split]),
        metric.evaluate_batch(logits[split:], torch.empty(0), targets[split:]),
    )

    assert accumulated.token_count == whole.token_count
    assert metric.score(accumulated) == pytest.approx(metric.score(whole), rel=1e-5)


def test_perplexity_evaluate_batch_non_contiguous_logits():
    # e.g. a caller dropping the final position of a model's output.
    full_logits, full_targets = _random_lm_batch(seed=1, batch=2, seq=5, vocab=7)
    logits = full_logits[:, :-1]
    targets = full_targets[:, :-1]
    assert not logits.is_contiguous()

    result = Perplexity().evaluate_batch(logits, torch.empty(0), targets)
    expected = Perplexity().evaluate_batch(
        logits.contiguous(), torch.empty(0), targets.contiguous()
    )

    assert result == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_perplexity_evaluate_batch_low_precision_logits(dtype: torch.dtype):
    logits, targets = _random_lm_batch(seed=2, batch=2, seq=6, vocab=9)
    low = logits.to(dtype)

    result = Perplexity().evaluate_batch(low, torch.empty(0), targets)
    # Same (already rounded) values in fp32: the upcast must make these equal.
    expected = Perplexity().evaluate_batch(low.float(), torch.empty(0), targets)

    assert result == expected


def test_perplexity_evaluate_batch_rejects_integer_logits():
    logits = torch.zeros(1, 2, 3, dtype=torch.int64)
    targets = torch.tensor([[0, 1]])

    with pytest.raises(ValueError, match="floating point"):
        Perplexity().evaluate_batch(logits, torch.empty(0), targets)


def test_perplexity_evaluate_batch_rejects_non_int64_targets():
    logits = torch.zeros(1, 2, 3)
    targets = torch.tensor([[0, 1]], dtype=torch.int32)

    with pytest.raises(ValueError, match="int64"):
        Perplexity().evaluate_batch(logits, torch.empty(0), targets)


@pytest.mark.parametrize(
    ("value", "index", "expected_non_finite_count"),
    [
        (math.nan, 0, 1),
        (math.nan, 1, 1),
        (math.inf, 0, 1),
        (math.inf, 1, 1),
        (-math.inf, 0, 1),
        (-math.inf, 1, 0),
    ],
    ids=[
        "nan-target",
        "nan-other",
        "+inf-target",
        "+inf-other",
        "-inf-target",
        "-inf-other",
    ],
)
def test_perplexity_non_finite_logits(
    value: float, index: int, expected_non_finite_count: int
):
    # Documented in the "Non-finite logits" section of `Perplexity`. The
    # target of the first position is token 0.
    logits = torch.zeros(1, 2, 3)
    logits[0, 0, index] = value
    targets = torch.tensor([[0, 1]])

    result = Perplexity().evaluate_batch(logits, torch.empty(0), targets)
    perplexity = Perplexity().score(result)

    assert result.token_count == 2
    assert result.non_finite_count == expected_non_finite_count
    assert math.isfinite(result.cross_entropy_sum)
    if expected_non_finite_count > 0:
        assert perplexity == math.inf
    else:
        assert math.isfinite(perplexity)


def test_perplexity_non_finite_positions_excluded_from_sum():
    vocab = 3
    logits = torch.zeros(1, 2, vocab)
    logits[0, 0, 0] = math.nan
    targets = torch.tensor([[0, 1]])

    result = Perplexity().evaluate_batch(logits, torch.empty(0), targets)

    # Only the second, uniform position contributes to the sum.
    assert result.cross_entropy_sum == pytest.approx(math.log(vocab))
    assert result.token_count == 2
    assert result.non_finite_count == 1


def test_perplexity_non_finite_logits_at_ignored_position():
    logits = torch.zeros(1, 2, 3)
    logits[0, 1] = math.nan
    targets = torch.tensor([[0, Perplexity.IGNORED_TARGET]])

    result = Perplexity().evaluate_batch(logits, torch.empty(0), targets)

    assert result.non_finite_count == 0
    assert Perplexity().score(result) == pytest.approx(3.0)


def test_perplexity_huge_finite_losses_do_not_overflow_sum():
    # A bit flip in an exponent can produce a logit close to the fp32 maximum.
    logits = torch.zeros(1, 2, 3)
    logits[0, :, 1] = 3e38
    targets = torch.tensor([[0, 0]])

    result = Perplexity().evaluate_batch(logits, torch.empty(0), targets)

    assert result.non_finite_count == 0
    assert result.cross_entropy_sum == pytest.approx(6e38)
    assert Perplexity().score(result) == math.inf


def test_perplexity_accumulate():
    first = PerplexityResult(cross_entropy_sum=1.0, token_count=2, non_finite_count=1)
    second = PerplexityResult(cross_entropy_sum=2.0, token_count=3, non_finite_count=0)

    result = Perplexity().accumulate(first, second)

    assert result == PerplexityResult(
        cross_entropy_sum=3.0, token_count=5, non_finite_count=1
    )


def test_perplexity_accumulate_any_non_finite_token_makes_score_inf():
    # A single corrupted token anywhere in the run makes the run's perplexity
    # inf, regardless of how many finite batches there are.
    finite = PerplexityResult(
        cross_entropy_sum=1.0, token_count=1000, non_finite_count=0
    )
    faulty = PerplexityResult(cross_entropy_sum=0.0, token_count=1, non_finite_count=1)

    result = Perplexity().accumulate(finite, faulty)

    assert Perplexity().score(result) == math.inf


def test_perplexity_score():
    result = PerplexityResult(
        cross_entropy_sum=2 * math.log(2), token_count=2, non_finite_count=0
    )

    assert Perplexity().score(result) == pytest.approx(2.0)


def test_perplexity_score_perfect_prediction():
    # log(1) == 0 total surprise -> perplexity of 1 (best possible).
    result = PerplexityResult(cross_entropy_sum=0.0, token_count=5, non_finite_count=0)

    assert Perplexity().score(result) == pytest.approx(1.0)


def test_perplexity_score_no_tokens_raises():
    result = PerplexityResult(cross_entropy_sum=0.0, token_count=0, non_finite_count=0)

    with pytest.raises(ValueError):
        Perplexity().score(result)


def test_perplexity_score_overflow_is_inf():
    # exp(1000) overflows a float; a fault can make the model this wrong.
    result = PerplexityResult(
        cross_entropy_sum=1000.0, token_count=1, non_finite_count=0
    )

    assert Perplexity().score(result) == math.inf
