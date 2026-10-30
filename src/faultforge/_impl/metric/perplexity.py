"""See [`faultforge.metric`]"""

import math
from dataclasses import dataclass
from typing import ClassVar, final, override

import torch
from torch import Tensor, nn

from faultforge._impl.fingerprint import Fingerprint
from faultforge._impl.metric.base import Metric, ScoreDirection


@final
@dataclass(slots=True, frozen=True)
class PerplexityResult:
    """Token-weighted cross-entropy totals for computing [`Perplexity`]."""

    cross_entropy_sum: float
    """The summed cross-entropy loss of the model's predictions over all accumulated batches.

    Only includes positions with a finite loss, so this is always finite.
    """
    token_count: int
    """The number of scored tokens (targets other than [`IGNORED_TARGET`]) over all accumulated batches.

    Includes the tokens counted by `non_finite_count`.

    [`IGNORED_TARGET`]: faultforge.metric.Perplexity.IGNORED_TARGET
    """
    non_finite_count: int
    """The number of scored tokens with a non-finite (`NaN` or `inf`) loss over all accumulated batches."""


@final
class Perplexity(Metric[PerplexityResult]):
    """Perplexity metric for evaluating LLMs

    Intuitively, perplexity quantifies how uncertain or "surprised" the model
    is when predicting the text. A perplexity of k means that the model is, on
    average, as uncertain as if it were choosing uniformly among k equally
    likely tokens. Lower perplexity therefore indicates better predictions. A
    perfect model, which always assigns probability 1 to the correct token, has
    a perplexity of 1.

    An LLM is trained to predict the next token given a sequence of preceding
    context tokens. At each position `i`, the model outputs a probability
    distribution over all tokens in its vocabulary. Because the reference text
    is known, the token that actually follows (`x_i`) is known as well. The
    quality of the prediction can therefore be measured by the cross-entropy
    between the predicted distribution and the correct token. Since the correct
    answer is a single token, the cross-entropy reduces to `-log p(x_i |
    x_{<i})`, the negative log-probability that the model assigns to the correct
    token given its context. The less probability the model assigns to the
    correct token, the higher this loss. Averaging the loss over all N tokens of
    the evaluation text and taking the exponential yields the perplexity.

    Batches are weighted by their number of scored tokens, not averaged per
    batch, so the score doesn't depend on how the text is split into batches.

    This metric is based on [Huggingface's perplexity calculation].

    # Inputs

    - `model_output` - logits of shape `(..., vocabulary)`, any floating point
      dtype.
    - `targets` - integer (`int64`) token ids with the shape of `model_output`
      minus the vocabulary dimension. Targets are expected to be **already
      shifted**: `targets[..., i]` is the token that should be predicted at
      position `i`, i.e. the token following input `i`. Positions with the
      value [`IGNORED_TARGET`] are ignored (padding, the final position,
      overlap of sliding windows, ...); all other positions are *scored*.

    # Non-finite logits

    Faults can produce non-finite logits. Only scored positions affect the
    result; at a scored position:

    - `NaN` or `+inf` anywhere in the position's logits make its loss `NaN`
      (log-softmax computes `inf - inf` for `+inf`).
    - `-inf` for the target token makes its loss `inf` (the correct token has
      probability 0).
    - `-inf` for other tokens is harmless (they just have probability 0).

    A `NaN` loss means the model's prediction carries no usable probability
    for the correct token, so it's treated the same as an `inf` loss: any
    position with a non-finite loss makes the score `inf`, the worst possible
    perplexity. Unlike `NaN`, `inf` still orders runs correctly, e.g. for
    medians or percentiles over many fault injection runs.

    Affected positions are counted in [`PerplexityResult.non_finite_count`]
    and left out of [`PerplexityResult.cross_entropy_sum`], so the result
    still records how many tokens a fault corrupted.

    Finite but huge losses whose mean overflows `exp` also make the score
    `inf`.

    Example uses:
    - decoder based large language models

    [`IGNORED_TARGET`]: Perplexity.IGNORED_TARGET
    [`PerplexityResult.non_finite_count`]: faultforge.metric.PerplexityResult.non_finite_count
    [`PerplexityResult.cross_entropy_sum`]: faultforge.metric.PerplexityResult.cross_entropy_sum
    [Huggingface's perplexity calculation]: https://huggingface.co/docs/transformers/perplexity
    """

    IGNORED_TARGET: ClassVar[int] = -100
    """Target value marking a position that isn't scored.

    Matches the default `ignore_index` of [`cross_entropy`] and the label
    convention of HuggingFace's transformers.

    [`cross_entropy`]: torch.nn.functional.cross_entropy
    """

    @override
    def evaluate_batch(
        self,
        batch_model_output: Tensor,
        batch_golden: Tensor,
        batch_targets: Tensor,
    ) -> PerplexityResult:
        _ = batch_golden

        if not batch_model_output.is_floating_point():
            raise ValueError(
                f"expected floating point logits, got {batch_model_output.dtype}"
            )
        if batch_targets.dtype != torch.int64:
            raise ValueError(
                f"expected int64 target token ids, got {batch_targets.dtype}"
            )

        vocabulary_size = batch_model_output.shape[-1]

        # Upcast so the summed loss doesn't lose precision or overflow: summing
        # in bf16 is off by several percent and fp16 overflows to inf once the
        # sum exceeds 65504 (a few tens of thousands of tokens).
        logits = batch_model_output.float()

        targets = batch_targets.reshape(-1)

        # literally: how different the probability distribution of the model
        # is compared to the ideal, for every token in the batch. Ignored
        # positions get a loss of 0.
        # Modeled after huggingface's `ForCausalLMLoss` when given pre-shifted
        # `shift_labels`:
        # https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py
        losses = nn.functional.cross_entropy(
            logits.reshape(-1, vocabulary_size),
            targets,
            ignore_index=Perplexity.IGNORED_TARGET,
            reduction="none",
        )

        scored = targets != Perplexity.IGNORED_TARGET
        finite = torch.isfinite(losses)

        # NOTE: summed in fp64 so the sum can't overflow, even for huge
        # finite per-token losses.
        cross_entropy_sum = losses[finite].double().sum()

        return PerplexityResult(
            cross_entropy_sum=cross_entropy_sum.item(),
            token_count=int(scored.sum().item()),
            non_finite_count=int((scored & ~finite).sum().item()),
        )

    @override
    def requires_golden(self) -> bool:
        return False

    @override
    def accumulate(
        self, existing: PerplexityResult, new: PerplexityResult
    ) -> PerplexityResult:
        return PerplexityResult(
            cross_entropy_sum=existing.cross_entropy_sum + new.cross_entropy_sum,
            token_count=existing.token_count + new.token_count,
            non_finite_count=existing.non_finite_count + new.non_finite_count,
        )

    @override
    def score(self, result: PerplexityResult) -> float:
        if result.token_count == 0:
            raise ValueError(
                "cannot compute perplexity: no scored tokens (all targets are "
                f"{Perplexity.IGNORED_TARGET})"
            )
        # See "Non-finite logits" in the docstring.
        if result.non_finite_count > 0:
            return math.inf
        try:
            return math.exp(result.cross_entropy_sum / result.token_count)
        except OverflowError:
            return math.inf

    @override
    def direction(self) -> ScoreDirection:
        return ScoreDirection.LowerIsBetter

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="perplexity")

    @override
    def display_name(self) -> str:
        return "Perplexity"

    @override
    def display_unit(self) -> str | None:
        return None
