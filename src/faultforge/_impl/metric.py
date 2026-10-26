"""See [`faultforge.metric`]"""

import abc
import math
from collections.abc import Generator
from dataclasses import dataclass
from typing import ClassVar, final, override

import torch
from torch import Tensor, nn

from faultforge._impl.dataset import (
    BatchedDataset,
    DataBatch,
)
from faultforge._impl.experiment.config import ScoreDisplay
from faultforge._impl.fingerprint import Fingerprint
from faultforge._impl.progress import Progress, stage


class Metric[R](ScoreDisplay, abc.ABC):
    """A metric for evaluating the output of a DNN.

    In the end all metrics need to produce a numeric [`score`] but can use any
    intermediate representation (the generic type `R`) which is later turned
    into a scalar float by [`score`].

    Metric data (the result type `R`) is created by [`evaluate_batch`] which
    is expected to be called on every iteration over the [batches] of the
    [dataset]. [`evaluate_batch`] has three parameters:
    - `model_output` - The output of a fault-injected [`Module`] for a single
      batch of data.
    - `golden` - The output of a non-faulty [`Module`] for the same batch
      of data, passed through [`preprocess_golden`]. See the [Golden
      outputs](#golden-outputs) section below.
    - `targets` - the `targets` field from the current [`DataBatch`]. The
      expected output of the model as defined by the [dataset].

    [`evaluate_batch`] processes a result for just one batch. The final metric
    is produced by repeated runs of [`accumulate`] which takes two instances of
    the result (`R`) as arguments. The caller is expected to feed the combined
    result as the `existing` argument of [`accumulate`] and the new result as
    the [`new`] argument.

    # Golden outputs

    Computing `golden` may not be required for all metrics so the
    [`requies_golden`] method can be used to signal whether golden outputs
    should be computed. Metrics which don't need golden outputs can ignore the
    `golden` parameter on [`evaluate_batch`].

    In some cases you may want to run a processing step on the golden outputs
    which is the same for every run. In this case the metric can override
    [`preprocess_golden`] to define that processing step. This enables the
    caller to cache the processed golden values so they don't have to be
    recomputed every time [`evaluate_batch`] is called. The caller is always
    expected to pass the golden outputs through [`preprocess_golden`], the
    default implementation just returns the values without altering them.

    The [`GoldenCache`] type exists to simplify common workflows with golden
    outputs.

    [`accumulate`]: Metric.accumulate
    [`evaluate_batch`]: Metric.evaluate_batch
    [`preprocess_golden`]: Metric.preprocess_golden
    [`requires_golden`]: Metric.requires_golden
    [`score`]: Metric.score

    [`DataBatch`]: faultforge.dataset.DataBatch
    [`Module`]: torch.nn.Module
    [batches]: faultforge.dataset.DataBatch
    [dataset]: faultforge.dataset.BatchedDataset
    """

    @abc.abstractmethod
    def evaluate_batch(
        self,
        batch_model_output: Tensor,
        batch_golden: Tensor,
        batch_targets: Tensor,
    ) -> R:
        """Create result data for this metric from model output for one dataset batch.

        See the docstring on [`Metric`] for details on the parameters.
        """

    def preprocess_golden(self, golden: Tensor) -> Tensor:
        """Process the golden outputs of the model before they are passed to [`evaluate_batch`].

        See [Metric#golden-outputs] for details.

        [`evaluate_batch`]: Metric.evaluate_batch
        """
        return golden

    @abc.abstractmethod
    def requires_golden(self) -> bool:
        """Whether `evaluate_batch` requires golden inputs.

        The return value of this function assumed to be constant for any
        specific `Metric`.

        This is a signal for the user of the metric of whether to compute golden
        outputs. If this function returns false then the caller may pass an
        empty tensor to [`evaluate_batch`] because it's implied that it will be
        ignored anyway.

        [`evaluate_batch`]: Metric.evaluate_batch
        """

    @abc.abstractmethod
    def accumulate(self, existing: R, new: R) -> R:
        """Combine the result of the current iteration with existing ones.

        - `existing` is the combined result of the already completed iterations
        - `new` is the result of the current iterations
        """

    @abc.abstractmethod
    def score(self, result: R) -> float:
        """Give a numeric "score" for the result."""

    @abc.abstractmethod
    def fingerprint(self) -> Fingerprint:
        """Return a structural identity for the metric"""

    def init_or_accumulate(self, existing: R | None, new: R) -> R:
        """Helper for `accumulate` given an initial non-existing result."""
        if existing is None:
            return new
        else:
            return self.accumulate(existing, new)

    def evaluate_and_accumulate(
        self,
        batch_model_output: Tensor,
        batch_golden: Tensor,
        batch_targets: Tensor,
        existing_result: R | None,
    ) -> R:
        """Helper for running `evaluate_batch` and `init_or_accumulate` in sequence."""
        batch_result = self.evaluate_batch(
            batch_model_output, batch_golden, batch_targets
        )

        return self.init_or_accumulate(existing_result, batch_result)


@final
class GoldenCache[R]:
    """An iterable over a [`BatchedDataset`] that additionally yields an inference result for a golden model.

    Golden outputs are computed on the first iteration and reused by every
    iteration after that.

    [`BatchedDataset`]: faultforge.dataset.BatchedDataset
    """

    def __init__(
        self,
        metric: Metric[R],
        golden_model: nn.Module,
        dataset: BatchedDataset,
        *,
        progress: Progress | None,
    ) -> None:
        self._golden_outputs: list[Tensor] = []

        self._golden_model: nn.Module = golden_model
        self._metric: Metric[R] = metric
        self._dataset: BatchedDataset = dataset
        self._progress: Progress | None = progress

    def __iter__(self) -> Generator[tuple[Tensor, DataBatch]]:
        for index, data_batch in enumerate(self._dataset):
            golden_output = self._get_or_compute(index, data_batch)

            yield (golden_output, data_batch)

    def _get_or_compute(self, index: int, data_batch: DataBatch) -> Tensor:
        if not self._metric.requires_golden():
            return torch.empty(0)

        if index < len(self._golden_outputs):
            return self._golden_outputs[index]

        # NOTE: this function is only ever called for sequential values of
        # `index` so pushing a new value is always correct. We assert the
        # invariant here.
        assert index == len(self._golden_outputs)

        with stage(self._progress, f"Golden inference for batch {index}"):
            raw_output = self._golden_model.forward(data_batch.inputs)

        if not isinstance(raw_output, Tensor):
            raise TypeError("Expected model to return a tensor")

        processed = self._metric.preprocess_golden(raw_output)

        self._golden_outputs.append(processed)

        return processed

    def batch_count(self) -> int | None:
        """Return the batch count of the underlying dataset, if known."""
        return self._dataset.batch_count()


@final
@dataclass(frozen=True, slots=True)
class AccuracyResult:
    """A result type for metrics which count correct runs over the total number of iterations."""

    correct_count: int
    """Number of times the top-1 prediction matched `targets`."""
    total_count: int
    """Total number of predictions made."""


@final
class Accuracy(Metric[AccuracyResult]):
    """Percentage of top-1 predictions vs dataset targets over all iterations.

    Top-1 in this case means that we take the `argmax` over a  model's
    output logits.

    Example uses:
    - image classification models
    - encoder based transformers.
    """

    @override
    def evaluate_batch(
        self,
        batch_model_output: Tensor,
        batch_golden: Tensor,
        batch_targets: Tensor,
    ) -> AccuracyResult:
        _ = batch_golden

        top1_logit = top1_logit_from_output(batch_model_output)

        # NOTE: sum implicitly converts bools to ints.
        correct_count = (top1_logit == batch_targets).sum().item()
        assert isinstance(correct_count, int)

        return AccuracyResult(
            correct_count=correct_count,
            total_count=top1_logit.numel(),
        )

    @override
    def requires_golden(self) -> bool:
        return False

    @override
    def accumulate(
        self, existing: AccuracyResult, new: AccuracyResult
    ) -> AccuracyResult:
        return AccuracyResult(
            correct_count=existing.correct_count + new.correct_count,
            total_count=existing.total_count + new.total_count,
        )

    @override
    def score(self, result: AccuracyResult) -> float:
        return ratio_to_percent(result.correct_count / result.total_count)

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="accuracy")

    @override
    def display_name(self) -> str | None:
        return "Accuracy"

    @override
    def display_unit(self) -> str | None:
        return "%"


@final
@dataclass(frozen=True, slots=True)
class AccuracyDegradationResult:
    """The difference of accuracy between faulty and golden runs.

    Positive if the accuracy dropped due to fault injection.
    """

    correct_count_faulty: int
    """Number of times the top-1 prediction matched `targets` for the faulty run."""
    correct_count_golden: int
    """Number of times the top-1 prediction matched `targets` for the non-faulty (golden) run."""
    total_count: int
    """Total number of predictions made."""


@final
class AccuracyDegradation(Metric[AccuracyDegradationResult]):
    """The difference of [`Accuracy`] between faulty and golden runs.

    Positive if the accuracy dropped due to fault injection.

    Example uses:
    - image classification models
    - encoder based transformers.
    """

    @override
    def evaluate_batch(
        self,
        batch_model_output: Tensor,
        batch_golden: Tensor,
        batch_targets: Tensor,
    ) -> AccuracyDegradationResult:
        _ = batch_golden

        top1_logit = top1_logit_from_output(batch_model_output)

        # NOTE: sum implicitly converts bools to ints.
        correct_count_faulty = (top1_logit == batch_targets).sum().item()
        assert isinstance(correct_count_faulty, int)

        # NOTE: golden is preprocessed
        correct_count_golden = (batch_golden == batch_targets).sum().item()
        assert isinstance(correct_count_golden, int)

        return AccuracyDegradationResult(
            correct_count_faulty=correct_count_faulty,
            correct_count_golden=correct_count_golden,
            total_count=top1_logit.numel(),
        )

    @override
    def preprocess_golden(self, golden: Tensor) -> Tensor:
        return top1_logit_from_output(golden)

    @override
    def requires_golden(self) -> bool:
        return True

    @override
    def accumulate(
        self, existing: AccuracyDegradationResult, new: AccuracyDegradationResult
    ) -> AccuracyDegradationResult:
        return AccuracyDegradationResult(
            correct_count_faulty=existing.correct_count_faulty
            + new.correct_count_faulty,
            correct_count_golden=existing.correct_count_golden
            + new.correct_count_golden,
            total_count=existing.total_count + new.total_count,
        )

    @override
    def score(self, result: AccuracyDegradationResult) -> float:
        return ratio_to_percent(
            (result.correct_count_golden - result.correct_count_faulty)
            / result.total_count
        )

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="accuracy_degredation")

    @override
    def display_name(self) -> str | None:
        return "Accuracy Degredation"

    @override
    def display_unit(self) -> str | None:
        return "%"


@final
@dataclass(slots=True, frozen=True)
class SdcResult:
    non_matching_count: int
    """Number of matched elements"""
    total_count: int
    """Number of total elements"""


@final
class Sdc(Metric[SdcResult]):
    """Silent Data Corruption.

    Percentage of output values that have changed in the case of fault injection
    vs the golden outputs. Does not care about what is actually predicted
    (unlike Top1Sdc). Works with any model output shape.

    Example uses:
    - basically any model output.
    """

    @override
    def evaluate_batch(
        self,
        batch_model_output: Tensor,
        batch_golden: Tensor,
        batch_targets: Tensor,
    ) -> SdcResult:
        _ = batch_targets

        if batch_model_output.shape != batch_golden.shape:
            raise ValueError(
                f"Shape mismatch between shape of model_output ({batch_model_output.shape}) and golden ({batch_golden.shape})"
            )

        # NOTE: sum implicitly converts bools to ints.
        non_matching_count = (batch_model_output != batch_golden).sum().item()
        assert isinstance(non_matching_count, int)

        return SdcResult(
            non_matching_count=non_matching_count,
            total_count=batch_model_output.numel(),
        )

    @override
    def requires_golden(self) -> bool:
        return True

    @override
    def accumulate(self, existing: SdcResult, new: SdcResult) -> SdcResult:
        return SdcResult(
            non_matching_count=existing.non_matching_count + new.non_matching_count,
            total_count=existing.total_count + new.total_count,
        )

    @override
    def score(self, result: SdcResult) -> float:
        return ratio_to_percent(result.non_matching_count / result.total_count)

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="sdc")

    @override
    def display_name(self) -> str | None:
        return "SDC"

    @override
    def display_unit(self) -> str | None:
        return "%"


@final
class Top1Sdc(Metric[SdcResult]):
    """Silent Data Corruption for the prediction (top-1 logit).

    Percentage of output values that have changed in the case of fault injection
    vs the golden outputs. Only compares the actual prediction (`argmax` over
    the logits). See [`Sdc`] for the general version.

    Example uses:
    - image classification models
    - encoder based transformers.
    """

    @override
    def evaluate_batch(
        self,
        batch_model_output: Tensor,
        batch_golden: Tensor,
        batch_targets: Tensor,
    ) -> SdcResult:
        top1_logits = top1_logit_from_output(batch_model_output)

        return Sdc().evaluate_batch(top1_logits, batch_golden, batch_targets)

    @override
    def preprocess_golden(self, golden: Tensor) -> Tensor:
        return top1_logit_from_output(golden)

    @override
    def requires_golden(self) -> bool:
        return Sdc().requires_golden()

    @override
    def accumulate(self, existing: SdcResult, new: SdcResult) -> SdcResult:
        return Sdc().accumulate(existing, new)

    @override
    def score(self, result: SdcResult) -> float:
        return Sdc().score(result)

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="top1-sdc")

    @override
    def display_name(self) -> str | None:
        return "Top-1 SDC"

    @override
    def display_unit(self) -> str | None:
        return "%"


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
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="perplexity")

    @override
    def display_name(self) -> str:
        return "Perplexity"

    @override
    def display_unit(self) -> str | None:
        return None


def ratio_to_percent(ratio: float) -> float:
    return ratio * 100


def top1_logit_from_output(batch_model_output: Tensor) -> Tensor:
    """Extract the top-1 logit from the model output (batched).

    Input n-d, output (n-1)-d
    """
    if len(batch_model_output.shape) < 2:
        raise ValueError(
            f"Logit extraction expects >=2 dimensional tensors, got shape {batch_model_output.shape}"
        )

    return batch_model_output.argmax(dim=1)
