"""See [`faultforge.metric`]"""

from dataclasses import dataclass
from typing import final, override

from torch import Tensor

from faultforge._impl.fingerprint import Fingerprint
from faultforge._impl.metric.base import Metric, ScoreDirection
from faultforge._impl.metric.helpers import ratio_to_percent, top1_logit_from_output


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
    def direction(self) -> ScoreDirection:
        return ScoreDirection.HigherIsBetter

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="accuracy")

    @override
    def display_name(self) -> str | None:
        return "Accuracy"

    @override
    def display_unit(self) -> str | None:
        return "%"
