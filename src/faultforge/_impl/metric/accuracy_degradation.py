"""See [`faultforge.metric`]"""

from dataclasses import dataclass
from typing import final, override

from torch import Tensor

from faultforge._impl.fingerprint import Fingerprint
from faultforge._impl.metric.base import Metric, ScoreDirection
from faultforge._impl.metric.helpers import ratio_to_percent, top1_logit_from_output


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
    def direction(self) -> ScoreDirection:
        return ScoreDirection.LowerIsBetter

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="accuracy_degredation")

    @override
    def display_name(self) -> str | None:
        return "Accuracy Degredation"

    @override
    def display_unit(self) -> str | None:
        return "%"
