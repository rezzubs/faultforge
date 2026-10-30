"""See [`faultforge.metric`]"""

from dataclasses import dataclass
from typing import final, override

from torch import Tensor

from faultforge._impl.fingerprint import Fingerprint
from faultforge._impl.metric.base import Metric, ScoreDirection
from faultforge._impl.metric.helpers import ratio_to_percent


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
    def direction(self) -> ScoreDirection:
        return ScoreDirection.LowerIsBetter

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="sdc")

    @override
    def display_name(self) -> str | None:
        return "SDC"

    @override
    def display_unit(self) -> str | None:
        return "%"
