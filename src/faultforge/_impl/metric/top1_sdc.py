"""See [`faultforge.metric`]"""

from typing import final, override

from torch import Tensor

from faultforge._impl.fingerprint import Fingerprint
from faultforge._impl.metric.base import Metric, ScoreDirection
from faultforge._impl.metric.helpers import top1_logit_from_output
from faultforge._impl.metric.sdc import Sdc, SdcResult


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
    def direction(self) -> ScoreDirection:
        return ScoreDirection.LowerIsBetter

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="top1-sdc")

    @override
    def display_name(self) -> str | None:
        return "Top-1 SDC"

    @override
    def display_unit(self) -> str | None:
        return "%"
