"""An experiment for measuring model reliability under memory faults.

See `faultforge.experiments.encoded_memory` for a general overview.
"""

import copy
import enum
import logging
from dataclasses import dataclass
from typing import final, override

import torch
from torch import Tensor, nn

from faultforge._internal.common import DEFAULT_BATCH_SIZE, DEFAULT_DEVICE, DeviceLike
from faultforge._internal.dataset import BatchedDataset
from faultforge._internal.encoding.abc import Encoder
from faultforge._internal.encoding.nn import EncodedModule
from faultforge._internal.experiment import (
    Data,
    DisplayConfig,
    Experiment,
)
from faultforge._internal.fault import BitFlip
from faultforge._internal.fingerprint import Fingerprint
from faultforge._internal.loading.abc import ModelBundle
from faultforge._rust import Picker

logger = logging.getLogger(__name__)


class ReliabilityMetric(enum.Enum):
    """Ways to measure the reliability of a fault-injected model."""

    Accuracy = "accuracy"
    """Correct predictions / total predictions.

    Whether or not a prediction is "correct" is defined by the dataset's
    ground-truth labels (targets).
    """
    AccuracyDegradation = "accuracy_degradation"
    """Golden model accuracy - faulty model accuracy."""
    Sdc = "sdc"
    """Silent Data Corruption. Change in any output logit vs the golden model."""
    Top1Sdc = "top1_sdc"
    """Critical Silent Data Corruption. Change in top-1 logit (the prediction) vs the golden model."""

    def requires_golden(self) -> bool:
        """Whether this metric also requires evaluating results on a golden model."""

        return self.value in {
            ReliabilityMetric.AccuracyDegradation.value,
            ReliabilityMetric.Sdc.value,
            ReliabilityMetric.Top1Sdc.value,
        }

    def score_name(self) -> str:
        """The name of the score for this metric."""
        match self:
            case ReliabilityMetric.Accuracy:
                return "Accuracy"
            case ReliabilityMetric.AccuracyDegradation:
                return "Accuracy Degradation"
            case ReliabilityMetric.Sdc:
                return "SDC"
            case ReliabilityMetric.Top1Sdc:
                return "Top-1 SDC"


@final
@dataclass(slots=True)
class BatchReliability:
    correct: int
    """Number of correct results as defined by the metric."""
    total: int
    """Total number of "items" in the batch. Metric dependent."""

    def __add__(self, other: BatchReliability) -> BatchReliability:
        return BatchReliability(
            correct=self.correct + other.correct, total=self.total + other.total
        )

    def __sub__(self, other: BatchReliability) -> BatchReliability:
        return BatchReliability(
            correct=self.correct - other.correct, total=self.total - other.total
        )


@final
class EncodedFaultInjection(Experiment[int, int | None]):
    """An experiment which emulates single-event upsets in the encoded memory that stores model parameters.

    `data.context` stores the total number of items processed (depending on the
    chosen metric); this is available after the first run.
    """

    _model: EncodedModule
    _dataset: BatchedDataset
    _device: torch.device
    _reliability_metric: ReliabilityMetric
    _faulty_bit_count: int

    _unencoded_golden: nn.Module | None

    # populated during first run
    _golden_results: list[Tensor] = []

    def __init__(
        self,
        bundle: ModelBundle,
        encoder: Encoder,
        reliability_metric: ReliabilityMetric,
        *,
        golden_is_encoded: bool = False,
        faults: int | float = 1,
        preload_dataset: bool = True,
        dataset_batch_limit: int | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: DeviceLike = DEFAULT_DEVICE,
    ) -> None:
        model = bundle.load_model(device)
        if golden_is_encoded:
            self._unencoded_golden = copy.deepcopy(model)
        else:
            self._unencoded_golden = None

        self._model = EncodedModule(model, encoder)
        self._device = torch.device(device)
        self._reliability_metric = reliability_metric

        self._dataset = bundle.load_dataset(batch_size, device)
        if dataset_batch_limit is not None and not preload_dataset:
            logger.warning(
                "preload_dataset is set to False but dataset_limit forces a preload anyway"
            )
            preload_dataset = True
        if preload_dataset:
            self._dataset = self._dataset.precompute(dataset_batch_limit)

        fingerprint = Fingerprint(
            kind="encoded_memory_fault_injection",
            scalars={
                "reliability_metric": reliability_metric.value,
                "golden": "encoded" if golden_is_encoded else "unencoded",
            },
            children={
                "bundle": [bundle.fingerprint()],
                "encoder": [encoder.fingerprint()],
            },
        )

        test_image_limit = (
            dataset_batch_limit * batch_size
            if dataset_batch_limit is not None
            else None
        )
        if test_image_limit is not None:
            fingerprint.scalars["test_image_limit"] = test_image_limit

        if isinstance(faults, int):
            if faults > self._model.bit_count():
                raise ValueError(
                    f"`faults` ({faults}) is greater than the number of bits in model parameters ({self._model.bit_count()})"
                )

            fingerprint.scalars["faults"] = faults
            self._faulty_bit_count = faults
        elif isinstance(faults, float):
            if faults > 1.0:
                raise ValueError(
                    f"`faults` ({faults}) is greater than 1.0 (floats are interpreted as the bit error rate)"
                )

            fingerprint.scalars["bit_error_rate"] = faults
            self._faulty_bit_count = int(round(faults * self._model.bit_count()))
            logger.debug(
                f"Resolved bit error rate {faults} to {self._faulty_bit_count} faults"
            )

        super().__init__(
            data=Data[int, int | None](
                fingerprint=fingerprint,
                context=None,
                results={},
            ),
            display=DisplayConfig(
                score_name=reliability_metric.score_name(),
                score_unit="%",
            ),
        )

    def _process_golden(self, golden_result: Tensor) -> Tensor:
        """Run a function on the golden result after computing it.

        This enables processing the results only once. The result will be given to
        the batch reliability functions.
        """
        match self._reliability_metric:
            case (
                ReliabilityMetric.Top1Sdc
                | ReliabilityMetric.Accuracy
                | ReliabilityMetric.AccuracyDegradation
            ):
                return golden_result.argmax(dim=1)
            case ReliabilityMetric.Sdc:
                return golden_result

    def _populate_golden(self):
        """Populate the golden results for SDC mode.

        Additionally sets context to the total number of predictions; this is
        used for computing SDC scores as well as the number of injected faults.
        """
        logger.debug("Computing golden results")

        total_items = 0

        golden: nn.Module = self._unencoded_golden or self._model

        try:
            for batch in self._dataset:
                logits = golden.forward(batch.inputs)
                processed = self._process_golden(logits)
                total_items += processed.numel()
                self._golden_results.append(processed)
        finally:
            self._dataset.reset()

        if self.data.context is None:
            self.data.context = total_items
        else:
            assert self.data.context == total_items, (
                "data.context mismatch vs previous run"
            )

    @override
    def result_score(self, result: int) -> float:
        if self.data.context is None:
            raise RuntimeError("Unable to score a result before the first run")

        match self._reliability_metric:
            case ReliabilityMetric.Sdc | ReliabilityMetric.Top1Sdc:
                return 100 - float(result) / float(self.data.context) * 100
            case ReliabilityMetric.Accuracy | ReliabilityMetric.AccuracyDegradation:
                return float(result) / float(self.data.context) * 100

    @override
    def latest_result(self) -> int | None:
        if not self.data.results:
            return None
        return len(self.data.results) - 1

    @override
    def run(self) -> None:
        if not self._golden_results and self._reliability_metric.requires_golden():
            self._populate_golden()

        picker = Picker(self._model.bit_count())
        model = self._model.clone()
        for _ in range(self._faulty_bit_count):
            try:
                fault_target = next(picker)
            except StopIteration:
                raise RuntimeError(
                    "Expected fault targets to be within range but picker is exhausted"
                )

            model.apply_fault(BitFlip(), fault_target)

        result = BatchReliability(correct=0, total=0)
        for batch_index, batch in enumerate(self._dataset):
            # n_batches x n_classes
            logits = model.forward(batch.inputs)

            match self._reliability_metric:
                case ReliabilityMetric.Accuracy:
                    batch_result = _batch_accuracy(logits, batch.targets)
                case ReliabilityMetric.AccuracyDegradation:
                    batch_result = _batch_accuracy_degradation(
                        logits, self._golden_results[batch_index], batch.targets
                    )
                case ReliabilityMetric.Sdc:
                    batch_result = _batch_sdc(logits, self._golden_results[batch_index])
                case ReliabilityMetric.Top1Sdc:
                    batch_result = _batch_critical_sdc(
                        logits, self._golden_results[batch_index]
                    )

            result += batch_result

        self._dataset.reset()

        if self.data.context is None:
            self.data.context = result.total
            assert not self._reliability_metric.requires_golden(), (
                "context should be set by populate_golden"
            )

        if result.total != self.data.context:
            raise RuntimeError(
                "Computed {} elements from the golden results, model returned {}",
                self.data.context,
                result.total,
            )

        self.data.results[len(self.data.results)] = result.correct


def _batch_critical_sdc(
    logits: Tensor, golden_classifications: Tensor
) -> BatchReliability:
    """Compute the critical SDC of a result. Used for ReliabilityMetric.CriticalSdc."""

    classifications = logits.argmax(dim=1)

    assert golden_classifications.shape == classifications.shape
    # bool is a subclass of int, so sum converts bools to ints.
    correct = int((classifications == golden_classifications).sum().item())
    total = golden_classifications.numel()

    return BatchReliability(correct=correct, total=total)


def _batch_sdc(logits: Tensor, golden_logits: Tensor) -> BatchReliability:
    """Compute the SDC of a result. Used for ReliabilityMetric.Sdc."""

    assert golden_logits.shape == logits.shape
    # bool is a subclass of int, so sum converts bools to ints.
    correct = int((logits == golden_logits).sum().item())
    total = golden_logits.numel()

    return BatchReliability(correct=correct, total=total)


def _batch_accuracy_degradation(
    logits: Tensor,
    golden_classifications: Tensor,
    targets: Tensor,
) -> BatchReliability:
    """Compute the accuracy degradation of a result. Used for ReliabilityMetric.AccuracyDegradation."""

    classifications = logits.argmax(dim=1)
    assert golden_classifications.shape == classifications.shape

    correct = int((classifications == targets).sum().item())
    golden_correct = int((golden_classifications == targets).sum().item())
    total = classifications.numel()

    run_accuracy = BatchReliability(correct=correct, total=total)
    golden_accuracy = BatchReliability(correct=golden_correct, total=total)

    return golden_accuracy - run_accuracy


def _batch_accuracy(logits: Tensor, targets: Tensor) -> BatchReliability:
    """Compute the accuracy of a result. Used for ReliabilityMetric.Accuracy."""
    classifications = logits.argmax(dim=1)

    correct = int((classifications == targets).sum().item())
    total = classifications.numel()

    return BatchReliability(correct=correct, total=total)
