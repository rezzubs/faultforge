"""An experiment for measuring model reliability under memory faults.

See `faultforge.experiments.encoded_memory` for a general overview.
"""

import copy
import enum
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import final, override

import torch
from pydantic import BaseModel
from torch import Tensor, nn

from faultforge._internal.common import DEFAULT_BATCH_SIZE, DEFAULT_DEVICE, DeviceLike
from faultforge._internal.dataset import BatchedDataset
from faultforge._internal.encoding.abc import Encoder
from faultforge._internal.encoding.nn import EncodedModule
from faultforge._internal.experiment import (
    Experiment,
    ExperimentDisplay,
)
from faultforge._internal.fault import BitFlip
from faultforge._internal.fingerprint import Fingerprint
from faultforge._internal.loading.abc import ModelBundle
from faultforge._internal.progress import Progress, stage
from faultforge._rust import Picker

logger = logging.getLogger(__name__)


class ReliabilityMetric(enum.StrEnum):
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


class _SavedData(BaseModel):
    """The on-disk shape of an `EncodedFaultInjection`'s results."""

    fingerprint: Fingerprint
    total_items: int | None
    results: list[int]


class _Display(ExperimentDisplay):
    """`EncodedFaultInjection`'s display: names/units the score per metric."""

    def __init__(self, metric: ReliabilityMetric) -> None:
        self._metric = metric

    @override
    def score_name(self) -> str | None:
        return self._metric.score_name()

    @override
    def score_unit(self) -> str | None:
        return "%"


@final
class EncodedFaultInjection(Experiment):
    """An experiment which emulates single-event upsets in the encoded memory that stores model parameters."""

    _model: EncodedModule
    _dataset: BatchedDataset
    _device: torch.device
    _reliability_metric: ReliabilityMetric
    _faulty_bit_count: int
    _progress: Progress | None
    _fingerprint: Fingerprint

    _unencoded_golden: nn.Module | None

    # populated during first run
    _golden_results: list[Tensor]
    _total_items: int | None
    _results: list[int]

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
        progress: Progress | None = None,
    ) -> None:
        self._progress = progress
        self._golden_results = []
        self._total_items = None
        self._results = []

        model = bundle.load_model(device, progress=progress)
        if golden_is_encoded:
            self._unencoded_golden = None
        else:
            self._unencoded_golden = copy.deepcopy(model)

        self._model = EncodedModule(model, encoder, progress=progress)
        self._device = torch.device(device)
        self._reliability_metric = reliability_metric

        self._dataset = bundle.load_dataset(batch_size, device, progress=progress)
        if dataset_batch_limit is not None and not preload_dataset:
            logger.warning(
                "preload_dataset is set to False but dataset_limit forces a preload anyway"
            )
            preload_dataset = True
        if preload_dataset:
            self._dataset = self._dataset.precompute(
                dataset_batch_limit, progress=progress
            )

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

        self._fingerprint = fingerprint

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
        """Populate the golden results.

        Additionally sets `_total_items` to the total number of predictions;
        this is used for computing SDC scores as well as the number of
        injected faults.
        """
        total_items = 0

        golden: nn.Module = self._unencoded_golden or self._model

        try:
            with stage(
                self._progress,
                "Computing golden results",
                total=self._dataset.batch_count(),
            ) as s:
                for batch in self._dataset:
                    logits = golden.forward(batch.inputs)
                    processed = self._process_golden(logits)
                    total_items += processed.numel()
                    self._golden_results.append(processed)
                    s.advance()
        finally:
            self._dataset.reset()

        if self._total_items is None:
            self._total_items = total_items
        else:
            assert self._total_items == total_items, (
                "_total_items mismatch vs previous run"
            )

    def _score(self, correct: int) -> float:
        if self._total_items is None:
            raise RuntimeError("Unable to score a result before the first run")

        match self._reliability_metric:
            case ReliabilityMetric.Sdc | ReliabilityMetric.Top1Sdc:
                return 100 - float(correct) / float(self._total_items) * 100
            case ReliabilityMetric.Accuracy | ReliabilityMetric.AccuracyDegradation:
                return float(correct) / float(self._total_items) * 100

    @override
    def scores(self) -> Sequence[float]:
        if self._total_items is None:
            return []
        return [self._score(correct) for correct in self._results]

    @override
    def display(self) -> ExperimentDisplay:
        return _Display(self._reliability_metric)

    @override
    def serialize(self) -> str:
        return _SavedData(
            fingerprint=self._fingerprint,
            total_items=self._total_items,
            results=self._results,
        ).model_dump_json()

    @override
    def deserialize(self, content: str) -> None:
        loaded = _SavedData.model_validate_json(content)
        self._fingerprint.raise_if_differs(loaded.fingerprint)
        self._total_items = loaded.total_items
        self._results = loaded.results

    @override
    def run(self) -> None:
        if not self._golden_results and self._reliability_metric.requires_golden():
            self._populate_golden()

        picker = Picker(self._model.bit_count())
        model = self._model.clone()
        with stage(self._progress, "Fault Injection"):
            fault_targets: list[tuple[BitFlip, int]] = []
            for _ in range(self._faulty_bit_count):
                try:
                    fault_target = next(picker)
                except StopIteration:
                    raise RuntimeError(
                        "Expected fault targets to be within range but picker is exhausted"
                    )
                fault_targets.append((BitFlip(), fault_target))

            model.apply_faults(fault_targets)

        result = BatchReliability(correct=0, total=0)
        with stage(self._progress, "Inference", total=self._dataset.batch_count()) as s:
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
                        batch_result = _batch_sdc(
                            logits, self._golden_results[batch_index]
                        )
                    case ReliabilityMetric.Top1Sdc:
                        batch_result = _batch_critical_sdc(
                            logits, self._golden_results[batch_index]
                        )

                result += batch_result
                s.advance()

        self._dataset.reset()

        if self._total_items is None:
            self._total_items = result.total
            assert not self._reliability_metric.requires_golden(), (
                "_total_items should be set by _populate_golden"
            )

        if result.total != self._total_items:
            raise RuntimeError(
                f"Computed {self._total_items} elements from the golden results, "
                f"model returned {result.total}"
            )

        self._results.append(result.correct)


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

    return BatchReliability(correct=golden_correct - correct, total=total)


def _batch_accuracy(logits: Tensor, targets: Tensor) -> BatchReliability:
    """Compute the accuracy of a result. Used for ReliabilityMetric.Accuracy."""
    classifications = logits.argmax(dim=1)

    correct = int((classifications == targets).sum().item())
    total = classifications.numel()

    return BatchReliability(correct=correct, total=total)
