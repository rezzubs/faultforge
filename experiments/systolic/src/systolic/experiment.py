"""An experiment for measuring model reliability under systolic-array register faults.

See `systolic` for a general overview.
"""

lazy import copy
lazy import logging
lazy from collections.abc import Sequence
lazy from pathlib import Path
lazy from typing import Literal, final, override

lazy import torch
lazy from pydantic import BaseModel
lazy from torch import nn

lazy from faultforge import Fingerprint, Picker, tensor_list_dtype
lazy from faultforge.dataset import DEFAULT_BATCH_SIZE, DEFAULT_DEVICE, DeviceLike
lazy from faultforge.experiment import (
    Experiment,
    ExperimentDisplay,
    RecordedRuns,
    ScoreDisplay,
    StopCondition,
)
lazy from faultforge.io import AnyPath, is_compressed, open_text
lazy from faultforge.loading import ModelBundle
lazy from faultforge.metric import GoldenCache, Metric
lazy from faultforge.progress import Progress, stage
lazy from systolic._rust import ArrayConfig, Fault
lazy from systolic.backend import SystolicBackend
lazy from systolic.fault import RegisterFaults
lazy from systolic.lifted_backend import LiftedBackend
lazy from systolic.model import BackendModel
lazy from systolic.simulated_backend import SimulatedBackend

logger = logging.getLogger(__name__)

BackendKind = Literal["simulated", "lifted"]


class RunResult(BaseModel):
    """A single run's drawn fault and score."""

    fault_id: int
    """The id of the fault active during the run, see `RegisterFaults.fault_from_id`."""
    score: float
    """The `score` as determined by the `Metric`."""


class SavedResult(BaseModel):
    """The on-disk shape of a `SystolicFaultInjection`'s results.

    Standalone-loadable: everything needed to inspect the scores and resume
    the fault `Picker` lives here, so a result file can be inspected without
    reconstructing the model/dataset that produced it.
    """

    fingerprint: Fingerprint
    runs: list[RunResult]
    """One entry per run, in run order."""
    metric_display_name: str
    """The display name of the metric, used for plotting."""
    golden_score: float | None = None
    """The metric's score for the fault-free golden model.

    Only computed when an estimator needs it (see
    `SystolicFaultInjection.golden_score`), so it's `None` otherwise.
    """

    @classmethod
    def load(cls, path: AnyPath) -> SavedResult:
        """Load a single result file previously written by
        `Experiment.save`/`save_atomic`."""
        path = Path(path).expanduser()
        with open_text(path, "rt", compressed=is_compressed(path)) as f:
            return cls.model_validate_json(f.read())

    def scores(self) -> list[float]:
        """Every recorded run's score, in run order."""
        return [run.score for run in self.runs]


class SystolicFaultInjectionDisplay(ExperimentDisplay):
    """`SystolicFaultInjection`'s display: names and units the score per metric."""

    def __init__(self, metric: Metric) -> None:
        self._metric = metric

    @override
    def score_display(self) -> ScoreDisplay:
        return self._metric


@final
class SystolicFaultInjection[R](Experiment):
    """An experiment which injects a single stuck-at register fault per run
    into a weight-stationary systolic array evaluating the model.

    One fault is active per run (not several simultaneous faults, unlike
    `encoded_memory`'s `EncodedFaultInjection`): each run draws one fault id
    from a `Picker` over `fault.radix(array)`, applies it via
    `backend.set_fault`, and scores one inference pass over the dataset. How
    many runs happen is controlled entirely by stop conditions.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        array: tuple[int, int],
        reliability_metric: Metric[R],
        *,
        fault: RegisterFaults | None = None,
        backend: BackendKind = "lifted",
        preload_dataset: bool = True,
        dataset_batch_limit: int | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: DeviceLike = DEFAULT_DEVICE,
        progress: Progress | None = None,
    ) -> None:
        self._progress = progress
        self._runs: list[RunResult] = []
        self._golden_score: float | None = None
        """Computed lazily, see `golden_score`."""
        self._golden_score_unsaved = False
        """Whether `_golden_score` was computed since the last save."""
        self._reliability_metric = reliability_metric
        self._fault_config = fault if fault is not None else RegisterFaults()

        nrows, ncols = array
        self._array = ArrayConfig(nrows, ncols, 32)

        model = bundle.load_model(device, progress=progress)
        dtype = tensor_list_dtype(list(model.parameters()))
        if dtype != torch.float32:
            raise ValueError(
                f"SystolicFaultInjection only supports float32, got {dtype}"
            )
        self._golden_model = copy.deepcopy(model)

        backend_instance: SystolicBackend
        match backend:
            case "simulated":
                backend_instance = SimulatedBackend(nrows, ncols)
            case "lifted":
                backend_instance = LiftedBackend(nrows, ncols)
        self._backend = backend_instance
        self._model = BackendModel(model, self._backend, progress=progress)

        dataset = bundle.load_dataset(batch_size, device, progress=progress)
        if dataset_batch_limit is not None and not preload_dataset:
            logger.warning(
                "preload_dataset is set to False but dataset_limit forces a preload anyway"
            )
            preload_dataset = True
        if preload_dataset:
            dataset = dataset.precompute(dataset_batch_limit, progress=progress)

        self._golden_cache: GoldenCache = GoldenCache(
            metric=reliability_metric,
            golden_model=self._golden_model,
            dataset=dataset,
            progress=progress,
        )

        fingerprint = Fingerprint(
            kind="systolic_fault_injection",
            scalars={
                "backend": backend,
                "array_nrows": nrows,
                "array_ncols": ncols,
                "registers": self._fault_config.fingerprint_scalar(),
            },
            children={
                "reliability_metric": [reliability_metric.fingerprint()],
                "bundle": [bundle.fingerprint()],
            },
        )
        sample_limit = (
            dataset_batch_limit * batch_size
            if dataset_batch_limit is not None
            else None
        )
        if sample_limit is not None:
            fingerprint.scalars["sample_limit"] = sample_limit
        self._fingerprint = fingerprint

        self._picker = Picker(self._fault_config.radix(self._array))

    @override
    def scores(self) -> Sequence[float]:
        return [run.score for run in self._runs]

    @override
    def display(self) -> ExperimentDisplay:
        return SystolicFaultInjectionDisplay(self._reliability_metric)

    @override
    def stop_conditions(self) -> Sequence[StopCondition]:
        def picker_exhausted(runs: RecordedRuns) -> str | None:
            _ = runs
            if self._picker.size == 0:
                return "Picker exhausted: every fault in the (restricted) fault space has been sampled"
            return None

        return (picker_exhausted,)

    @override
    def serialize(self) -> str:
        return SavedResult(
            fingerprint=self._fingerprint,
            runs=self._runs,
            metric_display_name=self._reliability_metric.display_name() or "",
            golden_score=self._golden_score,
        ).model_dump_json()

    @override
    def deserialize(self, content: str) -> None:
        loaded = SavedResult.model_validate_json(content)
        self._fingerprint.raise_if_differs(loaded.fingerprint)
        self._runs = loaded.runs
        # NOTE: the fingerprint matches, so a loaded golden score is still
        # valid. Keep a computed one if the file doesn't have it yet.
        if loaded.golden_score is not None:
            self._golden_score = loaded.golden_score
        self._picker = Picker.from_returned(
            self._fault_config.radix(self._array),
            {run.fault_id for run in self._runs},
        )

    @override
    def golden_score(self) -> float:
        """The metric's score for the fault-free golden model.

        Computed with one inference pass over the dataset on the first call,
        unless it was loaded from a file, and cached after that.
        """
        if self._golden_score is None:
            with stage(self._progress, "Golden score"):
                self._golden_score = self._reliability_metric.score(
                    self._infer(self._golden_model)
                )
            self._golden_score_unsaved = True
        return self._golden_score

    @override
    def has_unsaved_changes(self) -> bool:
        return self._golden_score_unsaved

    @override
    def on_saved(self) -> None:
        self._golden_score_unsaved = False

    def _draw_fault(self) -> tuple[int, Fault]:
        try:
            fault_id = next(self._picker)
        except StopIteration:
            raise RuntimeError(
                "Expected a fault id to be available but the picker is exhausted"
            ) from None
        return fault_id, self._fault_config.fault_from_id(fault_id, self._array)

    def _infer(self, model: nn.Module) -> R:
        """Run inference on `model` over the dataset, scored by `self._reliability_metric`."""
        result = None

        with (
            stage(
                self._progress, "Inference", total=self._golden_cache.batch_count()
            ) as s,
            torch.no_grad(),
        ):
            for golden, data_batch in self._golden_cache:
                logits = model.forward(data_batch.inputs)

                result = self._reliability_metric.evaluate_and_accumulate(
                    logits,
                    golden,
                    data_batch.targets,
                    result,
                )

                s.advance()

        if result is None:
            raise RuntimeError("dataset didn't produce any data")

        return result

    @override
    def run(self) -> None:
        fault_id, fault = self._draw_fault()
        self._backend.set_fault(fault)

        result = self._infer(self._model)
        self._runs.append(
            RunResult(fault_id=fault_id, score=self._reliability_metric.score(result))
        )
