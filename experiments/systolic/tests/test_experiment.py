"""Tests for `SystolicFaultInjection`'s bookkeeping (results, resuming, golden score)."""

from typing import override

import pytest
import torch
from systolic import (
    EvaluatorKind,
    Fault,
    Index2,
    PeRegisterKind,
    RegisterFaults,
    StuckAtKind,
    SystolicFaultInjection,
)
from torch import nn
from torch.utils.data import TensorDataset

from faultforge import Fingerprint
from faultforge.dataset import BatchedDataset, DeviceLike
from faultforge.loading import ModelBundle
from faultforge.metric import (
    Accuracy,
    Metric,
    Sdc,
)
from faultforge.progress import Progress

_IN_FEATURES = 4
_CLASSES = 3
_SAMPLES = 8


class _FakeBundle(ModelBundle):
    """A tiny deterministic in-memory model/dataset bundle."""

    @override
    def load_model(
        self,
        device: DeviceLike,
        *,
        progress: Progress | None = None,
    ) -> nn.Module:
        _ = progress
        generator = torch.Generator().manual_seed(0)
        model = nn.Linear(_IN_FEATURES, _CLASSES)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator))
        return model.to(device=device)

    @override
    def load_dataset(
        self,
        batch_size: int,
        device: DeviceLike,
        *,
        shuffle: bool = False,
        seed: int | None = None,
        progress: Progress | None = None,
    ) -> BatchedDataset:
        _ = progress
        generator = torch.Generator().manual_seed(1)
        inputs = torch.randn(_SAMPLES, _IN_FEATURES, generator=generator)
        targets = torch.randint(0, _CLASSES, (_SAMPLES,), generator=generator)
        return BatchedDataset.from_dataset(
            TensorDataset(inputs, targets),
            batch_size,
            device,
            shuffle=shuffle,
            seed=seed,
        )

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="fake_bundle")


def _make_experiment(
    metric: Metric | None = None, evaluator: EvaluatorKind = "lifted"
) -> SystolicFaultInjection:
    # The smallest fault space that still has more than one fault.
    return SystolicFaultInjection(
        _FakeBundle(),
        (1, 1),
        metric if metric is not None else Accuracy(),
        fault=RegisterFaults(registers=frozenset({PeRegisterKind.Weight})),
        evaluator=evaluator,
        batch_size=4,
    )


def test_serialize_deserialize_round_trip() -> None:
    experiment = _make_experiment()
    experiment.run()
    experiment.run()
    serialized = experiment.serialize()

    reloaded = _make_experiment()
    reloaded.deserialize(serialized)

    assert reloaded.serialize() == serialized
    assert reloaded.scores() == experiment.scores()


def test_resumed_experiment_draws_each_fault_once() -> None:
    first = _make_experiment()
    first.run()
    first.run()

    resumed = _make_experiment()
    resumed.deserialize(first.serialize())
    (picker_exhausted,) = resumed.stop_conditions()
    while picker_exhausted(resumed) is None:
        resumed.run()

    fault_ids = [run.fault_id for run in resumed._runs]
    assert len(fault_ids) == len(set(fault_ids))
    assert sorted(fault_ids) == list(range(len(fault_ids)))


def test_golden_score_is_saved_and_loaded() -> None:
    experiment = _make_experiment()
    assert not experiment.has_unsaved_changes()

    golden_score = experiment.golden_score()
    assert experiment.has_unsaved_changes()
    experiment.on_saved()
    assert not experiment.has_unsaved_changes()

    reloaded = _make_experiment()
    reloaded.deserialize(experiment.serialize())
    assert reloaded.golden_score() == golden_score
    assert not reloaded.has_unsaved_changes()


@pytest.mark.parametrize("evaluator", ["lifted", "simulated"])
def test_fault_free_array_matches_the_golden_model(evaluator: EvaluatorKind) -> None:
    # The golden model runs on the same array. The simulated array sums in a
    # different order than torch, so a golden model off the array would
    # differ from it in the last bits.
    experiment = _make_experiment(Sdc(), evaluator)
    assert experiment._reliability_metric.score(experiment._infer(None)) == 0.0


@pytest.mark.parametrize("evaluator", ["lifted", "simulated"])
def test_fault_reaches_the_model(evaluator: EvaluatorKind) -> None:
    experiment = _make_experiment(Sdc(), evaluator)
    # A high bit of the exponent changes every product it's part of.
    fault = Fault.Register(
        target=Index2(x=0, y=0),
        register=PeRegisterKind.Weight,
        stuck_at=StuckAtKind.One,
        bit_index=30,
    )
    assert experiment._reliability_metric.score(experiment._infer(fault)) > 0.0
