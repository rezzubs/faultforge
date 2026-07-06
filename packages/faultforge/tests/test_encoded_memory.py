"""Tests for EncodedFaultInjection's mutually exclusive result kinds."""

import json
from typing import override

import torch
from faultforge import Fingerprint
from faultforge._internal.common import DeviceLike
from faultforge._internal.dataset import BatchedDataset
from faultforge._internal.loading.abc import ModelBundle
from faultforge._internal.progress import Progress
from faultforge.encoding import IdentityEncoder
from faultforge.experiments.encoded_memory import (
    EncodedFaultInjection,
    ReliabilityMetric,
)
from torch import nn
from torch.utils.data import TensorDataset


class _FakeBundle(ModelBundle):
    """A tiny in-memory model/dataset bundle, just enough to drive `EncodedFaultInjection`."""

    def __init__(
        self, in_features: int, out_features: int, batch_size: int, num_batches: int
    ) -> None:
        self._in_features = in_features
        self._out_features = out_features
        self._batch_size = batch_size
        self._num_batches = num_batches

    @override
    def load_model(
        self, device: DeviceLike, *, progress: Progress | None = None
    ) -> nn.Module:
        return nn.Linear(self._in_features, self._out_features)

    @override
    def load_dataset(
        self, batch_size: int, device: DeviceLike, *, progress: Progress | None = None
    ) -> BatchedDataset:
        n = self._batch_size * self._num_batches
        inputs = torch.randn(n, self._in_features)
        targets = torch.randint(0, self._out_features, (n,))
        dataset = TensorDataset(inputs, targets)
        return BatchedDataset.from_dataset(dataset, batch_size, device)

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="fake_bundle")


def _make_experiment(
    *, compare_bitwise: bool, faults: int = 1
) -> EncodedFaultInjection:
    bundle = _FakeBundle(in_features=4, out_features=3, batch_size=2, num_batches=2)
    return EncodedFaultInjection(
        bundle,
        IdentityEncoder(),
        ReliabilityMetric.Accuracy,
        faults=faults,
        compare_bitwise=compare_bitwise,
        batch_size=2,
    )


def _result(experiment: EncodedFaultInjection) -> dict:
    return json.loads(experiment.serialize())["result"]


def test_simple_result_by_default():
    experiment = _make_experiment(compare_bitwise=False)
    experiment.run()
    assert _result(experiment)["kind"] == "simple"


def test_detailed_result_when_compare_bitwise():
    experiment = _make_experiment(compare_bitwise=True)
    experiment.run()

    result = _result(experiment)
    assert result["kind"] == "detailed"
    assert len(result["results"]) == 1
    # A bit flip always changes the underlying bit pattern, so the faulty
    # parameter tensors must disagree with the golden ones somewhere.
    assert len(result["results"][0]["bitmask"]) > 0


def test_multiple_runs_append_one_result_each():
    experiment = _make_experiment(compare_bitwise=True)
    experiment.run()
    experiment.run()

    assert len(_result(experiment)["results"]) == 2


def test_discard_bitmasks_converts_to_simple():
    experiment = _make_experiment(compare_bitwise=True)
    experiment.run()
    before = _result(experiment)

    experiment.discard_bitmasks()
    after = _result(experiment)

    assert after["kind"] == "simple"
    assert after["results"] == [run["correct_count"] for run in before["results"]]


def test_discard_bitmasks_is_noop_for_simple():
    experiment = _make_experiment(compare_bitwise=False)
    experiment.run()

    experiment.discard_bitmasks()

    assert _result(experiment)["kind"] == "simple"


def test_discard_bitmasks_updates_fingerprint_compare_bitwise_scalar():
    experiment = _make_experiment(compare_bitwise=True)
    experiment.run()

    experiment.discard_bitmasks()

    scalars = json.loads(experiment.serialize())["fingerprint"]["scalars"]
    assert scalars["compare_bitwise"] is False


def test_deserialize_after_discard_bitmasks_matches_simple_experiment():
    detailed = _make_experiment(compare_bitwise=True)
    detailed.run()
    detailed.discard_bitmasks()
    serialized = detailed.serialize()

    # Must not raise FingerprintError: the discarded result's fingerprint
    # should now agree with an experiment that was never recording bitmasks.
    simple = _make_experiment(compare_bitwise=False)
    simple.deserialize(serialized)
    assert json.loads(simple.serialize())["result"]["kind"] == "simple"


def test_serialize_deserialize_round_trip_detailed():
    experiment = _make_experiment(compare_bitwise=True)
    experiment.run()
    serialized = experiment.serialize()

    reloaded = _make_experiment(compare_bitwise=True)
    reloaded.deserialize(serialized)

    assert reloaded.serialize() == serialized


def test_serialize_deserialize_round_trip_simple():
    experiment = _make_experiment(compare_bitwise=False)
    experiment.run()
    serialized = experiment.serialize()

    reloaded = _make_experiment(compare_bitwise=False)
    reloaded.deserialize(serialized)

    assert reloaded.serialize() == serialized
