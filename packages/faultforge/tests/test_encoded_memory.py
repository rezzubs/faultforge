"""Tests for EncodedFaultInjection's mutually exclusive result kinds."""

import json
from compression import zstd
from typing import override

import pytest
import torch
from faultforge import Fingerprint
from faultforge._internal.common import DeviceLike
from faultforge._internal.dataset import BatchedDataset
from faultforge._internal.experiments.encoded_memory import (
    BatchReliability,
    _batch_accuracy,
    _batch_accuracy_degradation,
    _batch_critical_sdc,
    _batch_sdc,
)
from faultforge._internal.loading.abc import ModelBundle
from faultforge._internal.progress import Progress
from faultforge.encoding import IdentityEncoder
from faultforge.experiments.encoded_memory import (
    EncodedFaultInjection,
    ReliabilityMetric,
    discard_bitmasks_in_file,
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
        self,
        device: DeviceLike,
        *,
        dtype: torch.dtype = torch.float32,
        progress: Progress | None = None,
    ) -> nn.Module:
        return nn.Linear(self._in_features, self._out_features).to(
            device=device, dtype=dtype
        )

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
    *,
    compare_bitwise: bool,
    faults: int | float = 1,
    golden_is_encoded: bool = False,
    dataset_batch_limit: int | None = None,
    reliability_metric: ReliabilityMetric = ReliabilityMetric.Accuracy,
    dtype: torch.dtype = torch.float32,
    fault_summary: bool = False,
) -> EncodedFaultInjection:
    bundle = _FakeBundle(in_features=4, out_features=3, batch_size=2, num_batches=2)
    return EncodedFaultInjection(
        bundle,
        IdentityEncoder(),
        reliability_metric,
        golden_is_encoded=golden_is_encoded,
        faults=faults,
        compare_bitwise=compare_bitwise,
        fault_summary=fault_summary,
        dataset_batch_limit=dataset_batch_limit,
        batch_size=2,
        dtype=dtype,
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


def test_detailed_result_when_compare_bitwise_f16():
    # Regression test: `_populate_golden`/`run` must cast the dataset's
    # (always-float32) inputs to the model's dtype before calling forward,
    # otherwise this raises a dtype-mismatch RuntimeError.
    experiment = _make_experiment(compare_bitwise=True, dtype=torch.float16)
    experiment.run()

    result = _result(experiment)
    assert result["kind"] == "detailed"
    assert len(result["results"]) == 1
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


def test_discard_bitmasks_in_file(tmp_path):
    experiment = _make_experiment(compare_bitwise=True)
    experiment.run()
    before = _result(experiment)

    path = tmp_path / "result.json"
    experiment.save(path)

    discard_bitmasks_in_file(path)

    saved = json.loads(path.read_text())
    assert saved["result"]["kind"] == "simple"
    assert saved["result"]["results"] == [
        run["correct_count"] for run in before["results"]
    ]
    assert saved["fingerprint"]["scalars"]["compare_bitwise"] is False


def test_discard_bitmasks_in_file_is_noop_for_simple_results(tmp_path):
    experiment = _make_experiment(compare_bitwise=False)
    experiment.run()

    path = tmp_path / "result.json"
    experiment.save(path)
    original = path.read_text()

    discard_bitmasks_in_file(path)

    assert path.read_text() == original


def test_load_from_after_discard_bitmasks_in_file(tmp_path):
    experiment = _make_experiment(compare_bitwise=True)
    experiment.run()

    path = tmp_path / "result.json"
    experiment.save(path)
    discard_bitmasks_in_file(path)

    # Must not raise FingerprintError: the discarded file's fingerprint should
    # now agree with an experiment that was never recording bitmasks.
    simple = _make_experiment(compare_bitwise=False)
    simple.load_from(path)
    assert json.loads(simple.serialize())["result"]["kind"] == "simple"


_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def test_discard_bitmasks_in_file_preserves_compressed_format(tmp_path):
    experiment = _make_experiment(compare_bitwise=True)
    experiment.run()
    before = _result(experiment)

    path = tmp_path / "result.json"
    experiment.save_atomic(path, compressed=True)

    discard_bitmasks_in_file(path)

    assert path.read_bytes()[:4] == _ZSTD_MAGIC
    with zstd.open(path, "rt") as f:
        saved = json.loads(f.read())
    assert saved["result"]["kind"] == "simple"
    assert saved["result"]["results"] == [
        run["correct_count"] for run in before["results"]
    ]
    assert saved["fingerprint"]["scalars"]["compare_bitwise"] is False


def test_discard_bitmasks_in_file_preserves_uncompressed_format(tmp_path):
    experiment = _make_experiment(compare_bitwise=True)
    experiment.run()

    path = tmp_path / "result.json"
    experiment.save(path)

    discard_bitmasks_in_file(path)

    assert path.read_bytes()[:4] != _ZSTD_MAGIC


def test_discard_bitmasks_in_file_is_noop_for_simple_results_compressed(tmp_path):
    experiment = _make_experiment(compare_bitwise=False)
    experiment.run()

    path = tmp_path / "result.json"
    experiment.save_atomic(path, compressed=True)
    original = path.read_bytes()

    discard_bitmasks_in_file(path)

    assert path.read_bytes() == original


def test_load_from_after_discard_bitmasks_in_file_compressed(tmp_path):
    experiment = _make_experiment(compare_bitwise=True)
    experiment.run()

    path = tmp_path / "result.json"
    experiment.save_atomic(path, compressed=True)
    discard_bitmasks_in_file(path)

    # Must not raise FingerprintError: the discarded file's fingerprint should
    # now agree with an experiment that was never recording bitmasks.
    simple = _make_experiment(compare_bitwise=False)
    simple.load_from(path)
    assert json.loads(simple.serialize())["result"]["kind"] == "simple"


# SECTION Per-metric batch reliability functions


def test_batch_accuracy_counts_matching_predictions():
    logits = torch.tensor(
        [
            [0.1, 0.9],  # predicts class 1
            [0.8, 0.2],  # predicts class 0
            [0.3, 0.7],  # predicts class 1
        ]
    )
    targets = torch.tensor([1, 0, 0])  # last one is wrong

    assert _batch_accuracy(logits, targets) == BatchReliability(correct=2, total=3)


def test_batch_accuracy_degradation_is_golden_minus_faulty_correct():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])  # predicts 1, 0, 1
    golden_classifications = torch.tensor([1, 0, 0])
    targets = torch.tensor([1, 0, 0])

    # faulty correct: [1, 0, 1] vs [1, 0, 0] -> 2
    # golden correct: [1, 0, 0] vs [1, 0, 0] -> 3
    # degradation: golden_correct - correct = 1
    result = _batch_accuracy_degradation(logits, golden_classifications, targets)
    assert result == BatchReliability(correct=1, total=3)


def test_batch_sdc_counts_matching_logits_elementwise():
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    golden_logits = torch.tensor([[1.0, 0.0], [3.0, 0.0]])

    # elementwise equality: [[T, F], [T, F]] -> 2 correct out of 4
    result = _batch_sdc(logits, golden_logits)
    assert result == BatchReliability(correct=2, total=4)


def test_batch_critical_sdc_counts_matching_top1_predictions():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])  # predicts 1, 0, 1
    golden_classifications = torch.tensor([1, 0, 0])  # matches the first two only

    result = _batch_critical_sdc(logits, golden_classifications)
    assert result == BatchReliability(correct=2, total=3)


# SECTION End-to-end coverage across metrics and golden_is_encoded


@pytest.mark.parametrize(
    "metric",
    [
        ReliabilityMetric.Accuracy,
        ReliabilityMetric.AccuracyDegradation,
        ReliabilityMetric.Sdc,
        ReliabilityMetric.Top1Sdc,
    ],
)
@pytest.mark.parametrize("golden_is_encoded", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_all_reliability_metrics_run_end_to_end(
    metric: ReliabilityMetric, golden_is_encoded: bool, dtype: torch.dtype
):
    experiment = _make_experiment(
        compare_bitwise=False,
        golden_is_encoded=golden_is_encoded,
        reliability_metric=metric,
        dtype=dtype,
    )
    experiment.run()

    scores = experiment.scores()
    assert len(scores) == 1
    if metric == ReliabilityMetric.AccuracyDegradation:
        # Signed: the faulty model can outscore golden on a tiny random
        # dataset, which shows up as a negative "degradation".
        assert -100.0 <= scores[0] <= 100.0
    else:
        assert 0.0 <= scores[0] <= 100.0


def test_golden_is_encoded_produces_bitmask_against_self_model():
    experiment = _make_experiment(compare_bitwise=True, golden_is_encoded=True)
    experiment.run()

    result = _result(experiment)
    assert result["kind"] == "detailed"
    # Bitwise comparison must decode `self._model` (there's no unencoded
    # golden to fall back on) and still find the injected bit flip.
    assert len(result["results"][0]["bitmask"]) > 0


# --- `faults` resolution and validation -----------------------------------


def test_full_bit_error_rate_flips_every_bit():
    experiment = _make_experiment(compare_bitwise=True, faults=1.0)
    experiment.run()

    bitmask = _result(experiment)["results"][0]["bitmask"]
    # in_features=4, out_features=3 -> 4*3 weight + 3 bias = 15 float32
    # elements. A bit error rate of 1.0 flips every bit in the encoded
    # memory exactly once (Picker is a full permutation), so every
    # element's bit pattern ends up fully complemented, i.e. all 32 bits
    # set. Stored as the unsigned bit pattern rather than the signed int32
    # view's `-1`.
    assert len(bitmask) == 15
    assert all(value == (1 << 32) - 1 for value in bitmask)


def test_faults_greater_than_bit_count_raises():
    with pytest.raises(ValueError, match="faults"):
        _make_experiment(compare_bitwise=False, faults=100_000)


def test_bit_error_rate_greater_than_one_raises():
    with pytest.raises(ValueError, match="bit error rate"):
        _make_experiment(compare_bitwise=False, faults=1.5)


# SECTION Fault-injection summary display


def test_fault_summary_off_by_default():
    experiment = _make_experiment(compare_bitwise=True)
    experiment.run()

    assert experiment.display().extra() is None


def test_fault_summary_without_compare_bitwise_only_shows_ber_line():
    experiment = _make_experiment(compare_bitwise=False, fault_summary=True, faults=5)
    experiment.run()

    extra = experiment.display().extra()
    assert extra is not None
    assert "Flipped 5/480 bits - BER: 1.04e-02" in extra
    assert "parameters were affected" not in extra
    assert "faulty bit" not in extra


def test_fault_summary_with_compare_bitwise_includes_histogram():
    experiment = _make_experiment(compare_bitwise=True, fault_summary=True, faults=1.0)
    experiment.run()

    extra = experiment.display().extra()
    assert extra is not None
    assert "Flipped 480/480 bits - BER: 1.00e+00" in extra
    assert "15 parameters were affected" in extra
    assert "480 bits were measured faulty (0.00% masked)" in extra
    # Every element gets all 32 bits flipped, so the whole histogram is one bucket.
    assert "15 parameters had 32 faulty bits" in extra


def test_fault_summary_histogram_counts_sum_to_bitmask_length():
    experiment = _make_experiment(compare_bitwise=True, fault_summary=True, faults=200)
    experiment.run()

    bitmask = _result(experiment)["results"][0]["bitmask"]
    extra = experiment.display().extra()
    assert extra is not None

    histogram_lines = [line for line in extra.splitlines() if "faulty bit" in line]
    total = sum(int(line.split()[0]) for line in histogram_lines)
    assert total == len(bitmask)


def test_fault_summary_updates_after_each_run():
    experiment = _make_experiment(compare_bitwise=False, fault_summary=True, faults=3)
    experiment.run()
    first = experiment.display().extra()
    experiment.run()
    second = experiment.display().extra()

    assert first is not None
    assert second is not None
    assert "Flipped 3/480 bits" in first
    assert "Flipped 3/480 bits" in second


# --- Fingerprint content ---------------------------------------------------


def _fingerprint_scalars(experiment: EncodedFaultInjection) -> dict:
    return json.loads(experiment.serialize())["fingerprint"]["scalars"]


def test_fingerprint_records_int_faults_not_bit_error_rate():
    scalars = _fingerprint_scalars(_make_experiment(compare_bitwise=False, faults=3))
    assert scalars["faults"] == 3
    assert "bit_error_rate" not in scalars


def test_fingerprint_records_bit_error_rate_not_faults():
    scalars = _fingerprint_scalars(_make_experiment(compare_bitwise=False, faults=0.5))
    assert scalars["bit_error_rate"] == 0.5
    assert "faults" not in scalars


def test_fingerprint_records_golden_and_compare_bitwise_and_metric():
    scalars = _fingerprint_scalars(
        _make_experiment(
            compare_bitwise=True,
            golden_is_encoded=True,
            reliability_metric=ReliabilityMetric.Sdc,
        )
    )
    assert scalars["golden"] == "encoded"
    assert scalars["compare_bitwise"] is True
    assert scalars["reliability_metric"] == "sdc"


def test_fingerprint_records_dtype():
    scalars = _fingerprint_scalars(_make_experiment(compare_bitwise=False))
    assert scalars["dtype"] == "f32"

    scalars = _fingerprint_scalars(
        _make_experiment(compare_bitwise=False, dtype=torch.float16)
    )
    assert scalars["dtype"] == "f16"


def test_fingerprint_omits_test_image_limit_by_default():
    scalars = _fingerprint_scalars(_make_experiment(compare_bitwise=False))
    assert "test_image_limit" not in scalars


def test_fingerprint_records_test_image_limit_from_batch_limit():
    scalars = _fingerprint_scalars(
        _make_experiment(compare_bitwise=False, dataset_batch_limit=1)
    )
    # batch_size=2 (fixed in `_make_experiment`) * dataset_batch_limit=1
    assert scalars["test_image_limit"] == 2
