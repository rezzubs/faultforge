"""Tests for `faultforge._internal.metric.GoldenCache`."""

from typing import override

import pytest
import torch
from faultforge._internal.dataset import BatchedDataset
from faultforge._internal.fingerprint import Fingerprint
from faultforge._internal.metric import GoldenCache, Metric
from torch import Tensor, nn
from torch.utils.data import Dataset

# The following classes are `_` prefixed to not interpret them as Test classes.


class _CountingDataset(Dataset):
    """A map-style dataset yielding `index` as both input and target."""

    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    @override
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return torch.tensor([float(index)]), torch.tensor(index)


class _DoublingModel(nn.Module):
    """A golden model returning `inputs * 2`, counting how often it ran."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count: int = 0

    @override
    def forward(self, inputs: Tensor) -> Tensor:
        self.call_count += 1
        return inputs * 2


class _NonTensorModel(nn.Module):
    """A golden model returning something that isn't a `Tensor`."""

    @override
    def forward(self, inputs: Tensor) -> object:
        _ = inputs
        return "not a tensor"


class _StubMetric(Metric[int]):
    """A minimal `Metric` whose golden requirement and preprocessing are configurable."""

    def __init__(
        self, *, requires_golden: bool, preprocess_offset: float = 0.0
    ) -> None:
        self._requires_golden = requires_golden
        self._preprocess_offset = preprocess_offset

    @override
    def evaluate_batch(
        self,
        batch_model_output: Tensor,
        batch_golden: Tensor,
        batch_targets: Tensor,
    ) -> int:
        _ = (batch_model_output, batch_golden, batch_targets)
        return 0

    @override
    def preprocess_golden(self, golden: Tensor) -> Tensor:
        return golden + self._preprocess_offset

    @override
    def requires_golden(self) -> bool:
        return self._requires_golden

    @override
    def accumulate(self, existing: int, new: int) -> int:
        return existing + new

    @override
    def score(self, result: int) -> float:
        return float(result)

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="stub")


def _make_cache(
    metric: Metric[int],
    golden_model: nn.Module,
    *,
    item_count: int = 4,
) -> tuple[GoldenCache[int], _CountingDataset]:
    dataset = _CountingDataset(item_count)
    cache = GoldenCache(
        metric=metric,
        golden_model=golden_model,
        dataset=BatchedDataset.from_dataset(dataset, batch_size=1),
        progress=None,
    )
    return cache, dataset


def test_yields_one_pair_per_batch() -> None:
    cache, _ = _make_cache(_StubMetric(requires_golden=True), _DoublingModel())

    items = list(cache)

    assert len(items) == 4
    assert [int(batch.targets.item()) for _, batch in items] == [0, 1, 2, 3]


def test_golden_is_model_output_for_the_paired_batch() -> None:
    cache, _ = _make_cache(_StubMetric(requires_golden=True), _DoublingModel())

    for golden, batch in cache:
        assert torch.equal(golden, batch.inputs * 2)


def test_applies_preprocess_golden() -> None:
    metric = _StubMetric(requires_golden=True, preprocess_offset=1.0)
    cache, _ = _make_cache(metric, _DoublingModel())

    for golden, batch in cache:
        assert torch.equal(golden, batch.inputs * 2 + 1.0)


def test_second_pass_reuses_cached_golden_outputs() -> None:
    model = _DoublingModel()
    cache, _ = _make_cache(_StubMetric(requires_golden=True), model)

    first_pass = list(cache)
    assert model.call_count == 4

    second_pass = list(cache)
    assert model.call_count == 4

    for (first_golden, _), (second_golden, _) in zip(
        first_pass, second_pass, strict=True
    ):
        assert first_golden is second_golden


def test_golden_model_not_run_when_metric_does_not_require_golden() -> None:
    model = _DoublingModel()
    cache, _ = _make_cache(_StubMetric(requires_golden=False), model)

    items = list(cache)

    assert len(items) == 4
    assert model.call_count == 0
    assert all(golden.numel() == 0 for golden, _ in items)


def test_non_tensor_model_output_raises() -> None:
    cache, _ = _make_cache(_StubMetric(requires_golden=True), _NonTensorModel())

    with pytest.raises(ValueError, match="tensor"):
        _ = list(cache)


def test_batch_count_delegates_to_dataset() -> None:
    cache, _ = _make_cache(
        _StubMetric(requires_golden=True), _DoublingModel(), item_count=5
    )

    assert cache.batch_count() == 5


def test_reiteration_starts_from_the_first_batch() -> None:
    cache, _ = _make_cache(_StubMetric(requires_golden=True), _DoublingModel())

    first_pass = [int(batch.targets.item()) for _, batch in cache]
    second_pass = [int(batch.targets.item()) for _, batch in cache]

    assert first_pass == [0, 1, 2, 3]
    assert second_pass == [0, 1, 2, 3]


def test_abandoned_iteration_does_not_misalign_cached_goldens() -> None:
    cache, _ = _make_cache(_StubMetric(requires_golden=True), _DoublingModel())

    for _, batch in cache:
        if int(batch.targets.item()) == 1:
            break

    for golden, batch in cache:
        assert torch.equal(golden, batch.inputs * 2)
