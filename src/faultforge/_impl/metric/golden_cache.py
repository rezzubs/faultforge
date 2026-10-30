"""See [`faultforge.metric`]"""

from collections.abc import Generator
from typing import final

import torch
from torch import Tensor, nn

from faultforge._impl.dataset import (
    BatchedDataset,
    DataBatch,
)
from faultforge._impl.metric.base import Metric
from faultforge._impl.progress import Progress, stage


@final
class GoldenCache[R]:
    """An iterable over a [`BatchedDataset`] that additionally yields an inference result for a golden model.

    Golden outputs are computed on the first iteration and reused by every
    iteration after that.

    [`BatchedDataset`]: faultforge.dataset.BatchedDataset
    """

    def __init__(
        self,
        metric: Metric[R],
        golden_model: nn.Module,
        dataset: BatchedDataset,
        *,
        progress: Progress | None,
    ) -> None:
        self._golden_outputs: list[Tensor] = []

        self._golden_model: nn.Module = golden_model
        self._metric: Metric[R] = metric
        self._dataset: BatchedDataset = dataset
        self._progress: Progress | None = progress

    def __iter__(self) -> Generator[tuple[Tensor, DataBatch]]:
        for index, data_batch in enumerate(self._dataset):
            golden_output = self._get_or_compute(index, data_batch)

            yield (golden_output, data_batch)

    def _get_or_compute(self, index: int, data_batch: DataBatch) -> Tensor:
        if not self._metric.requires_golden():
            return torch.empty(0)

        if index < len(self._golden_outputs):
            return self._golden_outputs[index]

        # NOTE: this function is only ever called for sequential values of
        # `index` so pushing a new value is always correct. We assert the
        # invariant here.
        assert index == len(self._golden_outputs)

        with stage(self._progress, f"Golden inference for batch {index}"):
            raw_output = self._golden_model.forward(data_batch.inputs)

        if not isinstance(raw_output, Tensor):
            raise TypeError("Expected model to return a tensor")

        processed = self._metric.preprocess_golden(raw_output)

        self._golden_outputs.append(processed)

        return processed

    def batch_count(self) -> int | None:
        """Return the batch count of the underlying dataset, if known."""
        return self._dataset.batch_count()
