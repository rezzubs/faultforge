"""Validate that a model on a fault-free array reproduces baseline accuracy.

Not part of pytest/CI: downloads a real model and dataset. Run by hand.
"""

import torch
from systolic import (
    Array,
    LiftedEvaluator,
    OnArray,
)
from torch import nn

from faultforge.dataset import DEFAULT_BATCH_SIZE, DEFAULT_DEVICE, BatchedDataset
from faultforge.loading import Cifar, CifarDataset, CifarModel


def _accuracy(model: nn.Module, dataset: BatchedDataset) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataset:
            logits = model.forward(batch.inputs)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == batch.targets).sum().item())
            total += batch.targets.shape[0]
    return correct / total


def main() -> None:
    # ResNet20: groups=1 throughout. OnArray does not support grouped
    # convolutions yet (see systolic-roadmap.md's open questions), so
    # MobileNetV2/ShuffleNetV2 variants of this bundle can't be used here.
    bundle = Cifar(CifarModel.ResNet20, CifarDataset.Cifar10)
    reference = bundle.load_model(DEFAULT_DEVICE).eval()
    dataset = bundle.load_dataset(DEFAULT_BATCH_SIZE, DEFAULT_DEVICE)

    reference_accuracy = _accuracy(reference, dataset)

    with OnArray(LiftedEvaluator(Array(16, 16)), None):
        on_array_accuracy = _accuracy(reference, dataset)

    print(f"reference accuracy: {reference_accuracy:.4f}")
    print(f"on array accuracy:  {on_array_accuracy:.4f}")
    assert reference_accuracy == on_array_accuracy, (
        "the accuracy on a fault-free array diverged from the reference model"
    )


if __name__ == "__main__":
    main()
