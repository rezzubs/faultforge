"""Tests for `faultforge._internal.loading.transform`."""

import torch
from faultforge._internal.loading.transform import dtype_transforms
from torchvision.transforms import Compose


def test_none_adds_no_steps() -> None:
    assert dtype_transforms(None) == []


def test_casts_without_rescaling() -> None:
    pipeline = Compose([*dtype_transforms(torch.float16)])
    image = torch.tensor([[-2.5, 0.0, 1.75]])

    converted = pipeline(image)

    assert converted.dtype == torch.float16
    assert torch.equal(converted.to(torch.float32), image)


def test_splices_onto_existing_pipeline() -> None:
    pipeline = Compose([lambda t: t * 2, *dtype_transforms(torch.float64)])
    image = torch.tensor([1.0, 2.0])

    converted = pipeline(image)

    assert converted.dtype == torch.float64
    assert torch.equal(converted, torch.tensor([2.0, 4.0], dtype=torch.float64))
