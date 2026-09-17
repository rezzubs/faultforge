"""Tests for `faultforge._internal.dtype`."""

import pytest
import torch
from faultforge._internal.dtype import dtype_from_name, dtype_name


@pytest.mark.parametrize(
    ("dtype", "name"),
    [
        (torch.float32, "float32"),
        (torch.float16, "float16"),
        (torch.bfloat16, "bfloat16"),
        (torch.uint8, "uint8"),
    ],
)
def test_dtype_name_round_trips(dtype: torch.dtype, name: str) -> None:
    assert dtype_name(dtype) == name
    assert dtype_from_name(name) is dtype


def test_dtype_from_name_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="not the name of a torch dtype"):
        _ = dtype_from_name("float128")


def test_dtype_from_name_rejects_non_dtype_attribute() -> None:
    with pytest.raises(ValueError, match="not the name of a torch dtype"):
        _ = dtype_from_name("nn")
