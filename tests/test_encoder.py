"""Tests for the `Encoder.encode` contract: inputs are only read."""

import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given
from torch import Tensor

from faultforge import BitFlip
from faultforge.encoding import (
    CepEncoder,
    Encoder,
    EncoderSequence,
    IdentityEncoder,
    MsetEncoder,
    SecdedEncoder,
)

_ENCODERS = [
    IdentityEncoder(),
    SecdedEncoder(bits_per_chunk=64),
    CepEncoder(),
    MsetEncoder(),
    EncoderSequence(head=[IdentityEncoder()], tail=IdentityEncoder()),
    EncoderSequence(head=[MsetEncoder()], tail=SecdedEncoder(bits_per_chunk=64)),
]


@st.composite
def _tensor_lists(draw: st.DrawFn) -> list[Tensor]:
    dtype = draw(st.sampled_from([torch.float32, torch.float16]))
    sizes = draw(
        st.lists(st.integers(min_value=1, max_value=64), min_size=1, max_size=4)
    )
    requires_grad = draw(st.booleans())
    generator = torch.Generator().manual_seed(draw(st.integers(0, 2**32 - 1)))
    return [
        torch.randn(size, dtype=dtype, generator=generator).requires_grad_(
            requires_grad
        )
        for size in sizes
    ]


def _bits(ts: list[Tensor]) -> list[Tensor]:
    """Copy the bit patterns of `ts`, so NaNs compare equal."""
    int_dtype = torch.int32 if ts[0].dtype == torch.float32 else torch.int16
    return [t.detach().view(int_dtype).clone() for t in ts]


def _assert_bits_equal(actual: list[Tensor], expected: list[Tensor]) -> None:
    for actual_bits, expected_bits in zip(_bits(actual), expected, strict=True):
        assert torch.equal(actual_bits, expected_bits)


@pytest.mark.parametrize("encoder", _ENCODERS)
@given(ts=_tensor_lists(), data=st.data())
def test_encoding_does_not_modify_inputs(
    encoder: Encoder, ts: list[Tensor], data: st.DataObject
) -> None:
    original = _bits(ts)

    encoding = encoder.encode(ts)
    _assert_bits_equal(ts, original)

    target_bits = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=encoding.bit_count() - 1),
            min_size=1,
            max_size=8,
        )
    )
    encoding.apply_faults([(BitFlip(), bit) for bit in target_bits])
    _ = encoding.decode()
    _assert_bits_equal(ts, original)


@pytest.mark.parametrize("encoder", _ENCODERS)
@given(ts=_tensor_lists())
def test_encoding_does_not_depend_on_inputs_after_encode(
    encoder: Encoder, ts: list[Tensor]
) -> None:
    with torch.no_grad():
        copies = [t.detach().clone() for t in ts]
    expected = _bits(encoder.encode(copies).decode())

    encoding = encoder.encode(ts)
    with torch.no_grad():
        for t in ts:
            _ = t.fill_(1.0)

    _assert_bits_equal(encoding.decode(), expected)
