"""Tests for `Encoding.decode` and `Encoding.decode_into`."""

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
    TensorEncoder,
)

_ENCODERS = [
    IdentityEncoder(),
    SecdedEncoder(bits_per_chunk=64),
    CepEncoder(),
    MsetEncoder(),
    EncoderSequence(head=[], tail=SecdedEncoder(bits_per_chunk=64)),
    EncoderSequence(head=[IdentityEncoder()], tail=IdentityEncoder()),
    EncoderSequence(head=[MsetEncoder()], tail=SecdedEncoder(bits_per_chunk=64)),
    EncoderSequence(
        head=[CepEncoder(), MsetEncoder()], tail=SecdedEncoder(bits_per_chunk=64)
    ),
]


@st.composite
def _tensor_lists(draw: st.DrawFn) -> list[Tensor]:
    dtype = draw(st.sampled_from([torch.float32, torch.float16]))
    shapes = draw(
        st.lists(
            st.tuples(st.integers(1, 8), st.integers(1, 8)), min_size=1, max_size=4
        )
    )
    generator = torch.Generator().manual_seed(draw(st.integers(0, 2**32 - 1)))
    return [torch.randn(shape, dtype=dtype, generator=generator) for shape in shapes]


def _bits(ts: list[Tensor]) -> list[Tensor]:
    """Copy the bit patterns of `ts`, so NaNs compare equal."""
    int_dtype = torch.int32 if ts[0].dtype == torch.float32 else torch.int16
    return [t.detach().contiguous().view(int_dtype).clone() for t in ts]


def _assert_bits_equal(actual: list[Tensor], expected: list[Tensor]) -> None:
    for actual_bits, expected_bits in zip(_bits(actual), _bits(expected), strict=True):
        assert torch.equal(actual_bits, expected_bits)


@pytest.mark.parametrize("encoder", _ENCODERS)
@given(ts=_tensor_lists(), data=st.data())
def test_decode_into_matches_decode(
    encoder: Encoder, ts: list[Tensor], data: st.DataObject
) -> None:
    encoding = encoder.encode(ts)
    target_bits = data.draw(
        st.lists(st.integers(0, encoding.bit_count() - 1), max_size=8)
    )
    encoding.apply_faults([(BitFlip(), bit) for bit in target_bits])

    expected = encoding.decode()
    for t, expected_tensor in zip(ts, expected, strict=True):
        assert t.shape == expected_tensor.shape
        assert t.dtype == expected_tensor.dtype

    # A different value everywhere checks that every element is written.
    contiguous = [torch.full_like(t, 7.0) for t in ts]
    encoding.decode_into(contiguous)
    _assert_bits_equal(contiguous, expected)

    # Every other element of a larger tensor isn't contiguous.
    strided = [torch.full((*t.shape, 2), 7.0, dtype=t.dtype)[..., 0] for t in ts]
    encoding.decode_into(strided)
    _assert_bits_equal(strided, expected)

    # Decoding again gives the same result.
    _assert_bits_equal(encoding.decode(), expected)


@pytest.mark.parametrize("encoder", _ENCODERS)
def test_decode_returns_new_tensors(encoder: Encoder) -> None:
    encoding = encoder.encode([torch.ones(3, 2)])
    first = encoding.decode()
    expected = [t.clone() for t in first]

    with torch.no_grad():
        _ = first[0].fill_(5.0)

    _assert_bits_equal(encoding.decode(), expected)


@pytest.mark.parametrize("encoder", _ENCODERS)
def test_clone_decodes_independently(encoder: Encoder) -> None:
    encoding = encoder.encode([torch.ones(4), torch.ones(4)])
    expected = encoding.decode()

    clone = encoding.clone()
    clone.apply_faults([(BitFlip(), bit) for bit in range(clone.bit_count())])
    _ = clone.decode()

    _assert_bits_equal(encoding.decode(), expected)


@pytest.mark.parametrize("encoder", _ENCODERS)
def test_decode_into_rejects_wrong_destinations(encoder: Encoder) -> None:
    encoding = encoder.encode([torch.zeros(4), torch.zeros(2, 3)])

    with pytest.raises(ValueError, match="Expected 2 destination tensors"):
        encoding.decode_into([torch.zeros(4)])
    with pytest.raises(ValueError, match="Destination tensor 1 has shape"):
        encoding.decode_into([torch.zeros(4), torch.zeros(3, 2)])
    with pytest.raises(ValueError, match="dtype torch.float16"):
        encoding.decode_into([torch.zeros(4), torch.zeros(2, 3, dtype=torch.float16)])


@pytest.mark.parametrize("encoder", _ENCODERS)
def test_single_element_with_stride(encoder: Encoder) -> None:
    # Torch treats a single element as contiguous whatever its strides are.
    source = torch.full((1, 1, 2), 3.0)[..., 0]
    assert source.is_contiguous()
    assert source.stride() != (1, 1)
    encoding = encoder.encode([source])

    dest = torch.zeros(1, 1, 2)[..., 0]
    encoding.decode_into([dest])

    _assert_bits_equal([dest], encoding.decode())


def _reference_head_decode(head: list[TensorEncoder], ts: list[Tensor]) -> list[Tensor]:
    """Encode with each head encoder in turn and decode in reverse.

    Like a sequence with a lossless tail and no faults.
    """
    encodings = []
    for encoder in head:
        encoding = encoder.encode(ts)
        encodings.append(encoding)
        ts = encoding.encoded_tensors()

    decoded = [t.clone() for t in ts]
    for encoding in reversed(encodings):
        with torch.no_grad():
            for encoded, value in zip(encoding.encoded_tensors(), decoded, strict=True):
                _ = encoded.copy_(value)
        decoded = encoding.decode()
    return decoded


@pytest.mark.parametrize(
    "head",
    [[MsetEncoder()], [CepEncoder(), MsetEncoder()], [MsetEncoder(), CepEncoder()]],
)
@given(ts=_tensor_lists())
def test_sequence_with_lossless_tail_decodes_like_head(
    head: list[TensorEncoder], ts: list[Tensor]
) -> None:
    sequence = EncoderSequence(head=head, tail=SecdedEncoder(bits_per_chunk=64))

    _assert_bits_equal(sequence.encode(ts).decode(), _reference_head_decode(head, ts))
