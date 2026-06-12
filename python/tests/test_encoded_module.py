"""Property tests for EncodedModule: forward output must match the plain module."""

import copy

import hypothesis.strategies as st
import pytest
import torch
from faultforge.encoding import (
    CepEncoder,
    EncodedModule,
    Encoder,
    IdentityEncoder,
    MsetEncoder,
    SecdedEncoder,
)
from hypothesis import given, settings
from torch import nn

_DTYPES = st.sampled_from([torch.float32, torch.float16])


@given(
    in_features=st.integers(min_value=1, max_value=16),
    out_features=st.integers(min_value=1, max_value=16),
    batch_size=st.integers(min_value=1, max_value=8),
    dtype=_DTYPES,
)
@settings(max_examples=50)
def test_encoded_module_forward_matches_plain(
    in_features: int,
    out_features: int,
    batch_size: int,
    dtype: torch.dtype,
) -> None:
    module = nn.Linear(in_features, out_features).to(dtype=dtype)
    reference = copy.deepcopy(module)

    encoded = EncodedModule(module, IdentityEncoder())

    x = torch.randn(batch_size, in_features, dtype=dtype)

    with torch.no_grad():
        expected = reference.forward(x)
        actual = encoded.forward(x)

    assert torch.eq(expected, actual).all()


@given(
    in_features=st.integers(min_value=1, max_value=16),
    out_features=st.integers(min_value=1, max_value=16),
    batch_size=st.integers(min_value=1, max_value=8),
    bits_per_chunk=st.sampled_from([16, 32, 64, 256]),
    dtype=_DTYPES,
)
def test_secded_corrects_single_bit_flip(
    in_features: int,
    out_features: int,
    batch_size: int,
    bits_per_chunk: int,
    dtype: torch.dtype,
) -> None:
    module = nn.Linear(in_features, out_features).to(dtype=dtype)
    reference = copy.deepcopy(module)
    encoded = EncodedModule(module, SecdedEncoder(bits_per_chunk=bits_per_chunk))

    encoded.flip_bits(1)

    x = torch.randn(batch_size, in_features, dtype=dtype)
    with torch.no_grad():
        expected = reference.forward(x)
        actual = encoded.forward(x)

    assert torch.eq(expected, actual).all()


@pytest.mark.parametrize(
    "encoder",
    [
        IdentityEncoder(),
        SecdedEncoder(bits_per_chunk=64),
        CepEncoder(),
        MsetEncoder(),
    ],
)
@given(
    in_features=st.integers(min_value=1, max_value=16),
    out_features=st.integers(min_value=1, max_value=16),
    batch_size=st.integers(min_value=1, max_value=8),
    dtype=_DTYPES,
)
def test_clone_fault_does_not_affect_original(
    encoder: Encoder,
    in_features: int,
    out_features: int,
    batch_size: int,
    dtype: torch.dtype,
) -> None:
    module = nn.Linear(in_features, out_features).to(dtype=dtype)
    encoded = EncodedModule(module, encoder)

    x = torch.randn(batch_size, in_features, dtype=dtype)
    with torch.no_grad():
        before = encoded.forward(x).clone()

    clone = encoded.clone()
    clone.flip_bits(1)
    with torch.no_grad():
        clone.forward(x)

    with torch.no_grad():
        after = encoded.forward(x)

    assert torch.eq(before, after).all()
