"""Property tests for EncodedModule: forward output must match the plain module."""

import copy

import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given, settings
from torch import Tensor, nn

from faultforge import BitFlip
from faultforge.encoding import (
    CepEncoder,
    EncodedModule,
    Encoder,
    EncoderSequence,
    Encoding,
    IdentityEncoder,
    MsetEncoder,
    SecdedEncoder,
    SecdedEncoding,
)

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
    data=st.data(),
)
def test_secded_corrects_single_bit_flip(
    in_features: int,
    out_features: int,
    batch_size: int,
    bits_per_chunk: int,
    dtype: torch.dtype,
    data: st.DataObject,
) -> None:
    module = nn.Linear(in_features, out_features).to(dtype=dtype)
    reference = copy.deepcopy(module)
    encoded = EncodedModule(module, SecdedEncoder(bits_per_chunk=bits_per_chunk))

    target_bit = data.draw(st.integers(min_value=0, max_value=encoded.bit_count() - 1))
    encoded.apply_fault(BitFlip(), target_bit)

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
    data=st.data(),
)
def test_clone_fault_does_not_affect_original(
    encoder: Encoder,
    in_features: int,
    out_features: int,
    batch_size: int,
    dtype: torch.dtype,
    data: st.DataObject,
) -> None:
    module = nn.Linear(in_features, out_features).to(dtype=dtype)
    encoded = EncodedModule(module, encoder)

    x = torch.randn(batch_size, in_features, dtype=dtype)
    with torch.no_grad():
        before = encoded.forward(x).clone()

    clone = encoded.clone()
    target_bit = data.draw(st.integers(min_value=0, max_value=clone.bit_count() - 1))
    clone.apply_fault(BitFlip(), target_bit)
    with torch.no_grad():
        clone.forward(x)

    with torch.no_grad():
        after = encoded.forward(x)

    assert torch.eq(before, after).all()


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
    dtype=_DTYPES,
    data=st.data(),
)
def test_apply_faults_matches_sequential_apply_fault(
    encoder: Encoder,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    data: st.DataObject,
) -> None:
    module = nn.Linear(in_features, out_features).to(dtype=dtype)
    encoded = EncodedModule(module, encoder)

    bit_count = encoded.bit_count()
    target_bits = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=bit_count - 1),
            min_size=1,
            max_size=min(bit_count, 8),
            unique=True,
        )
    )

    batched = encoded.clone()
    batched.apply_faults([(BitFlip(), bit) for bit in target_bits])

    sequential = encoded.clone()
    for bit in target_bits:
        sequential.apply_fault(BitFlip(), bit)

    # Compare decoded parameters bit-exactly rather than via a forward pass:
    # a bit flip can legitimately produce NaN, and NaN != NaN under normal
    # float comparison even when the underlying bit patterns are identical.
    int_dtype = torch.int32 if dtype == torch.float32 else torch.int16
    batched_params = list(batched.decode().parameters())
    sequential_params = list(sequential.decode().parameters())
    for batched_param, sequential_param in zip(
        batched_params, sequential_params, strict=True
    ):
        assert torch.equal(
            batched_param.view(int_dtype), sequential_param.view(int_dtype)
        )


_ENCODERS = [
    IdentityEncoder(),
    SecdedEncoder(bits_per_chunk=64),
    CepEncoder(),
    MsetEncoder(),
    EncoderSequence(head=[MsetEncoder()], tail=SecdedEncoder(bits_per_chunk=64)),
]


def _params_match(module: nn.Module, expected: list[Tensor]) -> bool:
    """Compare the parameters of `module` to `expected` bit-exactly, so NaNs match."""
    params = list(module.parameters())
    int_dtype = torch.int32 if params[0].dtype == torch.float32 else torch.int16
    return all(
        torch.equal(param.view(int_dtype), expected_tensor.view(int_dtype))
        for param, expected_tensor in zip(params, expected, strict=True)
    )


def _assert_params_match(module: nn.Module, expected: list[Tensor]) -> None:
    assert _params_match(module, expected)


@pytest.mark.parametrize("encoder", _ENCODERS)
@given(
    in_features=st.integers(min_value=1, max_value=16),
    out_features=st.integers(min_value=1, max_value=16),
    dtype=_DTYPES,
    data=st.data(),
)
def test_decode_after_clones_and_faults_matches_full_decode(
    encoder: Encoder,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    data: st.DataObject,
) -> None:
    module = nn.Linear(in_features, out_features).to(dtype=dtype)
    reference = encoder.encode([p.detach().clone() for p in module.parameters()])

    # Each module next to an encoding with the same faults, decoded in full.
    modules = [(EncodedModule(module, encoder), reference)]
    # Faults close together often hit the same chunk. Later faults in a chunk
    # can make it decode to its original data again.
    bit_count = reference.bit_count()
    window_start = data.draw(st.integers(0, bit_count - 1))
    target_bit_strategy = st.integers(
        window_start, min(window_start + 150, bit_count - 1)
    )

    for _ in range(data.draw(st.integers(1, 12))):
        index = data.draw(st.integers(0, len(modules) - 1))
        encoded, expected = modules[index]
        match data.draw(st.sampled_from(["clone", "fault", "decode"])):
            case "clone":
                modules.append((encoded.clone(), expected.clone()))
            case "fault":
                target_bits = data.draw(st.lists(target_bit_strategy, max_size=8))
                faults = [(BitFlip(), bit) for bit in target_bits]
                encoded.apply_faults(faults)
                expected.apply_faults(faults)
            case "decode":
                _assert_params_match(encoded.decode(), expected.decode())

    for encoded, expected in modules:
        _assert_params_match(encoded.decode(), expected.decode())


def test_clone_decodes_only_what_faults_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    golden = EncodedModule(nn.Linear(4, 4), SecdedEncoder(bits_per_chunk=64))
    clone = golden.clone()

    def fail(self: Encoding, dest: list[Tensor]) -> None:
        raise AssertionError("expected a decode of only what faults changed")

    monkeypatch.setattr(SecdedEncoding, "decode_into", fail)

    clone.apply_fault(BitFlip(), 0)
    _ = clone.decode()
    _ = golden.clone().decode()


def test_decode_after_faults_are_undone() -> None:
    golden = EncodedModule(nn.Linear(4, 4), SecdedEncoder(bits_per_chunk=64))
    expected = [p.detach().clone() for p in golden.decode().parameters()]

    clone = golden.clone()
    # Bits 3 and 5 hold data. Two errors in a chunk can't be corrected.
    clone.apply_faults([(BitFlip(), 3), (BitFlip(), 5)])
    assert not _params_match(clone.decode(), expected)
    # One error can be, so the chunk decodes to its original data again.
    clone.apply_fault(BitFlip(), 5)
    _assert_params_match(clone.decode(), expected)
