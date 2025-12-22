import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np

from faultforge.encoding.embedded_parity import EmbeddedParityEncoder, EpScheme


def input_arrays(dtype: type[np.float32] | type[np.float16]):
    return st.lists(st_np.arrays(dtype, st_np.array_shapes()), min_size=1)


def _test_embedded_parity_no_faults(
    arrs: list[np.ndarray],
    scheme: EpScheme,
    mask: int,
    view_type: torch.dtype,
) -> None:
    ts = [torch.from_numpy(t) for t in arrs]  # pyright: ignore[reportUnknownMemberType]

    encoder = EmbeddedParityEncoder(scheme)

    encoded = encoder.tensor_encoder_encode_tensor_list(ts)

    decoded = encoded.encoding_decode_tensor_list()
    for t, d in zip(ts, decoded):
        mapped = t.view(view_type) & mask
        assert torch.equal(mapped, d.view(view_type))


@given(input_arrays(np.float32))
def test_f32e3p1_no_faults(arrs: list[np.ndarray]):
    _test_embedded_parity_no_faults(
        arrs,
        EpScheme.D3P1,
        0b1111_1111_1111_1111_1111_1111_0000_0000,
        torch.uint32,
    )


@given(input_arrays(np.float32))
def test_f32e7p1_no_faults(arrs: list[np.ndarray]):
    _test_embedded_parity_no_faults(
        arrs,
        EpScheme.D7P1,
        0b1111_1111_1111_1111_1111_1111_1111_0000,
        torch.uint32,
    )


@given(input_arrays(np.float32))
def test_f32e15p1_no_faults(arrs: list[np.ndarray]):
    _test_embedded_parity_no_faults(
        arrs,
        EpScheme.D15P1,
        0b1111_1111_1111_1111_1111_1111_1111_1100,
        torch.uint32,
    )


@given(input_arrays(np.float16))
def test_f16e3p1_no_faults(arrs: list[np.ndarray]):
    _test_embedded_parity_no_faults(
        arrs,
        EpScheme.D3P1,
        0b1111_1111_1111_0000,
        torch.uint16,
    )


@given(input_arrays(np.float16))
def test_f16e7p1_no_faults(arrs: list[np.ndarray]):
    _test_embedded_parity_no_faults(
        arrs,
        EpScheme.D7P1,
        0b1111_1111_1111_1100,
        torch.uint16,
    )


@given(input_arrays(np.float16))
def test_f16e15p1_no_faults(arrs: list[np.ndarray]):
    _test_embedded_parity_no_faults(
        arrs,
        EpScheme.D15P1,
        0b1111_1111_1111_1110,
        torch.uint16,
    )
