import numpy as np

from faultforge._rust import Fault
from faultforge._rust.secded import (
    Encoding,
    encode_f32,
)

# 16 data bits are protected by 5 parity bits and 1 overall parity bit.
_DATA_BITS_PER_CHUNK = 16
_ENCODED_BITS_PER_CHUNK = 22


def _encode() -> tuple[np.ndarray, Encoding]:
    source = np.arange(8, dtype=np.float32)
    return source, encode_f32([source], _DATA_BITS_PER_CHUNK)


def test_double_errors_not_collected_by_default():
    _, encoding = _encode()

    _, double_error_chunks = encoding.decode_f32()

    assert double_error_chunks is None


def test_no_double_errors_without_faults():
    source, encoding = _encode()

    (decoded,), double_error_chunks = encoding.decode_f32(collect_double_errors=True)

    assert double_error_chunks is not None
    assert double_error_chunks.dtype == np.uint64
    assert double_error_chunks.size == 0
    np.testing.assert_array_equal(decoded, source)


def test_double_error_chunk_indices():
    _, encoding = _encode()
    assert encoding.bit_count() % _ENCODED_BITS_PER_CHUNK == 0

    # Two flips in the same chunk are a detectable double error.
    for chunk_index in (1, 3):
        chunk_start = chunk_index * _ENCODED_BITS_PER_CHUNK
        encoding.apply_faults(
            [
                (Fault.flip(), chunk_start + 3),
                (Fault.flip(), chunk_start + 5),
            ]
        )

    _, double_error_chunks = encoding.decode_f32(collect_double_errors=True)

    assert double_error_chunks is not None
    assert double_error_chunks.dtype == np.uint64
    np.testing.assert_array_equal(double_error_chunks, [1, 3])
