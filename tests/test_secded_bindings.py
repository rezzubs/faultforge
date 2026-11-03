import numpy as np
import pytest

from faultforge._rust import Fault
from faultforge._rust.secded import (
    Encoding,
    encode,
)

# 16 data bits are protected by 5 parity bits and 1 overall parity bit.
_DATA_BITS_PER_CHUNK = 16
_ENCODED_BITS_PER_CHUNK = 22


def _encode() -> tuple[np.ndarray, Encoding]:
    source = np.arange(8, dtype=np.float32)
    return source, encode([source.view(np.uint8)], _DATA_BITS_PER_CHUNK)


def _decode(
    encoding: Encoding, *, collect_double_errors: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    (byte_count,) = encoding.array_byte_counts()
    # Start from a different value to check that every byte is written.
    decoded = np.full(byte_count, 0xFF, dtype=np.uint8)
    double_error_chunks = encoding.decode_into(
        0, decoded, collect_double_errors=collect_double_errors
    )
    return decoded.view(np.float32), double_error_chunks


def test_double_errors_not_collected_by_default():
    _, encoding = _encode()

    _, double_error_chunks = _decode(encoding)

    assert double_error_chunks is None


def test_no_double_errors_without_faults():
    source, encoding = _encode()

    decoded, double_error_chunks = _decode(encoding, collect_double_errors=True)

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

    _, double_error_chunks = _decode(encoding, collect_double_errors=True)

    assert double_error_chunks is not None
    assert double_error_chunks.dtype == np.uint64
    np.testing.assert_array_equal(double_error_chunks, [1, 3])


def test_encode_rejects_bits_per_chunk_not_multiple_of_8():
    with pytest.raises(ValueError, match="multiple of 8"):
        _ = encode([np.arange(8, dtype=np.uint8)], 12)


def test_arrays_are_decoded_separately():
    # 3 and 5 bytes in chunks of 2 bytes: chunk 1 holds the end of the first
    # array and the start of the second.
    first = np.arange(3, dtype=np.uint8)
    second = np.arange(10, 15, dtype=np.uint8)
    encoding = encode([first, second], _DATA_BITS_PER_CHUNK)
    assert encoding.array_byte_counts() == [3, 5]

    # Two flips in chunk 1 are a detectable double error.
    chunk_start = 1 * _ENCODED_BITS_PER_CHUNK
    encoding.apply_faults(
        [(Fault.flip(), chunk_start + 3), (Fault.flip(), chunk_start + 5)]
    )

    decoded_second = np.zeros(5, dtype=np.uint8)
    second_double_errors = encoding.decode_into(
        1, decoded_second, collect_double_errors=True
    )
    decoded_first = np.zeros(3, dtype=np.uint8)
    first_double_errors = encoding.decode_into(
        0, decoded_first, collect_double_errors=True
    )

    # Chunk 1 is reported for both arrays.
    assert first_double_errors is not None
    assert second_double_errors is not None
    np.testing.assert_array_equal(first_double_errors, [1])
    np.testing.assert_array_equal(second_double_errors, [1])
    # Only chunk 1 has faults, so the bytes in other chunks are intact.
    np.testing.assert_array_equal(decoded_first[:2], first[:2])
    np.testing.assert_array_equal(decoded_second[1:], second[1:])


def test_decode_rejects_wrong_dest_length():
    _, encoding = _encode()

    with pytest.raises(ValueError, match="should have 32 bytes, got 31"):
        _ = encoding.decode_into(0, np.zeros(31, dtype=np.uint8))


def test_decode_rejects_out_of_bounds_array():
    _, encoding = _encode()

    with pytest.raises(IndexError, match="out of bounds"):
        _ = encoding.decode_into(1, np.zeros(32, dtype=np.uint8))


def test_decode_rejects_non_contiguous_dest():
    _, encoding = _encode()

    with pytest.raises(ValueError, match="not contiguous"):
        _ = encoding.decode_into(0, np.zeros(64, dtype=np.uint8)[::2])


def test_encode_rejects_non_contiguous_input():
    with pytest.raises(ValueError, match="not contiguous"):
        _ = encode([np.zeros(64, dtype=np.uint8)[::2]], _DATA_BITS_PER_CHUNK)
