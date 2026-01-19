"""Bit pattern based encoding for Systems."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from typing_extensions import override

import faultforge._core
from faultforge.dtype import DnnDtype
from faultforge.encoding.encoding import Encoder, Encoding
from faultforge.tensor_ops import tensor_list_dtype

_logger = logging.getLogger(__name__)


@dataclass
class _RangeInclusive:
    start: int
    end: int

    def __init__(self, start: int | str, end: int | str) -> None:
        try:
            if (start := int(start)) < 0:
                raise ValueError("Range values must be non-negative")
        except ValueError as e:
            raise ValueError(f"Invalid range start value `{start}`\n-> {e}") from e

        try:
            if (end := int(end)) < 0:
                raise ValueError("Range values must be non-negative")
        except ValueError as e:
            raise ValueError(f"Invalid range end value `{end}`\n-> {e}") from e

        self.start = start
        self.end = end

    @override
    def __repr__(self) -> str:
        if self.start == self.end:
            return str(self.start)
        elif self.end > self.start:
            return f"{self.start}-{self.end}"
        else:
            raise RuntimeError(f"Invaild range {self.start}-{self.end}")

    def __iter__(self) -> Iterator[int]:
        return range(self.start, self.end + 1).__iter__()


def parse_range_or_int(text: str) -> _RangeInclusive | int:
    split = text.split("-")
    if len(split) == 1:
        try:
            return int(split[0])
        except ValueError as e:
            raise ValueError(
                f"Expected an integer or a `-` separated range, got `{text}`\n-> {e}"
            ) from e

    try:
        [start, end] = split
        return _RangeInclusive(start, end)
    except ValueError as e:
        raise ValueError(
            f"Invalid range `{text}`, expected two integers separated by a `-`\n-> {e}"
        ) from e


@dataclass
class _Ranges:
    ranges: list[_RangeInclusive]

    @override
    def __repr__(self) -> str:
        return "_".join([r.__repr__() for r in self.ranges])

    def start_new(self, start: int) -> None:
        self.ranges.append(_RangeInclusive(start, start))

    def extend_last(self) -> None:
        self.ranges[-1].end += 1


@dataclass
class BitPattern:
    """A pattern of bits to protect.

    The length is inferred from usage.
    """

    bits: set[int]

    def bits_count(self) -> int:
        return len(self.bits)

    @override
    def __repr__(self) -> str:
        if self.bits_count() == 0:
            return "empty_bit_pattern"

        bits = list(self.bits)
        bits.sort()

        ranges = _Ranges([])
        previous = None
        for bit in bits:
            if previous is None:
                ranges.start_new(bit)
                previous = bit
                continue

            if previous == bit - 1:
                ranges.extend_last()
            else:
                ranges.start_new(bit)

            previous = bit

        return ranges.__repr__()

    @classmethod
    def parse(cls, text: str) -> BitPattern:
        bits: set[int] = set()

        try:
            for part in text.split("_"):
                range_or_int = parse_range_or_int(part)

                if isinstance(range_or_int, int):
                    bits.add(range_or_int)
                    continue
                assert isinstance(range_or_int, _RangeInclusive)

                for i in range_or_int:
                    bits.add(i)
        except ValueError as e:
            raise ValueError(
                f"Invalid bit pattern `{text}`, expected a `_` separated list of integers or ranges\n-> {e}"
            ) from e

        return cls(bits)


@dataclass
class BitPatternEncoder(Encoder):
    """An encoder for :class:`BitPatternEncoding`."""

    pattern: BitPattern
    pattern_length: int
    bits_per_chunk: int

    @override
    def encoder_encode_tensor_list(self, ts: list[torch.Tensor]) -> Encoding:
        dtype = tensor_list_dtype(ts)
        if dtype is None:
            raise ValueError("Cannot encode an empty buffer")

        flattened = [t.flatten() for t in ts]

        match DnnDtype.from_torch(dtype):
            case DnnDtype.Float32:
                with torch.no_grad():
                    rust_input = [t.numpy(force=True) for t in flattened]
                    data = faultforge._core.encode_bit_pattern_f32(
                        rust_input,
                        self.pattern.bits,
                        self.pattern_length,
                        self.bits_per_chunk,
                    )
            case DnnDtype.Float16:
                with torch.no_grad():
                    rust_input = [
                        t.view(torch.uint16).numpy(force=True) for t in flattened
                    ]
                    data = faultforge._core.encode_bit_pattern_u16(
                        rust_input,
                        self.pattern.bits,
                        self.pattern_length,
                        self.bits_per_chunk,
                    )

        # These will be decoded into.
        decoded_tensors = [t.clone() for t in ts]
        return BitPatternEncoding(data, decoded_tensors, dtype)

    @override
    def encoder_add_metadata(self, metadata: dict[str, str]) -> None:
        metadata["bit_pattern"] = f"{self.pattern}({self.pattern_length})"
        metadata["chunk_size"] = str(self.bits_per_chunk)


@dataclass
class BitPatternEncoding(Encoding):
    """Bit pattern based SECDED encoding.

    This encoding provides Hamming Code based encoding for Systems. The memory
    will be partitioned into protected and unprotected regions by a bit pattern.

    See also: :mod:`faultforge.encoding.secded`.
    """

    _encoded_data: faultforge._core.BitPatternEncoding
    _decoded_tensors: list[torch.Tensor]
    _dtype: torch.dtype
    _needs_recompute: bool = False

    @override
    def encoding_decode_tensor_list(self) -> list[torch.Tensor]:
        if not self._needs_recompute:
            _logger.debug("Using cached decoded tensors")
            return self._decoded_tensors
        _logger.debug("Recomputing decoded tensors")

        element_counts = [t.numel() for t in self._decoded_tensors]
        match DnnDtype.from_torch(self._dtype):
            case DnnDtype.Float32:
                decoded, ded_results = self._encoded_data.decode_bit_pattern_f32(
                    element_counts
                )
                # HACK: There's nothing we can do about this warning without an upstream fix.
                torch_decoded = [
                    torch.from_numpy(t)  # pyright: ignore[reportUnknownMemberType]
                    for t in decoded
                ]

            case DnnDtype.Float16:
                decoded, ded_results = self._encoded_data.decode_bit_pattern_u16(
                    element_counts
                )
                # HACK: There's nothing we can do about this warning without an upstream fix.
                torch_decoded = [
                    torch.from_numpy(t).view(torch.float16)  # pyright: ignore[reportUnknownMemberType]
                    for t in decoded
                ]

        # Update cached decoded tensors in-place
        for output, decoded in zip(self._decoded_tensors, torch_decoded, strict=True):
            with torch.no_grad():
                _ = output.flatten().copy_(decoded)

        # TODO: we discard the double error detection results for now but may
        # want to do something with them in the future.
        _ = ded_results

        self._needs_recompute = False
        return self._decoded_tensors

    @override
    def encoding_flip_n_bits(self, n: int):
        _logger.debug("Invalidating decoded tensor cached due to fault injection.")
        self._needs_recompute = True
        self._encoded_data.flip_n_bits(n)

    @override
    def encoding_clone(self) -> BitPatternEncoding:
        copied_tensors = [t.clone() for t in self._decoded_tensors]
        return BitPatternEncoding(
            self._encoded_data.clone(),
            copied_tensors,
            self._dtype,
            self._needs_recompute,
        )

    @override
    def encoding_bits_count(self) -> int:
        return self._encoded_data.bits_count()
