"""Single Error Correction Double Error Detection (SECDED) encoding."""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

import torch

from faultforge._impl.dtype import EncodingDtype
from faultforge._impl.encoding.abc import Encoder, Encoding
from faultforge._impl.fault import (
    Fault,
    fault_to_rust,
)
from faultforge._impl.fingerprint import Fingerprint
from faultforge._impl.progress import Progress, stage
from faultforge._impl.tensor import tensor_list_dtype
from faultforge._rust import secded

_logger = logging.getLogger(__name__)


@dataclass
class SecdedEncoder(Encoder):
    """Encodes tensors with Single Error Correction Double Error Detection (SECDED) hamming codes.

    Each chunk of `bits_per_chunk` data bits gets its own hamming code, which
    allows single-bit errors within the chunk to be corrected and double-bit
    errors to be detected during decoding. Double-error detection results are
    not currently used.
    """

    bits_per_chunk: int
    """The number of data bits protected by a single hamming code.

    Equivalent to the memory line size in hardware. Must be a positive
    multiple of 8.
    """

    def __post_init__(self) -> None:
        if self.bits_per_chunk <= 0 or self.bits_per_chunk % 8 != 0:
            raise ValueError(
                "bits_per_chunk must be a positive multiple of 8, "
                f"got {self.bits_per_chunk}"
            )

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(
            kind="secded", scalars={"bits_per_chunk": self.bits_per_chunk}
        )

    @override
    def encode(
        self, ts: list[torch.Tensor], *, progress: Progress | None = None
    ) -> Encoding:
        dtype = tensor_list_dtype(ts)
        if dtype is None:
            raise ValueError("Cannot encode an empty buffer")
        dtype = EncodingDtype.from_torch(dtype)

        # SECDED is lossless, so copies of the inputs are the decode of the
        # encoding without faults. The inputs themselves can't be kept, see
        # `Encoder.encode`.
        decoded_tensors = [t.detach().clone() for t in ts]

        # The encoding only sees the bytes of the tensors in memory. Fault
        # targets address them in little-endian order, and changing that would
        # change the results of existing experiments.
        if sys.byteorder != "little":
            raise NotImplementedError("SECDED needs a little-endian host")

        with stage(progress, "Encoding (SECDED)"), torch.no_grad():
            rust_input = [
                _byte_view(t.detach().flatten()).numpy(force=True) for t in ts
            ]
            encoded_data = secded.encode(rust_input, self.bits_per_chunk)

        return SecdedEncoding(
            encoded_data,
            decoded_tensors,
            dtype,
        )


@dataclass
class SecdedEncoding(Encoding):
    """The encoding produced by `SecdedEncoder`. See `SecdedEncoder` for details."""

    _encoded_data: secded.Encoding
    """The blob that stores raw encoded data."""
    _decoded_tensors: list[torch.Tensor]
    """These are updated in-place during decoding."""
    _dtype: EncodingDtype
    _needs_recompute: bool = False

    @override
    def decode(self) -> list[torch.Tensor]:
        if not self._needs_recompute:
            _logger.debug("Using cached decoded tensors")
            return self._decoded_tensors
        _logger.debug("Recomputing decoded tensors")

        _decode_into(self._encoded_data, self._decoded_tensors)

        self._needs_recompute = False
        return self._decoded_tensors

    @override
    def clone(self) -> SecdedEncoding:
        return SecdedEncoding(
            self._encoded_data.clone(),
            [t.clone() for t in self._decoded_tensors],
            self._dtype,
            self._needs_recompute,
        )

    def _invalidate_decoded_cache(self) -> None:
        if not self._needs_recompute:
            _logger.debug("Invalidating decoded tensor cache due to fault injection.")
        self._needs_recompute = True

    @override
    def apply_fault(self, fault: Fault, target_bit: int) -> None:
        self._invalidate_decoded_cache()
        self._encoded_data.apply_fault(fault_to_rust(fault), target_bit)

    @override
    def apply_faults(self, faults: Sequence[tuple[Fault, int]]) -> None:
        self._invalidate_decoded_cache()
        self._encoded_data.apply_faults(
            [(fault_to_rust(fault), target_bit) for fault, target_bit in faults]
        )

    @override
    def bit_count(self) -> int:
        return self._encoded_data.bit_count()


def _byte_view(t: torch.Tensor) -> torch.Tensor:
    """View the bytes of a contiguous tensor as a flat uint8 tensor."""
    assert t.is_contiguous()
    # `view(-1)` can keep a stride other than 1: torch treats a tensor with a
    # single element as contiguous whatever its strides are. Its element is
    # still at the storage offset, so the explicit strides are valid.
    return t.as_strided((t.numel(),), (1,)).view(torch.uint8)


def _decode_into(encoded: secded.Encoding, dest: list[torch.Tensor]) -> None:
    """Decode `encoded` into `dest`, one tensor per encoded array.

    The bindings can only write into contiguous CPU memory, so every tensor is
    decoded into a CPU staging buffer first and then copied into place. The
    buffer is reused, so it only needs to fit the largest tensor. The copy is
    much faster than the decode.

    `dest` must have the layout of the encoded tensors.
    """
    byte_counts = encoded.array_byte_counts()
    assert len(dest) == len(byte_counts)
    staging = torch.empty(max(byte_counts), dtype=torch.uint8)

    # NOTE: Double error detection results aren't collected because nothing
    # uses them yet. At scale, collecting them can cost a lot of memory.
    with torch.no_grad():
        for index, (t, byte_count) in enumerate(zip(dest, byte_counts, strict=True)):
            staged = staging[:byte_count]
            encoded.decode_into(index, staged.numpy())
            _ = t.copy_(staged.view(t.dtype).view(t.shape))
