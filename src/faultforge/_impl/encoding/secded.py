"""Single Error Correction Double Error Detection (SECDED) encoding."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

import torch

from faultforge._impl.dtype import EncodingDtype
from faultforge._impl.encoding.abc import (
    Encoder,
    Encoding,
    check_decode_dest,
)
from faultforge._impl.fault import (
    Fault,
    fault_to_rust,
)
from faultforge._impl.fingerprint import Fingerprint
from faultforge._impl.progress import Progress, stage
from faultforge._impl.tensor import tensor_list_dtype
from faultforge._rust import secded


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
            [t.shape for t in ts],
            dtype,
        )


@dataclass
class SecdedEncoding(Encoding):
    """The encoding produced by `SecdedEncoder`. See `SecdedEncoder` for details."""

    _encoded_data: secded.Encoding
    """The blob that stores raw encoded data."""
    _shapes: list[torch.Size]
    """The shapes of the encoded tensors."""
    _dtype: EncodingDtype

    @override
    def decode_into(self, dest: list[torch.Tensor]) -> None:
        dtype = self._dtype.to_torch()
        check_decode_dest(dest, [(shape, dtype) for shape in self._shapes])
        _decode_into(self._encoded_data, dest)

    @override
    def decode_changes_into(self, dest: list[torch.Tensor]) -> None:
        dtype = self._dtype.to_torch()
        check_decode_dest(dest, [(shape, dtype) for shape in self._shapes])
        _decode_changes_into(self._encoded_data, dest)

    @override
    def empty_decoded(self) -> list[torch.Tensor]:
        dtype = self._dtype.to_torch()
        return [torch.empty(shape, dtype=dtype) for shape in self._shapes]

    @override
    def clone(self) -> SecdedEncoding:
        return SecdedEncoding(
            self._encoded_data.clone(),
            self._shapes,
            self._dtype,
        )

    @override
    def apply_fault(self, fault: Fault, target_bit: int) -> None:
        self._encoded_data.apply_fault(fault_to_rust(fault), target_bit)

    @override
    def apply_faults(self, faults: Sequence[tuple[Fault, int]]) -> None:
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

    `dest` must have the layout of the encoded tensors, see
    `SecdedEncoding.decode_into`.
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


def _decode_changes_into(encoded: secded.Encoding, dest: list[torch.Tensor]) -> None:
    """Write the bytes of `encoded` that faults changed into `dest`.

    Only the changed bytes are moved to the device of each tensor, so this is
    cheap on CUDA too.

    `dest` must have the layout of the encoded tensors and hold the decode
    without faults, see `SecdedEncoding.decode_changes_into`.
    """
    assert len(dest) == len(encoded.array_byte_counts())

    with torch.no_grad():
        for index, t in enumerate(dest):
            byte_offsets, byte_values = encoded.decoded_changes(index)
            if byte_offsets.size == 0:
                continue

            # Only contiguous tensors can be viewed as bytes. A contiguous copy
            # also holds the decode without faults, so the changes can be
            # written into it and copied back.
            contiguous = t if t.is_contiguous() else t.contiguous()
            byte_view = _byte_view(contiguous)
            byte_view[torch.from_numpy(byte_offsets).to(t.device)] = torch.from_numpy(
                byte_values
            ).to(t.device)
            if contiguous is not t:
                _ = t.copy_(contiguous)
