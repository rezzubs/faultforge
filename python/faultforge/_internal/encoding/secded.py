"""Single Error Correction Double Error Detection (SECDED) encoding."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import override

import torch

from faultforge import _rust
from faultforge._internal.dtype import EncodingDtype
from faultforge._internal.encoding.abc import Encoder, Encoding
from faultforge._internal.tensor import tensor_list_dtype

logger = logging.getLogger(__name__)


@dataclass
class SecdedEncoder(Encoder):
    """The encoder for `SecdedEncoding`.

    Parameters:
        bits_per_chunk:
            The number of data bits to protect with a single hamming code.
            Equivalent to the memory line size in hardware. Can be any positive
            integer but multiples of 8 bits have better encoding/decoding
            performance.
    """

    bits_per_chunk: int

    @override
    def encode(self, ts: list[torch.Tensor]) -> Encoding:
        dtype = tensor_list_dtype(ts)
        if dtype is None:
            raise ValueError("Cannot encode an empty buffer")
        dtype = EncodingDtype.from_torch(dtype)

        # Store decoded tensor copies
        decoded_tensors = [t.clone() for t in ts]

        match dtype:
            case EncodingDtype.F32:
                with torch.no_grad():
                    rust_input = [t.flatten().numpy(force=True) for t in ts]
                encoded_data = _rust.encode_full_f32(rust_input, self.bits_per_chunk)
            case EncodingDtype.F16:
                with torch.no_grad():
                    rust_input = [
                        t.flatten().view(torch.uint16).numpy(force=True) for t in ts
                    ]
                encoded_data = _rust.encode_full_u16(rust_input, self.bits_per_chunk)

        return SecdedEncoding(
            encoded_data,
            decoded_tensors,
            dtype,
        )


@dataclass
class SecdedEncoding(Encoding):
    """A Single Error Correction Double Error Detection (SECDED) encoding based on hamming codes.

    This encoding is used to detect and correct single-bit errors in a memory
    line and detect double-bit errors. Note that the double error detection
    results are not currently used.
    """

    _encoded_data: _rust.FullEncoding
    """The blob that stores raw encoded data."""
    _decoded_tensors: list[torch.Tensor]
    """These are updated in-place during decoding."""
    _dtype: EncodingDtype
    _needs_recompute: bool = False

    @override
    def decode(self) -> list[torch.Tensor]:
        if not self._needs_recompute:
            logger.debug("Using cached decoded tensors")
            return self._decoded_tensors
        logger.debug("Recomputing decoded tensors")

        match self._dtype:
            case EncodingDtype.F32:
                decoded, ded_results = self._encoded_data.decode_full_f32()
                torch_decoded = [torch.from_numpy(t) for t in decoded]

            case EncodingDtype.F16:
                decoded, ded_results = self._encoded_data.decode_full_u16()
                torch_decoded = [
                    torch.from_numpy(t).view(torch.float16) for t in decoded
                ]

        # Update cached decoded tensors in-place
        for cached, decoded in zip(self._decoded_tensors, torch_decoded, strict=True):
            with torch.no_grad():
                _ = cached.flatten().copy_(decoded)

        # We're discarding the double error detection results for now but may
        # want to do something with them in the future.
        _ = ded_results

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

    @override
    def flip_bits(self, n: int) -> None:
        logger.debug("Invalidating decoded tensor cache due to fault injection.")
        self._needs_recompute = True
        self._encoded_data.flip_n_bits(n)

    @override
    def bit_count(self) -> int:
        return self._encoded_data.bit_count()
