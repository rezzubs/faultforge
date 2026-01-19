"""Single Error Correction Double Error Detection (SECDED) encoding."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import override

import torch

import faultforge._core
from faultforge.dtype import DnnDtype
from faultforge.encoding.encoding import Encoder, Encoding
from faultforge.tensor_ops import tensor_list_dtype

_logger = logging.getLogger(__name__)


@dataclass
class SecdedEncoder(Encoder):
    """The encoder for :class:`SecdedEncoding`.

    :param bits_per_chunk: The number of data bits to protect with a single
    hamming code. Equivalent to the memory line size in hardware. Can be any
    positive integer but multiples of 8 bits have better performance (CPU time
    not accuracy).
    """

    bits_per_chunk: int

    @override
    def encoder_encode_tensor_list(self, ts: list[torch.Tensor]) -> Encoding:
        dtype = tensor_list_dtype(ts)
        if dtype is None:
            raise ValueError("Cannot encode an empty buffer")

        # Store decoded tensor copies
        decoded_tensors = [t.clone() for t in ts]

        match DnnDtype.from_torch(dtype):
            case DnnDtype.Float32:
                with torch.no_grad():
                    rust_input = [t.flatten().numpy(force=True) for t in ts]
                    encoded_data = faultforge._core.encode_full_f32(
                        rust_input, self.bits_per_chunk
                    )
            case DnnDtype.Float16:
                with torch.no_grad():
                    rust_input = [
                        t.flatten().view(torch.uint16).numpy(force=True) for t in ts
                    ]
                    encoded_data = faultforge._core.encode_full_u16(
                        rust_input, self.bits_per_chunk
                    )

        return SecdedEncoding(
            encoded_data,
            decoded_tensors,
            dtype,
        )

    @override
    def encoder_add_metadata(self, metadata: dict[str, str]) -> None:
        metadata["chunk_size"] = str(self.bits_per_chunk)


@dataclass
class SecdedEncoding(Encoding):
    """A Single Error Correction Double Error Detection (SECDED) encoding based on hamming codes.

    This encoding is used to detect and correct single-bit errors in a memory
    line and detect double-bit errors. Note that the double error detection
    results are not currently used.
    """

    _encoded_data: faultforge._core.FullEncoding
    _decoded_tensors: list[torch.Tensor]
    _dtype: torch.dtype
    _needs_recompute: bool = False

    @override
    def encoding_decode_tensor_list(self) -> list[torch.Tensor]:
        if not self._needs_recompute:
            _logger.debug("Using cached decoded tensors")
            return self._decoded_tensors
        _logger.debug("Recomputing decoded tensors")

        match DnnDtype.from_torch(self._dtype):
            case DnnDtype.Float32:
                decoded, ded_results = self._encoded_data.decode_full_f32()
                # HACK: There's nothing we can do about this warning without an upstream fix.
                torch_decoded = [
                    torch.from_numpy(t)  # pyright: ignore[reportUnknownMemberType]
                    for t in decoded
                ]

            case DnnDtype.Float16:
                decoded, ded_results = self._encoded_data.decode_full_u16()
                torch_decoded = [
                    torch.from_numpy(t).view(torch.float16)  # pyright: ignore[reportUnknownMemberType]
                    for t in decoded
                ]

        # Update cached decoded tensors in-place
        for cached, decoded in zip(self._decoded_tensors, torch_decoded, strict=True):
            with torch.no_grad():
                _ = cached.flatten().copy_(decoded)

        # TODO: we discard the double error detection results for now but may
        # want to do something with them in the future.
        _ = ded_results

        self._needs_recompute = False
        return self._decoded_tensors

    @override
    def encoding_clone(self) -> SecdedEncoding:
        return SecdedEncoding(
            self._encoded_data.clone(),
            self._decoded_tensors,
            self._dtype,
            self._needs_recompute,
        )

    @override
    def encoding_flip_n_bits(self, n: int):
        _logger.debug("Invalidating decoded tensor cache due to fault injection.")
        self._needs_recompute = True
        self._encoded_data.flip_n_bits(n)

    @override
    def encoding_bits_count(self) -> int:
        return self._encoded_data.bits_count()
