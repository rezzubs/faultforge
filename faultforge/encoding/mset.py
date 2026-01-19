"""Most Significant Exponent bit Triplication (MSET)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

import faultforge._core
from faultforge.encoding._tensor import TensorEncoderHelper, TensorEncodingHelper
from faultforge.encoding.sequence import TensorEncoding

_logger = logging.getLogger(__name__)


class MsetEncoder(TensorEncoderHelper):
    """An encoder for :class:`MsetEncoding`."""

    @override
    def encode_float32(self, t: Tensor) -> Tensor:
        with torch.no_grad():
            t_np = t.numpy(force=True)
        faultforge._core.bit30_encode_f32(t_np)
        return torch.from_numpy(t_np)  # pyright: ignore[reportUnknownMemberType]

    @override
    def encode_float16(self, t: Tensor) -> Tensor:
        with torch.no_grad():
            t_np = t.view(torch.uint16).numpy(force=True)
        faultforge._core.bit14_encode_u16(t_np)
        return torch.from_numpy(t_np).view(torch.float16)  # pyright: ignore[reportUnknownMemberType]

    @override
    def create_encoding(
        self,
        data: list[Tensor],
        bits_count: int,
        decoded_tensors: list[Tensor],
        dtype: torch.dtype,
    ) -> TensorEncoding:
        return MsetEncoding(data, bits_count, decoded_tensors, dtype, True)

    @override
    def encoder_add_metadata(self, metadata: dict[str, str]) -> None:
        metadata["msb_duplicated"] = "true"


@dataclass
class MsetEncoding(TensorEncodingHelper):
    """MSET based encoding.

    MSET stands for Most Significant Exponent bit Triplication. Depending on the
    data type, the second highest bit will be copied to the lowest bits. A
    majority voting scheme will be used to determine the final value of the
    exponent bit.
    """

    @override
    def decode_float16(self, t: Tensor) -> Tensor:
        encoded_np = t.view(torch.uint16).numpy(force=True).copy()
        faultforge._core.bit14_decode_u16(encoded_np)
        return torch.from_numpy(encoded_np).view(torch.float16)  # pyright: ignore[reportUnknownMemberType]

    @override
    def decode_float32(self, t: Tensor) -> Tensor:
        encoded_np = t.numpy(force=True).copy()
        faultforge._core.bit30_decode_f32(encoded_np)
        return torch.from_numpy(encoded_np)  # pyright: ignore[reportUnknownMemberType]

    @override
    def encoding_clone(self) -> MsetEncoding:
        copied_data = [t.clone() for t in self._encoded_data]
        copied_decoded = [t.clone() for t in self._decoded_tensors]
        return self.__class__(
            copied_data,
            self._bits_count,
            copied_decoded,
            self._dtype,
            self._needs_recompute,
        )
