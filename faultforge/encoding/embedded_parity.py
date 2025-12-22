import logging
from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

import faultforge._core
from faultforge.encoding._tensor import TensorEncoderHelper, TensorEncodingHelper
from faultforge.encoding.sequence import TensorEncoding

_logger = logging.getLogger(__name__)


class EmbeddedParityEncoder(TensorEncoderHelper):
    @override
    def encode_float32(self, t: Tensor) -> Tensor:
        with torch.no_grad():
            t_np = t.numpy(force=True)
        faultforge._core.embedded_parity_encode_f32(t_np)
        return torch.from_numpy(t_np)  # pyright: ignore[reportUnknownMemberType]

    @override
    def encode_float16(self, t: Tensor) -> Tensor:
        with torch.no_grad():
            t_np = t.view(torch.uint16).numpy(force=True)
        faultforge._core.embedded_parity_encode_u16(t_np)
        return torch.from_numpy(t_np).view(torch.float16)  # pyright: ignore[reportUnknownMemberType]

    @override
    def create_encoding(
        self,
        data: list[Tensor],
        bits_count: int,
        decoded_tensors: list[Tensor],
        dtype: torch.dtype,
    ) -> TensorEncoding:
        return EmbeddedParityEncoding(data, bits_count, decoded_tensors, dtype)

    @override
    def encoder_add_metadata(self, metadata: dict[str, str]) -> None:
        metadata["embedded_parity"] = "true"


@dataclass
class EmbeddedParityEncoding(TensorEncodingHelper):
    @override
    def decode_float16(self, t: Tensor) -> Tensor:
        encoded_np = t.view(torch.uint16).numpy(force=True).copy()
        faultforge._core.embedded_parity_decode_u16(encoded_np)
        return torch.from_numpy(encoded_np).view(torch.float16)  # pyright: ignore[reportUnknownMemberType]

    @override
    def decode_float32(self, t: Tensor) -> Tensor:
        encoded_np = t.numpy(force=True).copy()
        faultforge._core.embedded_parity_decode_f32(encoded_np)
        return torch.from_numpy(encoded_np)  # pyright: ignore[reportUnknownMemberType]
