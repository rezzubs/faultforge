import enum
import logging
from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

from faultforge import _core
from faultforge.encoding._tensor import TensorEncoderHelper, TensorEncodingHelper
from faultforge.encoding.sequence import TensorEncoding

_logger = logging.getLogger(__name__)


class EpScheme(enum.Enum):
    D3P1 = "d3p1"
    D7P1 = "d7p1"
    D15P1 = "d15p1"

    def _core(self) -> _core.EpScheme:
        match self:
            case EpScheme.D3P1:
                return _core.EpScheme.D3P1
            case EpScheme.D7P1:
                return _core.EpScheme.D7P1
            case EpScheme.D15P1:
                return _core.EpScheme.D15P1

    @staticmethod
    def default() -> EpScheme:
        """Return the default scheme."""
        return EpScheme.D3P1


@dataclass
class EmbeddedParityEncoder(TensorEncoderHelper):
    scheme: EpScheme = EpScheme.D3P1

    @override
    def encode_float32(self, t: Tensor) -> Tensor:
        with torch.no_grad():
            t_np = t.numpy(force=True)
        _core.embedded_parity_encode_f32(t_np, self.scheme._core())
        return torch.from_numpy(t_np)  # pyright: ignore[reportUnknownMemberType]

    @override
    def encode_float16(self, t: Tensor) -> Tensor:
        with torch.no_grad():
            t_np = t.view(torch.uint16).numpy(force=True)
        _core.embedded_parity_encode_u16(t_np, self.scheme._core())
        return torch.from_numpy(t_np).view(torch.float16)  # pyright: ignore[reportUnknownMemberType]

    @override
    def create_encoding(
        self,
        data: list[Tensor],
        bits_count: int,
        decoded_tensors: list[Tensor],
        dtype: torch.dtype,
    ) -> TensorEncoding:
        return EmbeddedParityEncoding(
            data,
            bits_count,
            decoded_tensors,
            dtype,
            True,
            _scheme=self.scheme,
        )

    @override
    def encoder_add_metadata(self, metadata: dict[str, str]) -> None:
        metadata["embedded_parity"] = "true"
        metadata["embedded_parity_scheme"] = self.scheme.value


@dataclass
class EmbeddedParityEncoding(TensorEncodingHelper):
    _scheme: EpScheme

    @override
    def encoding_clone(self) -> EmbeddedParityEncoding:
        data_tensors = [t.clone() for t in self._encoded_data]
        decoded_tensors = [t.clone() for t in self._decoded_tensors]
        return EmbeddedParityEncoding(
            data_tensors,
            self._bits_count,
            decoded_tensors,
            self._dtype,
            self._needs_recompute,
            self._scheme,
        )

    @override
    def decode_float16(self, t: Tensor) -> Tensor:
        encoded_np = t.view(torch.uint16).numpy(force=True).copy()
        _core.embedded_parity_decode_u16(encoded_np, self._scheme._core())
        return torch.from_numpy(encoded_np).view(torch.float16)  # pyright: ignore[reportUnknownMemberType]

    @override
    def decode_float32(self, t: Tensor) -> Tensor:
        encoded_np = t.numpy(force=True).copy()
        _core.embedded_parity_decode_f32(encoded_np, self._scheme._core())
        return torch.from_numpy(encoded_np)  # pyright: ignore[reportUnknownMemberType]
