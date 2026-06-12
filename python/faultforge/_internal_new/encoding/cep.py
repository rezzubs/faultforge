"""Chunked Embedded Parity (CEP)."""

import enum
import logging
from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

from faultforge import _rust
from faultforge._internal_new.dtype import EncodingDtype
from faultforge._internal_new.encoding.abc import (
    InPlaceEncoder,
    InPlaceEncoding,
    TensorEncoding,
)

_logger = logging.getLogger(__name__)


class EpScheme(enum.Enum):
    """How many data bits to use per parity bit.

    D3P1 should result in the best accuracy in most cases because it results in
    the biggest number of chunks per parameter - it can mitigate more faults.
    """

    D3P1 = "d3p1"
    D7P1 = "d7p1"
    D15P1 = "d15p1"

    def _to_rust(self) -> _rust.EpScheme:
        match self:
            case EpScheme.D3P1:
                return _rust.EpScheme.D3P1
            case EpScheme.D7P1:
                return _rust.EpScheme.D7P1
            case EpScheme.D15P1:
                return _rust.EpScheme.D15P1

    @staticmethod
    def default() -> EpScheme:
        """Return the default scheme."""
        return EpScheme.D3P1


@dataclass
class CepEncoder(InPlaceEncoder):
    """An encoder for `CepEncoding`."""

    scheme: EpScheme = EpScheme.D3P1

    @override
    def encode_float32(self, t: Tensor) -> Tensor:
        with torch.no_grad():
            t_np = t.numpy(force=True)
        _rust.embedded_parity_encode_f32(t_np, self.scheme._to_rust())
        return torch.from_numpy(t_np)

    @override
    def encode_float16(self, t: Tensor) -> Tensor:
        with torch.no_grad():
            t_np = t.view(torch.uint16).numpy(force=True)
        _rust.embedded_parity_encode_u16(t_np, self.scheme._to_rust())
        return torch.from_numpy(t_np).view(torch.float16)

    @override
    def create_encoding(
        self,
        data: list[Tensor],
        bits_count: int,
        dtype: EncodingDtype,
    ) -> TensorEncoding:
        return CepEncoding(
            _encoded_data=data,
            _bits_count=bits_count,
            _decoded_tensors=None,
            _dtype=dtype,
            _scheme=self.scheme,
        )


@dataclass
class CepEncoding(InPlaceEncoding):
    """An encoding that embeds parity bits into the data.

    The higher bits of the data are chunked based on the provided scheme and
    each chunk will be given a matching parity bit which is embedded inside the
    lower bits. If a parity bit doesn't match during decoding, the corresponding
    chunk will be set to zero.
    """

    _scheme: EpScheme

    @override
    def clone(self) -> CepEncoding:
        cloned_data = [t.clone() for t in self._encoded_data]
        if self._decoded_tensors is not None:
            cloned_decoded = [t.clone() for t in self._decoded_tensors]
        else:
            cloned_decoded = None

        return CepEncoding(
            _encoded_data=cloned_data,
            _bits_count=self._bits_count,
            _decoded_tensors=cloned_decoded,
            _dtype=self._dtype,
            _scheme=self._scheme,
        )

    @override
    def decode_float16(self, t: Tensor) -> Tensor:
        encoded_np = t.view(torch.uint16).numpy(force=True).copy()
        _rust.embedded_parity_decode_u16(encoded_np, self._scheme._to_rust())
        return torch.from_numpy(encoded_np).view(torch.float16)

    @override
    def decode_float32(self, t: Tensor) -> Tensor:
        encoded_np = t.numpy(force=True).copy()
        _rust.embedded_parity_decode_f32(encoded_np, self._scheme._to_rust())
        return torch.from_numpy(encoded_np)
