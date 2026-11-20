"""Chunked Embedded Parity (CEP)."""

lazy import logging
lazy from dataclasses import dataclass
lazy from typing import override

lazy import numpy as np
lazy import torch
lazy from torch import Tensor

lazy from faultforge._impl.dtype import EncodingDtype
lazy from faultforge._impl.encoding.abc import (
    InPlaceEncoder,
    InPlaceEncoding,
    TensorEncoding,
)
lazy from faultforge._impl.encoding.cep_scheme import CepScheme
lazy from faultforge._impl.fingerprint import Fingerprint
lazy from faultforge._impl.tensor import to_owned_numpy
lazy from faultforge._rust import cep

_logger = logging.getLogger(__name__)


@dataclass
class CepEncoder(InPlaceEncoder):
    """Encodes tensors with Chunked Embedded Parity (CEP).

    The higher bits of each element are chunked based on `scheme` and each
    chunk is given a matching parity bit embedded inside the lower bits. If a
    parity bit doesn't match during decoding, the corresponding chunk is set
    to zero.
    """

    scheme: CepScheme = CepScheme.D3P1

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="cep", scalars={"scheme": self.scheme.value})

    @override
    def encode_float32(self, t: Tensor) -> Tensor:
        t_np = to_owned_numpy(t)
        cep.encode_f32(t_np, self.scheme._to_rust())
        return torch.from_numpy(t_np)

    @override
    def encode_float16(self, t: Tensor) -> Tensor:
        t_np = to_owned_numpy(t).view(np.uint16)
        cep.encode_u16(t_np, self.scheme._to_rust())
        return torch.from_numpy(t_np).view(torch.float16)

    @override
    def create_encoding(
        self,
        data: list[Tensor],
        bit_count: int,
        dtype: EncodingDtype,
    ) -> TensorEncoding:
        return CepEncoding(
            _encoded_data=data,
            _bit_count=bit_count,
            _dtype=dtype,
            _scheme=self.scheme,
        )


@dataclass
class CepEncoding(InPlaceEncoding):
    """The encoding produced by `CepEncoder`. See `CepEncoder` for details."""

    _scheme: CepScheme

    @override
    def clone(self) -> CepEncoding:
        return CepEncoding(
            _encoded_data=[t.clone() for t in self._encoded_data],
            _bit_count=self._bit_count,
            _dtype=self._dtype,
            _scheme=self._scheme,
        )

    @override
    def decode_float16(self, t: Tensor) -> Tensor:
        encoded_np = t.view(torch.uint16).numpy(force=True).copy()
        cep.decode_u16(encoded_np, self._scheme._to_rust())
        return torch.from_numpy(encoded_np).view(torch.float16)

    @override
    def decode_float32(self, t: Tensor) -> Tensor:
        encoded_np = t.numpy(force=True).copy()
        cep.decode_f32(encoded_np, self._scheme._to_rust())
        return torch.from_numpy(encoded_np)
