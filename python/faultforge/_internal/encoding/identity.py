from dataclasses import dataclass
from typing import override

from torch import Tensor

from faultforge._internal.dtype import EncodingDtype
from faultforge._internal.encoding.abc import TensorEncoder, TensorEncoding
from faultforge._internal.tensor import (
    tensor_list_dtype,
    tensor_list_fault_injection,
)


class IdentityEncoder(TensorEncoder):
    @override
    def encode(self, ts: list[Tensor]) -> IdentityEncoding:
        dtype = tensor_list_dtype(ts)
        if dtype is None:
            raise ValueError("Cannot encode an empty list")
        dtype = EncodingDtype.from_torch(dtype)
        bit_count = sum(t.numel() for t in ts) * dtype.bit_count()
        return IdentityEncoding(_tensors=list(ts), _bit_count=bit_count)


@dataclass
class IdentityEncoding(TensorEncoding):
    _tensors: list[Tensor]
    _bit_count: int

    @override
    def encoded_tensors(self) -> list[Tensor]:
        return self._tensors

    @override
    def trigger_recompute(self) -> None:
        pass

    @override
    def decode(self) -> list[Tensor]:
        return self._tensors

    @override
    def clone(self) -> IdentityEncoding:
        return IdentityEncoding(
            _tensors=[t.clone() for t in self._tensors], _bit_count=self._bit_count
        )

    @override
    def flip_bits(self, n: int) -> None:
        tensor_list_fault_injection(self._tensors, n)

    @override
    def bit_count(self) -> int:
        return self._bit_count
