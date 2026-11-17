"""An encoder that stores tensors unmodified, without any protection."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

from faultforge._impl.dtype import EncodingDtype
from faultforge._impl.encoding.abc import (
    TensorEncoder,
    TensorEncoding,
    check_decode_dest,
)
from faultforge._impl.fault import Fault
from faultforge._impl.fingerprint import Fingerprint
from faultforge._impl.progress import Progress
from faultforge._impl.tensor import (
    tensor_list_dtype,
    tensor_list_fault,
    tensor_list_faults,
)


class IdentityEncoder(TensorEncoder):
    """An encoder for `IdentityEncoding`."""

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="identity")

    @override
    def encode(
        self, ts: list[Tensor], *, progress: Progress | None = None
    ) -> IdentityEncoding:
        _ = progress
        dtype = tensor_list_dtype(ts)
        if dtype is None:
            raise ValueError("Cannot encode an empty list")
        dtype = EncodingDtype.from_torch(dtype)
        bit_count = sum(t.numel() for t in ts) * dtype.bit_count()
        return IdentityEncoding(
            _tensors=[t.detach().clone() for t in ts], _bit_count=bit_count
        )


@dataclass
class IdentityEncoding(TensorEncoding):
    """An encoding that stores tensors unmodified, without any protection.

    This can be used in place of an actual encoder for functions which expect
    it. Useful as a baseline to compare protected encodings against.
    """

    _tensors: list[Tensor]
    _bit_count: int

    @override
    def encoded_tensors(self) -> list[Tensor]:
        return self._tensors

    @override
    def decode_into(self, dest: list[Tensor]) -> None:
        check_decode_dest(dest, [(t.shape, t.dtype) for t in self._tensors])
        with torch.no_grad():
            for t, dest_tensor in zip(self._tensors, dest, strict=True):
                _ = dest_tensor.copy_(t)

    @override
    def empty_decoded(self) -> list[Tensor]:
        return [torch.empty_like(t) for t in self._tensors]

    @override
    def clone(self) -> IdentityEncoding:
        return IdentityEncoding(
            _tensors=[t.clone() for t in self._tensors], _bit_count=self._bit_count
        )

    @override
    def apply_fault(self, fault: Fault, target_bit: int) -> None:
        tensor_list_fault(self._tensors, fault, target_bit)

    @override
    def apply_faults(self, faults: Sequence[tuple[Fault, int]]) -> None:
        tensor_list_faults(self._tensors, list(faults))

    @override
    def bit_count(self) -> int:
        return self._bit_count
