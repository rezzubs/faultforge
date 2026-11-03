"""Abstract base classes for encoders and encodings.

See `faultforge.encoding` for a general overview.
"""

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

from faultforge._impl.dtype import EncodingDtype
from faultforge._impl.fault import Fault
from faultforge._impl.fingerprint import Fingerprint
from faultforge._impl.progress import Progress, stage
from faultforge._impl.tensor import (
    tensor_list_dtype,
    tensor_list_fault,
    tensor_list_faults,
)


class Encoder(abc.ABC):
    """An encoder for lists of tensors."""

    @abc.abstractmethod
    def encode(self, ts: list[Tensor], *, progress: Progress | None = None) -> Encoding:
        """Encode a list of tensors.

        The inputs are only read. Implementations must not modify them or keep
        references to them. The caller can keep using the inputs, and changes
        to them don't affect the encoding.
        """
        ...

    @abc.abstractmethod
    def fingerprint(self) -> Fingerprint:
        """Return a structural identity for this encoder.

        Two encoders that encode tensors identically should produce equal
        fingerprints. Composite encoders nest the fingerprints of their parts.
        """
        ...


class Encoding:
    """An encoded list of tensors.

    Created by an `Encoder`.
    """

    def decode(self) -> list[Tensor]:
        """Decode into a new list of tensors.

        See `decode_into` to reuse existing tensors.
        """
        dest = self.empty_decoded()
        self.decode_into(dest)
        return dest

    @abc.abstractmethod
    def decode_into(self, dest: list[Tensor]) -> None:
        """Decode into existing tensors.

        `dest` must have one tensor per encoded tensor, with the same shape and
        dtype. The tensors can be on any device and don't need to be
        contiguous.

        Raises:
            ValueError: If `dest` doesn't match the encoded tensors.
        """
        ...

    @abc.abstractmethod
    def empty_decoded(self) -> list[Tensor]:
        """Allocate uninitialized tensors that fit the decoded data.

        The result is a valid `dest` for `decode_into`.
        """
        ...

    @abc.abstractmethod
    def clone(self) -> Encoding:
        """Clone the encoded data."""
        ...

    @abc.abstractmethod
    def apply_fault(self, fault: Fault, target_bit: int) -> None:
        """Apply a fault to the encoded data at the given bit index."""
        ...

    def apply_faults(self, faults: Sequence[tuple[Fault, int]]) -> None:
        """Apply multiple faults at once.

        The default implementation just calls `apply_fault` in a loop.
        Encodings that can apply a batch more efficiently than one fault at a
        time (e.g. by only paying a per-call conversion overhead once for the
        whole batch) should override this.
        """
        for fault, target_bit in faults:
            self.apply_fault(fault, target_bit)

    @abc.abstractmethod
    def bit_count(self) -> int:
        """Return the number of bits in the encoded data."""
        ...


class TensorEncoder(Encoder):
    """An Encoder which produces a `TensorEncoding`."""

    @override
    @abc.abstractmethod
    def encode(
        self, ts: list[Tensor], *, progress: Progress | None = None
    ) -> TensorEncoding: ...


class TensorEncoding(Encoding):
    """An encoding which uses tensors as the underlying data structure.

    These kinds of encodings can be sequenced.
    """

    @abc.abstractmethod
    def encoded_tensors(self) -> list[Tensor]:
        """Access the encoded tensors.

        Modifying them modifies the encoding, like a fault would.
        """
        ...

    @override
    @abc.abstractmethod
    def clone(self) -> TensorEncoding:
        """Clone the encoding."""
        ...


class InPlaceEncoder(TensorEncoder):
    """A helper class for implementing `TensorEncoder`."""

    @abc.abstractmethod
    def encode_float32(self, t: Tensor) -> Tensor:
        """Encode a float32 tensor into a new tensor.

        `t` must not be modified, see `Encoder.encode`.
        """
        ...

    @abc.abstractmethod
    def encode_float16(self, t: Tensor) -> Tensor:
        """Encode a float16 tensor into a new tensor.

        `t` must not be modified, see `Encoder.encode`.
        """
        ...

    @abc.abstractmethod
    def create_encoding(
        self,
        data: list[Tensor],
        bit_count: int,
        dtype: EncodingDtype,
    ) -> TensorEncoding:
        """Create the concrete encoding instance."""
        ...

    @override
    def encode(
        self, ts: list[Tensor], *, progress: Progress | None = None
    ) -> TensorEncoding:
        dtype = tensor_list_dtype(ts)
        if dtype is None:
            raise ValueError("Cannot encode an empty list")

        dtype = EncodingDtype.from_torch(dtype)

        bit_count = 0
        data: list[Tensor] = []
        element_bit_count = dtype.bit_count()

        match dtype:
            case EncodingDtype.F32:
                encode = self.encode_float32
            case EncodingDtype.F16:
                encode = self.encode_float16

        with stage(
            progress, f"Encoding ({self.fingerprint().kind})", total=len(ts)
        ) as s:
            for t in ts:
                bit_count += t.numel() * element_bit_count
                data.append(encode(t))
                s.advance()

        return self.create_encoding(data, bit_count, dtype)


@dataclass
class InPlaceEncoding(TensorEncoding):
    """A helper base class for implementing TensorEncoding"""

    _encoded_data: list[Tensor]
    _bit_count: int
    _dtype: EncodingDtype

    @abc.abstractmethod
    def decode_float32(self, t: Tensor) -> Tensor:
        """Decode a float32 tensor into a new tensor.

        `t` must not be modified.
        """
        ...

    @abc.abstractmethod
    def decode_float16(self, t: Tensor) -> Tensor:
        """Decode a float16 tensor into a new tensor.

        `t` must not be modified.
        """
        ...

    @override
    def encoded_tensors(self) -> list[Tensor]:
        return self._encoded_data

    @override
    def decode_into(self, dest: list[Tensor]) -> None:
        check_decode_dest(dest, [(t.shape, t.dtype) for t in self._encoded_data])

        match self._dtype:
            case EncodingDtype.F16:
                decode = self.decode_float16
            case EncodingDtype.F32:
                decode = self.decode_float32

        with torch.no_grad():
            for encoded, dest_tensor in zip(self._encoded_data, dest, strict=True):
                _ = dest_tensor.copy_(decode(encoded))

    @override
    def empty_decoded(self) -> list[Tensor]:
        return [torch.empty_like(t) for t in self._encoded_data]

    @override
    def apply_fault(self, fault: Fault, target_bit: int) -> None:
        tensor_list_fault(self._encoded_data, fault, target_bit)

    @override
    def apply_faults(self, faults: Sequence[tuple[Fault, int]]) -> None:
        tensor_list_faults(self._encoded_data, list(faults))

    @override
    def bit_count(self) -> int:
        return self._bit_count


def check_decode_dest(
    dest: Sequence[Tensor], layout: Sequence[tuple[torch.Size, torch.dtype]]
) -> None:
    """Check that `dest` has a tensor for each shape and dtype in `layout`.

    Raises:
        ValueError: If it doesn't.
    """
    if len(dest) != len(layout):
        raise ValueError(f"Expected {len(layout)} destination tensors, got {len(dest)}")
    for index, (t, (shape, dtype)) in enumerate(zip(dest, layout, strict=True)):
        if t.shape != shape or t.dtype != dtype:
            raise ValueError(
                f"Destination tensor {index} has shape {tuple(t.shape)} and "
                f"dtype {t.dtype}, expected {tuple(shape)} and {dtype}"
            )
