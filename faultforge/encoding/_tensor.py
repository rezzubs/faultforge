import abc
import logging
from dataclasses import dataclass
from typing import Self, override

import torch
from torch import Tensor

from faultforge.encoding.sequence import TensorEncoder, TensorEncoding
from faultforge.tensor_ops import tensor_list_dtype, tensor_list_fault_injection
from faultforge.utils import dtype_bits_count

_logger = logging.getLogger(__name__)


class TensorEncoderHelper(TensorEncoder, abc.ABC):
    """A base class for encoders which encode tensors in place."""

    @abc.abstractmethod
    def encode_float32(self, t: Tensor) -> Tensor:
        """Encode a float32 tensor in place and return it."""
        ...

    @abc.abstractmethod
    def encode_float16(self, t: Tensor) -> Tensor:
        """Encode a float16 tensor in place and return it."""
        ...

    @abc.abstractmethod
    def create_encoding(
        self,
        data: list[Tensor],
        bits_count: int,
        decoded_tensors: list[Tensor],
        dtype: torch.dtype,
    ) -> TensorEncoding:
        """Create the concrete encoding instance."""
        ...

    @override
    def tensor_encoder_encode_tensor_list(self, ts: list[Tensor]) -> TensorEncoding:
        dtype = tensor_list_dtype(ts)

        match dtype:
            case None:
                raise ValueError("Cannot encode an empty list")
            case torch.float32:
                encode = self.encode_float32
            case torch.float16:
                encode = self.encode_float16
            case _:
                raise ValueError(f"Cannot encode dtype={dtype} tensors.")

        # Store decoded tensor copies
        decoded_tensors = [t.clone() for t in ts]

        bits_count = 0
        data: list[Tensor] = []
        item_bits_count = dtype_bits_count(dtype)

        for t in ts:
            bits_count += t.numel() * item_bits_count
            data.append(encode(t))

        return self.create_encoding(data, bits_count, decoded_tensors, dtype)


@dataclass
class TensorEncodingHelper(TensorEncoding, abc.ABC):
    """A base class enocders which encode tensors in place."""

    _encoded_data: list[Tensor]
    _bits_count: int
    _decoded_tensors: list[Tensor]
    _dtype: torch.dtype
    _needs_recompute: bool

    @abc.abstractmethod
    def decode_float32(self, t: Tensor) -> Tensor: ...

    @abc.abstractmethod
    def decode_float16(self, t: Tensor) -> Tensor: ...

    @override
    def tensor_encoding_tensors(self) -> list[Tensor]:
        return self._encoded_data

    @override
    def tensor_encoding_trigger_recompute(self) -> None:
        self._needs_recompute = True

    @override
    def encoding_decode_tensor_list(self) -> list[Tensor]:
        if not self._needs_recompute:
            return self._decoded_tensors

        match self._dtype:
            case torch.float16:
                decode = self.decode_float16

            case torch.float32:
                decode = self.decode_float32

            case _:
                raise RuntimeError(f"Unsupported dtype: {self._dtype}")

        for output, encoded in zip(
            self._decoded_tensors, self._encoded_data, strict=True
        ):
            with torch.no_grad():
                decoded = decode(encoded)
                _ = output.copy_(decoded)

        self._needs_recompute = False
        return self._decoded_tensors

    @override
    def encoding_flip_n_bits(self, n: int) -> None:
        _logger.debug("Invalidating decoded tensors due to fault injection")
        self.tensor_encoding_trigger_recompute()
        tensor_list_fault_injection(self._encoded_data, n)

    @override
    def encoding_bits_count(self) -> int:
        return self._bits_count

    @override
    def encoding_clone(self) -> Self:
        copied_data = [t.clone() for t in self._encoded_data]
        copied_decoded = [t.clone() for t in self._decoded_tensors]
        return self.__class__(
            copied_data,
            self._bits_count,
            copied_decoded,
            self._dtype,
            self._needs_recompute,
        )
