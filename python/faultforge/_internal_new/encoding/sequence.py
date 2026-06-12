"""Sequencing encoders."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

from torch import Tensor

from faultforge._internal_new.encoding.abc import (
    Encoder,
    Encoding,
    TensorEncoder,
    TensorEncoding,
)


@dataclass(slots=True)
class EncoderSequence(Encoder):
    """An encoder which composes other encoders by applying them sequentially.

    The first encoder of `head` will be applied first, followed by the second,
    and so on. The `tail` encoder will be applied last.
    """

    head: Sequence[TensorEncoder]
    tail: Encoder

    @override
    def encode(self, ts: list[Tensor]) -> Encoding:
        head_encodings: list[TensorEncoding] = []

        for encoder in self.head:
            encoding = encoder.encode(ts)
            head_encodings.append(encoding)
            ts = encoding.encoded_tensors()

        tail_encoding = self.tail.encode(ts)

        return EncodingSequence(head_encodings, tail_encoding)


@dataclass
class EncodingSequence(Encoding):
    """An encoding which is created by applying multiple encoders in sequence."""

    _head: list[TensorEncoding]
    _tail: Encoding

    @override
    def decode(self) -> list[Tensor]:
        ts = self._tail.decode()

        for encoding in reversed(self._head):
            for original, updated in zip(encoding.encoded_tensors(), ts, strict=True):
                _ = original.copy_(updated)
            encoding.trigger_recompute()

            ts = encoding.decode()

        return ts

    @override
    def flip_bits(self, n: int) -> None:
        self._tail.flip_bits(n)

    @override
    def bit_count(self) -> int:
        return self._tail.bit_count()

    @override
    def clone(self) -> EncodingSequence:
        return EncodingSequence([h.clone() for h in self._head], self._tail.clone())
