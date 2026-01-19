"""Encoding of sequences."""

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

from torch import Tensor

from faultforge.encoding.encoding import Encoder, Encoding


class TensorEncoder(Encoder, abc.ABC):
    """An encoder that outputs a `TensorEncoding`."""

    @abc.abstractmethod
    def tensor_encoder_encode_tensor_list(self, ts: list[Tensor]) -> TensorEncoding:
        """Encode a list of tensors.

        By default `encode_tensor_list` is implemented to return the result of
        this function.
        """
        ...

    @override
    def encoder_encode_tensor_list(self, ts: list[Tensor]) -> Encoding:
        return self.tensor_encoder_encode_tensor_list(ts)


class TensorEncoding(Encoding, abc.ABC):
    """An encoding that uses tensors as the underlying data structure.

    If your encoding format stores tensors directly then it's appropriate to
    subclass this class instead of :class:`Encoding`.
    """

    @abc.abstractmethod
    def tensor_encoding_tensors(self) -> list[Tensor]:
        """Returns the tensors that make up the encoded data.

        Changing these tensors is expected to have an effect on decoding.
        `tensor_encoding_trigger_recompute` should be called after these tensors
        are altered.
        """
        ...

    @abc.abstractmethod
    def tensor_encoding_trigger_recompute(self) -> None:
        """Trigger a recomputation of the decoded tensors as a result of changes to the encoded tensors."""
        ...


@dataclass
class EncoderSequence(Encoder):
    """An encoder which composes other encoders by applying them sequentially.

    The first encoder of `head` will be applied first, followed by the second,
    and so on. The `tail` encoder will be applied last.
    """

    head: Sequence[TensorEncoder]
    tail: Encoder

    @override
    def encoder_encode_tensor_list(self, ts: list[Tensor]) -> EncodingSequence:
        head_encodings: list[TensorEncoding] = []

        for encoder in self.head:
            encoding = encoder.tensor_encoder_encode_tensor_list(ts)
            head_encodings.append(encoding)
            ts = encoding.tensor_encoding_tensors()

        tail_encoding = self.tail.encoder_encode_tensor_list(ts)

        return EncodingSequence(head_encodings, tail_encoding)

    @override
    def encoder_add_metadata(self, metadata: dict[str, str]) -> None:
        for encoder in self.head:
            encoder.encoder_add_metadata(metadata)
        self.tail.encoder_add_metadata(metadata)


@dataclass
class EncodingSequence(Encoding):
    """An encoding which is composed of other sequentially applied encodings."""

    _head: list[TensorEncoding]
    _tail: Encoding

    @override
    def encoding_decode_tensor_list(self) -> list[Tensor]:
        ts = self._tail.encoding_decode_tensor_list()

        for h in reversed(self._head):
            for original, updated in zip(h.tensor_encoding_tensors(), ts, strict=True):
                _ = original.copy_(updated)
            h.tensor_encoding_trigger_recompute()

            ts = h.encoding_decode_tensor_list()

        return ts

    @override
    def encoding_flip_n_bits(self, n: int) -> None:
        for h in self._head:
            h.tensor_encoding_trigger_recompute()
        self._tail.encoding_flip_n_bits(n)

    @override
    def encoding_bits_count(self) -> int:
        return self._tail.encoding_bits_count()

    @override
    def encoding_clone(self) -> EncodingSequence:
        return EncodingSequence(
            [h.encoding_clone() for h in self._head], self._tail.encoding_clone()
        )
