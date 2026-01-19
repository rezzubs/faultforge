import copy
import logging
from typing import override

import torch
from torch import Tensor

from faultforge.encoding.encoding import Encoder, Encoding
from faultforge.system import System

logger = logging.getLogger(__name__)


class EncodedSystem[T](System[Encoding]):
    """Apply an **encoding** on top of the data tensors.

    This class expects its ``base`` to be a system where:
    - The tensors returned by :func:`System.system_data_tensors()` are the
      actual source of truth.
    - Modifying those tensors directly changes the system state and by
      extension, fault injection into those tensors affects system behavior.

    This is unlike ``EncodedSystem`` itself where the source of truth is more
    abstract and depends on the specific encoding used. For this reason,
    EncodedSystem cannot be used as a `base` of another `EncodedSystem`. For
    composing encoders, see :mod:`faultforge.encoding.sequence` instead.
    """

    def __init__(
        self,
        base: System[T],
        encoder: Encoder,
    ) -> None:
        self.base: System[T] = base
        self.encoded_data: Encoding | None = None
        self.encoder: Encoder = encoder

    def _encode_base(self) -> Encoding:
        """Apply the encoder to the base system data tensors."""

        logger.debug("Encoding data tensors")
        data_tensors = self.base.system_data_tensors(self.base.system_data())
        return self.encoder.encoder_encode_tensor_list(data_tensors)

    def decoded_data(self, data: Encoding) -> T:
        """Decode the encoded data and return the original data structure."""

        decoded_tensors = data.encoding_decode_tensor_list()

        # Create a T structure and populate its tensors with decoded values
        # This is necessary because we need to return T (e.g., nn.Module), not just tensors
        inner_data = self.base.system_data()
        inner_data_tensors = self.base.system_data_tensors(inner_data)

        for target, decoded in zip(inner_data_tensors, decoded_tensors, strict=True):
            with torch.no_grad():
                _ = target.copy_(decoded)

        return inner_data

    @override
    def system_data(self) -> Encoding:
        if self.encoded_data is None:
            self.encoded_data = self._encode_base()

        return self.encoded_data

    @override
    def system_accuracy(self, data: Encoding) -> float:
        return self.base.system_accuracy(self.decoded_data(data))

    @override
    def system_data_tensors(self, data: Encoding) -> list[Tensor]:
        return self.base.system_data_tensors(self.decoded_data(data))

    @override
    def system_inject_n_faults(self, data: Encoding, n: int):
        data.encoding_flip_n_bits(n)

    @override
    def system_metadata(self) -> dict[str, str]:
        metadata = copy.deepcopy(self.base.system_metadata())

        self.encoder.encoder_add_metadata(metadata)

        overhead = (
            self.system_total_bits_count() / self.base.system_total_bits_count() - 1
        ) * 100
        metadata["memory_overhead"] = f"{overhead:.1f}%"
        metadata["protected"] = "true"

        return metadata

    @override
    def system_clone_data(self, data: Encoding) -> Encoding:
        return data.encoding_clone()

    @override
    def system_total_bits_count(self) -> int:
        return self.system_data().encoding_bits_count()
