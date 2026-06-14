"""PyTorch Modules with encoded parameters"""

from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor, nn

from faultforge._internal.encoding.abc import (
    Encoder,
    Encoding,
)
from faultforge._internal.fault import Fault


@dataclass(slots=True)
class EncodedModule(nn.Module):
    """A wrapper for a PyTorch module where parameters are stored in simulated encoded memory.

    Supports fault injection in the encoded memory.
    """

    _module: nn.Module
    _memory: Encoding

    def __init__(self, module: nn.Module, encoder: Encoder):
        nn.Module.__init__(self)
        self._module = module
        parameters: list[Tensor] = [p.clone() for p in module.parameters()]

        self._memory = encoder.encode(parameters)

    @classmethod
    def _from_parts(cls, module: nn.Module, data: Encoding) -> EncodedModule:
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance._module = module
        instance._memory = data
        return instance

    def decode(self) -> nn.Module:
        decoded = self._memory.decode()
        for param, decoded_param in zip(self._module.parameters(), decoded):
            assert param.shape == decoded_param.shape
            assert param.dtype == decoded_param.dtype

            with torch.no_grad():
                param.copy_(decoded_param)
        return self._module

    def clone(self) -> "EncodedModule":
        return EncodedModule._from_parts(self._module, self._memory.clone())

    def apply_fault(self, fault: Fault, target_bit: int) -> None:
        """Apply a fault at the given bit index.

        The fault is expected to be in the range `[0, bit_count)`.
        """
        self._memory.apply_fault(fault, target_bit)

    def bit_count(self) -> int:
        """Return the number of bits in the encoded data."""
        return self._memory.bit_count()

    @override
    def forward(self, t: Tensor) -> Tensor:
        result = self.decode().forward(t)
        assert isinstance(result, Tensor)

        return result

    @override
    def to(self, *args, **kwargs) -> EncodedModule:
        raise RuntimeError("to() is not supported on EncodedModule")
