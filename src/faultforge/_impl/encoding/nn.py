"""PyTorch Modules with encoded parameters"""

import copy
from collections.abc import Sequence
from typing import override

import torch
from torch import Tensor, nn

from faultforge._impl.encoding.abc import (
    Encoder,
    Encoding,
)
from faultforge._impl.fault import Fault
from faultforge._impl.progress import Progress


class EncodedModule(nn.Module):
    """A wrapper for a PyTorch module where parameters are stored in simulated encoded memory.

    Supports fault injection in the encoded memory.

    The wrapped module holds the only decoded copy of the parameters. `decode`
    writes into its parameters in place and returns it. Its parameters must
    not be modified, because later decodes assume a clean base.

    What `decode` does:

    - If the module was decoded and no faults were applied since, it returns
      the module as it is.
    - If faults were applied and the parameters hold the decode without
      faults, it only writes the data that faults changed. This is much faster
      at low bit error rates, for encodings that support it, see
      `Encoding.decode_changes_into`.
    - Otherwise it decodes everything. This includes the first decode of a new
      `EncodedModule`.

    The parameters hold the decode without faults after a decode with no
    faults applied, and in clones of such a module. `clone` decodes a module
    without faults first, so its clones always hold the decode without faults.
    For example:

    ```python
    golden = EncodedModule(model, encoder)
    faulty = golden.clone()  # Decodes `golden` before cloning.
    faulty.apply_faults(first_faults)
    faulty.apply_faults(second_faults)
    faulty.decode()  # Writes what the faults changed.
    faulty.apply_faults(third_faults)
    faulty.decode()  # Decodes everything.
    ```

    To decode only what faults changed every time, apply faults to fresh
    clones of a module without faults instead of reusing a faulty one.
    """

    _module: nn.Module
    _memory: Encoding
    _device: torch.device | None
    """The decoded module will be sent to this device."""
    _dirty: bool
    """Whether the decoded data needs to be refreshed."""
    _params_are_pristine: bool
    """Whether the parameters hold the decode of the memory before any faults.

    Then a decode only needs to write what faults changed, see
    `Encoding.decode_changes_into`.
    """
    _has_faults: bool
    """Whether any faults have been applied to the memory."""

    def __init__(
        self,
        module: nn.Module,
        encoder: Encoder,
        *,
        inherit_device: bool = True,
        progress: Progress | None = None,
    ):
        nn.Module.__init__(self)
        self._module = module
        # Encoders don't modify or keep their inputs, so the parameters can be
        # passed directly.
        parameters: list[Tensor] = list(module.parameters())

        if not inherit_device or len(parameters) == 0:
            self._device = None
        elif len(parameters) > 0:
            self._device = parameters[0].device

        self._memory = encoder.encode(parameters, progress=progress)
        self._dirty = True
        # We have to assume that the parameters are not initially pristine
        # because lossy encodings don't decode to their input.
        self._params_are_pristine = False
        self._has_faults = False

    @classmethod
    def _from_parts(
        cls,
        module: nn.Module,
        data: Encoding,
        device: torch.device | None,
        *,
        dirty: bool,
        params_are_pristine: bool,
        has_faults: bool,
    ) -> EncodedModule:
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance._module = module
        instance._memory = data
        instance._device = device
        instance._dirty = dirty
        instance._params_are_pristine = params_are_pristine
        instance._has_faults = has_faults
        return instance

    def force_decode(self) -> nn.Module:
        """Decode the memory, even if nothing changed since the last decode.

        Decodes like `decode` does when faults were applied, see
        `EncodedModule`.
        """
        # The parameters are the only decoded copy. The encoding decodes
        # straight into them, wherever they are.
        parameters: list[Tensor] = list(self._module.parameters())
        if self._params_are_pristine:
            self._memory.decode_changes_into(parameters)
        else:
            self._memory.decode_into(parameters)

        self._dirty = False
        self._params_are_pristine = not self._has_faults

        if self._device is not None:
            return self._module.to(self._device)
        else:
            return self._module

    def decode(self) -> nn.Module:
        """Decode the memory into the wrapped module and return it.

        Reuses the previous decode if no faults were applied since. The
        parameters of the returned module must not be modified. See
        `EncodedModule` for what a decode writes.
        """
        if self._dirty:
            self.force_decode()
        return self._module

    def clone(self) -> EncodedModule:
        """Clone this module, including its encoded memory.

        The underlying `nn.Module` is deep-copied too, not just the encoded
        memory: `force_decode` writes decoded parameters into it in place, so
        sharing it between clones would let one clone's fault corrupt the
        other's decoded output.

        If no faults were applied to this module, it's decoded first. Then the
        first decode of the clone only writes what its faults changed, see
        `EncodedModule`.
        """
        # Decoding first gives the clone parameters that hold the decode
        # without faults, so its first decode only writes what its faults
        # changed. Decoding here once is cheaper than in every clone.
        if self._dirty and not self._has_faults:
            self.force_decode()

        return EncodedModule._from_parts(
            copy.deepcopy(self._module),
            self._memory.clone(),
            self._device,
            dirty=self._dirty,
            params_are_pristine=self._params_are_pristine,
            has_faults=self._has_faults,
        )

    def apply_fault(self, fault: Fault, target_bit: int) -> None:
        """Apply a fault at the given bit index.

        The fault is expected to be in the range `[0, bit_count)`.
        """
        self._dirty = True
        self._has_faults = True
        self._memory.apply_fault(fault, target_bit)

    def apply_faults(self, faults: Sequence[tuple[Fault, int]]) -> None:
        """Apply multiple faults at once.

        Equivalent to calling `apply_fault` in a loop, but faster: see
        `Encoding.apply_faults`.
        """
        self._dirty = True
        self._has_faults = True
        self._memory.apply_faults(faults)

    def bit_count(self) -> int:
        """Return the number of bits in the encoded data."""
        return self._memory.bit_count()

    @override
    # torch declares `forward` as a mutable `Callable[..., Any]` attribute, so ty
    # rejects any narrower method. Narrowing `forward` is how torch is meant to be used.
    def forward(self, t: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        result = self.decode().forward(t)
        assert isinstance(result, Tensor)

        return result

    @override
    def to(self, *args, **kwargs) -> EncodedModule:
        """Move the module to the specified device.

        Updating the data type after initialization is not supported unlike
        other nn.Module variants.
        """
        device = None
        dtype = None

        if (
            len(args) >= 2
            and isinstance(args[0], (torch.device, str, int))
            and isinstance(args[1], torch.dtype)
        ):
            device = torch.device(torch.device(args[0]))
            dtype = args[1]
        elif len(args) == 1 and isinstance(args[0], torch.dtype):
            dtype = args[0]
        elif len(args) == 1 and isinstance(args[0], (torch.device, str, int)):
            device = torch.device(torch.device(args[0]))
        else:
            raise ValueError("EncodedModule.to only supports a device as an agrument")

        if device is not None:
            self._device = device

        if dtype is not None:
            raise ValueError("Updating the dtype of an EncodedModule is not supported")

        return self
