"""Operations on tensors."""

import numpy as np
import torch

from faultforge import _rust
from faultforge._internal.dtype import FiDtype
from faultforge._internal.fault import (
    Fault,
    fault_to_rust,
)


def tensor_list_dtype(ts: list[torch.Tensor]) -> torch.dtype | None:
    """Confirms that all tensors in `ts` have the same datatype.

    Returns:
        The common dtype of `ts` or None if `ts` is empty.

    Raises:
        ValueError: If dtype values don't match.
    """

    dtype = None

    for i, t in enumerate(ts):
        if dtype is None:
            dtype = t.dtype
        else:
            if dtype != t.dtype:
                raise ValueError(
                    f"dtype=`{t.dtype}` for tensor {i} while all previous values had dtype=`{dtype}`"
                )
    return dtype


def tensor_list_fault(ts: list[torch.Tensor], fault: Fault, target_bit: int):
    """Apply a fault at specific bit position.

    Raises:
        ValueError:
            - If values in `ts` don't all have the same data type.
            - If the data type is unsupported. See `FiDtype`.
    """

    dtype = tensor_list_dtype(ts)

    if dtype is None:
        raise ValueError("`ts` is empty")

    fault: _rust.Fault = fault_to_rust(fault)

    # NOTE: the length checks are handled in rust.
    match FiDtype.from_torch(dtype):
        case FiDtype.F32:
            with torch.no_grad():
                np_array = [t.numpy(force=True) for t in ts]
                _rust.list_of_array_fault_f32(np_array, fault, target_bit)

        case FiDtype.F16:
            with torch.no_grad():
                np_array = [t.numpy(force=True).view(np.uint16) for t in ts]
                _rust.list_of_array_fault_u16(np_array, fault, target_bit)
                np_array = [t.view(np.float16) for t in np_array]
        case FiDtype.U8:
            with torch.no_grad():
                np_array = [t.numpy(force=True) for t in ts]
                _rust.list_of_array_fault_u8(np_array, fault, target_bit)

    for original, updated in zip(ts, np_array, strict=True):
        # We have to assert because Tensor.numpy returns `Unknown`.
        assert isinstance(updated, np.ndarray)
        updated = torch.from_numpy(updated)
        assert updated.shape == original.shape

        with torch.no_grad():
            _ = original.copy_(updated)
