"""Operations on tensors."""

import logging

import torch

from faultforge import _rust
from faultforge._internal_new.dtype import FiDtype

logger = logging.getLogger(__name__)


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


def tensor_list_fault_injection(ts: list[torch.Tensor], faults_count: int):
    """Flip `faults_count` random unique bits in `ts`.

    Raises:
        ValueError:
            - If values in `ts` don't all have the same data type.
            - If faults_count is greater than the number of bits `ts`.
            - If the data type is unsupported. See `FiDtype`.
    """

    dtype = tensor_list_dtype(ts)

    if dtype is None:
        logger.warning("Skipping fault injection because the input buffer is empty")
        return

    flattened = [t.flatten() for t in ts]

    # NOTE: the length checks are handled in rust.
    match FiDtype.from_torch(dtype):
        case FiDtype.F32:
            with torch.no_grad():
                rust_input = [t.numpy(force=True) for t in flattened]
                result = _rust.f32_array_list_fi(rust_input, faults_count)
                torch_result = [
                    # HACK: There's nothing we can do about this warning without an upstream fix.
                    torch.from_numpy(t)
                    for t in result
                ]

        case FiDtype.F16:
            with torch.no_grad():
                rust_input = [t.cpu().view(torch.uint16).numpy() for t in flattened]

                result = _rust.u16_array_list_fi(rust_input, faults_count)
                torch_result = [
                    # HACK: There's nothing we can do about this warning without an upstream fix.
                    torch.from_numpy(t).view(torch.float16)
                    for t in result
                ]

                for original, updated in zip(flattened, torch_result, strict=True):
                    _ = original.copy_(updated)
        case FiDtype.U8:
            with torch.no_grad():
                rust_input = [t.numpy(force=True) for t in flattened]

                result = _rust.u8_array_list_fi(rust_input, faults_count)
                torch_result = [
                    # HACK: There's nothing we can do about this warning without an upstream fix.
                    torch.from_numpy(t)
                    for t in result
                ]

    for original, updated in zip(flattened, torch_result, strict=True):
        with torch.no_grad():
            _ = original.copy_(updated)
