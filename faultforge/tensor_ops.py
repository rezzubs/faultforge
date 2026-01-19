"""Utility functions for tensors and lists of tensors."""

import logging

import torch

import faultforge._core
from faultforge._utils import dtype_bits_count

from .dtype import DnnDtype, FiDtype


class DtypeError(Exception):
    """An error related to tensor data types."""


_logger = logging.getLogger(__name__)


def tensor_list_dtype(ts: list[torch.Tensor]) -> torch.dtype | None:
    """Confirms that all tensors in `ts` have the same datatype.

    :returns: The common dtype of `ts` or None if `ts` is empty.

    :raises DtypeError: If dtype values don't match.
    """

    dtype = None

    for i, t in enumerate(ts):
        if dtype is None:
            dtype = t.dtype
        else:
            if dtype != t.dtype:
                raise DtypeError(
                    f"dtype=`{t.dtype}` for tensor {i} while all previous values had dtype=`{dtype}`"
                )
    return dtype


def tensor_list_bits_count(ts: list[torch.Tensor]) -> int:
    """Get the total number of bits in a list of tensors."""

    total = 0

    for t in ts:
        total += dtype_bits_count(t.dtype) * t.numel()

    return total


def tensor_list_fault_injection(ts: list[torch.Tensor], faults_count: int):
    """Flip `faults_count` unique bits in ``ts``.

    :raises DtypeError: If values in ``ts`` don't all have the same data type.
    :raises ValueError:
        - If faults_count is greated than the number of bits ``ts``.
        - If the data type is unsupported. See :class:`FiDtype`.
    """

    dtype = tensor_list_dtype(ts)

    if dtype is None:
        _logger.warning("Skipping fault injection because the input buffer is empty")
        return

    flattened = [t.flatten() for t in ts]

    # NOTE: the length checks are handled in rust.
    match FiDtype.from_torch(dtype):
        case FiDtype.Float32:
            with torch.no_grad():
                rust_input = [t.numpy(force=True) for t in flattened]
                result = faultforge._core.f32_array_list_fi(rust_input, faults_count)
                torch_result = [
                    # HACK: There's nothing we can do about this warning without an upstream fix.
                    torch.from_numpy(t)  # pyright: ignore[reportUnknownMemberType]
                    for t in result
                ]

        case FiDtype.Float16:
            with torch.no_grad():
                rust_input = [t.cpu().view(torch.uint16).numpy() for t in flattened]

                result = faultforge._core.u16_array_list_fi(rust_input, faults_count)
                torch_result = [
                    # HACK: There's nothing we can do about this warning without an upstream fix.
                    torch.from_numpy(t).view(torch.float16)  # pyright: ignore[reportUnknownMemberType]
                    for t in result
                ]

                for original, updated in zip(flattened, torch_result, strict=True):
                    _ = original.copy_(updated)
        case FiDtype.Uint8:
            with torch.no_grad():
                rust_input = [t.numpy(force=True) for t in flattened]

                result = faultforge._core.u8_array_list_fi(rust_input, faults_count)
                torch_result = [
                    # HACK: There's nothing we can do about this warning without an upstream fix.
                    torch.from_numpy(t)  # pyright: ignore[reportUnknownMemberType]
                    for t in result
                ]

    for original, updated in zip(flattened, torch_result, strict=True):
        with torch.no_grad():
            _ = original.copy_(updated)


def tensor_list_compare_bitwise(
    left: list[torch.Tensor], right: list[torch.Tensor]
) -> list[int]:
    """Return a bitwise xor of all tensors in `left` and `right` as unsigned integers."""

    left_dtype = tensor_list_dtype(left)
    right_dtype = tensor_list_dtype(right)

    if left_dtype != right_dtype:
        raise DtypeError(
            f"The data types of the two lists don't match {left_dtype}!={right_dtype}"
        )

    if left_dtype is None:
        _logger.warning("compared empty tensor lists the result will be empty")
        return []

    match DnnDtype.from_torch(left_dtype):
        case DnnDtype.Float32:
            left_numpy = [t.flatten().numpy(force=True) for t in left]
            right_numpy = [t.flatten().numpy(force=True) for t in right]
            return faultforge._core.compare_array_list_bitwise_f32(
                left_numpy, right_numpy
            )
        case DnnDtype.Float16:
            #  A view is like a bitwise transmute. This is
            # necessary because rust's `f16` isn't yet stable. See:
            # https://github.com/rust-lang/rust/issues/116909
            left_numpy = [
                t.view(torch.uint16).flatten().numpy(force=True) for t in left
            ]
            right_numpy = [
                t.view(torch.uint16).flatten().numpy(force=True) for t in right
            ]
            return faultforge._core.compare_array_list_bitwise_u16(
                left_numpy, right_numpy
            )
