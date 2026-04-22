"""The supported data types for different functions grouped by use case."""

from __future__ import annotations

import enum

import torch


class DnnDtype(enum.Enum):
    """Supported data types for DNN evaluation."""

    Float32 = "float32"
    Float16 = "float16"

    def to_torch(self) -> torch.dtype:
        """Convert to an equivalent pytorch data type."""

        match self:
            case DnnDtype.Float32:
                return torch.float32
            case DnnDtype.Float16:
                return torch.float16

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> DnnDtype:
        """Get the equivalent of a pytorch data type.

        :raises ValueError: For incompatible data types
        """

        match dtype:
            case torch.float32:
                return DnnDtype.Float32
            case torch.float16:
                return DnnDtype.Float16
            case other:
                raise ValueError(f"Unsupported pytorch data type {other}")

    def bits_count(self) -> int:
        match self:
            case DnnDtype.Float32:
                return 32
            case DnnDtype.Float16:
                return 16


class FiDtype(enum.Enum):
    """Suppored data types for fault injection."""

    Float32 = enum.auto()
    Float16 = enum.auto()
    Uint8 = enum.auto()

    def to_torch(self) -> torch.dtype:
        """Convert to an equivalent pytorch data type."""
        match self:
            case FiDtype.Float32:
                return torch.float32
            case FiDtype.Float16:
                return torch.float16
            case FiDtype.Uint8:
                return torch.uint8

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> FiDtype:
        """Get the equivalent of a pytorch data type.

        :raises ValueError: For incompatible data types
        """

        match dtype:
            case torch.float32:
                return FiDtype.Float32
            case torch.float16:
                return FiDtype.Float16
            case torch.uint8:
                return FiDtype.Uint8
            case other:
                raise ValueError(f"Unsupported pytorch data type {other}")
