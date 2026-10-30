"""Parsing single command line values into typed values."""

import argparse
import enum
from collections.abc import Callable
from pathlib import Path

type Converter = Callable[[str], object]
"""Parses one command line value. Raises `ValueError` or `argparse.ArgumentTypeError` on bad input."""

_CONVERTERS: dict[type, Converter] = {
    str: str,
    int: int,
    float: float,
    Path: Path,
}
"""Converters for types that can be constructed directly from a string."""


def converter_for(value_type: object) -> Converter:
    """Get the converter for a field's value type.

    Raises `TypeError` if the type isn't supported.
    """
    if isinstance(value_type, type) and issubclass(value_type, enum.StrEnum):
        return _enum_converter(value_type)
    if isinstance(value_type, type) and value_type in _CONVERTERS:
        return _CONVERTERS[value_type]
    raise TypeError(f"no command line conversion for {value_type!r}")


def is_numeric(value_type: object) -> bool:
    """Whether bounds apply to values of the type."""
    return value_type in (int, float)


def bounded(
    converter: Converter, minimum: float | None, maximum: float | None
) -> Converter:
    """Wrap `converter` to reject values outside `minimum` and `maximum`."""

    def convert(text: str) -> object:
        value = converter(text)
        # NOTE: only used for numeric types, see `is_numeric`.
        assert isinstance(value, int | float)
        if minimum is not None and value < minimum:
            raise argparse.ArgumentTypeError(
                f"{text} is less than the minimum {minimum}"
            )
        if maximum is not None and value > maximum:
            raise argparse.ArgumentTypeError(
                f"{text} is more than the maximum {maximum}"
            )
        return value

    # argparse names the type in its error message for a `ValueError`.
    convert.__name__ = getattr(converter, "__name__", "value")
    return convert


def _enum_converter[E: enum.StrEnum](enum_type: type[E]) -> Callable[[str], E]:
    def convert(text: str) -> E:
        try:
            return enum_type(text)
        except ValueError:
            choices = ", ".join(repr(member.value) for member in enum_type)
            raise argparse.ArgumentTypeError(
                f"invalid choice: {text!r} (choose from {choices})"
            ) from None

    return convert
