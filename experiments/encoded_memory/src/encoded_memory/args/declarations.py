"""Markers for the fields of an options dataclass, used inside `Annotated`."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Option:
    """A field set with a flag, e.g. `--batch-size 64`.

    The flag is the field name with underscores replaced by dashes. A `bool`
    field becomes a `--name`/`--no-name` pair.
    """

    help: str
    """The description shown in the help."""
    aliases: tuple[str, ...] = ()
    """Additional flags for the same option, e.g. `("--ber",)`."""
    metavar: str | None = None
    """The placeholder for the value in the help. Defaults to the field name."""
    convert: Callable[[str], object] | None = None
    """Parses a single value instead of the converter for the field's type."""
    minimum: float | None = None
    """The smallest accepted value, for numeric fields."""
    maximum: float | None = None
    """The largest accepted value, for numeric fields."""


@dataclass(frozen=True)
class Positional:
    """A field set by position, e.g. the paths in `check a.json b.json`."""

    help: str
    """The description shown in the help."""
    metavar: str | None = None
    """The placeholder for the value in the help. Defaults to the field name."""
    convert: Callable[[str], object] | None = None
    """Parses a single value instead of the converter for the field's type."""


@dataclass(frozen=True)
class Group:
    """A heading in the help for the arguments of a nested dataclass."""

    title: str
    """The heading."""
    description: str | None = None
    """Text shown under the heading."""
