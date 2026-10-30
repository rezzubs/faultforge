"""Turning options dataclasses into `argparse` arguments and back."""

import argparse
import dataclasses
import enum
import types
import typing
from collections.abc import Sequence
from typing import (
    Annotated,
    Any,
    get_args,
    get_origin,
    get_type_hints,
)

from encoded_memory.args.conversion import (
    bounded,
    converter_for,
    is_numeric,
)
from encoded_memory.args.declarations import (
    Group,
    Option,
    Positional,
)
from encoded_memory.args.errors import UsageError

_DEST_ROOT = "options"
"""The first part of every argument's `dest`.

Each `dest` is the dotted path of the field from the root dataclass. That keeps
fields with the same name in different nested dataclasses apart.
"""


def add_arguments(
    parser: argparse.ArgumentParser, options_type: type
) -> list[argparse.Action]:
    """Add the arguments of the options dataclass `options_type` to `parser`.

    Returns the added actions. Raises `TypeError` if a field can't be turned
    into an argument.
    """
    return _add_fields(parser, parser, options_type, (_DEST_ROOT,))


def construct[T](options_type: type[T], namespace: argparse.Namespace) -> T:
    """Build the options from a namespace parsed with `add_arguments`.

    A `UsageError` raised by the dataclass's `__post_init__` is passed on.
    """
    return _construct(options_type, namespace, (_DEST_ROOT,))


def parse[T](
    options_type: type[T],
    argv: Sequence[str] | None = None,
    *,
    prog: str | None = None,
    description: str | None = None,
) -> T:
    """Parse a command line into the options dataclass `options_type`.

    `argv` defaults to `sys.argv[1:]`. Exits like `argparse` on `--help` and on
    a usage error, including a `UsageError` from `__post_init__`.
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_arguments(parser, options_type)
    namespace = parser.parse_args(argv)
    try:
        return construct(options_type, namespace)
    except UsageError as error:
        parser.error(str(error))


def _add_fields(
    parser: argparse.ArgumentParser,
    container: argparse._ActionsContainer,
    options_type: type,
    path: tuple[str, ...],
) -> list[argparse.Action]:
    if not dataclasses.is_dataclass(options_type):
        raise TypeError(f"{options_type!r} is not a dataclass")

    hints = get_type_hints(options_type, include_extras=True)
    actions: list[argparse.Action] = []
    for field in dataclasses.fields(options_type):
        if not field.init:
            continue
        value_type, markers = _split_annotated(hints[field.name])
        field_path = (*path, field.name)

        if isinstance(value_type, type) and dataclasses.is_dataclass(value_type):
            group = next(
                (marker for marker in markers if isinstance(marker, Group)), None
            )
            # NOTE: argparse doesn't support nesting groups, so a group is
            # always added to the parser itself.
            nested_container = (
                container
                if group is None
                else parser.add_argument_group(group.title, group.description)
            )
            actions.extend(
                _add_fields(parser, nested_container, value_type, field_path)
            )
            continue

        declarations = [
            marker for marker in markers if isinstance(marker, Option | Positional)
        ]
        if len(declarations) != 1:
            raise TypeError(
                f"field {field.name} of {options_type.__name__} needs exactly one "
                "Option or Positional annotation"
            )
        actions.append(
            _add_argument(
                container, field, value_type, declarations[0], ".".join(field_path)
            )
        )
    return actions


def _add_argument(
    container: argparse._ActionsContainer,
    field: dataclasses.Field[Any],
    value_type: object,
    declaration: Option | Positional,
    dest: str,
) -> argparse.Action:
    element_type, is_list = _element_type(value_type)
    default = _default(field)
    is_required = default is dataclasses.MISSING
    help_suffixes: list[str] = []
    if not is_required and default is not None and default != []:
        help_suffixes.append(f"default: {_format_default(default)}")

    keywords: dict[str, Any] = {}
    if not is_required:
        keywords["default"] = default

    if element_type is bool and not is_list:
        if not isinstance(declaration, Option):
            raise TypeError(f"bool field {field.name} must be an Option")
        return container.add_argument(
            *_flags(field.name, declaration),
            dest=dest,
            action=argparse.BooleanOptionalAction,
            required=is_required,
            help=_help(declaration.help, help_suffixes),
            **keywords,
        )

    converter = declaration.convert or converter_for(element_type)
    if isinstance(declaration, Option) and (
        declaration.minimum is not None or declaration.maximum is not None
    ):
        if declaration.convert is None and not is_numeric(element_type):
            raise TypeError(f"bounds on non-numeric field {field.name}")
        converter = bounded(converter, declaration.minimum, declaration.maximum)
        help_suffixes.extend(_format_bounds(declaration.minimum, declaration.maximum))
    keywords["type"] = converter

    if (
        declaration.convert is None
        and isinstance(element_type, type)
        and issubclass(element_type, enum.StrEnum)
    ):
        # NOTE: argparse lists the choices in place of a metavar. An explicit
        # metavar hides them.
        keywords["choices"] = list(element_type)
        if declaration.metavar is not None:
            keywords["metavar"] = declaration.metavar
    else:
        keywords["metavar"] = declaration.metavar or field.name.upper()

    help_text = _help(declaration.help, help_suffixes)
    match declaration:
        case Option():
            if is_list:
                keywords["nargs"] = "+"
            return container.add_argument(
                *_flags(field.name, declaration),
                dest=dest,
                required=is_required,
                help=help_text,
                **keywords,
            )
        case Positional():
            if is_list:
                keywords["nargs"] = "+" if is_required else "*"
            elif not is_required:
                keywords["nargs"] = "?"
            # NOTE: argparse takes the dest of a positional as its name.
            return container.add_argument(dest, help=help_text, **keywords)


def _construct(
    options_type: type, namespace: argparse.Namespace, path: tuple[str, ...]
) -> Any:
    if not dataclasses.is_dataclass(options_type):
        raise TypeError(f"{options_type!r} is not a dataclass")
    hints = get_type_hints(options_type, include_extras=True)
    values: dict[str, object] = {}
    for field in dataclasses.fields(options_type):
        if not field.init:
            continue
        value_type, _ = _split_annotated(hints[field.name])
        field_path = (*path, field.name)
        if isinstance(value_type, type) and dataclasses.is_dataclass(value_type):
            values[field.name] = _construct(value_type, namespace, field_path)
        else:
            values[field.name] = getattr(namespace, ".".join(field_path))
    return options_type(**values)


def _split_annotated(hint: object) -> tuple[object, tuple[object, ...]]:
    """Split `Annotated[T, *markers]` into `T` and the markers."""
    if get_origin(hint) is Annotated:
        value_type, *markers = get_args(hint)
        return value_type, tuple(markers)
    return hint, ()


def _element_type(value_type: object) -> tuple[object, bool]:
    """Get the type of a single value and whether the field takes a list.

    `None` is dropped from optional types. The default is what makes an
    argument optional.
    """
    if get_origin(value_type) in (types.UnionType, typing.Union):
        members = [
            member for member in get_args(value_type) if member is not types.NoneType
        ]
        if len(members) != 1:
            raise TypeError(f"unsupported union {value_type!r}, only `T | None` is")
        value_type = members[0]
    if get_origin(value_type) is list:
        (element_type,) = get_args(value_type)
        return element_type, True
    return value_type, False


def _default(field: dataclasses.Field[Any]) -> object:
    """Get the field's default, or `dataclasses.MISSING` if it has none."""
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return dataclasses.MISSING


def _flags(field_name: str, option: Option) -> list[str]:
    return [f"--{field_name.replace('_', '-')}", *option.aliases]


def _format_default(default: object) -> str:
    if isinstance(default, list):
        return " ".join(str(value) for value in default)
    return str(default)


def _format_bounds(minimum: float | None, maximum: float | None) -> list[str]:
    bounds: list[str] = []
    if minimum is not None:
        bounds.append(f"minimum: {minimum}")
    if maximum is not None:
        bounds.append(f"maximum: {maximum}")
    return bounds


def _help(text: str, suffixes: list[str]) -> str:
    if suffixes:
        text = f"{text} ({', '.join(suffixes)})"
    # NOTE: argparse applies %-formatting to help strings, so a literal % must
    # be escaped.
    return text.replace("%", "%%")
