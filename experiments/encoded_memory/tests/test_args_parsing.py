"""Tests for turning options dataclasses into arguments and back."""

import argparse
import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import pytest
from encoded_memory.args import (
    Group,
    Option,
    Positional,
    UsageError,
    add_arguments,
    parse,
)
from hypothesis import given
from hypothesis import strategies as st


class Color(enum.StrEnum):
    Red = "red"
    LightBlue = "light-blue"


@dataclass(frozen=True)
class Scalars:
    name: Annotated[str, Option(help="A name.")]
    count: Annotated[int, Option(help="A count.")] = 1
    ratio: Annotated[float | None, Option(help="A ratio.")] = None
    path: Annotated[Path, Option(help="A path.")] = Path("default.json")
    color: Annotated[Color, Option(help="A color.")] = Color.Red
    verbose: Annotated[bool, Option(help="Talk more.")] = False


def test_defaults():
    assert parse(Scalars, ["--name", "a"]) == Scalars(name="a")


def test_values_are_converted():
    options = parse(
        Scalars,
        [
            "--name",
            "a",
            "--count",
            "3",
            "--ratio",
            "0.5",
            "--path",
            "out.json",
            "--color",
            "light-blue",
            "--verbose",
        ],
    )
    assert options == Scalars(
        name="a",
        count=3,
        ratio=0.5,
        path=Path("out.json"),
        color=Color.LightBlue,
        verbose=True,
    )


def test_bool_can_be_turned_off():
    @dataclass(frozen=True)
    class Flag:
        enabled: Annotated[bool, Option(help="Enable it.")] = True

    assert parse(Flag, ["--no-enabled"]) == Flag(enabled=False)


@given(st.integers())
def test_any_int_round_trips(value: int):
    assert parse(Scalars, ["--name", "a", "--count", str(value)]).count == value


@given(st.floats(allow_nan=False))
def test_any_float_round_trips(value: float):
    # NOTE: argparse reads values like `-inf` or `-1e-05` as flags unless they
    # are attached with `=`.
    assert parse(Scalars, ["--name", "a", f"--ratio={value!r}"]).ratio == value


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ([], "the following arguments are required: --name"),
        (["--name", "a", "--count", "x"], "invalid int value: 'x'"),
        (["--name", "a", "--color", "green"], "invalid choice: 'green'"),
    ],
)
def test_reports_invalid_arguments(
    arguments: list[str], message: str, capsys: pytest.CaptureFixture[str]
):
    with pytest.raises(SystemExit) as exit_info:
        _ = parse(Scalars, arguments)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_help_shows_flags_choices_and_defaults(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit):
        _ = parse(Scalars, ["--help"])
    help_text = capsys.readouterr().out
    assert "--name NAME" in help_text
    assert "{red,light-blue}" in help_text
    assert "(default: red)" in help_text
    assert "--verbose, --no-verbose" in help_text
    # `None` defaults aren't worth showing.
    assert "default: None" not in help_text


def test_percent_signs_in_help_are_literal(capsys: pytest.CaptureFixture[str]):
    @dataclass(frozen=True)
    class Margin:
        margin: Annotated[float, Option(help="At most N% of the value.")] = 5.0

    with pytest.raises(SystemExit):
        _ = parse(Margin, ["--help"])
    assert "At most N% of the value." in capsys.readouterr().out


@dataclass(frozen=True)
class Bounded:
    rate: Annotated[float, Option(help="A rate.", minimum=0.0, maximum=1.0)] = 0.0


@pytest.mark.parametrize("value", ["0", "0.5", "1"])
def test_accepts_values_within_bounds(value: str):
    assert parse(Bounded, ["--rate", value]).rate == float(value)


@pytest.mark.parametrize(
    ("value", "message"),
    [("-0.1", "less than the minimum 0.0"), ("1.5", "more than the maximum 1.0")],
)
def test_rejects_values_out_of_bounds(
    value: str, message: str, capsys: pytest.CaptureFixture[str]
):
    with pytest.raises(SystemExit):
        _ = parse(Bounded, ["--rate", value])
    assert message in capsys.readouterr().err


def test_bounds_require_a_numeric_type():
    @dataclass(frozen=True)
    class Invalid:
        name: Annotated[str, Option(help="A name.", minimum=1)] = "a"

    with pytest.raises(TypeError, match="non-numeric"):
        _ = parse(Invalid, [])


def test_aliases():
    @dataclass(frozen=True)
    class Aliased:
        bit_error_rate: Annotated[
            float | None, Option(help="A rate.", aliases=("--ber",))
        ] = None

    assert parse(Aliased, ["--ber", "0.1"]).bit_error_rate == 0.1
    assert parse(Aliased, ["--bit-error-rate", "0.1"]).bit_error_rate == 0.1


def test_custom_converter():
    @dataclass(frozen=True)
    class Hexadecimal:
        value: Annotated[
            int, Option(help="A number.", convert=lambda text: int(text, 16))
        ]

    assert parse(Hexadecimal, ["--value", "ff"]).value == 255


@dataclass(frozen=True)
class Files:
    paths: Annotated[list[Path], Positional(help="The files.")]
    tags: Annotated[list[str], Option(help="Tags.")] = field(default_factory=list)


def test_lists():
    options = parse(Files, ["a", "b", "--tags", "x", "y"])
    assert options == Files(paths=[Path("a"), Path("b")], tags=["x", "y"])
    assert parse(Files, ["a"]).tags == []


def test_required_positional_list_needs_a_value(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit):
        _ = parse(Files, [])
    assert "required: PATHS" in capsys.readouterr().err


def test_optional_positional():
    @dataclass(frozen=True)
    class MaybeFile:
        path: Annotated[Path | None, Positional(help="A file.")] = None

    assert parse(MaybeFile, []).path is None
    assert parse(MaybeFile, ["a"]).path == Path("a")


@dataclass(frozen=True)
class Limits:
    max_runs: Annotated[int | None, Option(help="At most N runs.")] = None


@dataclass(frozen=True)
class Grouped:
    limits: Annotated[Limits, Group("Limits", "When to stop.")]
    name: Annotated[str, Option(help="A name.")] = "a"


def test_nested_dataclasses_are_constructed():
    options = parse(Grouped, ["--max-runs", "3"])
    assert options == Grouped(limits=Limits(max_runs=3))


def test_groups_appear_in_the_help(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit):
        _ = parse(Grouped, ["--help"])
    help_text = capsys.readouterr().out
    assert "Limits:\n  When to stop." in help_text
    # The flag isn't prefixed with the field name of the group.
    assert "--max-runs" in help_text


def test_nested_dataclass_without_group_is_flattened():
    @dataclass(frozen=True)
    class Flattened:
        limits: Limits

    parser = argparse.ArgumentParser()
    add_arguments(parser, Flattened)
    assert [group.title for group in parser._action_groups] == [
        "positional arguments",
        "options",
    ]


def test_duplicate_flags_are_rejected():
    @dataclass(frozen=True)
    class Duplicate:
        limits: Limits
        max_runs: Annotated[int, Option(help="Clashes.")] = 1

    with pytest.raises(argparse.ArgumentError, match="conflicting option"):
        _ = parse(Duplicate, [])


def test_fields_without_a_declaration_are_rejected():
    @dataclass(frozen=True)
    class Undeclared:
        name: str = "a"

    with pytest.raises(TypeError, match="Option or Positional"):
        _ = parse(Undeclared, [])


def test_unsupported_types_are_rejected():
    @dataclass(frozen=True)
    class Unsupported:
        value: Annotated[complex, Option(help="A value.")] = 0j

    with pytest.raises(TypeError, match="no command line conversion"):
        _ = parse(Unsupported, [])


def test_usage_errors_from_post_init_are_reported(capsys: pytest.CaptureFixture[str]):
    @dataclass(frozen=True)
    class Exclusive:
        first: Annotated[bool, Option(help="First.")] = False
        second: Annotated[bool, Option(help="Second.")] = False

        def __post_init__(self) -> None:
            if self.first and self.second:
                raise UsageError("can't be combined with --first", option="--second")

    assert parse(Exclusive, ["--first"]) == Exclusive(first=True)
    with pytest.raises(SystemExit) as exit_info:
        _ = parse(Exclusive, ["--first", "--second"])
    assert exit_info.value.code == 2
    assert (
        "argument --second: can't be combined with --first" in capsys.readouterr().err
    )
