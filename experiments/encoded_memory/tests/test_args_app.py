"""Tests for running subcommands with `encoded_memory.args.App`."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import pytest
from encoded_memory.args import (
    App,
    Option,
    Positional,
    UsageError,
)


@dataclass(frozen=True)
class GreetOptions:
    name: Annotated[str, Option(help="Who to greet.")] = "world"


@dataclass(frozen=True)
class CountOptions:
    paths: Annotated[list[Path], Positional(help="Files to count.")]


def _app(calls: list[object]) -> App:
    app = App(prog="tool")

    @app.command
    def greet(options: GreetOptions) -> None:
        """Print a greeting.

        Longer description
        on two lines.
        """
        calls.append(options)

    @app.command
    def count_files(options: CountOptions) -> int:
        """Count files."""
        calls.append(options)
        return len(options.paths)

    @app.command
    def fail(options: GreetOptions) -> None:
        """Always reports a usage error."""
        raise UsageError("failed on purpose", option="--name")

    return app


def test_dispatches_to_the_handler():
    calls: list[object] = []
    assert _app(calls).run(["greet", "--name", "you"]) == 0
    assert calls == [GreetOptions(name="you")]


def test_handler_return_value_is_the_exit_code():
    calls: list[object] = []
    assert _app(calls).run(["count-files", "a", "b", "c"]) == 3


def test_command_list_shows_summaries(capsys: pytest.CaptureFixture[str]):
    assert _app([]).run(["--help"]) == 0
    help_text = capsys.readouterr().out
    assert "count-files" in help_text
    assert "Print a greeting." in help_text
    assert "Longer description" not in help_text


def test_command_help_keeps_docstring_layout(capsys: pytest.CaptureFixture[str]):
    assert _app([]).run(["greet", "--help"]) == 0
    assert "Longer description\non two lines." in capsys.readouterr().out


def test_no_arguments_prints_help(capsys: pytest.CaptureFixture[str]):
    assert _app([]).run([]) == 2
    assert "count-files" in capsys.readouterr().err


def test_command_with_required_arguments_prints_help(
    capsys: pytest.CaptureFixture[str],
):
    calls: list[object] = []
    assert _app(calls).run(["count-files"]) == 2
    assert "Files to count." in capsys.readouterr().err
    assert calls == []


def test_command_without_required_arguments_runs():
    calls: list[object] = []
    assert _app(calls).run(["greet"]) == 0
    assert calls == [GreetOptions()]


def test_parse_errors_return_2(capsys: pytest.CaptureFixture[str]):
    assert _app([]).run(["greet", "--unknown"]) == 2
    assert "unrecognized arguments: --unknown" in capsys.readouterr().err


def test_usage_errors_from_handlers_return_2(capsys: pytest.CaptureFixture[str]):
    assert _app([]).run(["fail"]) == 2
    error = capsys.readouterr().err
    assert "usage: tool fail" in error
    assert "argument --name: failed on purpose" in error


def test_handler_must_take_an_options_dataclass():
    app = App()

    def no_parameters() -> None: ...

    def not_a_dataclass(options: int) -> None: ...

    with pytest.raises(TypeError, match="exactly one parameter"):
        _ = app.command(no_parameters)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="dataclass"):
        _ = app.command(not_a_dataclass)


def test_duplicate_commands_are_rejected():
    app = App()

    def greet(options: GreetOptions) -> None: ...

    _ = app.command(greet)
    with pytest.raises(ValueError, match="already registered"):
        _ = app.command(greet)
