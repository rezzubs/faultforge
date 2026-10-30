"""Programs made of subcommands."""

import argparse
import dataclasses
import inspect
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    get_type_hints,
)

from encoded_memory.args.errors import UsageError
from encoded_memory.args.parsing import (
    add_arguments,
    construct,
)

_COMMAND_DEST = "command"
"""The `dest` of the subcommand name. Never clashes with an options dest."""


@dataclass(frozen=True)
class _Command:
    name: str
    """The name used on the command line."""
    handler: Callable[[Any], int | None]
    """The function that runs the command."""
    options_type: type
    """The options dataclass the handler takes."""


@dataclass(frozen=True)
class _BuiltCommand:
    command: _Command
    """The registered command."""
    parser: argparse.ArgumentParser
    """The subparser of the command."""
    has_required_arguments: bool
    """Whether the command can't run without arguments."""


class App:
    """A program whose subcommands are handlers taking an options dataclass.

    ```python
    app = App()

    @app.command
    def check(options: CheckOptions) -> int: ...

    sys.exit(app.run())
    ```
    """

    def __init__(
        self, *, prog: str | None = None, description: str | None = None
    ) -> None:
        """Create a program without commands.

        `prog` is the program name in the help. Defaults to the name it was
        started with.
        """
        self._prog: str | None = prog
        """The program name in the help."""
        self._description: str | None = description
        """The description in the top level help."""
        self._commands: dict[str, _Command] = {}
        """The registered commands by name."""

    def command[T](
        self, handler: Callable[[T], int | None]
    ) -> Callable[[T], int | None]:
        """Register `handler` as a subcommand. Returns `handler` unchanged.

        - The command's name is the function's name with underscores replaced
          by dashes.
        - The function's only parameter must be annotated with an options
          dataclass.
        - The docstring is the help. Its first paragraph is the summary in the
          command list.
        - The return value is the exit code. `None` means 0.

        Raises `TypeError` if the handler doesn't fit and `ValueError` if a
        command with the same name exists.
        """
        function_name = getattr(handler, "__name__", None)
        if function_name is None:
            raise TypeError(f"{handler!r} must be a function")
        parameters = list(inspect.signature(handler).parameters)
        if len(parameters) != 1:
            raise TypeError(f"{function_name} must take exactly one parameter")
        options_type = get_type_hints(handler).get(parameters[0])
        if not (
            isinstance(options_type, type) and dataclasses.is_dataclass(options_type)
        ):
            raise TypeError(
                f"the parameter of {function_name} must be annotated with a dataclass"
            )

        name = function_name.replace("_", "-")
        if name in self._commands:
            raise ValueError(f"command {name} is already registered")
        self._commands[name] = _Command(name, handler, options_type)
        return handler

    def build_parser(self) -> argparse.ArgumentParser:
        """Build the parser for all registered commands."""
        parser, _ = self._build()
        return parser

    def run(self, argv: Sequence[str] | None = None) -> int:
        """Parse a command line, run its command and return the exit code.

        `argv` defaults to `sys.argv[1:]`. Help and usage errors also return
        their exit code rather than exiting.

        Without arguments, the help is printed with exit code 2. The same goes
        for a command that has required arguments.
        """
        arguments = list(sys.argv[1:] if argv is None else argv)
        parser, built_commands = self._build()

        if not arguments:
            parser.print_help(sys.stderr)
            return 2
        if len(arguments) == 1 and arguments[0] in built_commands:
            built = built_commands[arguments[0]]
            if built.has_required_arguments:
                built.parser.print_help(sys.stderr)
                return 2

        try:
            namespace = parser.parse_args(arguments)
            built = built_commands[getattr(namespace, _COMMAND_DEST)]
            try:
                options = construct(built.command.options_type, namespace)
                exit_code = built.command.handler(options)
            except UsageError as error:
                built.parser.error(str(error))
        except SystemExit as system_exit:
            return _exit_code(system_exit)
        return 0 if exit_code is None else exit_code

    def _build(self) -> tuple[argparse.ArgumentParser, dict[str, _BuiltCommand]]:
        parser = argparse.ArgumentParser(
            prog=self._prog,
            description=self._description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        subparsers = parser.add_subparsers(
            dest=_COMMAND_DEST, required=True, metavar="COMMAND", title="commands"
        )

        built_commands: dict[str, _BuiltCommand] = {}
        for command in self._commands.values():
            docstring = inspect.getdoc(command.handler)
            summary = (
                None
                if docstring is None
                else " ".join(docstring.split("\n\n")[0].split())
            )
            subparser = subparsers.add_parser(
                command.name,
                help=summary,
                description=docstring,
                formatter_class=argparse.RawDescriptionHelpFormatter,
            )
            actions = add_arguments(subparser, command.options_type)
            built_commands[command.name] = _BuiltCommand(
                command,
                subparser,
                has_required_arguments=any(action.required for action in actions),
            )
        return parser, built_commands


def _exit_code(system_exit: SystemExit) -> int:
    """Get the exit code `sys.exit` would use for `system_exit`."""
    match system_exit.code:
        case None:
            return 0
        case int():
            return system_exit.code
        case _:
            # NOTE: `sys.exit` prints any other value and exits with 1.
            print(system_exit.code, file=sys.stderr)
            return 1
