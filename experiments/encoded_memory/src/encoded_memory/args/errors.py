"""Errors reported to the command line user."""

from typing import override


class UsageError(Exception):
    """Invalid command line usage that the parser can't detect by itself.

    Raise it from an options dataclass's `__post_init__` or from a command
    handler. It's reported with the usage of the command and exit code 2.
    """

    def __init__(self, message: str, *, option: str | None = None) -> None:
        """Create the error.

        `option` names the argument at fault, e.g. `--max-runs`.
        """
        super().__init__(message)
        self.message: str = message
        """What's wrong with the command line."""
        self.option: str | None = option
        """The argument at fault, if a single one is."""

    @override
    def __str__(self) -> str:
        # Matches how argparse reports errors in a single argument.
        if self.option is None:
            return self.message
        return f"argument {self.option}: {self.message}"
