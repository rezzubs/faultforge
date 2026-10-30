"""Typed command line parsing on top of `argparse`.

The options of a command are declared as a dataclass:

- A field annotated with `Option` or `Positional` becomes an argument. The
  field's type decides how the argument is parsed and its default makes it
  optional.
- A field whose type is another dataclass adds that dataclass's arguments.
  Annotating it with `Group` puts them under a heading in the help.

Declaring the dataclass with `kw_only=True` lets fields with and without
defaults appear in any order.

`parse` turns a command line into an instance of one such dataclass. `App`
dispatches subcommands to handlers that each take one.

Errors that the parser can't catch by itself, like options that conflict with
each other, are raised as `UsageError`. They are reported like any other
command line error.

The parser is a regular `argparse.ArgumentParser`. Its help formatting,
including colors, is the standard one.
"""

from encoded_memory.args.app import App
from encoded_memory.args.declarations import (
    Group,
    Option,
    Positional,
)
from encoded_memory.args.errors import UsageError
from encoded_memory.args.parsing import (
    add_arguments,
    construct,
    parse,
)

__all__ = [
    "App",
    "Group",
    "Option",
    "Positional",
    "UsageError",
    "add_arguments",
    "construct",
    "parse",
]
