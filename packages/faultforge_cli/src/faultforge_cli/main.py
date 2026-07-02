import typer

from faultforge_cli import encoded_memory
from faultforge_cli.logging import setup_logging

app = typer.Typer(
    pretty_exceptions_enable=False,
)

app.add_typer(
    encoded_memory.app,
    name="encoded-memory",
    help="Commands for encoded memory experiments.",
)


def main() -> None:
    setup_logging()
    app()


if __name__ == "__main__":
    main()
