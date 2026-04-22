import typer

from . import draw, record, strip_faults
from .utils import setup_logging

app = typer.Typer()

app.add_typer(record.app)
app.add_typer(strip_faults.app)
app.add_typer(
    draw.app,
    name="draw",
    help="Subcommands for making visualizations from recorded data",
)


def main():
    setup_logging()
    app()


if __name__ == "__main__":
    main()
