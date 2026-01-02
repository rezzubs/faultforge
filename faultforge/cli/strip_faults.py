import logging
from collections.abc import Generator
from pathlib import Path
from typing import Annotated

import typer

from faultforge.stats import Stats

_logger = logging.getLogger(__name__)

app = typer.Typer()


def recursive_paths(root: Path) -> Generator[Path]:
    if not root.is_dir():
        yield root
        return

    for path in root.iterdir():
        yield from recursive_paths(path)


@app.command()
def strip_faults(
    paths: Annotated[list[Path], typer.Argument(help="Paths to stat files")],
):
    """Strip the recorded faults from the provided stat files"""

    paths = [leaf for root in paths for leaf in recursive_paths(root)]

    for path in paths:
        _logger.info(f"Stripping faults from {path}")
        try:
            stats = Stats.load(path)
            _logger.debug("Loading finished")
        except Exception as e:
            _logger.warning(f"Failed to load statistics from {path}: {e}")
            continue
        for entry in stats.entries:
            entry.faulty_parameters = []
        stats.save(path)
        del stats
