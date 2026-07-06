"""The `encoded-memory` CLI: recording experiments and plotting their results.

`commands` holds the `typer` commands; `results` and `plots` hold the
`compare`/`heatmap` workflow (see `results`' module docstring for a tour).
"""

from faultforge_cli.encoded_memory.commands import app

__all__ = ["app"]
