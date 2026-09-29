# AGENTS.md

Guidance for agents working in this repository.

This repository contains a library for reproducible hardware fault injection.
experiments as well as experiments which use that library.

## Project structure

- `src/faultforge` - The python package of the faultforge library.
  - Top-level modules under `faultforge/*.py` are thin, documented re-export
    shims; the real implementation lives in `faultforge/_internal/`. This design
    makes it simple to avoid import errors while keeping the public API shape
    distinct from the one that's convenient for implementation.
- `crates/` - Cargo workspace. Rust crates that are either standalone or
  consumed as PyO3 extension modules.
  - `crates/bindings` - A PyO3 extension exposed to python as
    `faultforge._rust`.
- `experiments/` - experiment packages built on `faultforge`.
  - Each one depends on a semver-compatible range of `faultforge`, written out
    explicitly (e.g. `>=0.3.0,<0.4` or `>=1.2.3,<2`). Compatible releases are
    picked up without changes. Installs from outside the repository (e.g. a git
    tag) get a matching version from PyPI.
  - Experiments that are uv workspace members use the local `faultforge` source
    instead, so library changes are checked against them immediately. uv checks
    the range against the local version.
  - To decouple an experiment from library changes, remove it from the
    workspace, drop its `tool.uv.sources` entry and give it its own lockfile.

## Commands

- Common project commands are exposed in a `justfile`. Prefer these over running
  tools directly. `just --list` can be used for an overview.
- You should run the various test/check commands after completing a significant
  change. `just ci` runs everything CI checks.
- The project uses `uv` as the python project/package manager. When you need
  options the recipes don't cover, use `uv run --all-packages <tool>` rather
  than binaries from `.venv`. `uv run` rebuilds the Rust extension when Rust
  sources change, so `.venv` binaries may run against stale bindings.
- The project also uses nix. You can run `nix develop` to get access to project
  tooling if not already in a nix development shell.

## Testing conventions

- Rust: `proptest` for property-based tests.
- Python: `hypothesis` for property-based tests.
