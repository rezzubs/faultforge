# AGENTS.md

Guidance for agents working in this repository.

This repository contains FaultForge, a toolkit for reproducible hardware
fault-injection experiments, and a reference experiment built on it.

## Project structure

- `src/faultforge` - The python package of the faultforge library.
  - Top-level modules under `faultforge/*.py` are thin, documented re-export
    shims; the real implementation lives in `faultforge/_impl/`. This design
    makes it simple to avoid import errors while keeping the public API shape
    distinct from the one that's convenient for implementation.
- `crates/` - Cargo workspace. Rust crates that are either standalone or
  consumed as PyO3 extension modules.
  - `crates/bindings` - A PyO3 extension exposed to python as
    `faultforge._rust`.
  - The crates are internal and unpublished. Breaking changes in them don't
    need a version bump unless they change the public Python API.
- `experiments/encoded_memory` - The reference experiment built on
  `faultforge`.
  - It depends on a semver-compatible range of `faultforge`, written out
    explicitly (e.g. `>=0.3.0,<0.4` or `>=1.2.3,<2`). Installs from outside the
    repository (e.g. a git tag) get a matching version from PyPI.
  - It's a uv workspace member, so it uses the local `faultforge` source and
    library changes are checked against it immediately. uv checks the range
    against the local version. It must stay in the workspace.
- `docs/` - User-facing documentation. `README.md` uses absolute GitHub URLs
  because it's also the PyPI description. Other docs use relative links.

## Versioning

The library, the reference experiment and the Rust crates share one version,
bumped together on release. The release workflow checks that all of them
match the tag.

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
