# AGENTS.md

Guidance for agents working in this repository.

## Overview

FaultForge simulates fault mitigation (error-correcting codes, bit-flip/stuck-at
fault injection) for PyTorch models. It's a hybrid project: performance-critical
bit-level encoding/fault logic lives in Rust, exposed to Python via PyO3, and the
experiment framework, model/dataset loading, and CLI live in Python on top of it.

- `crates/` - Cargo workspace (Rust): `picker`, `memory`, `bindings` (the PyO3
  extension, compiled as `faultforge._rust`).
- `packages/` - `uv` workspace (Python): `faultforge` (the library) and
  `faultforge_cli` (a thin `typer` CLI).

**After changing any Rust code, rebuild the extension before running Python
tests**, otherwise Python will import the stale compiled `.so`:

```sh
.venv/bin/maturin develop -m packages/faultforge/pyproject.toml
```

## Commands

`uv` by default creates a venv in `.venv`. Use the binaries in `.venv/bin`
directly rather than `uv run ...`.

### Rust

```sh
cargo clippy --workspace -- -D warnings   # lint (CI treats warnings as errors)
cargo test --workspace                    # test (default; don't assume nextest is installed)
cargo doc --workspace --no-deps --document-private-items  # doc check (CI runs this)
```

CI (`.github/workflows/rust.yml`) uses `cargo nextest` instead of `cargo test`;
use it if it's already installed, but don't require it.

### Python

```sh
.venv/bin/ty check
.venv/bin/ruff check .
.venv/bin/ruff format .
.venv/bin/pytest
```

CI (`.github/workflows/python.yml`) runs the equivalent via `uv run`.

## Architecture

### Rust (`crates/`)

- `picker` - a Fisher-Yates-based random-permutation iterator, with support
  for resuming from a partial result.
- `memory` - bit-level buffer types and error-correcting-code encodings
  (see the crate-level doc comment in `crates/memory/src/lib.rs` for details)
  used to simulate protected memory and inject faults into it.
- `bindings` - the PyO3 crate exposing the above to Python as `faultforge._rust`.

### Python (`packages/`)

- `faultforge`: top-level modules under `faultforge/*.py` are thin, documented
  re-export shims; the real implementation lives in `faultforge/_internal/`.
  When changing behavior, edit `_internal`; when adding a public symbol,
  re-export it from the matching top-level shim. See `faultforge/__init__.py`'s
  module docstring for a tour of the library's key parts (experiments,
  encoding, dataset/model loading, fault injection primitives).
- `faultforge_cli`: `typer`-based CLI entry point (`faultforge_cli.main:app`).

### Testing conventions already in use

- Rust: `proptest` for property-based tests.
- Python: `hypothesis` for property-based tests.

## Releasing

`main` is the development branch. The `latest` branch tracks the most recent
tagged release. To cut a release, on `main`:

1. Move the `CHANGELOG.md` `## [Unreleased]` section to a new
   `## [X.Y.Z] - YYYY-MM-DD` heading, leaving a fresh empty `[Unreleased]`
   above it.
2. Bump `version` in `packages/faultforge/pyproject.toml` and
   `packages/faultforge_cli/pyproject.toml` to `X.Y.Z` (kept in lockstep;
   `Cargo.toml`'s `workspace.package.version` stays `0.0.0`, since the Rust
   crates aren't independently versioned or published).
3. Commit, tag the commit `vX.Y.Z`, and push both the commit and the tag.

Pushing a `vX.Y.Z` tag triggers `.github/workflows/release.yml`, which builds
wheels/sdists for `faultforge` and `faultforge-cli`, publishes both to PyPI
via Trusted Publishing, creates the GitHub release (using the matching
`CHANGELOG.md` section as the release body), and fast-forwards `latest` to
the new tag.
