# Rebuild the `faultforge._rust` extension from current Rust sources. Needed
# before Python tests can see Rust changes.
build:
    .venv/bin/maturin develop

# Lint Rust with clippy.
lint-rust:
    cargo clippy --workspace -- -D warnings

# Check Rust formatting without modifying files.
fmt-check-rust:
    cargo fmt --all -- --check

# Format Rust in place.
fmt-rust:
    cargo fmt --all

# Run Rust tests.
test-rust:
    cargo nextest run --workspace

# Check that Rust documentation builds without warnings.
doc-rust:
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items

# Type-check Python with ty.
typecheck-python:
    .venv/bin/ty check

# Lint Python with ruff.
lint-python:
    .venv/bin/ruff check .

# Check Python formatting without modifying files.
fmt-check-python:
    .venv/bin/ruff format --check .

# Format Python in place.
fmt-python:
    .venv/bin/ruff format .

# Run Python tests.
test-python:
    .venv/bin/pytest

# Everything CI checks for the `crates/` workspace.
check-rust: lint-rust fmt-check-rust test-rust doc-rust

# Everything CI checks for the Python side. Rebuilds the extension first so
# tests run against current Rust code.
check-python: build typecheck-python lint-python fmt-check-python test-python

# Format both Rust and Python in place.
fmt: fmt-rust fmt-python

# Full pre-push sanity check: Rust and Python, lint/format/types/tests/docs.
commit-checklist: check-rust check-python

# Lighter check for Python-only changes, skipping the Rust suite.
commit-checklist-python: check-python
