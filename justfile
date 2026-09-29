# Full pre-push sanity check: everything CI checks for Rust and Python.
[group('general')]
ci: ci-rust ci-python

# Everything CI checks for the `crates/` workspace.
[group('rust')]
ci-rust: lint-rust fmt-check-rust test-rust doc-rust

# Everything CI checks for the Python side.
[group('python')]
ci-python: typecheck-python lint-python fmt-check-python test-python

# Format both Rust and Python in place.
[group('general')]
fmt: fmt-rust fmt-python

# Format Rust in place.
[group('rust')]
fmt-rust:
    cargo fmt --all

# Format Python in place.
[group('python')]
fmt-python:
    uv run --all-packages ruff format .

# Check Rust and Python formatting without modifying files.
[group('general')]
fmt-check: fmt-check-rust fmt-check-python

# Check Rust formatting without modifying files.
[group('rust')]
fmt-check-rust:
    cargo fmt --all -- --check

# Check Python formatting without modifying files.
[group('python')]
fmt-check-python:
    uv run --all-packages ruff format --check .

# Lint Rust and Python.
[group('general')]
lint: lint-rust lint-python

# Lint Rust with clippy.
[group('rust')]
lint-rust:
    cargo clippy --workspace -- -D warnings

# Lint Python with ruff.
[group('python')]
lint-python:
    uv run --all-packages ruff check .

# Run Rust and Python tests.
[group('general')]
test: test-rust test-python

# Run Rust tests.
[group('rust')]
test-rust:
    cargo nextest run --workspace

# Run Python tests.
[group('python')]
test-python:
    uv run --all-packages pytest

# Type-check Python with ty.
[group('python')]
typecheck-python:
    uv run --all-packages ty check

# Check that Rust documentation builds without warnings.
[group('rust')]
doc-rust:
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items
