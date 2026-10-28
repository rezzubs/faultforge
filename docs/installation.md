# Installation

FaultForge requires Python 3.14 or newer.

## From PyPI

```sh
pip install faultforge
# or
uv add faultforge
```

Prebuilt wheels are available for:

- Linux (x86_64 and aarch64)
- macOS (Apple silicon)
- Windows (x86_64)

On other platforms pip builds from the source distribution, which needs a
Rust toolchain, see [below](#from-source).

## From source

Building from source requires a Rust toolchain. Install one with your system
package manager or via [rustup](https://rustup.rs/).

The repository's root package is the `faultforge` library. Point pip at
whichever revision you want:

```sh
# latest release (the latest branch tracks the most recent tagged release)
pip install 'faultforge @ git+https://github.com/rezzubs/faultforge.git@latest'

# a specific released version
pip install 'faultforge @ git+https://github.com/rezzubs/faultforge.git@vX.Y.Z'

# a specific commit
pip install 'faultforge @ git+https://github.com/rezzubs/faultforge.git@<commit-sha>'

# development (main can break at any point)
pip install 'faultforge @ git+https://github.com/rezzubs/faultforge.git'
```

## Reference experiment

The [Encoded Memory](../experiments/encoded_memory/README.md) experiment isn't
published to PyPI. Its README covers installing it from the repository.
