# FaultForge

Fault mitigation simulations for PyTorch models.

## Highlights

- Apply various encodings on PyTorch models.
- A CLI for running fault injection experiments with the provided techniques
- Designed to be extended. Supports:
  - Custom encodings
  - Custom DNN models
  - Custom test benches.
- Many common DNN models supported out of the box for ImageNet and CIFAR10/100.

## Installation

The only system dependency is a rust compiler (only for installation).
You can install rust using your system package manager or with [rustup](https://rustup.rs/).

To install the faultforge package into your virtual environment, run:

```sh
pip install 'git+https://github.com/rezzubs/faultforge.git[all]'
```

The main branch is kept in sync with the latest version. To install a specific version (tag) run:

```sh
pip install 'git+https://github.com/rezzubs/faultforge.git@0.1.0[all]'
```

The tag can also be replaced with a commit hash

### Optional features

Some features are optional and dependencies for them are not installed by default.
These can be enabled by specifying the relevant groups within during pip install (`[group1,group2]`)

| group name | description |
| - | - |
| all | Install all optional dependencies |
| imagenet | Add support for ImageNet models |
| cli | Enables the usage of the CLI |
| cifar | Add support for CIFAR models |

By default, only the dependencies required by the core library installed.

## CLI Usage

The CLI is documented in help pages:

```sh
faultforge --help
```

TODO

## Library Architecture

TODO

## Encoding Formats

TODO

## Supported Models

TODO
