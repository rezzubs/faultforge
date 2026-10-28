# FaultForge

[![CI](https://github.com/rezzubs/faultforge/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rezzubs/faultforge/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Frezzubs%2Ffaultforge%2Fmain%2Fpyproject.toml&query=%24.project.license&label=license)](https://github.com/rezzubs/faultforge/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/faultforge)](https://pypi.org/project/faultforge/)
[![Python](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Frezzubs%2Ffaultforge%2Fmain%2Fpyproject.toml&query=%24.project%5B%27requires-python%27%5D&label=python)](https://github.com/rezzubs/faultforge/blob/main/pyproject.toml)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Built with Nix](https://img.shields.io/badge/built%20with-Nix-5277C3?logo=nixos&logoColor=white)](https://builtwithnix.org)

A toolkit for reproducible hardware fault-injection experiments on PyTorch
models.

## Design principles

- **Standalone parts.** Each component is useful by itself. Encode a tensor,
  inject faults into it, or summarize a list of scores without setting up
  anything else.
- **Built to fit together.** The same parts combine into complete,
  resumable experiments with little glue code.
- **Use as much or as little as you need.** Nothing forces you to adopt the
  rest of the toolkit to use one piece of it.
- **Extensible.** The built-in models, encodings, metrics and experiments
  implement public base classes. Your own implementations plug in the same
  way.

## Highlights

Building blocks:

- [**Fault model**](https://github.com/rezzubs/faultforge/blob/main/docs/library.md#fault-injection-primitives) -
  bit flips and stuck-at faults, a resumable repeat-free location sampler and
  batched injection.
- [**Encodings**](https://github.com/rezzubs/faultforge/blob/main/docs/library.md#encoder--encoding) -
  SECDED (Hamming codes), MSET and CEP, usable alone or chained together, and
  a wrapper that stores a model's parameters in encoded memory.
- [**Metrics**](https://github.com/rezzubs/faultforge/blob/main/src/faultforge/metric.py) -
  a generic system for reliability metrics.
- [**Model loading**](https://github.com/rezzubs/faultforge/blob/main/docs/library.md#modelbundle) -
  models bundled with their evaluation datasets. Image classifiers on
  CIFAR-10/100 and ImageNet, and HuggingFace language models on WikiText, are
  built in.

Experiment harness:

- [**Experiments**](https://github.com/rezzubs/faultforge/blob/main/docs/library.md#experiment) -
  automatic run loops with clean Ctrl+C handling, and atomic, resumable,
  optionally zstd-compressed saving.
- [**Estimators**](https://github.com/rezzubs/faultforge/blob/main/docs/library.md#estimators) -
  mean and failure rate, with 95% confidence intervals.
- [**Stop conditions**](https://github.com/rezzubs/faultforge/blob/main/docs/library.md#stop-conditions) -
  run until an estimate is stable, for a fixed number of runs, or until
  interrupted.
- [**Fingerprints**](https://github.com/rezzubs/faultforge/blob/main/docs/library.md#fingerprint) -
  resuming a saved result with a changed configuration fails with a precise
  diff instead of silently mixing incompatible data.

Performance-critical paths are implemented in Rust.

[`docs/library.md`](https://github.com/rezzubs/faultforge/blob/main/docs/library.md)
walks through each part and ends with a complete example.

## Installation

```sh
pip install faultforge
# or
uv add faultforge
```

See [`docs/installation.md`](https://github.com/rezzubs/faultforge/blob/main/docs/installation.md)
for building from source and installing specific revisions.

## Reference experiment

[Encoded Memory](https://github.com/rezzubs/faultforge/blob/main/experiments/encoded_memory/README.md)
injects faults into ECC-protected model parameters. It's a complete
experiment built on FaultForge and the reference implementation to follow when
writing your own. It also provides an `encoded-memory` CLI for recording
results.

It lives in this repository and is released together with the library, but
isn't published to PyPI. Its README covers installation.

## Stability

FaultForge is experimental. Expect breaking changes in most releases, and
check the [changelog](https://github.com/rezzubs/faultforge/blob/main/CHANGELOG.md)
when upgrading. Versions follow [semver](https://semver.org/): while the
version is `0.x`, minor releases can break and patch releases can't. A range
like `faultforge>=0.3.0,<0.4.0` picks up only compatible releases.

- `main` is the development branch and can break at any point. The
  [`latest`](https://github.com/rezzubs/faultforge/tree/latest) branch tracks
  the most recent release.
- The Rust crates in this repository are internal and can change at any time.
  They aren't published, and their version only mirrors the library's.

## Publications

FaultForge has been used to develop and evaluate the techniques in several
papers. See [`docs/publications.md`](https://github.com/rezzubs/faultforge/blob/main/docs/publications.md)
for links and citations.

## License

FaultForge is licensed under the [Universal Permissive License (UPL),
Version 1.0](https://github.com/rezzubs/faultforge/blob/main/LICENSE). It's
an [OSI-approved](https://opensource.org/license/upl) permissive license that
is about as short as MIT, but:

- includes an explicit patent grant, like Apache-2.0.
- is compatible with both GPLv2 and GPLv3. Apache-2.0 isn't compatible with
  GPLv2, which is why many projects dual-license under "MIT OR Apache-2.0".
- extends the patent grant to larger works that include the software, not
  only the software itself.

The main condition is the same as MIT's: keep the copyright and license notice
in copies of the software.
