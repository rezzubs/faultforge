# FaultForge

A framework for running reproducible hardware fault-injection experiments on
PyTorch models - with error-corrected memory reliability testing built in as
its first experiment.

> [!NOTE]
> `main` is the active development branch. For the latest release, see the [`latest`](https://github.com/rezzubs/faultforge/tree/latest) branch.

## Highlights

- A reusable experiment framework ([`Experiment`](docs/library.md#experiment)):
  automatic run loops, statistical stop conditions (run until a stable mean,
  a fixed count, or Ctrl+C), and atomic, resumable, optionally
  zstd-compressed saving.
- [`Fingerprint`](docs/library.md#fingerprint)-based verification: resuming or
  comparing a saved result against a changed configuration fails loudly with
  a precise diff, instead of silently mixing incompatible data.
- A composable [encoding framework](docs/experiments/encoded_memory.md#encoding-techniques)
  (`Encoder`/`Encoding`, chainable via `EncoderSequence`) with three built-in
  ECC-style techniques - SECDED (Hamming codes), MSET, and CEP - usable
  standalone or combined.
- An explicit fault model (bit flips and stuck-at faults) with a resumable,
  repeat-free fault-location sampler, and a batched injection API built for
  performance.
- One ready-made experiment today,
  [`encoded_memory`](docs/experiments/encoded_memory.md): fault injection into
  ECC-protected model parameters, with built-in CIFAR-10/100 and ImageNet
  model/dataset loading and a CLI for recording and plotting results. It
  doubles as the reference implementation to follow when adding a new
  experiment.
- Performance-critical bit-buffer, encoding, and fault-injection logic is
  implemented in Rust and exposed to Python via PyO3.

## Installation

FaultForge is split into two packages:

- [`faultforge`](https://pypi.org/project/faultforge/) - the library.
- [`faultforge-cli`](https://pypi.org/project/faultforge-cli/) - a CLI built
  on top of it (installs a `faultforge` command).

```sh
pip install faultforge        # library only
pip install faultforge-cli    # adds the `faultforge` CLI command
```

Building from source requires a Rust toolchain, since the library's
performance-critical parts are a PyO3 extension compiled with
[maturin](https://www.maturin.rs/). Install one with your system package
manager or via [rustup](https://rustup.rs/). No toolchain is needed when
installing a prebuilt wheel from PyPI.

### Installing from source

Both packages live in this repository, under `packages/faultforge` and
`packages/faultforge_cli`. Point pip at a subdirectory of whichever revision
you want:

```sh
# latest (main is kept in sync with the newest code going forward)
pip install 'faultforge @ git+https://github.com/rezzubs/faultforge.git#subdirectory=packages/faultforge'

# latest release (the latest branch tracks the most recent tagged release)
pip install 'faultforge @ git+https://github.com/rezzubs/faultforge.git@latest#subdirectory=packages/faultforge'

# a specific released version
pip install 'faultforge @ git+https://github.com/rezzubs/faultforge.git@v0.2#subdirectory=packages/faultforge'

# a specific commit
pip install 'faultforge @ git+https://github.com/rezzubs/faultforge.git@<commit-sha>#subdirectory=packages/faultforge'
```

Substitute `packages/faultforge_cli` for `packages/faultforge` (and
`faultforge-cli` for the package name before `@`) to install the CLI the same
way.

FaultForge requires Python 3.14 or newer.

## Quick example

```python
from faultforge.encoding import SecdedEncoder
from faultforge.experiment import MaxRuns
from faultforge.experiments.encoded_memory import (
    EncodedFaultInjection,
    ReliabilityMetric,
)
from faultforge.loading import Cifar, CifarDataset, CifarModel

bundle = Cifar(model=CifarModel.ResNet20, dataset=CifarDataset.Cifar10)
encoder = SecdedEncoder(bits_per_chunk=64)

experiment = EncodedFaultInjection(
    bundle,
    encoder,
    ReliabilityMetric.Accuracy,
    faults=1e-3,  # a bit error rate; pass an int instead for an exact fault count
)
experiment.run_loop(stop_conditions=[MaxRuns(total=20)])

print("Mean accuracy:", experiment.mean_score())
```

See [`docs/library.md`](docs/library.md) for a full walkthrough of the
framework's pieces and how they compose.

## Experiments

FaultForge ships one ready-made experiment today, built as the reference
implementation for adding your own:

- [Encoded Memory](docs/experiments/encoded_memory.md) - fault injection into
  ECC-protected model parameters, covering the available encoding techniques,
  the `faultforge encoded-memory` CLI, and using the experiment directly as a
  library.

## Citation

### [Encoded Memory](docs/experiments/encoded_memory.md)

There are two papers related to the encoded memory experiment.


#### Paper in DATE 2026

Introduced the MSET technique as a zero cost alternative to ECCs.

```bibtex
@inproceedings{ahmadilivani2026late,
  title={Late Breaking Results: Uncovering the Limits of ECCs in Vision Transformers and a Zero-Cost Reliability Enhancement},
  author={Ahmadilivani, Mohammad Hasan and Roots, Marten and Restifo, Marco and Loorits, Sven-Markus and Di Mauro, Luca and Raik, Jaan},
  booktitle={2026 Design, Automation \& Test in Europe Conference (DATE)},
  pages={1--3},
  year={2026},
  organization={IEEE},
  link={https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11539549}
}
```


#### Paper in IOLTS 2026

Introduced the CEP technique:

```bitext
@article{ahmadilivani2026effective,
  title={Effective and Memory-Efficient Alternatives to ECC for Reliable Large-Scale DNNs},
  author={Ahmadilivani, Mohammad Hasan and Roots, Marten and Restifo, Marco and Loorits, Sven-Markus and Di Mauro, Luca and Raik, Jaan},
  booktitle={The 32nd IEEE International Symposium on On-Line Testing and Robust System Design (IOLTS)},
  year={2026},
  organization={IEEE},
  link={arXiv preprint arXiv:2605.07417}
}
```

## License

[UPL-1.0](LICENSE)
