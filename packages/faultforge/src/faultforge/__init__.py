"""A framework for running reproducible experiments on PyTorch models.

faultforge provides the general infrastructure for repeating an experiment
until a statistically stable result is reached, saving and resuming its
progress, and verifying a saved result against the configuration that
produced it. Here is a general overview of the key parts of the library; see
submodules for details.

Key Components:
- `faultforge.experiment`: `Experiment`, the base class for running an
  operation repeatedly until a stable result is reached, with save/resume
  support tied to a `faultforge.fingerprint.Fingerprint`.
- `faultforge.dataset`: `BatchedDataset`/`CachedDataset` for iterating over a
  dataset in fixed-size batches.
- `faultforge.loading`: Provides a `ModelBundle` for loading a model together
  with its evaluation dataset.

Built on the framework, `faultforge.experiments` holds ready-made
experiments. Currently there is one: encoded-memory fault injection, which
measures model reliability under simulated single-event upsets in
error-corrected parameter memory. Its supporting pieces:

- `faultforge.encoding`: `Encoder`/`Encoding` pairs that transform tensors into
  a protected representation and back, plus `EncodedModule` for wrapping a
  `torch.nn.Module` so its parameters live in simulated encoded memory.
- `faultforge.fault`: `Fault` variants (`BitFlip`, `StuckAt`) describing what
  happens to a targeted bit.
- `faultforge.picker`: `Picker`, a Fisher-Yates based sampler used to pick
  fault locations without repeats.

To add a new kind of experiment, subclass `Experiment` and reuse
`faultforge.loading`/`faultforge.dataset` for model and data handling; see
`faultforge.experiments.encoded_memory` for a complete example.
"""

import sys

from . import _rust


def _register_submodules(module, prefix, seen=None):
    def is_module(obj):
        return isinstance(obj, type(sys))

    if seen is None:
        seen = set()

    if id(module) in seen:
        return
    seen.add(id(module))

    for name in dir(module):
        if name.startswith("__"):
            continue
        obj = getattr(module, name)
        if is_module(obj):
            full_name = f"{prefix}.{name}"
            sys.modules[full_name] = obj
            _register_submodules(obj, full_name, seen)


_register_submodules(_rust, f"{__name__}._rust")
