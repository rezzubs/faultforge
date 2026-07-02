"""A framework for running reproducible experiments on PyTorch models.

faultforge provides the general infrastructure for repeating an experiment
until a statistically stable result is reached, saving and resuming its
progress, and verifying a saved result against the configuration that
produced it. Here is a general overview of the key parts of the library; see
submodules for details.

**Framework**

- `Fingerprint`: a structural, serializable identity for an experiment
  component, used to verify a saved result against the configuration that
  produced it. See `faultforge.fingerprint` for related helpers.
- `faultforge.experiment`: `Experiment`, the base class for running an
  operation repeatedly until a stable result is reached, with save/resume
  support tied to a `Fingerprint`.
- `faultforge.dataset`: `BatchedDataset`/`CachedDataset` for iterating over a
  dataset in fixed-size batches.
- `faultforge.loading`: Provides a `ModelBundle` for loading a model together
  with its evaluation dataset.
- `faultforge.progress`: `Progress`, for reporting on long-running operations
  (dataset loading, encoding, fault injection, ...) via periodic log messages.

To add a new kind of experiment, subclass `Experiment` and reuse
`faultforge.loading`/`faultforge.dataset` for model and data handling.

**Ready-made experiments**

`faultforge.experiments` holds experiments built on the framework above.
Currently there is one, `faultforge.experiments.encoded_memory`: encoded-
memory fault injection, which measures model reliability under simulated
single-event upsets in error-corrected parameter memory. It's a complete
example to follow when adding a new experiment. It's built on:

- `faultforge.encoding`: `Encoder`/`Encoding` pairs that transform tensors into
  a protected representation and back, plus `EncodedModule` for wrapping a
  `torch.nn.Module` so its parameters live in simulated encoded memory.

**Fault injection primitives**

A few lower-level pieces used to build the above are exposed directly at the
root rather than in their own submodule:

- `Fault` (`BitFlip`, `StuckAt`): describes what happens to a targeted bit.
- `Picker`: a Fisher-Yates based sampler used to pick fault locations
  without repeats.
- `tensor_list_dtype`/`tensor_list_fault`: the tensor-level operations that
  back fault injection.
"""

import sys

from faultforge._internal.common import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    AnyPath,
    DeviceLike,
)
from faultforge._internal.fault import BitFlip, Fault, StuckAt
from faultforge._internal.fingerprint import Fingerprint
from faultforge._internal.tensor import tensor_list_dtype, tensor_list_fault
from faultforge._rust import Picker

from . import _rust

__all__ = [
    "AnyPath",
    "BitFlip",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DEVICE",
    "DeviceLike",
    "Fault",
    "Fingerprint",
    "Picker",
    "StuckAt",
    "tensor_list_dtype",
    "tensor_list_fault",
]


# This function exists to set up the rust extension modules so they can be
# imported like any other python module.
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
