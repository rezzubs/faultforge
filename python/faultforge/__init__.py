"""Error correction and fault simulation for PyTorch.

Systems
-------
At the core of this library is the :class:`system.System` class, which
represents a neural network model and its associated data. The system needs to
expose a core data object and provide a method to get an accuracy metric based
on that object. The system is also responsible for fault injection.

The :class:`experiment.Experiment` class can be used to record experiments for a
system over many runs of fault injection.

System Implementations
----------------------
- The :mod:`imagenet` package provides a **system** implementation for ImageNet
  models and related types.
- The :mod:`cifar` package provides a **system** implementation for CIFAR models
  and related types.
- The :mod:`encoding` package provides different *encoded* **system** types
  which wrap other **system** implementations.

Utilities
---------
The module :mod:`tensor_ops` provides utility functions for tensors and lists of
tensors including fault injection and bitwise comparison.

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
