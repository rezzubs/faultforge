"""Error correction and fault simulation for PyTorch."""

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
