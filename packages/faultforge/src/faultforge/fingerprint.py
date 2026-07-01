"""Structural fingerprints for identifying experiments and their components.

A `Fingerprint` is a small, serializable description of the *semantics* of an
experiment component (an encoder, a model/dataset bundle, ...). It captures only
the parts that change what the component does. Environmental knobs like the
device, batch size or filesystem paths should be deliberately left out.

Fingerprints are never turned back into live objects. They exist purely so that
a result file saved to disk can be matched against the configuration that is
about to be used, and so that a mismatch can be reported in a way that points at
exactly what changed.
"""

from faultforge._internal.fingerprint import (
    ABSENT,
    Fingerprint,
    FingerprintDifference,
    format_differences,
)

__all__ = [
    "ABSENT",
    "Fingerprint",
    "FingerprintDifference",
    "format_differences",
]
