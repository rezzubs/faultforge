"""An experiment for measuring model reliability under memory faults.

`EncodedFaultInjection` runs a model whose parameters are stored through a
`faultforge.encoding.Encoder`, injects bit flips into that encoded memory, and
scores the result according to a `ReliabilityMetric`.
"""

from faultforge._internal.experiments.encoded_memory import (
    DetailedResult,
    DetailedRunResult,
    EncodedFaultInjection,
    ReliabilityMetric,
    SimpleResult,
    discard_bitmasks_in_file,
)

__all__ = [
    "DetailedResult",
    "DetailedRunResult",
    "EncodedFaultInjection",
    "ReliabilityMetric",
    "SimpleResult",
    "discard_bitmasks_in_file",
]
