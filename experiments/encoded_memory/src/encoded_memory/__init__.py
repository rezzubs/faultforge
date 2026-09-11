"""An experiment for measuring model reliability under memory faults.

`EncodedFaultInjection` runs a model whose parameters are stored through a
`faultforge.encoding.Encoder`, injects bit flips into that encoded memory, and
scores the result according to a `ReliabilityMetric`.
"""

from encoded_memory.experiment import (
    DetailedResults,
    DetailedRunResult,
    EncodedFaultInjection,
    SavedResult,
    SimpleResults,
    discard_bitmasks_in_file,
)

__all__ = [
    "DetailedResults",
    "DetailedRunResult",
    "EncodedFaultInjection",
    "ReliabilityMetric",
    "SavedResult",
    "SimpleResults",
    "discard_bitmasks_in_file",
]
