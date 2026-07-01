"""An experiment for measuring model reliability under memory faults.

`EncodedFaultInjection` runs a model whose parameters are stored through a
`faultforge.encoding.Encoder`, injects bit flips into that encoded memory, and
scores the result according to a `ReliabilityMetric`.
"""

from faultforge._internal.experiments.encoded_memory import (
    EncodedFaultInjection,
    ReliabilityMetric,
)

__all__ = [
    "EncodedFaultInjection",
    "ReliabilityMetric",
]
