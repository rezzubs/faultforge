"""An experiment for measuring model reliability under memory faults.

`EncodedFaultInjection` runs a model whose parameters are stored through a
`faultforge.encoding.Encoder`, injects bit flips into that encoded memory, and
scores the result according to a `faultforge.metric.Metric`.

`check_saved` tells whether saved results still need more runs, without
loading the model or dataset.
"""

from encoded_memory.check import (
    CheckResult,
    CheckStatus,
    check_saved,
)
from encoded_memory.experiment import (
    DetailedResults,
    DetailedRunResult,
    EncodedFaultInjection,
    EncodedFaultInjectionDisplay,
    SavedResult,
    SimpleResults,
    discard_bitmasks_in_file,
)
from encoded_memory.stability import StabilityLimits

__all__ = [
    "CheckResult",
    "CheckStatus",
    "DetailedResults",
    "DetailedRunResult",
    "EncodedFaultInjection",
    "EncodedFaultInjectionDisplay",
    "SavedResult",
    "SimpleResults",
    "StabilityLimits",
    "check_saved",
    "discard_bitmasks_in_file",
]
