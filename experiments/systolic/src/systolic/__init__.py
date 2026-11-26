"""An experiment for measuring model reliability under systolic-array
register faults.

`SystolicFaultInjection` runs the matrix multiplications of a model on a
systolic array (see `OnArray`), injects a stuck-at register fault into one
processing element, and scores the result according to a
`faultforge.metric.Metric`.

`OnArray` runs the matrix multiplications inside it on an `Array`, using an
`Evaluator` to compute each one as an `ArrayMatmul`:

- `SimulatedEvaluator` simulates the array cycle by cycle. It's the oracle.
- `LiftedEvaluator` applies the effect of a fault with torch, without
  simulating the array. It's the fast workhorse (Rust computes the lift,
  torch applies it).
- `ProfilingEvaluator` records the inputs of the processing elements.

`run_profiling` runs a model over a random dataset subsample on a
`ProfilingEvaluator`, pooling the MAC inputs of every matrix multiplication
into one `ProfilingArtifact` - the per-PE input distribution later
fault-modeling work samples from. `save_profiling_artifact`/
`load_profiling_artifact` round-trip that artifact to a single `.npz` file.

`gap_grids` compares each PE's profiled inputs against the pooled array (see
`GapGrid`), a first, cheap look at whether faults need modeling per row,
column, or PE rather than for the array as a whole. `build_gap_heatmap_figure`
draws the result.
"""

lazy from systolic._rust import (
    AccumulatorFaultPart,
    ArrayConfig,
    Fault,
    Index2,
    LiftedFault,
    Mapping,
    Pass,
    PeRegisterKind,
    StuckAtKind,
    WeightFaultPart,
)
lazy from systolic.array import Array, ArrayMatmul
lazy from systolic.evaluator import Evaluator
lazy from systolic.experiment import (
    EvaluatorKind,
    RunResult,
    SavedResult,
    SystolicFaultInjection,
    SystolicFaultInjectionDisplay,
)
lazy from systolic.fault import RegisterFaults
lazy from systolic.lifted_evaluator import LiftedEvaluator
lazy from systolic.on_array import OnArray
lazy from systolic.profiling import (
    ProfilingMetadata,
    load_profiling_artifact,
    save_profiling_artifact,
)
lazy from systolic.profiling_driver import run_profiling
lazy from systolic.profiling_evaluator import ProfilingEvaluator
lazy from systolic.profiling_plots import build_gap_heatmap_figure
lazy from systolic.profiling_similarity import GapGrid, gap_grids
lazy from systolic.regime import Regime, Variable
lazy from systolic.simulated_evaluator import SimulatedEvaluator
lazy from systolic.stability import StabilityLimits

__all__ = [
    "AccumulatorFaultPart",
    "Array",
    "ArrayConfig",
    "ArrayMatmul",
    "Evaluator",
    "EvaluatorKind",
    "Fault",
    "GapGrid",
    "Index2",
    "LiftedEvaluator",
    "LiftedFault",
    "Mapping",
    "OnArray",
    "Pass",
    "PeRegisterKind",
    "ProfilingEvaluator",
    "ProfilingMetadata",
    "Regime",
    "RegisterFaults",
    "RunResult",
    "SavedResult",
    "SimulatedEvaluator",
    "StabilityLimits",
    "StuckAtKind",
    "SystolicFaultInjection",
    "SystolicFaultInjectionDisplay",
    "Variable",
    "WeightFaultPart",
    "build_gap_heatmap_figure",
    "gap_grids",
    "load_profiling_artifact",
    "run_profiling",
    "save_profiling_artifact",
]
