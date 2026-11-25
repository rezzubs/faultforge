use std::collections::HashSet;

use pyo3::prelude::*;
use systolic::fault::{
    AccumulatorFaultPart as RustAccumulatorFaultPart, LiftedRegisterFaultData,
    WeightFaultPart as RustWeightFaultPart,
};

/// The weights of one output row that pass through a faulty weight register
/// in one pass. `for_activations` is `(start, end)`, a half-open range.
// See `systolic::fault::WeightFaultPart`'s doc comment. The range is a tuple
// for the same reason as in `AccumulatorFaultPart`.
#[pyclass(eq, from_py_object, name = "WeightFaultPart")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightFaultPart {
    #[pyo3(get)]
    pub affected_output_row: usize,
    #[pyo3(get)]
    pub for_activations: (usize, usize),
}

impl From<RustWeightFaultPart> for WeightFaultPart {
    fn from(value: RustWeightFaultPart) -> Self {
        Self {
            affected_output_row: value.affected_output_row,
            for_activations: (value.for_activations.start, value.for_activations.end),
        }
    }
}

/// The accumulated-fault fix-up for one output row in one pass.
/// `for_activations` is `(start, end)`, a half-open range.
// See `systolic::fault::AccumulatorFaultPart`'s doc comment for the
// physical meaning; `for_activations` is a tuple rather than a Python
// `range` since that's all the torch applier needs it for (slicing).
#[pyclass(eq, from_py_object, name = "AccumulatorFaultPart")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccumulatorFaultPart {
    #[pyo3(get)]
    pub affected_output_row: usize,
    #[pyo3(get)]
    pub for_activations: (usize, usize),
}

impl From<RustAccumulatorFaultPart> for AccumulatorFaultPart {
    fn from(value: RustAccumulatorFaultPart) -> Self {
        Self {
            affected_output_row: value.affected_output_row,
            for_activations: (value.for_activations.start, value.for_activations.end),
        }
    }
}

/// A register fault lifted to matrix space: an equivalent description of
/// the fault in terms of operations on the weight/activation/output
/// matrices rather than the array's physical registers.
// See `docs/fault-lifting.md` for the theory and
// `systolic::fault::LiftedRegisterFaultData` for the Rust-side source of
// truth this mirrors.
#[pyclass(eq, name = "LiftedFault")]
#[derive(Debug, Clone, PartialEq)]
pub enum LiftedFault {
    Weight {
        parts: Vec<WeightFaultPart>,
    },
    Activation {
        affected_activation_rows: HashSet<usize>,
        affected_output_rows: HashSet<usize>,
    },
    Accumulator {
        parts: Vec<AccumulatorFaultPart>,
    },
}

impl From<LiftedRegisterFaultData> for LiftedFault {
    fn from(value: LiftedRegisterFaultData) -> Self {
        match value {
            LiftedRegisterFaultData::Weight { parts } => LiftedFault::Weight {
                parts: parts.into_iter().map(WeightFaultPart::from).collect(),
            },
            LiftedRegisterFaultData::Activation {
                affected_activation_rows,
                affected_output_rows,
            } => LiftedFault::Activation {
                affected_activation_rows,
                affected_output_rows,
            },
            LiftedRegisterFaultData::Accumulator { parts } => LiftedFault::Accumulator {
                parts: parts.into_iter().map(AccumulatorFaultPart::from).collect(),
            },
        }
    }
}
