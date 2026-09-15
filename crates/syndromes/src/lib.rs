//! Syndrome distribution generation for stuck-at faults in a netlist.
#![warn(missing_docs)]

mod bit;
pub mod computer;
pub mod fault_selection;
pub mod histogram;
mod netlist;

pub use bit::{Bit, UnknownBitError};

/// The bitwise difference between a correct and a faulty output.
pub type Syndrome = u32;
