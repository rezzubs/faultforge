//! Syndrome distribution generation for stuck-at faults in a netlist.
#![warn(missing_docs)]

mod bit;
pub mod computer;
pub mod fault_selection;
pub mod generation;
pub mod histogram;
pub mod input_source;
mod netlist;
#[cfg(test)]
mod test_files;

pub use bit::{Bit, UnknownBitError};

/// The bitwise difference between a correct and a faulty output.
pub type Syndrome = u32;

/// One multiply-add input: `activation * weight + partial_sum`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triple {
    /// The activation operand of the multiplication.
    pub activation: f32,
    /// The weight operand of the multiplication.
    pub weight: f32,
    /// The value added to the product.
    pub partial_sum: f32,
}
