use crate::Bit;

/// A bit-level fault that can be applied to a [`BitBuffer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fault {
    /// The bit will be flipped.
    Flip,
    /// The bit will be stuck at a specific value.
    ///
    /// Note that this doesn't prevent further writes to the bit and only
    /// affects the current value.
    StuckAt(Bit),
}
