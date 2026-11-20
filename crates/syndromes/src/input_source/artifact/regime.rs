//! The phase of the computation an element was in.

use std::fmt;

/// The phase of the computation an element was in when its inputs were
/// recorded.
///
/// The zero regime is not listed: its input is always `(0, 0, 0)` and nothing
/// is recorded for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Regime {
    /// The first product of an accumulation, with a zero partial sum.
    First,
    /// A product added to a running partial sum.
    Active,
    /// A finished sum passed through, with zero activation and weight.
    Drain,
}

impl fmt::Display for Regime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::First => "first",
            Self::Active => "active",
            Self::Drain => "drain",
        })
    }
}
