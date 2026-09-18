//! The input source seam: something that turns an RNG into a triple.

pub mod artifact;
mod constant;
mod normal;
mod random_bits;

use crate::Triple;
use rand::Rng;

pub use constant::Constant;
pub use normal::{InvalidStandardDeviation, Normal, Scales, StandardDeviation};
pub use random_bits::RandomBits;

/// A source of multiply-add inputs.
///
/// A source is a pure function of the RNG it is handed. It holds shared
/// read-only data at most, never an RNG or other mutable state, so any
/// caller can reproduce any draw from the seed alone.
pub trait InputSource: Send + Sync {
    /// Draws one triple.
    fn triple(&self, rng: &mut dyn Rng) -> Triple;
}

#[cfg(test)]
mod test_draws {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    /// `count` triples from `source` with a fresh RNG seeded by `seed`.
    pub fn draws(source: &impl InputSource, seed: u64, count: usize) -> Vec<Triple> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..count).map(|_| source.triple(&mut rng)).collect()
    }
}
