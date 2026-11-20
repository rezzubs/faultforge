//! The input source seam: something that turns an RNG into a triple.

pub mod artifact;
mod normal;
mod random_bits;

use crate::Triple;
use rand::Rng;

pub use normal::{InvalidStandardDeviation, Normal, Scales, StandardDeviation};

/// A source of multiply-add inputs.
///
/// A source is a pure function of the RNG it is handed. It holds shared
/// read-only data at most, never an RNG or other mutable state, so any
/// caller can reproduce any draw from the seed alone.
#[derive(Debug, Clone, PartialEq)]
pub enum InputSource {
    /// Uniformly random bit patterns in every field.
    ///
    /// Not physically meaningful; NaN and infinities appear like any other
    /// pattern. Meant for testing.
    RandomBits,
    /// The same triple every time. `(0, 0, 0)` is the input of the zero
    /// regime.
    Constant(Triple),
    /// Independent normal draws per field.
    Normal(Normal),
    /// Recorded triples drawn whole.
    Joint(artifact::Joint),
    /// Recorded values drawn per field.
    Marginals(artifact::Marginals),
}

impl InputSource {
    /// Draws one triple.
    pub fn triple(&self, rng: &mut impl Rng) -> Triple {
        match self {
            Self::RandomBits => random_bits::triple(rng),
            Self::Constant(triple) => *triple,
            Self::Normal(normal) => normal.triple(rng),
            Self::Joint(joint) => joint.triple(rng),
            Self::Marginals(marginals) => marginals.triple(rng),
        }
    }
}

#[cfg(test)]
mod test_draws {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    /// `count` triples from `source` with a fresh RNG seeded by `seed`.
    pub fn draws(source: &InputSource, seed: u64, count: usize) -> Vec<Triple> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..count).map(|_| source.triple(&mut rng)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{test_draws::draws, *};

    #[test]
    fn constant_returns_its_triple() {
        let triple = Triple {
            activation: 1.5,
            weight: -2.0,
            partial_sum: 0.25,
        };
        for drawn in draws(&InputSource::Constant(triple), 3, 4) {
            assert_eq!(drawn, triple);
        }
    }
}
