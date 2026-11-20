//! Uniformly random bit patterns.

use crate::Triple;
use rand::{Rng, RngExt};

/// A triple with uniformly random bits in every field.
pub fn triple(rng: &mut impl Rng) -> Triple {
    Triple {
        activation: f32::from_bits(rng.random()),
        weight: f32::from_bits(rng.random()),
        partial_sum: f32::from_bits(rng.random()),
    }
}

#[cfg(test)]
mod tests {
    use crate::input_source::{InputSource, test_draws::draws};

    #[test]
    fn random_bits_vary_and_are_reproducible() {
        let first = draws(&InputSource::RandomBits, 1, 16);
        let second = draws(&InputSource::RandomBits, 1, 16);
        assert_eq!(
            first
                .iter()
                .map(|triple| triple.activation.to_bits())
                .collect::<Vec<_>>(),
            second
                .iter()
                .map(|triple| triple.activation.to_bits())
                .collect::<Vec<_>>()
        );
        let distinct: std::collections::HashSet<u32> = first
            .iter()
            .map(|triple| triple.activation.to_bits())
            .collect();
        assert!(distinct.len() > 1);
    }
}
