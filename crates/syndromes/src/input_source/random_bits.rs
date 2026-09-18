//! Uniformly random bit patterns.

use crate::{Triple, input_source::InputSource};
use rand::{Rng, RngExt};

/// Uniformly random bit patterns in every field.
///
/// Not physically meaningful; NaN and infinities appear like any other
/// pattern. Meant for testing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RandomBits;

impl InputSource for RandomBits {
    fn triple(&self, rng: &mut dyn Rng) -> Triple {
        Triple {
            activation: f32::from_bits(rng.random()),
            weight: f32::from_bits(rng.random()),
            partial_sum: f32::from_bits(rng.random()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input_source::test_draws::draws;

    #[test]
    fn random_bits_vary_and_are_reproducible() {
        let first = draws(&RandomBits, 1, 16);
        let second = draws(&RandomBits, 1, 16);
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
