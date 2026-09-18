//! The artifact sampled with replacement.

use super::Pool;
use crate::{Triple, input_source::InputSource};
use rand::Rng;

/// Recorded triples drawn whole, with replacement.
#[derive(Debug, Clone, PartialEq)]
pub struct Joint(Pool);

impl Joint {
    /// A source over `pool`.
    pub fn new(pool: Pool) -> Self {
        Self(pool)
    }
}

impl InputSource for Joint {
    fn triple(&self, rng: &mut dyn Rng) -> Triple {
        *self.0.pick(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        super::{Group, Regime, test_archives::*},
        *,
    };
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn joint_draws_pooled_triples() {
        let file = fixture_archive("joint");
        let pool = load(&file, Regime::Active, Group::Row(1)).expect("loads");
        let recorded = pool.triples().to_vec();
        let source = Joint::new(pool);
        let mut rng = StdRng::seed_from_u64(0);
        let mut seen = vec![false; recorded.len()];
        for _ in 0..500 {
            let triple = source.triple(&mut rng);
            let index = recorded
                .iter()
                .position(|candidate| *candidate == triple)
                .expect("drawn triple is in the pool");
            seen[index] = true;
        }
        assert!(seen.iter().all(|&seen| seen));
    }
}
