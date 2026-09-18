//! The artifact with each field sampled on its own.

use super::Pool;
use crate::{Triple, input_source::InputSource};
use rand::Rng;

/// Each field drawn from its own marginal, independently of the others.
///
/// Exists to test whether the joint distribution of the inputs matters or
/// only each input's own distribution does. In the drain regime it is the
/// same as [`Joint`](super::Joint), since activation and weight are always zero there.
#[derive(Debug, Clone, PartialEq)]
pub struct Marginals(Pool);

impl Marginals {
    /// A source over `pool`.
    pub fn new(pool: Pool) -> Self {
        Self(pool)
    }
}

impl InputSource for Marginals {
    fn triple(&self, rng: &mut dyn Rng) -> Triple {
        Triple {
            activation: self.0.pick(rng).activation,
            weight: self.0.pick(rng).weight,
            partial_sum: self.0.pick(rng).partial_sum,
        }
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
    fn marginals_mix_fields_of_pooled_triples() {
        let file = fixture_archive("marginals");
        let pool =
            load(&file, Regime::Active, Group::Element { row: 1, column: 1 }).expect("loads");
        let recorded = pool.triples().to_vec();
        let source = Marginals::new(pool);
        let mut rng = StdRng::seed_from_u64(0);
        let mut mixed = false;
        for _ in 0..500 {
            let triple = source.triple(&mut rng);
            assert!(
                recorded
                    .iter()
                    .any(|candidate| candidate.activation == triple.activation)
            );
            assert!(
                recorded
                    .iter()
                    .any(|candidate| candidate.weight == triple.weight)
            );
            assert!(
                recorded
                    .iter()
                    .any(|candidate| candidate.partial_sum == triple.partial_sum)
            );
            mixed |= !recorded.contains(&triple);
        }
        assert!(mixed, "every draw was a recorded triple");
    }
}
