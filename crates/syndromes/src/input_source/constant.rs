//! A fixed triple.

use crate::{Triple, input_source::InputSource};
use rand::Rng;

/// The same triple every time.
///
/// `(0, 0, 0)` is the input of the zero regime, whose histogram is exact
/// after one pass over every fault case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Constant(pub Triple);

impl InputSource for Constant {
    fn triple(&self, _rng: &mut dyn Rng) -> Triple {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input_source::test_draws::draws;

    #[test]
    fn constant_returns_its_triple() {
        let triple = Triple {
            activation: 1.5,
            weight: -2.0,
            partial_sum: 0.25,
        };
        for drawn in draws(&Constant(triple), 3, 4) {
            assert_eq!(drawn, triple);
        }
    }
}
