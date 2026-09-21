//! How the fault case for an evaluation is chosen.

use rand::{Rng, RngExt};

/// A fixed fault case that a computer does not have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
#[error("fault case {fault_case} is not below the case count {fault_case_count}")]
pub struct FaultCaseOutOfRange {
    /// The requested case.
    pub fault_case: usize,
    /// The number of cases the computer has.
    pub fault_case_count: usize,
}

/// The rule for picking a fault case per evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FaultSelection {
    /// Every case is equally likely.
    Uniform,
    /// The same case every time. Meant for diagnostics.
    Fixed(usize),
}

impl FaultSelection {
    /// A selection that always picks `fault_case`, checked against the count.
    pub fn fixed(fault_case: usize, fault_case_count: usize) -> Result<Self, FaultCaseOutOfRange> {
        if fault_case < fault_case_count {
            Ok(Self::Fixed(fault_case))
        } else {
            Err(FaultCaseOutOfRange {
                fault_case,
                fault_case_count,
            })
        }
    }

    /// Picks a case below `fault_case_count`.
    ///
    /// # Panics
    ///
    /// Panics if `fault_case_count` is zero, or if a fixed case is not below
    /// it.
    pub fn select(&self, fault_case_count: usize, rng: &mut impl Rng) -> usize {
        match *self {
            Self::Uniform => rng.random_range(0..fault_case_count),
            Self::Fixed(fault_case) => {
                assert!(
                    fault_case < fault_case_count,
                    "fixed fault case {fault_case} is not below the case count {fault_case_count}"
                );
                fault_case
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn fixed_returns_its_case() {
        let mut rng = StdRng::seed_from_u64(0);
        let selection = FaultSelection::fixed(3, 10).unwrap();
        for _ in 0..10 {
            assert_eq!(selection.select(10, &mut rng), 3);
        }
    }

    #[test]
    fn fixed_is_range_checked() {
        assert_eq!(
            FaultSelection::fixed(10, 10),
            Err(FaultCaseOutOfRange {
                fault_case: 10,
                fault_case_count: 10
            })
        );
    }

    #[test]
    fn uniform_stays_in_range_and_covers_it() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut seen = [false; 5];
        for _ in 0..200 {
            seen[FaultSelection::Uniform.select(5, &mut rng)] = true;
        }
        assert!(seen.iter().all(|&seen| seen));
    }
}
