//! When a histogram counts as stable.

use crate::histogram::Histogram;
use std::num::NonZeroU64;

/// A probability-scale threshold: a finite number in `0..=1`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Threshold(f64);

/// A number that is not a [`Threshold`].
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error("{0} is not a number in 0..=1")]
pub struct InvalidThreshold(pub f64);

impl Threshold {
    /// The threshold as a number.
    pub fn get(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for Threshold {
    type Error = InvalidThreshold;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        // NaN fails the range check on its own.
        if (0.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(InvalidThreshold(value))
        }
    }
}

/// The rule that ends a run.
///
/// Both statistics must be at or below their thresholds. The rule is asked
/// every `check_interval` jobs, so a run ends at a multiple of it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StoppingRule {
    /// Stop once the missing mass is at most this.
    pub missing_mass: Threshold,
    /// Stop once the self-split distance is at most this.
    pub self_split_distance: Threshold,
    /// Syndromes seen fewer times than this are left out of the distance.
    pub head_threshold: u64,
    /// Ask the rule every this many jobs.
    pub check_interval: NonZeroU64,
}

impl StoppingRule {
    /// Whether `histogram` is stable by this rule.
    pub fn stops(&self, histogram: &Histogram) -> bool {
        // An empty histogram has both statistics at zero, which is absence
        // of evidence rather than stability.
        !histogram.is_empty()
            && histogram.missing_mass() <= self.missing_mass.get()
            && histogram.self_split_distance(self.head_threshold) <= self.self_split_distance.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::histogram::Half;

    fn threshold(value: f64) -> Threshold {
        Threshold::try_from(value).expect("valid threshold")
    }

    fn rule(missing_mass: f64, self_split_distance: f64) -> StoppingRule {
        StoppingRule {
            missing_mass: threshold(missing_mass),
            self_split_distance: threshold(self_split_distance),
            head_threshold: 2,
            check_interval: NonZeroU64::new(1).expect("nonzero"),
        }
    }

    #[test]
    fn thresholds_are_probabilities() {
        assert_eq!(threshold(0.0).get(), 0.0);
        assert_eq!(threshold(1.0).get(), 1.0);
        assert_eq!(Threshold::try_from(-0.5), Err(InvalidThreshold(-0.5)));
        assert_eq!(Threshold::try_from(1.5), Err(InvalidThreshold(1.5)));
        assert!(Threshold::try_from(f64::NAN).is_err());
        assert!(Threshold::try_from(f64::INFINITY).is_err());
    }

    #[test]
    fn empty_histogram_never_stops() {
        assert!(!rule(1.0, 1.0).stops(&Histogram::new()));
    }

    #[test]
    fn both_statistics_must_pass() {
        // Half A: syndrome 0 three times, syndrome 1 once. Half B: syndrome 0
        // twice, syndrome 1 twice. Missing mass 0, distance 0.25 over the
        // head, as in the histogram tests.
        let mut histogram = Histogram::new();
        for _ in 0..3 {
            histogram.record(0, Half::A);
        }
        histogram.record(1, Half::A);
        for _ in 0..2 {
            histogram.record(0, Half::B);
            histogram.record(1, Half::B);
        }
        assert!(rule(0.0, 0.25).stops(&histogram));
        assert!(!rule(0.0, 0.2).stops(&histogram));

        // A singleton makes the missing mass 1/9.
        histogram.record(2, Half::A);
        assert!(rule(0.2, 1.0).stops(&histogram));
        assert!(!rule(0.1, 1.0).stops(&histogram));
    }
}
