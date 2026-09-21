//! Syndrome counts and the statistics that decide when to stop.

use crate::Syndrome;
use std::collections::HashMap;

/// One of the two interleaved halves of a run.
///
/// Each half is an independent estimate of the distribution; comparing them
/// measures how stable the estimate is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Half {
    /// Even-numbered jobs.
    A,
    /// Odd-numbered jobs.
    B,
}

impl Half {
    /// The half that job `index` belongs to.
    pub fn of_job(index: u64) -> Self {
        if index.is_multiple_of(2) {
            Self::A
        } else {
            Self::B
        }
    }

    fn slot(self) -> usize {
        match self {
            Self::A => 0,
            Self::B => 1,
        }
    }
}

/// Syndrome counts, kept per half.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Histogram {
    counts: HashMap<Syndrome, [u64; 2]>,
    evaluations: [u64; 2],
}

impl Histogram {
    /// An empty histogram.
    pub fn new() -> Self {
        Self::default()
    }

    /// Counts one evaluation.
    pub fn record(&mut self, syndrome: Syndrome, half: Half) {
        self.counts.entry(syndrome).or_default()[half.slot()] += 1;
        self.evaluations[half.slot()] += 1;
    }

    /// Adds every count of `other` to this histogram.
    pub fn merge(&mut self, other: &Histogram) {
        for (&syndrome, counts) in &other.counts {
            let entry = self.counts.entry(syndrome).or_default();
            entry[0] += counts[0];
            entry[1] += counts[1];
        }
        self.evaluations[0] += other.evaluations[0];
        self.evaluations[1] += other.evaluations[1];
    }

    /// The number of evaluations counted.
    pub fn evaluations(&self) -> u64 {
        self.evaluations[0] + self.evaluations[1]
    }

    /// The number of distinct syndromes seen.
    pub fn len(&self) -> usize {
        self.counts.len()
    }

    /// Whether no evaluation has been counted.
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// How often a syndrome was seen, over both halves.
    pub fn count(&self, syndrome: Syndrome) -> u64 {
        self.counts
            .get(&syndrome)
            .map_or(0, |counts| counts[0] + counts[1])
    }

    /// Every syndrome with its count over both halves, in no particular order.
    pub fn iter(&self) -> impl Iterator<Item = (Syndrome, u64)> + '_ {
        self.counts
            .iter()
            .map(|(&syndrome, counts)| (syndrome, counts[0] + counts[1]))
    }

    /// The estimated probability that the next evaluation produces a syndrome
    /// not seen so far.
    ///
    /// This is the Good-Turing estimate: the fraction of evaluations whose
    /// syndrome was seen exactly once. Zero for an empty histogram.
    pub fn missing_mass(&self) -> f64 {
        let singleton_count = self
            .counts
            .values()
            .filter(|counts| counts[0] + counts[1] == 1)
            .count();
        ratio(
            u64::try_from(singleton_count).expect("usize fits in u64"),
            self.evaluations(),
        )
    }

    /// Half of the L1 distance between the two halves' syndrome fractions,
    /// over syndromes seen at least `head_threshold` times in total.
    ///
    /// With no threshold this is the total variation distance. The threshold
    /// excludes the rare syndromes, since each of those lands in one half
    /// only and would contribute a fixed amount that says nothing about how
    /// stable the bulk of the distribution is. Zero if either half is empty.
    pub fn self_split_distance(&self, head_threshold: u64) -> f64 {
        let [evaluations_a, evaluations_b] = self.evaluations;
        if evaluations_a == 0 || evaluations_b == 0 {
            return 0.0;
        }

        // Total variation distance is defined as the largest disagreement
        // about the probability of any set of syndromes:
        //
        //   TV(A, B) = max over sets S of |p_A(S) - p_B(S)|
        //
        // For two full distributions this equals `sum |p_A(x) - p_B(x)| / 2`:
        // the maximising S is the set where p_A > p_B, and because both sides
        // sum to 1, the excess of A on that set equals the excess of B on its
        // complement, so the plain sum counts every displaced unit of mass
        // twice.
        //
        // Restricting to the head breaks that assumption: the head fractions
        // of each half no longer sum to 1, so the two excesses need not be
        // equal and the halved sum is not exactly a max over sets any more.
        // It is still the head's L1 distance on the `[0, 1]` scale, which is
        // what the stopping threshold is calibrated against.
        //
        // The terms are brought to the common denominator `N_A * N_B` and
        // summed as integers, so the result does not depend on the order the
        // map is iterated in. A float sum would, in its last bits, and the
        // stopping decision must be a function of the histogram alone.
        let [evaluations_a, evaluations_b] = [evaluations_a, evaluations_b].map(u128::from);
        let overflow = "a self-split term exceeds u128";
        let scaled_sum = self
            .counts
            .values()
            .filter(|counts| counts[0] + counts[1] >= head_threshold)
            .map(|counts| {
                let [count_a, count_b] = counts.map(u128::from);
                count_a
                    .checked_mul(evaluations_b)
                    .expect(overflow)
                    .abs_diff(count_b.checked_mul(evaluations_a).expect(overflow))
            })
            .try_fold(0u128, u128::checked_add)
            .expect(overflow);
        let denominator = evaluations_a
            .checked_mul(evaluations_b)
            .and_then(|product| product.checked_mul(2))
            .expect(overflow);
        // There is no lossless u128 to f64 conversion. Each cast rounds once,
        // deterministically, so the result is a function of the counts.
        scaled_sum as f64 / denominator as f64
    }
}

/// `numerator / denominator`, or zero when the denominator is zero.
fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        return 0.0;
    }
    // There is no lossless u64 to f64 conversion. Counts stay far below 2^53,
    // where the cast is exact.
    numerator as f64 / denominator as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: f64, right: f64) -> bool {
        (left - right).abs() < 1e-12
    }

    #[test]
    fn halves_alternate_by_job() {
        assert_eq!(Half::of_job(0), Half::A);
        assert_eq!(Half::of_job(1), Half::B);
        assert_eq!(Half::of_job(2), Half::A);
        assert_eq!(Half::of_job(3), Half::B);
        assert_eq!(Half::of_job(4), Half::A);
    }

    #[test]
    fn empty_statistics_are_zero() {
        let histogram = Histogram::new();
        assert!(histogram.is_empty());
        assert_eq!(histogram.evaluations(), 0);
        assert_eq!(histogram.missing_mass(), 0.0);
        assert_eq!(histogram.self_split_distance(1), 0.0);
    }

    #[test]
    fn counts_and_missing_mass() {
        let mut histogram = Histogram::new();
        for (syndrome, half) in [
            (0, Half::A),
            (0, Half::B),
            (0, Half::A),
            (1, Half::B),
            (2, Half::A),
        ] {
            histogram.record(syndrome, half);
        }
        assert_eq!(histogram.evaluations(), 5);
        assert_eq!(histogram.len(), 3);
        assert_eq!(histogram.count(0), 3);
        assert_eq!(histogram.count(1), 1);
        assert_eq!(histogram.count(9), 0);
        // Two singletons out of five evaluations.
        assert!(close(histogram.missing_mass(), 0.4));
    }

    #[test]
    fn self_split_distance_is_hand_computed() {
        let mut histogram = Histogram::new();
        // Half A: syndrome 0 three times, syndrome 1 once. Half B: syndrome 0
        // twice, syndrome 1 twice. Both halves hold four evaluations.
        for _ in 0..3 {
            histogram.record(0, Half::A);
        }
        histogram.record(1, Half::A);
        for _ in 0..2 {
            histogram.record(0, Half::B);
            histogram.record(1, Half::B);
        }
        // |3/4 - 2/4| + |1/4 - 2/4| = 1/2, halved.
        assert!(close(histogram.self_split_distance(1), 0.25));
    }

    #[test]
    fn head_threshold_excludes_rare_syndromes() {
        let mut histogram = Histogram::new();
        for _ in 0..10 {
            histogram.record(0, Half::A);
            histogram.record(0, Half::B);
        }
        // A singleton in one half only.
        histogram.record(7, Half::A);

        let all = histogram.self_split_distance(1);
        let head = histogram.self_split_distance(2);
        assert!(all > 0.0);
        // Only syndrome 0 remains: |10/11 - 10/10| / 2.
        assert!(close(head, (10.0 / 11.0 - 1.0f64).abs() / 2.0));
        assert!(head < all);
    }

    #[test]
    fn merge_adds_per_half() {
        let mut left = Histogram::new();
        left.record(0, Half::A);
        left.record(1, Half::B);

        let mut right = Histogram::new();
        right.record(0, Half::B);
        right.record(2, Half::A);

        let mut merged = left.clone();
        merged.merge(&right);

        let mut expected = Histogram::new();
        expected.record(0, Half::A);
        expected.record(1, Half::B);
        expected.record(0, Half::B);
        expected.record(2, Half::A);

        assert_eq!(merged, expected);
        assert_eq!(merged.evaluations(), 4);
    }
}
