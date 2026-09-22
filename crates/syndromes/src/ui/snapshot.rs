//! A histogram prepared for drawing.

use std::collections::HashMap;
use syndromes::{Syndrome, file::HistogramFile, histogram::Histogram, histogram::Statistics};

/// The number of bits in an output, and so the number of bars in the per-bit
/// view.
// A `usize` is what indexes an array; the cast is const and cannot truncate.
pub const BIT_COUNT: usize = Syndrome::BITS as usize;

/// One histogram, prepared for drawing.
///
/// Everything the views need is computed when the snapshot is built, never
/// while a frame is drawn.
#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot {
    name: String,
    seed: u64,
    entries: Vec<(Syndrome, u64)>,
    bit_flips: [u64; BIT_COUNT],
    corrupted: u64,
    statistics: Statistics,
}

impl Snapshot {
    /// The snapshot of a saved histogram.
    pub fn from_file(name: String, file: HistogramFile) -> Self {
        Self::new(name, file.seed, file.histogram, file.statistics)
    }

    /// The snapshot of a histogram being built.
    ///
    /// Sorting is linearithmic in the number of distinct syndromes, which is
    /// why a snapshot is taken at a fixed rate rather than at every change.
    // Only the tests take this path so far. The view over a running
    // generation is what needs it.
    #[allow(dead_code)]
    pub fn from_histogram(
        name: String,
        seed: u64,
        histogram: &Histogram,
        head_threshold: u64,
    ) -> Self {
        Self::new(
            name,
            seed,
            histogram.iter().collect(),
            histogram.statistics(head_threshold),
        )
    }

    fn new(
        name: String,
        seed: u64,
        mut entries: Vec<(Syndrome, u64)>,
        statistics: Statistics,
    ) -> Self {
        // The same order `HistogramFile::from_outcome` writes. A file is
        // already in it, but the views address entries by rank, so sorting
        // again costs one pass at load and does not trust the file's contents.
        entries.sort_by_key(|&(syndrome, count)| (std::cmp::Reverse(count), syndrome));

        let mut bit_flips = [0; BIT_COUNT];
        let mut corrupted = 0;
        for &(syndrome, count) in &entries {
            if syndrome == 0 {
                continue;
            }
            corrupted += count;
            for (index, flips) in bit_flips.iter_mut().enumerate() {
                if syndrome & (1 << index) != 0 {
                    *flips += count;
                }
            }
        }

        Self {
            name,
            seed,
            entries,
            bit_flips,
            corrupted,
            statistics,
        }
    }

    /// What to call this histogram in the window.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The seed the run derived from.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Every syndrome with its count, most frequent first.
    pub fn entries(&self) -> &[(Syndrome, u64)] {
        &self.entries
    }

    /// How many evaluations flipped each output bit.
    pub fn bit_flips(&self) -> &[u64; BIT_COUNT] {
        &self.bit_flips
    }

    /// The number of evaluations counted.
    pub fn evaluations(&self) -> u64 {
        self.statistics.evaluations
    }

    /// The number of evaluations whose syndrome was nonzero, that is where the
    /// fault changed the output at all.
    pub fn corrupted(&self) -> u64 {
        self.corrupted
    }

    /// The numbers that summarise this histogram.
    pub fn statistics(&self) -> &Statistics {
        &self.statistics
    }

    /// How far this histogram is from `other`, as a total variation distance.
    ///
    /// Zero when the two agree everywhere, one when they share no syndrome.
    /// Zero if either side counted nothing.
    pub fn total_variation_distance(&self, other: &Self) -> f64 {
        let (left, right) = (self.evaluations(), other.evaluations());
        if left == 0 || right == 0 {
            return 0.0;
        }
        let (left, right) = (u128::from(left), u128::from(right));

        // Brought to the common denominator `left * right` and summed as
        // integers, the same way `Histogram::self_split_distance` is, so the
        // number does not depend on the order the entries happen to be in.
        let overflow = "a distance term exceeds u128";
        let term = |mine: u128, theirs: u128| {
            mine.checked_mul(right)
                .expect(overflow)
                .abs_diff(theirs.checked_mul(left).expect(overflow))
        };

        // Draining as we go leaves exactly the syndromes only `other` has.
        let mut theirs: HashMap<Syndrome, u64> = other.entries.iter().copied().collect();
        let mut scaled_sum = 0u128;
        for &(syndrome, count) in &self.entries {
            let other_count = theirs.remove(&syndrome).unwrap_or(0);
            let value = term(u128::from(count), u128::from(other_count));
            scaled_sum = scaled_sum.checked_add(value).expect(overflow);
        }
        for count in theirs.into_values() {
            let value = term(0, u128::from(count));
            scaled_sum = scaled_sum.checked_add(value).expect(overflow);
        }

        let denominator = left
            .checked_mul(right)
            .and_then(|product| product.checked_mul(2))
            .expect(overflow);
        // There is no lossless u128 to f64 conversion. Each cast rounds once,
        // deterministically.
        scaled_sum as f64 / denominator as f64
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use std::num::NonZeroU64;
    use syndromes::{
        generation::{Outcome, StoppingRule, Threshold},
        histogram::Half,
    };

    /// How many times a syndrome has to be seen to count towards the
    /// self-split distance. Arbitrary; the tests here do not depend on it.
    const HEAD_THRESHOLD: u64 = 2;

    /// A snapshot holding each syndrome the given number of times.
    pub fn snapshot(name: &str, entries: &[(Syndrome, u64)]) -> Snapshot {
        Snapshot::from_histogram(name.to_owned(), 0, &histogram(entries), HEAD_THRESHOLD)
    }

    /// The same counts as a histogram, with the evaluations alternating
    /// between the halves the way a real run's do.
    fn histogram(entries: &[(Syndrome, u64)]) -> Histogram {
        let mut histogram = Histogram::new();
        let mut index = 0;
        for &(syndrome, count) in entries {
            for _ in 0..count {
                histogram.record(syndrome, Half::of_job(index));
                index += 1;
            }
        }
        histogram
    }

    fn rule() -> StoppingRule {
        StoppingRule {
            missing_mass: Threshold::try_from(0.01).expect("valid"),
            self_split_distance: Threshold::try_from(0.1).expect("valid"),
            head_threshold: HEAD_THRESHOLD,
            check_interval: NonZeroU64::new(64).expect("nonzero"),
        }
    }

    #[test]
    fn entries_come_out_most_frequent_first() {
        let file = HistogramFile {
            seed: 7,
            rule: rule(),
            statistics: histogram(&[(1, 1), (4, 6), (0, 3)]).statistics(HEAD_THRESHOLD),
            // Deliberately not in the order the file format documents.
            histogram: vec![(1, 1), (4, 6), (0, 3)],
        };
        let snapshot = Snapshot::from_file("file".to_owned(), file);
        assert_eq!(snapshot.entries(), [(4, 6), (0, 3), (1, 1)]);
        assert_eq!(snapshot.seed(), 7);
        assert_eq!(snapshot.name(), "file");
    }

    #[test]
    fn the_two_constructors_agree() {
        let entries = [(4, 6), (0, 3), (1, 1)];
        let histogram = histogram(&entries);
        let outcome = Outcome {
            statistics: histogram.statistics(HEAD_THRESHOLD),
            histogram,
        };
        let from_file = Snapshot::from_file(
            "same".to_owned(),
            HistogramFile::from_outcome(7, rule(), &outcome),
        );
        let from_histogram =
            Snapshot::from_histogram("same".to_owned(), 7, &outcome.histogram, HEAD_THRESHOLD);
        assert_eq!(from_file, from_histogram);
    }

    #[test]
    fn bits_are_counted_per_evaluation() {
        let snapshot = snapshot("bits", &[(0b0101, 6), (0, 3), (0b0100, 1)]);
        let flips = snapshot.bit_flips();
        assert_eq!(flips[0], 6);
        assert_eq!(flips[1], 0);
        assert_eq!(flips[2], 7);
        assert!(flips[3..].iter().all(|&count| count == 0));
        assert_eq!(snapshot.evaluations(), 10);
    }

    #[test]
    fn corruption_counts_every_nonzero_syndrome() {
        let some = snapshot("some", &[(0b001, 6), (0, 3), (0b100, 1)]);
        assert_eq!(some.corrupted(), 7);
        // No zero syndrome at all, so every evaluation was corrupted.
        let every = snapshot("every", &[(0b001, 6), (0b100, 4)]);
        assert_eq!(every.corrupted(), every.evaluations());
        let none = snapshot("none", &[(0, 10)]);
        assert_eq!(none.corrupted(), 0);
    }

    #[test]
    fn distance_is_zero_against_itself_and_symmetric() {
        let left = snapshot("left", &[(4, 6), (0, 3), (1, 1)]);
        assert_eq!(left.total_variation_distance(&left), 0.0);

        // The same distribution at twice the evaluations is still the same
        // distribution.
        let doubled = snapshot("doubled", &[(4, 12), (0, 6), (1, 2)]);
        assert_eq!(left.total_variation_distance(&doubled), 0.0);

        let right = snapshot("right", &[(4, 5), (0, 4), (1, 1)]);
        assert!(
            (left.total_variation_distance(&right) - right.total_variation_distance(&left)).abs()
                < 1e-12
        );
    }

    #[test]
    fn distance_is_hand_computed() {
        // 6/10 against 5/10, 3/10 against 4/10, 1/10 against 1/10.
        let left = snapshot("left", &[(4, 6), (0, 3), (1, 1)]);
        let right = snapshot("right", &[(4, 5), (0, 4), (1, 1)]);
        let expected = (0.1 + 0.1 + 0.0) / 2.0;
        assert!((left.total_variation_distance(&right) - expected).abs() < 1e-12);

        // Nothing in common is the largest the distance gets.
        let disjoint = snapshot("disjoint", &[(9, 10)]);
        assert!((left.total_variation_distance(&disjoint) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn distance_against_nothing_is_zero() {
        let left = snapshot("left", &[(4, 6)]);
        let empty =
            Snapshot::from_histogram("empty".to_owned(), 0, &Histogram::new(), HEAD_THRESHOLD);
        assert_eq!(left.total_variation_distance(&empty), 0.0);
        assert_eq!(empty.total_variation_distance(&left), 0.0);
    }
}
