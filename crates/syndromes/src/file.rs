//! The on-disk form of a histogram: one JSON document per run.

use crate::{
    Syndrome,
    generation::{Outcome, StoppingRule},
    histogram::Statistics,
};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

/// One histogram on disk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistogramFile {
    /// The seed the run was generated from.
    pub seed: u64,
    /// The rule that ended the run.
    pub rule: StoppingRule,
    /// The statistics of the histogram at stop.
    pub statistics: Statistics,
    /// `[syndrome, count]` pairs, most frequent first, ties by syndrome.
    pub histogram: Vec<(Syndrome, u64)>,
}

impl HistogramFile {
    /// The file for a finished run.
    pub fn from_outcome(seed: u64, rule: StoppingRule, outcome: &Outcome) -> Self {
        let mut histogram: Vec<_> = outcome.histogram.iter().collect();
        histogram.sort_by_key(|&(syndrome, count)| (std::cmp::Reverse(count), syndrome));
        Self {
            seed,
            rule,
            statistics: outcome.statistics,
            histogram,
        }
    }

    /// Writes the file as JSON.
    pub fn write(&self, writer: impl Write) -> Result<(), serde_json::Error> {
        serde_json::to_writer(writer, self)
    }

    /// Reads a file written by [`HistogramFile::write`].
    pub fn read(reader: impl Read) -> Result<Self, serde_json::Error> {
        serde_json::from_reader(reader)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        generation::Threshold,
        histogram::{Half, Histogram},
    };
    use std::num::NonZeroU64;

    fn rule() -> StoppingRule {
        StoppingRule {
            missing_mass: Threshold::try_from(0.01).expect("valid"),
            self_split_distance: Threshold::try_from(0.1).expect("valid"),
            head_threshold: 2,
            check_interval: NonZeroU64::new(64).expect("nonzero"),
        }
    }

    fn outcome() -> Outcome {
        let mut histogram = Histogram::new();
        for (syndrome, half) in [
            (5, Half::A),
            (5, Half::B),
            (5, Half::A),
            (9, Half::B),
            (9, Half::A),
            (1, Half::B),
            (7, Half::A),
        ] {
            histogram.record(syndrome, half);
        }
        Outcome {
            statistics: histogram.statistics(2),
            histogram,
        }
    }

    #[test]
    fn entries_are_sorted_by_count_then_syndrome() {
        let file = HistogramFile::from_outcome(3, rule(), &outcome());
        assert_eq!(file.histogram, [(5, 3), (9, 2), (1, 1), (7, 1)]);
        assert_eq!(file.statistics.evaluations, 7);
        assert_eq!(file.statistics.distinct_syndromes, 4);
    }

    #[test]
    fn round_trips_through_json() {
        let file = HistogramFile::from_outcome(3, rule(), &outcome());
        let mut bytes = Vec::new();
        file.write(&mut bytes).expect("writes");
        let text = String::from_utf8(bytes.clone()).expect("utf-8");
        assert!(text.contains("[5,3]"), "{text}");
        assert!(text.contains("\"seed\":3"), "{text}");
        assert_eq!(HistogramFile::read(bytes.as_slice()).expect("reads"), file);
    }

    #[test]
    fn an_invalid_threshold_is_rejected_on_read() {
        let text = r#"{"seed":0,"rule":{"missing_mass":1.5,"self_split_distance":0.1,"head_threshold":2,"check_interval":64},"statistics":{"evaluations":0,"distinct_syndromes":0,"missing_mass":0.0,"self_split_distance":0.0},"histogram":[]}"#;
        assert!(HistogramFile::read(text.as_bytes()).is_err());
    }
}
