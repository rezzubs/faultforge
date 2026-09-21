//! The aggregator: commits batches in job order and asks the stopping rule.

use super::{stopping::StoppingRule, worker::Batch};
use crate::{
    Syndrome,
    computer::EvaluationError,
    histogram::{Half, Histogram},
};
use std::{collections::BTreeMap, num::NonZeroU64};

/// A finished run.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    /// The syndrome counts over every committed job.
    pub histogram: Histogram,
    /// The missing mass at stop.
    pub missing_mass: f64,
    /// The self-split distance at stop.
    pub self_split_distance: f64,
}

/// What receiving a batch led to.
pub enum Progress {
    /// The run goes on.
    Continue(Aggregator),
    /// The stopping rule fired.
    Stopped(Outcome),
    /// A job before the stopping point failed.
    Failed(EvaluationError),
}

/// Records batches into one histogram, strictly in job order.
///
/// Batches may arrive in any order; each is held until every batch before
/// it has been recorded. The stopping rule sees only complete prefixes of
/// the job sequence, so the point at which a run stops is a function of
/// the seed rather than of scheduling.
pub struct Aggregator {
    histogram: Histogram,
    /// Batches that arrived before the ones preceding them.
    buffer: BTreeMap<u64, Result<Vec<Syndrome>, EvaluationError>>,
    next_batch: u64,
    batch_size: NonZeroU64,
    rule: StoppingRule,
}

impl Aggregator {
    /// An aggregator for batches of `batch_size` jobs.
    pub fn new(batch_size: NonZeroU64, rule: StoppingRule) -> Self {
        Self {
            histogram: Histogram::new(),
            buffer: BTreeMap::new(),
            next_batch: 0,
            batch_size,
            rule,
        }
    }

    /// Takes a batch in and records what is contiguous.
    ///
    /// Once the rule has fired, jobs past the stopping point are not
    /// recorded, including the rest of the batch that reached it. A failed
    /// batch is likewise reported only when its turn comes, so an error
    /// past the stopping point never surfaces.
    pub fn receive(mut self, batch: Batch) -> Progress {
        self.buffer.insert(batch.index, batch.result);
        while let Some(result) = self.buffer.remove(&self.next_batch) {
            let syndromes = match result {
                Ok(syndromes) => syndromes,
                Err(error) => return Progress::Failed(error),
            };
            let first_job = Batch::first_job_index(self.next_batch, self.batch_size);
            for (offset, syndrome) in syndromes.into_iter().enumerate() {
                let job = first_job + u64::try_from(offset).expect("a batch fits in u64");
                self.histogram.record(syndrome, Half::of_job(job));
                // Jobs `0..=job` are recorded now, which is `job + 1` of them.
                let committed_jobs = job + 1;
                if committed_jobs.is_multiple_of(self.rule.check_interval.get())
                    && self.rule.stops(&self.histogram)
                {
                    return Progress::Stopped(self.into_outcome());
                }
            }
            self.next_batch += 1;
        }
        Progress::Continue(self)
    }

    fn into_outcome(self) -> Outcome {
        Outcome {
            missing_mass: self.histogram.missing_mass(),
            self_split_distance: self.histogram.self_split_distance(self.rule.head_threshold),
            histogram: self.histogram,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Triple, bit::UnknownBitError, generation::stopping::Threshold};

    fn nonzero(value: u64) -> NonZeroU64 {
        NonZeroU64::new(value).expect("nonzero")
    }

    /// A rule whose thresholds are met by any non-empty histogram.
    fn always(check_interval: u64) -> StoppingRule {
        StoppingRule {
            missing_mass: Threshold::try_from(1.0).expect("valid"),
            self_split_distance: Threshold::try_from(1.0).expect("valid"),
            head_threshold: 1,
            check_interval: nonzero(check_interval),
        }
    }

    /// A rule that a histogram of distinct syndromes never meets.
    fn never(check_interval: u64) -> StoppingRule {
        StoppingRule {
            missing_mass: Threshold::try_from(0.0).expect("valid"),
            ..always(check_interval)
        }
    }

    /// Batch `index` of size `size` with a distinct syndrome per job.
    fn distinct(index: u64, size: u64) -> Batch {
        let first_job = Batch::first_job_index(index, nonzero(size));
        Batch {
            index,
            result: Ok((first_job..first_job + size)
                .map(|job| u32::try_from(job).expect("small"))
                .collect()),
        }
    }

    /// Batch `index` that failed to evaluate.
    fn failed(index: u64) -> Batch {
        Batch {
            index,
            result: Err(EvaluationError {
                triple: Triple {
                    activation: 0.0,
                    weight: 0.0,
                    partial_sum: 0.0,
                },
                fault_case: None,
                source: UnknownBitError { index: 0 },
            }),
        }
    }

    fn continues(progress: Progress) -> Aggregator {
        match progress {
            Progress::Continue(aggregator) => aggregator,
            Progress::Stopped(_) => panic!("stopped"),
            Progress::Failed(error) => panic!("failed: {error}"),
        }
    }

    fn stopped(progress: Progress) -> Outcome {
        match progress {
            Progress::Continue(_) => panic!("continued"),
            Progress::Stopped(outcome) => outcome,
            Progress::Failed(error) => panic!("failed: {error}"),
        }
    }

    /// Feeds `batches` to an aggregator that never stops and returns what
    /// it recorded.
    fn record(batches: impl IntoIterator<Item = Batch>) -> Outcome {
        let mut aggregator = Aggregator::new(nonzero(4), never(2));
        for batch in batches {
            aggregator = continues(aggregator.receive(batch));
        }
        aggregator.into_outcome()
    }

    #[test]
    fn batches_are_recorded_in_job_order() {
        let ordered = record([0, 1, 2].map(|index| distinct(index, 4)));
        let shuffled = record([2, 0, 1].map(|index| distinct(index, 4)));
        assert_eq!(ordered, shuffled);
        assert_eq!(ordered.histogram.evaluations(), 12);

        // Even jobs in half A, odd jobs in half B.
        let mut expected = Histogram::new();
        for job in 0..12 {
            expected.record(job, Half::of_job(u64::from(job)));
        }
        assert_eq!(ordered.histogram, expected);
    }

    #[test]
    fn early_batches_wait_for_their_predecessors() {
        let aggregator = Aggregator::new(nonzero(4), always(4));
        // Batch 1 alone cannot be recorded, so the rule is not asked.
        let aggregator = continues(aggregator.receive(distinct(1, 4)));
        // Batch 0 completes the prefix and the rule fires at job 4.
        let outcome = stopped(aggregator.receive(distinct(0, 4)));
        assert_eq!(outcome.histogram.evaluations(), 4);
    }

    #[test]
    fn stops_at_a_checkpoint_inside_a_batch() {
        let aggregator = Aggregator::new(nonzero(8), always(3));
        let outcome = stopped(aggregator.receive(distinct(0, 8)));
        assert_eq!(outcome.histogram.evaluations(), 3);
        assert_eq!(outcome.missing_mass, 1.0);
    }

    #[test]
    fn a_failure_waits_for_its_turn() {
        let aggregator = Aggregator::new(nonzero(4), always(4));
        let aggregator = continues(aggregator.receive(failed(1)));
        // The rule fires at job 4, before batch 1 is reached.
        let outcome = stopped(aggregator.receive(distinct(0, 4)));
        assert_eq!(outcome.histogram.evaluations(), 4);
    }

    #[test]
    fn a_failure_before_the_stopping_point_is_reported() {
        let aggregator = Aggregator::new(nonzero(4), always(4));
        let aggregator = continues(aggregator.receive(distinct(1, 4)));
        assert!(matches!(aggregator.receive(failed(0)), Progress::Failed(_)));
    }
}
