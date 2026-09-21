//! A worker: claims batches of jobs and evaluates them.

use super::{Configuration, job::Job};
use crate::{
    Syndrome,
    computer::{Computer, EvaluationError},
};
use std::{
    num::NonZeroU64,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
};

/// The result of one batch of consecutive jobs.
#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    /// The position of the batch in the run.
    pub index: u64,
    /// One syndrome per job, in job order, or the first evaluation error.
    pub result: Result<Vec<Syndrome>, EvaluationError>,
}

impl Batch {
    /// The index of the first job of batch `index`.
    pub fn first_job_index(index: u64, batch_size: NonZeroU64) -> u64 {
        index * batch_size.get()
    }
}

/// What every worker of a run sees.
pub struct Shared<'a> {
    /// The run being generated.
    pub configuration: &'a Configuration,
    /// The next unclaimed batch.
    pub next_batch: AtomicU64,
    /// Set once the aggregator has all it needs.
    pub stop: AtomicBool,
}

/// Claims and evaluates batches until told to stop.
///
/// A failed batch is sent with its error and ends the worker. The worker
/// also ends when the receiver is gone.
pub fn work(shared: &Shared<'_>, mut computer: Computer, sender: mpsc::Sender<Batch>) {
    let configuration = shared.configuration;
    let batch_size = configuration.batch_size;
    // Relaxed suffices: the counter only has to hand out each index once,
    // and the stop flag is a hint that may be seen a batch late.
    while !shared.stop.load(Ordering::Relaxed) {
        let index = shared.next_batch.fetch_add(1, Ordering::Relaxed);
        let first_job_index = Batch::first_job_index(index, batch_size);
        let result: Result<Vec<Syndrome>, EvaluationError> = (first_job_index
            ..first_job_index + batch_size.get())
            .map(|index| {
                Job { index }.evaluate(
                    configuration.seed,
                    &configuration.source,
                    configuration.selection,
                    &mut computer,
                )
            })
            .collect();
        let failed = result.is_err();
        if sender.send(Batch { index, result }).is_err() || failed {
            return;
        }
    }
}
