//! Generating a histogram: numbered jobs, workers, and an aggregator.
//!
//! Job `k` draws its inputs and fault case from an RNG seeded by the run
//! seed and `k`, and the stopping rule only ever sees complete prefixes of
//! the job sequence. The histogram and the point at which a run stops
//! therefore depend on the seed alone.

mod aggregator;
mod job;
mod stopping;
mod worker;

use crate::{
    computer::{Computer, EvaluationError},
    fault_selection::FaultSelection,
    input_source::InputSource,
};
use aggregator::{Aggregator, Progress};
use std::{
    num::{NonZeroU64, NonZeroUsize},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    thread,
};

pub use aggregator::Outcome;
pub use stopping::{InvalidThreshold, StoppingRule, Threshold};

/// Everything that defines a run.
///
/// The seed, source, selection and rule define the result. The number of
/// workers and the batch size only affect how fast it is reached and how
/// many evaluations past the stopping point are wasted.
#[derive(Debug, Clone, PartialEq)]
pub struct Configuration {
    /// The seed every job's RNG derives from.
    pub seed: u64,
    /// Where the inputs come from.
    pub source: InputSource,
    /// How the fault case of each job is chosen.
    pub selection: FaultSelection,
    /// When the run ends.
    pub rule: StoppingRule,
    /// The number of threads evaluating jobs.
    pub workers: NonZeroUsize,
    /// The number of consecutive jobs a worker claims at a time.
    pub batch_size: NonZeroU64,
}

/// Why a run produced no histogram.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GenerationError {
    /// The computer has no gates to fault.
    #[error("the computer has no fault cases")]
    NoFaultCases,
    /// An evaluation produced an undefined output bit.
    #[error(transparent)]
    Evaluation(#[from] EvaluationError),
}

impl Configuration {
    /// Runs the generation to its stopping point.
    ///
    /// Each worker gets a clone of `computer`. The calling thread aggregates.
    pub fn run(&self, computer: &Computer) -> Result<Outcome, GenerationError> {
        if computer.fault_case_count() == 0 {
            return Err(GenerationError::NoFaultCases);
        }

        let shared = worker::Shared {
            configuration: self,
            next_batch: AtomicU64::new(0),
            stop: AtomicBool::new(false),
        };
        let (sender, receiver) = mpsc::channel();
        let outcome = thread::scope(|scope| {
            for _ in 0..self.workers.get() {
                let shared = &shared;
                let computer = computer.clone();
                let sender = sender.clone();
                scope.spawn(move || worker::work(shared, computer, sender));
            }
            // Only the workers hold senders now, so the channel closes when
            // they are all gone.
            drop(sender);

            let mut aggregator = Aggregator::new(self.batch_size, self.rule);
            let outcome = loop {
                match receiver.recv() {
                    Ok(batch) => match aggregator.receive(batch) {
                        Progress::Continue(next) => aggregator = next,
                        Progress::Stopped(outcome) => break Some(Ok(outcome)),
                        Progress::Failed(error) => {
                            break Some(Err(GenerationError::Evaluation(error)));
                        }
                    },
                    // Every worker returned without an outcome or an error,
                    // which only a panic causes; the scope re-raises it.
                    Err(mpsc::RecvError) => break None,
                }
            };
            shared.stop.store(true, Ordering::Relaxed);
            outcome
        });
        outcome.expect("workers only exit after an outcome or an error")
    }
}

#[cfg(test)]
mod test_computers {
    use crate::{
        computer::{Computer, Fused, FusedDescription},
        netlist::test_netlists::{TemporaryFile, bitwise_xor3, write_temporary},
    };

    /// A fused computer over the three-input XOR netlist: 128 fault cases,
    /// every syndrome a single bit or zero.
    pub fn xor3(name: &str) -> (Computer, TemporaryFile) {
        let file = write_temporary(name, &bitwise_xor3());
        let computer = Fused::load(&FusedDescription {
            path: file.0.clone(),
            activation: "a".to_owned(),
            weight: "b".to_owned(),
            partial_sum: "c".to_owned(),
            output: "z".to_owned(),
            constants: Vec::new(),
        })
        .expect("test netlist loads");
        (Computer::Fused(computer), file)
    }
}

#[cfg(test)]
mod tests {
    use super::{test_computers::xor3, *};
    use crate::{
        computer::{AdderDescription, MultiplierDescription, Separate, SeparateDescription},
        netlist::test_netlists::{write_temporary, xor_with_undriven_bit},
    };

    fn configuration(workers: usize, batch_size: u64) -> Configuration {
        Configuration {
            seed: 7,
            source: InputSource::RandomBits,
            selection: FaultSelection::Uniform,
            rule: StoppingRule {
                missing_mass: Threshold::try_from(0.01).expect("valid"),
                self_split_distance: Threshold::try_from(0.1).expect("valid"),
                head_threshold: 2,
                check_interval: NonZeroU64::new(64).expect("nonzero"),
            },
            workers: NonZeroUsize::new(workers).expect("nonzero"),
            batch_size: NonZeroU64::new(batch_size).expect("nonzero"),
        }
    }

    #[test]
    fn outcome_does_not_depend_on_workers_or_batch_size() {
        let (computer, _file) = xor3("run");
        let reference = configuration(1, 64).run(&computer).expect("runs");
        assert!(reference.histogram.evaluations() > 64);
        assert!(reference.histogram.evaluations().is_multiple_of(64));
        assert!(reference.missing_mass <= 0.01);
        assert!(reference.self_split_distance <= 0.1);

        for (workers, batch_size) in [(4, 64), (1, 100), (3, 7)] {
            let outcome = configuration(workers, batch_size)
                .run(&computer)
                .expect("runs");
            assert_eq!(
                outcome, reference,
                "{workers} workers, batches of {batch_size}"
            );
        }
    }

    #[test]
    fn undefined_output_ends_the_run() {
        let file = write_temporary("run-undriven", &xor_with_undriven_bit());
        let computer = Separate::load(&SeparateDescription {
            multiplier: MultiplierDescription {
                path: file.0.clone(),
                activation: "a".to_owned(),
                weight: "b".to_owned(),
                product: "z".to_owned(),
                constants: Vec::new(),
            },
            adder: AdderDescription {
                path: file.0.clone(),
                product: "a".to_owned(),
                partial_sum: "b".to_owned(),
                sum: "z".to_owned(),
                constants: Vec::new(),
            },
        })
        .expect("loads");
        let error = configuration(2, 8)
            .run(&Computer::Separate(computer))
            .expect_err("fails");
        assert!(matches!(error, GenerationError::Evaluation(_)));
    }
}
