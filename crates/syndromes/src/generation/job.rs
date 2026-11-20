//! One evaluation, identified by its index.

use crate::{
    Syndrome,
    computer::{Computer, EvaluationError},
    fault_selection::FaultSelection,
    input_source::InputSource,
};
use rand::{SeedableRng, rngs::ChaCha8Rng};

/// One numbered evaluation of a run.
///
/// Everything about a job follows from the run seed and its index, so any
/// worker can evaluate any job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Job {
    /// The position of the job in the run.
    pub index: u64,
}

impl Job {
    /// The RNG every draw for this job comes from.
    fn rng(self, seed: u64) -> ChaCha8Rng {
        // The seed is the cipher key and the index the stream, so the jobs
        // of one run are independent streams of one key.
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        rng.set_stream(self.index);
        rng
    }

    /// Draws the inputs and the fault case, then evaluates them.
    pub fn evaluate(
        self,
        seed: u64,
        source: &InputSource,
        selection: FaultSelection,
        computer: &mut Computer,
    ) -> Result<Syndrome, EvaluationError> {
        let mut rng = self.rng(seed);
        // The draw order is part of what a job index means. Changing it
        // changes every histogram generated from a given seed.
        let triple = source.triple(&mut rng);
        let fault_case = selection.select(computer.fault_case_count(), &mut rng);
        Ok(computer.evaluate(triple, fault_case)?.syndrome())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::test_computers::xor3;

    #[test]
    fn a_job_is_a_function_of_seed_and_index() {
        let (mut computer, _file) = xor3("job");
        let evaluate = |seed, index, computer: &mut Computer| {
            Job { index }
                .evaluate(
                    seed,
                    &InputSource::RandomBits,
                    FaultSelection::Uniform,
                    computer,
                )
                .expect("evaluates")
        };
        // Only the syndrome is observable, and one syndrome says little, so
        // compare sequences: repeating jobs agrees exactly, shifting the
        // index or changing the seed disagrees somewhere.
        let repeated: Vec<Syndrome> = (0..32)
            .map(|index| evaluate(1, index, &mut computer))
            .collect();
        let again: Vec<Syndrome> = (0..32)
            .map(|index| evaluate(1, index, &mut computer))
            .collect();
        assert_eq!(repeated, again);

        let shifted: Vec<Syndrome> = (1..33)
            .map(|index| evaluate(1, index, &mut computer))
            .collect();
        assert_ne!(repeated, shifted);

        let other_seed: Vec<Syndrome> = (0..32)
            .map(|index| evaluate(2, index, &mut computer))
            .collect();
        assert_ne!(repeated, other_seed);
    }
}
