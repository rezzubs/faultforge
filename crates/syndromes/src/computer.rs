//! The computer seam: something that turns a triple into an output word.

use crate::{
    Syndrome, Triple,
    bit::UnknownBitError,
    netlist::{FaultCase, Netlist},
};
use std::path::PathBuf;

pub use crate::netlist::{ConstantAssignment, LoadError};

/// The outputs of one evaluation, as bit patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Evaluation {
    /// The output with no fault applied.
    pub correct: u32,
    /// The output with the fault case applied.
    pub faulty: u32,
}

impl Evaluation {
    /// The bits that the fault changed.
    pub fn syndrome(&self) -> Syndrome {
        self.correct ^ self.faulty
    }
}

/// An output bit had no defined value, which means the netlist is broken.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub struct EvaluationError {
    /// The inputs that were being evaluated.
    pub triple: Triple,
    /// The fault case that was applied, or `None` for the fault-free pass.
    pub fault_case: Option<usize>,
    /// The offending bit.
    #[source]
    pub source: UnknownBitError,
}

impl std::fmt::Display for EvaluationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "output bit {} is undefined for inputs {:?} ",
            self.source.index, self.triple
        )?;
        match self.fault_case {
            Some(fault_case) => write!(formatter, "under fault case {fault_case}"),
            None => write!(formatter, "with no fault applied"),
        }
    }
}

/// Computes `activation * weight + partial_sum` with an optional stuck-at
/// fault.
#[derive(Debug, Clone)]
#[expect(
    clippy::large_enum_variant,
    reason = "one computer exists per worker thread and is never moved in bulk"
)]
pub enum Computer {
    /// One netlist computing the whole multiply-add.
    Fused(Fused),
    /// A multiplier netlist feeding an adder netlist.
    Separate(Separate),
}

impl Computer {
    /// The number of fault cases. Valid cases are `0..count`.
    pub fn fault_case_count(&self) -> usize {
        match self {
            Self::Fused(fused) => fused.fault_case_count(),
            Self::Separate(separate) => separate.fault_case_count(),
        }
    }

    /// Computes the output with no fault and with `fault_case` applied.
    pub fn evaluate(
        &mut self,
        triple: Triple,
        fault_case: usize,
    ) -> Result<Evaluation, EvaluationError> {
        match self {
            Self::Fused(fused) => fused.evaluate(triple, fault_case),
            Self::Separate(separate) => separate.evaluate(triple, fault_case),
        }
    }
}

/// The netlist and bus names of a fused multiply-add.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusedDescription {
    /// The Verilog source file.
    pub path: PathBuf,
    /// The input bus carrying the activation.
    pub activation: String,
    /// The input bus carrying the weight.
    pub weight: String,
    /// The input bus carrying the partial sum.
    pub partial_sum: String,
    /// The output bus carrying the result.
    pub output: String,
    /// Input buses held at a fixed value.
    pub constants: Vec<ConstantAssignment>,
}

/// A single netlist computing the whole multiply-add.
#[derive(Debug, Clone)]
pub struct Fused {
    netlist: Netlist,
    activation: String,
    weight: String,
    partial_sum: String,
}

impl Fused {
    /// Loads the netlist.
    pub fn load(description: &FusedDescription) -> Result<Self, LoadError> {
        let netlist = Netlist::load(
            &description.path,
            &[
                &description.activation,
                &description.weight,
                &description.partial_sum,
            ],
            &description.output,
            &description.constants,
        )?;
        Ok(Self {
            netlist,
            activation: description.activation.clone(),
            weight: description.weight.clone(),
            partial_sum: description.partial_sum.clone(),
        })
    }

    /// The number of fault cases. Valid cases are `0..count`.
    pub fn fault_case_count(&self) -> usize {
        self.netlist.fault_case_count()
    }

    /// Computes the output with no fault and with `fault_case` applied.
    pub fn evaluate(
        &mut self,
        triple: Triple,
        fault_case: usize,
    ) -> Result<Evaluation, EvaluationError> {
        // Fault-free first: writing inputs re-propagates the whole netlist,
        // switching the fault afterwards only re-propagates its cone.
        self.netlist.clear_fault();
        self.netlist
            .write_input(&self.activation, triple.activation.to_bits());
        self.netlist
            .write_input(&self.weight, triple.weight.to_bits());
        self.netlist
            .write_input(&self.partial_sum, triple.partial_sum.to_bits());
        self.netlist.settle();
        let correct = self
            .netlist
            .read_output()
            .map_err(|source| EvaluationError {
                triple,
                fault_case: None,
                source,
            })?;

        self.netlist.apply_fault(FaultCase::from_index(fault_case));
        self.netlist.settle();
        let faulty = self
            .netlist
            .read_output()
            .map_err(|source| EvaluationError {
                triple,
                fault_case: Some(fault_case),
                source,
            })?;

        Ok(Evaluation { correct, faulty })
    }
}

/// The netlist and bus names of a multiplier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiplierDescription {
    /// The Verilog source file.
    pub path: PathBuf,
    /// The input bus carrying the activation.
    pub activation: String,
    /// The input bus carrying the weight.
    pub weight: String,
    /// The output bus carrying the product.
    pub product: String,
    /// Input buses held at a fixed value.
    pub constants: Vec<ConstantAssignment>,
}

/// The netlist and bus names of an adder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdderDescription {
    /// The Verilog source file.
    pub path: PathBuf,
    /// The input bus carrying the product.
    pub product: String,
    /// The input bus carrying the partial sum.
    pub partial_sum: String,
    /// The output bus carrying the sum.
    pub sum: String,
    /// Input buses held at a fixed value.
    pub constants: Vec<ConstantAssignment>,
}

/// The two netlists of a separate multiply-add.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeparateDescription {
    /// The multiplier.
    pub multiplier: MultiplierDescription,
    /// The adder.
    pub adder: AdderDescription,
}

/// A multiplier netlist feeding an adder netlist.
///
/// Fault cases below the multiplier's count belong to the multiplier, the
/// rest to the adder.
#[derive(Debug, Clone)]
pub struct Separate {
    multiplier: Netlist,
    activation: String,
    weight: String,
    adder: Netlist,
    product: String,
    partial_sum: String,
}

impl Separate {
    /// Loads both netlists.
    pub fn load(description: &SeparateDescription) -> Result<Self, LoadError> {
        let multiplier = &description.multiplier;
        let adder = &description.adder;
        Ok(Self {
            multiplier: Netlist::load(
                &multiplier.path,
                &[&multiplier.activation, &multiplier.weight],
                &multiplier.product,
                &multiplier.constants,
            )?,
            activation: multiplier.activation.clone(),
            weight: multiplier.weight.clone(),
            adder: Netlist::load(
                &adder.path,
                &[&adder.product, &adder.partial_sum],
                &adder.sum,
                &adder.constants,
            )?,
            product: adder.product.clone(),
            partial_sum: adder.partial_sum.clone(),
        })
    }

    /// The number of fault cases. Valid cases are `0..count`.
    pub fn fault_case_count(&self) -> usize {
        self.multiplier.fault_case_count() + self.adder.fault_case_count()
    }

    /// Computes the output with no fault and with `fault_case` applied.
    pub fn evaluate(
        &mut self,
        triple: Triple,
        fault_case: usize,
    ) -> Result<Evaluation, EvaluationError> {
        let fault_free = |source| EvaluationError {
            triple,
            fault_case: None,
            source,
        };
        let faulted = |source| EvaluationError {
            triple,
            fault_case: Some(fault_case),
            source,
        };

        self.multiplier.clear_fault();
        self.adder.clear_fault();

        self.multiplier
            .write_input(&self.activation, triple.activation.to_bits());
        self.multiplier
            .write_input(&self.weight, triple.weight.to_bits());
        self.multiplier.settle();
        let product = self.multiplier.read_output().map_err(fault_free)?;

        self.adder.write_input(&self.product, product);
        self.adder
            .write_input(&self.partial_sum, triple.partial_sum.to_bits());
        self.adder.settle();
        let correct = self.adder.read_output().map_err(fault_free)?;

        let multiplier_cases = self.multiplier.fault_case_count();
        if fault_case < multiplier_cases {
            self.multiplier
                .apply_fault(FaultCase::from_index(fault_case));
            self.multiplier.settle();
            let faulty_product = self.multiplier.read_output().map_err(faulted)?;
            self.adder.write_input(&self.product, faulty_product);
        } else {
            self.adder
                .apply_fault(FaultCase::from_index(fault_case - multiplier_cases));
        }
        self.adder.settle();
        let faulty = self.adder.read_output().map_err(faulted)?;

        Ok(Evaluation { correct, faulty })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::netlist::test_netlists::*;
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use std::path::Path;

    fn random_triple(rng: &mut StdRng) -> Triple {
        Triple {
            activation: f32::from_bits(rng.random()),
            weight: f32::from_bits(rng.random()),
            partial_sum: f32::from_bits(rng.random()),
        }
    }

    /// Both test netlists use `a`, `b` and `z` as their bus names.
    fn separate_description(multiplier: &Path, adder: &Path) -> SeparateDescription {
        SeparateDescription {
            multiplier: MultiplierDescription {
                path: multiplier.to_path_buf(),
                activation: "a".to_owned(),
                weight: "b".to_owned(),
                product: "z".to_owned(),
                constants: Vec::new(),
            },
            adder: AdderDescription {
                path: adder.to_path_buf(),
                product: "a".to_owned(),
                partial_sum: "b".to_owned(),
                sum: "z".to_owned(),
                constants: Vec::new(),
            },
        }
    }

    fn separate() -> (Computer, TemporaryFile, TemporaryFile) {
        let multiplier = write_temporary("separate-mul", &bitwise_xor());
        let adder = write_temporary("separate-add", &bitwise_and());
        let computer = Separate::load(&separate_description(&multiplier.0, &adder.0)).unwrap();
        (Computer::Separate(computer), multiplier, adder)
    }

    fn fused() -> (Computer, TemporaryFile) {
        let file = write_temporary("fused", &bitwise_xor3());
        let computer = Fused::load(&FusedDescription {
            path: file.0.clone(),
            activation: "a".to_owned(),
            weight: "b".to_owned(),
            partial_sum: "c".to_owned(),
            output: "z".to_owned(),
            constants: Vec::new(),
        })
        .unwrap();
        (Computer::Fused(computer), file)
    }

    /// `(a ^ b) & c` and `a ^ b ^ c` on the bit patterns.
    fn expected_separate(triple: Triple) -> u32 {
        (triple.activation.to_bits() ^ triple.weight.to_bits()) & triple.partial_sum.to_bits()
    }

    fn expected_fused(triple: Triple) -> u32 {
        triple.activation.to_bits() ^ triple.weight.to_bits() ^ triple.partial_sum.to_bits()
    }

    fn check_correct_outputs(computer: &mut Computer, expected: fn(Triple) -> u32) {
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..50 {
            let triple = random_triple(&mut rng);
            let fault_case = rng.random_range(0..computer.fault_case_count());
            let evaluation = computer.evaluate(triple, fault_case).unwrap();
            assert_eq!(evaluation.correct, expected(triple));
        }
    }

    /// Every gate in the test netlists drives one output bit, so a syndrome
    /// has at most one bit set, and together the cases reach every bit.
    fn check_single_bit_syndromes(computer: &mut Computer) {
        let mut rng = StdRng::seed_from_u64(2);
        let mut seen = 0;
        for fault_case in 0..computer.fault_case_count() {
            for _ in 0..4 {
                let syndrome = computer
                    .evaluate(random_triple(&mut rng), fault_case)
                    .unwrap()
                    .syndrome();
                assert!(syndrome.count_ones() <= 1, "syndrome {syndrome:#x}");
                seen |= syndrome;
            }
        }
        assert_eq!(seen, u32::MAX);
    }

    #[test]
    fn separate_correct_outputs() {
        let (mut computer, _multiplier, _adder) = separate();
        assert_eq!(computer.fault_case_count(), 128);
        check_correct_outputs(&mut computer, expected_separate);
    }

    #[test]
    fn separate_single_bit_syndromes() {
        let (mut computer, _multiplier, _adder) = separate();
        check_single_bit_syndromes(&mut computer);
    }

    #[test]
    fn separate_adder_cases_reach_the_adder() {
        let (mut computer, _multiplier, _adder) = separate();
        // With the partial sum all zero the AND adder masks every multiplier
        // fault, so only adder cases can produce a syndrome.
        let triple = Triple {
            activation: 0.0,
            weight: 0.0,
            partial_sum: 0.0,
        };
        let multiplier_cases = computer.fault_case_count() / 2;
        for fault_case in 0..multiplier_cases {
            assert_eq!(computer.evaluate(triple, fault_case).unwrap().syndrome(), 0);
        }
        let mut seen = 0;
        for fault_case in multiplier_cases..computer.fault_case_count() {
            seen |= computer.evaluate(triple, fault_case).unwrap().syndrome();
        }
        assert_eq!(seen, u32::MAX);
    }

    #[test]
    fn fused_correct_outputs() {
        let (mut computer, _file) = fused();
        assert_eq!(computer.fault_case_count(), 128);
        check_correct_outputs(&mut computer, expected_fused);
    }

    #[test]
    fn fused_single_bit_syndromes() {
        let (mut computer, _file) = fused();
        check_single_bit_syndromes(&mut computer);
    }

    #[test]
    fn faults_do_not_leak_between_evaluations() {
        let (mut computer, _file) = fused();
        let zero = Triple {
            activation: 0.0,
            weight: 0.0,
            partial_sum: 0.0,
        };
        // Repeating the same inputs means nothing but the fault removal can
        // restore a stuck bit before the fault-free read.
        for fault_case in 0..computer.fault_case_count() {
            let evaluation = computer.evaluate(zero, fault_case).unwrap();
            assert_eq!(evaluation.correct, 0, "case {fault_case}");
        }
    }

    #[test]
    fn undriven_output_names_the_inputs() {
        let file = write_temporary("computer-undriven", &xor_with_undriven_bit());
        let mut computer = Separate::load(&separate_description(&file.0, &file.0)).unwrap();
        let triple = Triple {
            activation: 1.0,
            weight: 2.0,
            partial_sum: 3.0,
        };
        let error = computer.evaluate(triple, 0).unwrap_err();
        assert_eq!(error.triple, triple);
        assert_eq!(error.fault_case, None);
        assert_eq!(error.source, UnknownBitError { index: 5 });
    }
}
