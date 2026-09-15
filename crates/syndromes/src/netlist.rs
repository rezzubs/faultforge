//! A netlist with 32-bit input and output buses and a stuck-at fault.

use crate::bit::{Bit, UnknownBitError, WORD_BITS, signals_from_word, word_from_signals};
use ariadne::sources;
use logic_simulation::{Signal, Simulation, fault::StuckAtFault, netlist::parse_simulation};
use std::{
    fs,
    ops::Range,
    path::{Path, PathBuf},
};

/// A constant value held on an input bus for the lifetime of the netlist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstantAssignment {
    /// The name of the input bus.
    pub bus: String,
    /// The value, least significant bit first. Must match the bus width.
    pub bits: Vec<Bit>,
}

/// Failures of loading a netlist.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// The source file could not be read.
    #[error("could not read {path}")]
    Read {
        /// The file that was requested.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },
    /// The source did not parse. The message holds the rendered reports.
    #[error("could not parse {path}:\n{reports}")]
    Parse {
        /// The file that was parsed.
        path: PathBuf,
        /// The parser's reports, rendered for a terminal.
        reports: String,
    },
    /// A named input bus does not exist in the netlist.
    #[error("{path} has no input bus named `{bus}`")]
    MissingInput {
        /// The file that was loaded.
        path: PathBuf,
        /// The requested bus.
        bus: String,
    },
    /// The named output bus does not exist in the netlist.
    #[error("{path} has no output bus named `{bus}`")]
    MissingOutput {
        /// The file that was loaded.
        path: PathBuf,
        /// The requested bus.
        bus: String,
    },
    /// A bus does not have the expected number of bits.
    #[error("bus `{bus}` in {path} is {actual} bits wide, expected {expected}")]
    Width {
        /// The file that was loaded.
        path: PathBuf,
        /// The bus in question.
        bus: String,
        /// The width required by the description.
        expected: usize,
        /// The width found in the netlist.
        actual: usize,
    },
}

/// One stuck-at fault: a gate output and the value it is stuck at.
///
/// Cases are numbered so that a netlist's cases are `0..fault_case_count`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FaultCase {
    /// The gate output, an index into the simulation's fault targets.
    pub target: usize,
    /// The value the output is stuck at.
    pub stuck_at: StuckAtFault,
}

impl FaultCase {
    /// The case with the given number.
    pub fn from_index(index: usize) -> Self {
        Self {
            target: index / 2,
            stuck_at: if index.is_multiple_of(2) {
                StuckAtFault::Low
            } else {
                StuckAtFault::High
            },
        }
    }
}

/// A loaded netlist ready for evaluation.
///
/// Inputs are written by bus name. Only buses named at load time may be
/// written.
#[derive(Debug, Clone)]
pub struct Netlist {
    simulation: Simulation<StuckAtFault>,
    output_bus: String,
    /// The bus indices of the output, cached so reads don't re-resolve them.
    output_range: Range<usize>,
    fault_case_count: usize,
}

impl Netlist {
    /// Loads a netlist, checks that the named buses exist and are 32 bits
    /// wide, then writes the constants.
    pub fn load(
        path: &Path,
        input_buses: &[&str],
        output_bus: &str,
        constants: &[ConstantAssignment],
    ) -> Result<Self, LoadError> {
        let source = fs::read_to_string(path).map_err(|source| LoadError::Read {
            path: path.to_path_buf(),
            source,
        })?;

        let mut simulation = parse(&source, path)?;

        for bus in input_buses {
            check_input_width(&mut simulation, path, bus, WORD_BITS)?;
        }

        let output_range = match simulation.output_bus(output_bus) {
            Ok(view) => view.range(),
            Err(_) => {
                return Err(LoadError::MissingOutput {
                    path: path.to_path_buf(),
                    bus: output_bus.to_owned(),
                });
            }
        };
        if output_range.len() != WORD_BITS {
            return Err(LoadError::Width {
                path: path.to_path_buf(),
                bus: output_bus.to_owned(),
                expected: WORD_BITS,
                actual: output_range.len(),
            });
        }

        for constant in constants {
            let range =
                check_input_width(&mut simulation, path, &constant.bus, constant.bits.len())?;
            let signals: Vec<Signal> = constant.bits.iter().map(|&bit| bit.into()).collect();
            simulation
                .input_bus(&constant.bus)
                .expect("the bus was just resolved")
                .write_vector(range, &signals)
                .expect("the width was just checked");
        }

        let fault_case_count = 2 * simulation.fault_radix();

        Ok(Self {
            simulation,
            output_bus: output_bus.to_owned(),
            output_range,
            fault_case_count,
        })
    }

    /// The number of stuck-at fault cases, two per gate output.
    pub fn fault_case_count(&self) -> usize {
        self.fault_case_count
    }

    /// Writes a word to an input bus.
    ///
    /// # Panics
    ///
    /// Panics if `bus_name` was not named at load time.
    pub fn write_input(&mut self, bus_name: &str, word: u32) {
        let mut view = self
            .simulation
            .input_bus(bus_name)
            .expect("input buses are resolved at load time");
        let range = view.range();
        view.write_vector(range, &signals_from_word(word))
            .expect("input widths are checked at load time");
    }

    /// Propagates pending changes until the netlist is stable.
    pub fn settle(&mut self) {
        self.simulation.settle();
    }

    /// Reads the output bus.
    ///
    /// Fails if any output bit has no defined value.
    pub fn read_output(&self) -> Result<u32, UnknownBitError> {
        let signals = self
            .simulation
            .output_bus(&self.output_bus)
            .expect("the output bus was resolved at load time")
            .read_vector(self.output_range.clone())
            .expect("the output width was checked at load time");
        word_from_signals(&signals)
    }

    /// Applies a fault case, replacing any existing one.
    ///
    /// # Panics
    ///
    /// Panics if the case is not below [`Self::fault_case_count`].
    pub fn apply_fault(&mut self, fault_case: FaultCase) {
        self.simulation
            .make_faulty(fault_case.target, fault_case.stuck_at)
            .expect("fault case is below the case count");
    }

    /// Removes the fault, if any.
    pub fn clear_fault(&mut self) {
        self.simulation.remove_fault();
    }
}

fn parse(source: &str, path: &Path) -> Result<Simulation<StuckAtFault>, LoadError> {
    let file_name = path.display().to_string();
    match parse_simulation(source, file_name.clone()) {
        Ok(simulation) => Ok(simulation),
        Err(reports) => {
            let mut rendered = Vec::new();
            for report in reports {
                report
                    .write(sources([(file_name.clone(), source)]), &mut rendered)
                    .expect("writing to a Vec cannot fail");
            }
            Err(LoadError::Parse {
                path: path.to_path_buf(),
                reports: String::from_utf8_lossy(&rendered).into_owned(),
            })
        }
    }
}

/// Resolves an input bus and checks its width, returning its range.
fn check_input_width(
    simulation: &mut Simulation<StuckAtFault>,
    path: &Path,
    bus: &str,
    expected: usize,
) -> Result<Range<usize>, LoadError> {
    let range = match simulation.input_bus(bus) {
        Ok(view) => view.range(),
        Err(_) => {
            return Err(LoadError::MissingInput {
                path: path.to_path_buf(),
                bus: bus.to_owned(),
            });
        }
    };
    if range.len() != expected {
        return Err(LoadError::Width {
            path: path.to_path_buf(),
            bus: bus.to_owned(),
            expected,
            actual: range.len(),
        });
    }
    Ok(range)
}

#[cfg(test)]
pub mod test_netlists {
    //! Verilog sources for small netlists, written to temporary files.

    use super::*;
    use std::fmt::Write;

    /// A file that is deleted when dropped.
    pub struct TemporaryFile(pub PathBuf);

    impl Drop for TemporaryFile {
        fn drop(&mut self) {
            _ = fs::remove_file(&self.0);
        }
    }

    /// Writes `source` to a fresh file in the system temporary directory.
    pub fn write_temporary(name: &str, source: &str) -> TemporaryFile {
        let path = std::env::temp_dir().join(format!(
            "syndromes-{}-{}-{name}.v",
            std::process::id(),
            // Tests run in parallel inside one process, so the thread
            // distinguishes files with the same name.
            format!("{:?}", std::thread::current().id()).replace(['(', ')'], "")
        ));
        fs::write(&path, source).expect("temporary directory is writable");
        TemporaryFile(path)
    }

    fn header(name: &str, ports: &[(&str, &str)]) -> String {
        let mut source = String::new();
        let names: Vec<&str> = ports.iter().map(|(_, name)| *name).collect();
        writeln!(source, "module {name}({});", names.join(", ")).expect("String write");
        for (direction, name) in ports {
            writeln!(source, "  {direction} [{}:0] {name};", WORD_BITS - 1).expect("String write");
        }
        source
    }

    /// `z[i] = a[i] ^ b[i]` for every bit.
    pub fn bitwise_xor() -> String {
        let mut source = header(
            "bitwise_xor",
            &[("input", "a"), ("input", "b"), ("output", "z")],
        );
        for index in 0..WORD_BITS {
            writeln!(
                source,
                "  XOR2_X1 xor{index}(.A (a[{index}]), .B (b[{index}]), .Z (z[{index}]));"
            )
            .expect("String write");
        }
        source.push_str("endmodule\n");
        source
    }

    /// `z[i] = a[i] & b[i]` for every bit.
    pub fn bitwise_and() -> String {
        let mut source = header(
            "bitwise_and",
            &[("input", "a"), ("input", "b"), ("output", "z")],
        );
        for index in 0..WORD_BITS {
            writeln!(
                source,
                "  AND2_X1 and{index}(.A1 (a[{index}]), .A2 (b[{index}]), .ZN (z[{index}]));"
            )
            .expect("String write");
        }
        source.push_str("endmodule\n");
        source
    }

    /// `z[i] = a[i] ^ b[i] ^ c[i]` for every bit.
    pub fn bitwise_xor3() -> String {
        let mut source = header(
            "bitwise_xor3",
            &[
                ("input", "a"),
                ("input", "b"),
                ("input", "c"),
                ("output", "z"),
            ],
        );
        writeln!(source, "  wire [{}:0] ab;", WORD_BITS - 1).expect("String write");
        for index in 0..WORD_BITS {
            writeln!(
                source,
                "  XOR2_X1 first{index}(.A (a[{index}]), .B (b[{index}]), .Z (ab[{index}]));"
            )
            .expect("String write");
            writeln!(
                source,
                "  XOR2_X1 second{index}(.A (ab[{index}]), .B (c[{index}]), .Z (z[{index}]));"
            )
            .expect("String write");
        }
        source.push_str("endmodule\n");
        source
    }

    /// `z[i] = a[i] ^ b[i]` with a 3-bit `mode` input that inverts bit 0 of
    /// the output when its low bit is set. The other mode bits are unused.
    pub fn xor_with_mode() -> String {
        let mut source = header(
            "xor_with_mode",
            &[("input", "a"), ("input", "b"), ("output", "z")],
        );
        source.push_str("  input [2:0] mode;\n  wire raw0;\n");
        writeln!(source, "  XOR2_X1 xor0(.A (a[0]), .B (b[0]), .Z (raw0));").expect("String write");
        writeln!(
            source,
            "  XOR2_X1 mode0(.A (raw0), .B (mode[0]), .Z (z[0]));"
        )
        .expect("String write");
        for index in 1..WORD_BITS {
            writeln!(
                source,
                "  XOR2_X1 xor{index}(.A (a[{index}]), .B (b[{index}]), .Z (z[{index}]));"
            )
            .expect("String write");
        }
        source.push_str("endmodule\n");
        source
    }

    /// Like [`bitwise_xor`] but output bit 5 is never driven.
    pub fn xor_with_undriven_bit() -> String {
        bitwise_xor().replace(
            "XOR2_X1 xor5(.A (a[5]), .B (b[5]), .Z (z[5]));",
            "XOR2_X1 xor5(.A (a[5]), .B (b[5]), .Z (unused));",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{test_netlists::*, *};

    fn load(path: &Path, inputs: &[&str]) -> Result<Netlist, LoadError> {
        Netlist::load(path, inputs, "z", &[])
    }

    #[test]
    fn loads_bitwise_xor() {
        let file = write_temporary("loads", &bitwise_xor());
        let mut netlist = load(&file.0, &["a", "b"]).unwrap();

        // 32 XOR gates, each with one output, two polarities each.
        assert_eq!(netlist.fault_case_count(), 64);

        netlist.write_input("a", 0xF0F0_F0F0);
        netlist.write_input("b", 0xFF00_FF00);
        netlist.settle();
        assert_eq!(netlist.read_output(), Ok(0xF0F0_F0F0 ^ 0xFF00_FF00));
    }

    #[test]
    fn missing_input_bus() {
        let file = write_temporary("missing-input", &bitwise_xor());
        let error = load(&file.0, &["a", "nope"]).unwrap_err();
        assert!(matches!(error, LoadError::MissingInput { bus, .. } if bus == "nope"));
    }

    #[test]
    fn missing_output_bus() {
        let file = write_temporary("missing-output", &bitwise_xor());
        let error = Netlist::load(&file.0, &["a", "b"], "nope", &[]).unwrap_err();
        assert!(matches!(error, LoadError::MissingOutput { bus, .. } if bus == "nope"));
    }

    #[test]
    fn wrong_input_width() {
        let file = write_temporary("input-width", &xor_with_mode());
        let error = load(&file.0, &["a", "mode"]).unwrap_err();
        assert!(matches!(
            error,
            LoadError::Width { bus, expected: 32, actual: 3, .. } if bus == "mode"
        ));
    }

    #[test]
    fn wrong_constant_width() {
        let file = write_temporary("constant-width", &xor_with_mode());
        let constants = [ConstantAssignment {
            bus: "mode".to_owned(),
            bits: vec![Bit::One],
        }];
        let error = Netlist::load(&file.0, &["a", "b"], "z", &constants).unwrap_err();
        assert!(matches!(
            error,
            LoadError::Width { bus, expected: 1, actual: 3, .. } if bus == "mode"
        ));
    }

    #[test]
    fn parse_failure_is_reported() {
        let file = write_temporary("parse", "module broken(a);\n  input a\nendmodule\n");
        let error = load(&file.0, &["a"]).unwrap_err();
        assert!(matches!(error, LoadError::Parse { .. }));
    }

    #[test]
    fn constants_persist_across_input_writes() {
        let file = write_temporary("constants", &xor_with_mode());
        let constants = [ConstantAssignment {
            bus: "mode".to_owned(),
            bits: vec![Bit::One, Bit::Zero, Bit::Zero],
        }];
        let mut netlist = Netlist::load(&file.0, &["a", "b"], "z", &constants).unwrap();

        for (a, b) in [(0, 0), (1, 0), (0xFFFF_FFFF, 0x1234_5678)] {
            netlist.write_input("a", a);
            netlist.write_input("b", b);
            netlist.settle();
            assert_eq!(netlist.read_output(), Ok((a ^ b) ^ 1));
        }
    }

    #[test]
    fn undriven_output_bit_is_an_error() {
        let file = write_temporary("undriven", &xor_with_undriven_bit());
        let mut netlist = load(&file.0, &["a", "b"]).unwrap();
        netlist.write_input("a", 1);
        netlist.write_input("b", 2);
        netlist.settle();
        assert_eq!(netlist.read_output(), Err(UnknownBitError { index: 5 }));
    }

    #[test]
    fn faults_apply_and_clear() {
        let file = write_temporary("faults", &bitwise_xor());
        let mut netlist = load(&file.0, &["a", "b"]).unwrap();
        netlist.write_input("a", 0);
        netlist.write_input("b", 0);
        netlist.settle();
        assert_eq!(netlist.read_output(), Ok(0));

        // Every stuck-at-one case flips exactly one bit of an all-zero output.
        let mut seen = 0;
        for index in (1..netlist.fault_case_count()).step_by(2) {
            netlist.apply_fault(FaultCase::from_index(index));
            netlist.settle();
            let output = netlist.read_output().unwrap();
            assert_eq!(output.count_ones(), 1);
            seen |= output;
        }
        assert_eq!(seen, u32::MAX);

        netlist.clear_fault();
        netlist.settle();
        assert_eq!(netlist.read_output(), Ok(0));
    }

    #[test]
    fn fault_case_from_index() {
        assert_eq!(FaultCase::from_index(0).stuck_at, StuckAtFault::Low);
        assert_eq!(FaultCase::from_index(1).stuck_at, StuckAtFault::High);
        assert_eq!(FaultCase::from_index(7).target, 3);
    }
}
