//! The TOML description of a computer.
//!
//! One of:
//!
//! ```toml
//! [fused]
//! path = "mac.v"
//! activation = "a"
//! weight = "b"
//! partial_sum = "c"
//! output = "z"
//! constants = { round = "000" }
//! ```
//!
//! ```toml
//! [separate.multiplier]
//! path = "mul.v"
//! activation = "a"
//! weight = "b"
//! product = "p"
//!
//! [separate.adder]
//! path = "add.v"
//! product = "p"
//! partial_sum = "c"
//! sum = "s"
//! ```
//!
//! Constants map a bus to its bits, most significant first. Paths are
//! relative to the description file.

use eyre::{Context, bail};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
use syndromes::{
    Bit,
    computer::{
        AdderDescription, Computer, ConstantAssignment, Fused, FusedDescription,
        MultiplierDescription, Separate, SeparateDescription,
    },
};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum ComputerFile {
    Fused(FusedFile),
    Separate(SeparateFile),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FusedFile {
    path: PathBuf,
    activation: String,
    weight: String,
    partial_sum: String,
    output: String,
    #[serde(default)]
    constants: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SeparateFile {
    multiplier: MultiplierFile,
    adder: AdderFile,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MultiplierFile {
    path: PathBuf,
    activation: String,
    weight: String,
    product: String,
    #[serde(default)]
    constants: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AdderFile {
    path: PathBuf,
    product: String,
    partial_sum: String,
    sum: String,
    #[serde(default)]
    constants: BTreeMap<String, String>,
}

/// Loads the computer described by the TOML file at `path`.
pub fn load(path: &Path) -> eyre::Result<Computer> {
    let text =
        fs::read_to_string(path).with_context(|| format!("could not read {}", path.display()))?;
    let file: ComputerFile =
        toml::from_str(&text).with_context(|| format!("could not parse {}", path.display()))?;
    // The description names netlists next to itself, not next to wherever
    // the command happens to be run from.
    let base = path.parent().unwrap_or(Path::new(""));
    let computer = match file {
        ComputerFile::Fused(fused) => Computer::Fused(Fused::load(&fused.description(base)?)?),
        ComputerFile::Separate(separate) => {
            Computer::Separate(Separate::load(&SeparateDescription {
                multiplier: separate.multiplier.description(base)?,
                adder: separate.adder.description(base)?,
            })?)
        }
    };
    Ok(computer)
}

impl FusedFile {
    fn description(self, base: &Path) -> eyre::Result<FusedDescription> {
        Ok(FusedDescription {
            path: base.join(self.path),
            activation: self.activation,
            weight: self.weight,
            partial_sum: self.partial_sum,
            output: self.output,
            constants: constants(self.constants)?,
        })
    }
}

impl MultiplierFile {
    fn description(self, base: &Path) -> eyre::Result<MultiplierDescription> {
        Ok(MultiplierDescription {
            path: base.join(self.path),
            activation: self.activation,
            weight: self.weight,
            product: self.product,
            constants: constants(self.constants)?,
        })
    }
}

impl AdderFile {
    fn description(self, base: &Path) -> eyre::Result<AdderDescription> {
        Ok(AdderDescription {
            path: base.join(self.path),
            product: self.product,
            partial_sum: self.partial_sum,
            sum: self.sum,
            constants: constants(self.constants)?,
        })
    }
}

fn constants(constants: BTreeMap<String, String>) -> eyre::Result<Vec<ConstantAssignment>> {
    constants
        .into_iter()
        .map(|(bus, text)| {
            let bits = bits(&text).with_context(|| format!("constant for bus {bus}"))?;
            Ok(ConstantAssignment { bus, bits })
        })
        .collect()
}

/// Parses bits written most significant first into least significant first.
fn bits(text: &str) -> eyre::Result<Vec<Bit>> {
    if text.is_empty() {
        bail!("no bits given");
    }
    text.chars()
        .rev()
        .map(|character| match character {
            '0' => Ok(Bit::Zero),
            '1' => Ok(Bit::One),
            other => bail!("{other:?} is not a bit"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_fused_description() {
        let text = r#"
            [fused]
            path = "mac.v"
            activation = "a"
            weight = "b"
            partial_sum = "c"
            output = "z"
            constants = { round = "010", mode = "1" }
        "#;
        let ComputerFile::Fused(fused) = toml::from_str(text).expect("parses") else {
            panic!("not fused");
        };
        let description = fused.description(Path::new("netlists")).expect("converts");
        assert_eq!(
            description,
            FusedDescription {
                path: PathBuf::from("netlists/mac.v"),
                activation: "a".to_owned(),
                weight: "b".to_owned(),
                partial_sum: "c".to_owned(),
                output: "z".to_owned(),
                constants: vec![
                    ConstantAssignment {
                        bus: "mode".to_owned(),
                        bits: vec![Bit::One],
                    },
                    ConstantAssignment {
                        bus: "round".to_owned(),
                        bits: vec![Bit::Zero, Bit::One, Bit::Zero],
                    },
                ],
            }
        );
    }

    #[test]
    fn parses_a_separate_description() {
        let text = r#"
            [separate.multiplier]
            path = "mul.v"
            activation = "a"
            weight = "b"
            product = "p"

            [separate.adder]
            path = "/elsewhere/add.v"
            product = "p"
            partial_sum = "c"
            sum = "s"
        "#;
        let ComputerFile::Separate(separate) = toml::from_str(text).expect("parses") else {
            panic!("not separate");
        };
        let multiplier = separate
            .multiplier
            .description(Path::new("netlists"))
            .expect("converts");
        let adder = separate
            .adder
            .description(Path::new("netlists"))
            .expect("converts");
        assert_eq!(multiplier.path, PathBuf::from("netlists/mul.v"));
        assert!(multiplier.constants.is_empty());
        // An absolute path is left alone.
        assert_eq!(adder.path, PathBuf::from("/elsewhere/add.v"));
        assert_eq!(adder.sum, "s");
    }

    #[test]
    fn rejects_unknown_fields_and_bad_bits() {
        let unknown = r#"
            [fused]
            path = "mac.v"
            activation = "a"
            weight = "b"
            partial_sum = "c"
            output = "z"
            rounding = "x"
        "#;
        assert!(toml::from_str::<ComputerFile>(unknown).is_err());
        assert!(bits("").is_err());
        assert!(bits("012").is_err());
        assert_eq!(bits("10").expect("parses"), [Bit::Zero, Bit::One]);
    }
}
