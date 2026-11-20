//! The flags that pick and configure an input source.

use clap::{Args, ValueEnum};
use eyre::{Context, eyre};
use std::{path::PathBuf, str::FromStr};
use syndromes::{
    Triple,
    input_source::{
        InputSource, Normal, Scales, StandardDeviation,
        artifact::{Group, Joint, Marginals, Pool, Regime},
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SourceKind {
    /// Uniformly random bit patterns. For testing.
    RandomBits,
    /// The same triple every time.
    Constant,
    /// Independent zero-mean normal draws per field.
    Normal,
    /// Recorded triples drawn whole from a profiling artifact.
    Joint,
    /// Recorded values drawn per field from a profiling artifact.
    Marginals,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum RegimeArgument {
    /// The first product of an accumulation.
    First,
    /// A product added to a running partial sum.
    Active,
    /// A finished sum passed through.
    Drain,
}

impl From<RegimeArgument> for Regime {
    fn from(regime: RegimeArgument) -> Self {
        match regime {
            RegimeArgument::First => Regime::First,
            RegimeArgument::Active => Regime::Active,
            RegimeArgument::Drain => Regime::Drain,
        }
    }
}

/// `all`, `row:N`, `column:N` or `element:R,C`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GroupArgument(Group);

impl FromStr for GroupArgument {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let index = |text: &str| {
            text.parse::<usize>()
                .map_err(|error| format!("{text:?} is not an index: {error}"))
        };
        let group = match text.split_once(':') {
            None if text == "all" => Group::Array,
            Some(("row", row)) => Group::Row(index(row)?),
            Some(("column", column)) => Group::Column(index(column)?),
            Some(("element", position)) => {
                let (row, column) = position
                    .split_once(',')
                    .ok_or_else(|| format!("{position:?} is not ROW,COLUMN"))?;
                Group::Element {
                    row: index(row)?,
                    column: index(column)?,
                }
            }
            _ => {
                return Err(format!(
                    "{text:?} is not all, row:N, column:N or element:R,C"
                ));
            }
        };
        Ok(Self(group))
    }
}

/// `ACTIVATION,WEIGHT,PARTIAL_SUM`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct TripleArgument(Triple);

impl FromStr for TripleArgument {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let mut fields = text.split(',').map(|field| {
            field
                .trim()
                .parse::<f32>()
                .map_err(|error| format!("{field:?} is not a number: {error}"))
        });
        let mut next = || {
            fields
                .next()
                .ok_or_else(|| "expected three numbers".to_owned())?
        };
        let triple = Triple {
            activation: next()?,
            weight: next()?,
            partial_sum: next()?,
        };
        if fields.next().is_some() {
            return Err("expected three numbers".to_owned());
        }
        Ok(Self(triple))
    }
}

fn standard_deviation(text: &str) -> Result<StandardDeviation, String> {
    let value: f32 = text.parse().map_err(|error| format!("{error}"))?;
    StandardDeviation::try_from(value).map_err(|error| error.to_string())
}

#[derive(Debug, Args)]
pub struct SourceArguments {
    /// Where the inputs come from.
    #[arg(long, value_enum)]
    source: SourceKind,
    /// The triple of a constant source, as ACTIVATION,WEIGHT,PARTIAL_SUM.
    #[arg(long, required_if_eq("source", "constant"))]
    triple: Option<TripleArgument>,
    /// The standard deviation of every field of a normal source. A
    /// per-field scale overrides it for that field.
    #[arg(long, value_parser = standard_deviation)]
    scale_all: Option<StandardDeviation>,
    /// The standard deviation of the activation of a normal source.
    #[arg(long, value_parser = standard_deviation)]
    scale_activation: Option<StandardDeviation>,
    /// The standard deviation of the weight of a normal source.
    #[arg(long, value_parser = standard_deviation)]
    scale_weight: Option<StandardDeviation>,
    /// The standard deviation of the partial sum of a normal source.
    #[arg(long, value_parser = standard_deviation)]
    scale_partial_sum: Option<StandardDeviation>,
    /// The profiling artifact of a joint or marginals source.
    #[arg(long, required_if_eq_any([("source", "joint"), ("source", "marginals")]))]
    artifact: Option<PathBuf>,
    /// The regime to draw from the artifact.
    #[arg(long, value_enum, required_if_eq_any([("source", "joint"), ("source", "marginals")]))]
    regime: Option<RegimeArgument>,
    /// The elements to pool from the artifact: all, row:N, column:N or
    /// element:R,C.
    #[arg(long, required_if_eq_any([("source", "joint"), ("source", "marginals")]))]
    group: Option<GroupArgument>,
}

impl SourceArguments {
    /// Builds the source, loading the artifact if one is named.
    pub fn build(&self) -> eyre::Result<InputSource> {
        // clap has already required the flags each kind needs, except the
        // scales, which have `--scale-all` to fall back on.
        let required = "required by clap for this source";
        let source = match self.source {
            SourceKind::RandomBits => InputSource::RandomBits,
            SourceKind::Constant => InputSource::Constant(self.triple.expect(required).0),
            SourceKind::Normal => InputSource::Normal(Normal::new(Scales {
                activation: self.scale("activation", self.scale_activation)?,
                weight: self.scale("weight", self.scale_weight)?,
                partial_sum: self.scale("partial-sum", self.scale_partial_sum)?,
            })),
            SourceKind::Joint => InputSource::Joint(Joint::new(self.pool()?)),
            SourceKind::Marginals => InputSource::Marginals(Marginals::new(self.pool()?)),
        };
        Ok(source)
    }

    /// The scale of one field of a normal source: its own flag, else
    /// `--scale-all`.
    fn scale(
        &self,
        field: &str,
        specific: Option<StandardDeviation>,
    ) -> eyre::Result<StandardDeviation> {
        specific
            .or(self.scale_all)
            .ok_or_else(|| eyre!("--source normal needs --scale-{field} or --scale-all"))
    }

    fn pool(&self) -> eyre::Result<Pool> {
        let required = "required by clap for this source";
        let path = self.artifact.as_deref().expect(required);
        let regime = self.regime.expect(required).into();
        let group = self.group.expect(required).0;
        Pool::load(path, regime, group)
            .with_context(|| format!("could not load {} from {}", group, path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Command};
    use clap::Parser;

    #[test]
    fn groups_parse() {
        let group = |text: &str| text.parse::<GroupArgument>().map(|group| group.0);
        assert_eq!(group("all"), Ok(Group::Array));
        assert_eq!(group("row:2"), Ok(Group::Row(2)));
        assert_eq!(group("column:0"), Ok(Group::Column(0)));
        assert_eq!(
            group("element:1,3"),
            Ok(Group::Element { row: 1, column: 3 })
        );
        assert!(group("rows").is_err());
        assert!(group("row:x").is_err());
        assert!(group("element:1").is_err());
    }

    #[test]
    fn triples_parse() {
        let triple = |text: &str| text.parse::<TripleArgument>().map(|triple| triple.0);
        assert_eq!(
            triple("1.5, -2, 0"),
            Ok(Triple {
                activation: 1.5,
                weight: -2.0,
                partial_sum: 0.0,
            })
        );
        assert!(triple("1,2").is_err());
        assert!(triple("1,2,3,4").is_err());
        assert!(triple("1,two,3").is_err());
    }

    /// Parses a full `generate` command line with the given source flags.
    fn parse(source_flags: &[&str]) -> Result<Cli, clap::Error> {
        let mut arguments = vec![
            "syndromes",
            "generate",
            "--computer",
            "mac.toml",
            "--missing-mass",
            "0.01",
            "--self-split-distance",
            "0.1",
            "--head-threshold",
            "2",
            "--check-interval",
            "64",
            "--output",
            "out.json",
        ];
        arguments.extend_from_slice(source_flags);
        Cli::try_parse_from(arguments)
    }

    #[test]
    fn each_source_requires_its_flags() {
        assert!(parse(&["--source", "random-bits"]).is_ok());
        assert!(parse(&["--source", "constant"]).is_err());
        assert!(parse(&["--source", "constant", "--triple", "0,0,0"]).is_ok());
        assert!(parse(&["--source", "normal", "--scale-activation", "-1"]).is_err());
        assert!(parse(&["--source", "joint", "--artifact", "a.npz"]).is_err());
        for kind in ["joint", "marginals"] {
            assert!(
                parse(&[
                    "--source",
                    kind,
                    "--artifact",
                    "a.npz",
                    "--regime",
                    "active",
                    "--group",
                    "row:1",
                ])
                .is_ok()
            );
        }
    }
    /// Builds the source of a `generate` command line with the given flags.
    fn build(source_flags: &[&str]) -> eyre::Result<InputSource> {
        let Command::Generate(arguments) = parse(source_flags).expect("parses").command;
        arguments.source.build()
    }

    fn normal(activation: f32, weight: f32, partial_sum: f32) -> InputSource {
        let deviation = |value: f32| StandardDeviation::try_from(value).expect("valid");
        InputSource::Normal(Normal::new(Scales {
            activation: deviation(activation),
            weight: deviation(weight),
            partial_sum: deviation(partial_sum),
        }))
    }

    #[test]
    fn normal_scales_come_from_the_field_or_from_scale_all() {
        assert!(build(&["--source", "normal"]).is_err());
        assert!(build(&["--source", "normal", "--scale-activation", "1"]).is_err());

        let all = build(&["--source", "normal", "--scale-all", "2"]).expect("builds");
        assert_eq!(all, normal(2.0, 2.0, 2.0));

        let mixed = build(&[
            "--source",
            "normal",
            "--scale-all",
            "2",
            "--scale-weight",
            "0.5",
        ])
        .expect("builds");
        assert_eq!(mixed, normal(2.0, 0.5, 2.0));

        let each = build(&[
            "--source",
            "normal",
            "--scale-activation",
            "1",
            "--scale-weight",
            "0.5",
            "--scale-partial-sum",
            "3",
        ])
        .expect("builds");
        assert_eq!(each, normal(1.0, 0.5, 3.0));
    }
}
