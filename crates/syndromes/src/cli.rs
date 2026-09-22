//! The commands and their arguments.

mod computer;
mod source;
mod status;
#[cfg(feature = "ui")]
mod view;

use clap::{Args, Parser, Subcommand};
use eyre::{Context, bail};
use std::{
    fs::File,
    io::BufWriter,
    num::{NonZeroU64, NonZeroUsize},
    path::{Path, PathBuf},
    thread,
};
use syndromes::{
    fault_selection::FaultSelection,
    file::HistogramFile,
    generation::{Configuration, StoppingRule, Threshold},
};

#[derive(Debug, Parser)]
#[command(version, about)]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Generate a syndrome histogram and save it as JSON.
    Generate(GenerateArguments),
    /// Show one saved histogram, or two side by side.
    #[cfg(feature = "ui")]
    View(view::ViewArguments),
}

impl Cli {
    pub fn run(self) -> eyre::Result<()> {
        match self.command {
            Command::Generate(arguments) => generate(&arguments),
            #[cfg(feature = "ui")]
            Command::View(arguments) => arguments.run(),
        }
    }
}

#[derive(Debug, Args)]
struct GenerateArguments {
    /// The TOML file describing the computer: its netlists, ports and
    /// constants.
    ///
    /// One fused multiply-add netlist:
    ///
    ///   [fused]
    ///   path = "mac.v"
    ///   activation = "a"
    ///   weight = "b"
    ///   partial_sum = "c"
    ///   output = "z"
    ///   constants = { round = "000" }
    ///
    /// Or a multiplier whose product feeds an adder:
    ///
    ///   [separate.multiplier]
    ///   path = "mul.v"
    ///   activation = "a"
    ///   weight = "b"
    ///   product = "p"
    ///
    ///   [separate.adder]
    ///   path = "add.v"
    ///   product = "p"
    ///   partial_sum = "c"
    ///   sum = "s"
    ///
    /// Every field except `constants` is required. Constants map a bus to
    /// its bits, most significant first. Paths are relative to the
    /// description file.
    #[arg(long, verbatim_doc_comment)]
    computer: PathBuf,
    #[command(flatten)]
    source: source::SourceArguments,
    /// Inject only this fault case instead of a uniformly chosen one.
    #[arg(long)]
    fault_case: Option<usize>,
    /// The seed the whole run derives from. Chosen at random if omitted and
    /// recorded in the output either way.
    #[arg(long)]
    seed: Option<u64>,
    /// Stop once the missing mass is at most this.
    #[arg(long, value_parser = threshold)]
    missing_mass: Threshold,
    /// Stop once the self-split distance is at most this.
    #[arg(long, value_parser = threshold)]
    self_split_distance: Threshold,
    /// Leave syndromes seen fewer times than this out of the self-split
    /// distance.
    #[arg(long)]
    head_threshold: u64,
    /// Ask the stopping rule every this many evaluations.
    #[arg(long, default_value_t = NonZeroU64::new(1000).expect("nonzero"))]
    check_interval: NonZeroU64,
    /// The number of worker threads.
    #[arg(long, default_value_t = default_workers())]
    workers: NonZeroUsize,
    /// The number of consecutive jobs a worker claims at a time.
    #[arg(long, default_value_t = NonZeroU64::new(256).expect("nonzero"))]
    batch_size: NonZeroU64,
    /// Where to write the histogram.
    #[arg(long)]
    output: PathBuf,
    /// Replace the output file if it exists.
    #[arg(long)]
    overwrite: bool,
}

fn threshold(text: &str) -> Result<Threshold, String> {
    let value: f64 = text.parse().map_err(|error| format!("{error}"))?;
    Threshold::try_from(value).map_err(|error| error.to_string())
}

fn default_workers() -> NonZeroUsize {
    thread::available_parallelism().unwrap_or(NonZeroUsize::MIN)
}

fn generate(arguments: &GenerateArguments) -> eyre::Result<()> {
    check_output(&arguments.output, arguments.overwrite)?;

    let computer = computer::load(&arguments.computer)?;
    let source = arguments.source.build()?;
    let selection = match arguments.fault_case {
        Some(fault_case) => FaultSelection::fixed(fault_case, computer.fault_case_count())?,
        None => FaultSelection::Uniform,
    };
    let rule = StoppingRule {
        missing_mass: arguments.missing_mass,
        self_split_distance: arguments.self_split_distance,
        head_threshold: arguments.head_threshold,
        check_interval: arguments.check_interval,
    };
    let seed = arguments.seed.unwrap_or_else(rand::random);
    let configuration = Configuration {
        seed,
        source,
        selection,
        rule,
        workers: arguments.workers,
        batch_size: arguments.batch_size,
    };

    let mut status = status::Status::new(rule);
    let outcome = configuration.run(&computer, |_, statistics| status.update(statistics))?;
    status.finish(&outcome.statistics);

    let file = HistogramFile::from_outcome(seed, rule, &outcome);
    let writer = File::create(&arguments.output)
        .with_context(|| format!("could not create {}", arguments.output.display()))?;
    file.write(BufWriter::new(writer))
        .with_context(|| format!("could not write {}", arguments.output.display()))?;
    eprintln!(
        "wrote {}: seed {seed}, {} evaluations, {} syndromes",
        arguments.output.display(),
        outcome.statistics.evaluations,
        outcome.statistics.distinct_syndromes
    );
    Ok(())
}

/// Fails now, before the run, if `output` could not be written afterwards.
fn check_output(output: &Path, overwrite: bool) -> eyre::Result<()> {
    if output.exists() && !overwrite {
        bail!(
            "{} exists; pass --overwrite to replace it",
            output.display()
        );
    }
    // An empty parent is the working directory.
    let parent = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    if let Some(parent) = parent
        && !parent.is_dir()
    {
        bail!("{} is not a directory", parent.display());
    }
    Ok(())
}
