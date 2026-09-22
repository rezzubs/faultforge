//! The `syndromes` command line.

mod cli;
mod format;
#[cfg(feature = "ui")]
mod ui;

use clap::Parser;

fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    cli::Cli::parse().run()
}
