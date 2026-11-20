//! The `syndromes` command line.

mod cli;

use clap::Parser;

fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    cli::Cli::parse().run()
}
