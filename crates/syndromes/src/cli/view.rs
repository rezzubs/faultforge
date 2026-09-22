//! The flags that open saved histograms in a window.

use crate::ui::{App, Feed, Snapshot};
use clap::Args;
use eyre::Context;
use std::{fs::File, io::BufReader, path::Path, path::PathBuf};
use syndromes::file::HistogramFile;

#[derive(Debug, Args)]
pub struct ViewArguments {
    /// The histogram to show.
    file: PathBuf,
    /// A second histogram to compare it against.
    #[arg(long)]
    compare: Option<PathBuf>,
}

impl ViewArguments {
    /// Loads the histograms and opens the window.
    pub fn run(self) -> eyre::Result<()> {
        // Both files are read before the window opens, so a bad path is an
        // error on the terminal rather than an empty window.
        let primary = load(&self.file)?;
        let secondary = self.compare.as_deref().map(load).transpose()?;

        let title = format!("syndromes - {}", primary.name());
        eframe::run_native(
            &title,
            eframe::NativeOptions::default(),
            Box::new(|_| Ok(Box::new(App::new(Feed::File, primary, secondary)))),
        )
        .map_err(|error| eyre::eyre!("could not open the window: {error}"))
    }
}

fn load(path: &Path) -> eyre::Result<Snapshot> {
    let read = || -> eyre::Result<HistogramFile> {
        let file = File::open(path)?;
        Ok(HistogramFile::read(BufReader::new(file))?)
    };
    let file = read().with_context(|| format!("could not read {}", path.display()))?;
    Ok(Snapshot::from_file(name(path), file))
}

/// What to call the histogram from this path.
fn name(path: &Path) -> String {
    path.file_name()
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_histogram_is_named_after_its_file() {
        assert_eq!(name(Path::new("runs/normal.json")), "normal.json");
        assert_eq!(name(Path::new("normal.json")), "normal.json");
        // No file name at all, so there is nothing to shorten to.
        assert_eq!(name(Path::new("..")), "..");
    }
}
