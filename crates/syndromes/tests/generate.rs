//! Runs the binary end to end on a small netlist.

use std::{fs, path::PathBuf, process::Command};
use syndromes::file::HistogramFile;

/// A fresh output path, deleted when dropped.
struct Output(PathBuf);

impl Drop for Output {
    fn drop(&mut self) {
        _ = fs::remove_file(&self.0);
    }
}

fn output(name: &str) -> Output {
    Output(std::env::temp_dir().join(format!(
        "syndromes-generate-{}-{name}.json",
        std::process::id()
    )))
}

fn generate(output: &Output, extra: &[&str]) -> std::process::Output {
    let computer = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/xor3.toml");
    Command::new(env!("CARGO_BIN_EXE_syndromes"))
        .args(["generate", "--computer"])
        .arg(&computer)
        .args([
            "--source",
            "random-bits",
            "--missing-mass",
            "0.01",
            "--self-split-distance",
            "0.1",
            "--head-threshold",
            "2",
            "--workers",
            "3",
            "--batch-size",
            "7",
            "--output",
        ])
        .arg(&output.0)
        .args(extra)
        .output()
        .expect("runs the binary")
}

const SEEDED: &[&str] = &["--seed", "7", "--check-interval", "64"];

#[test]
fn generates_a_histogram_file() {
    let output = output("basic");
    let run = generate(&output, SEEDED);
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );

    let file = HistogramFile::read(fs::File::open(&output.0).expect("opens")).expect("reads");
    assert_eq!(file.seed, 7);
    assert_eq!(file.rule.check_interval.get(), 64);
    let evaluations = file.statistics.evaluations;
    assert!(evaluations > 0 && evaluations.is_multiple_of(64));
    assert_eq!(
        file.histogram.iter().map(|(_, count)| count).sum::<u64>(),
        evaluations
    );
    assert!(file.histogram.windows(2).all(|pair| pair[0].1 >= pair[1].1));

    // The same seed reproduces the file byte for byte, and only with
    // --overwrite.
    let first = fs::read(&output.0).expect("reads");
    let refused = generate(&output, SEEDED);
    assert!(!refused.status.success());
    let second = generate(&output, &[SEEDED, &["--overwrite"]].concat());
    assert!(
        second.status.success(),
        "{}",
        String::from_utf8_lossy(&second.stderr)
    );
    assert_eq!(fs::read(&output.0).expect("reads"), first);
}

#[test]
fn defaults_to_a_random_seed() {
    let output = output("unseeded");
    let seeds: Vec<u64> = (0..2)
        .map(|_| {
            let run = generate(&output, &["--overwrite"]);
            assert!(
                run.status.success(),
                "{}",
                String::from_utf8_lossy(&run.stderr)
            );
            let file =
                HistogramFile::read(fs::File::open(&output.0).expect("opens")).expect("reads");
            assert_eq!(file.rule.check_interval.get(), 1000);
            file.seed
        })
        .collect();
    assert_ne!(seeds[0], seeds[1]);
}
