//! Runs the `view` subcommand far enough to fail before a window opens.
#![cfg(feature = "ui")]

use std::process::Command;

fn view(arguments: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_syndromes"))
        .arg("view")
        .args(arguments)
        .output()
        .expect("runs the binary")
}

#[test]
fn a_missing_file_is_an_error_on_the_terminal() {
    let missing = std::env::temp_dir().join("syndromes-view-does-not-exist.json");
    let run = view(&[missing.to_str().expect("utf-8")]);
    assert!(!run.status.success());
    let message = String::from_utf8_lossy(&run.stderr);
    assert!(message.contains("could not read"), "{message}");
    assert!(
        message.contains("syndromes-view-does-not-exist"),
        "{message}"
    );
}

#[test]
fn a_malformed_file_is_an_error_on_the_terminal() {
    let path = std::env::temp_dir().join(format!(
        "syndromes-view-malformed-{}.json",
        std::process::id()
    ));
    std::fs::write(&path, b"not json").expect("writes");
    let run = view(&[path.to_str().expect("utf-8")]);
    _ = std::fs::remove_file(&path);
    assert!(!run.status.success());
    let message = String::from_utf8_lossy(&run.stderr);
    assert!(message.contains("could not read"), "{message}");
}
