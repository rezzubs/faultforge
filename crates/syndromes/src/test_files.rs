//! Temporary files for tests that need something on disk.

use std::{fs, path::PathBuf};

/// A file that is deleted when dropped.
pub struct TemporaryFile(pub PathBuf);

impl Drop for TemporaryFile {
    fn drop(&mut self) {
        _ = fs::remove_file(&self.0);
    }
}

/// A fresh path in the system temporary directory.
///
/// The file is not created, only named; the returned guard deletes it.
pub fn temporary_path(name: &str, extension: &str) -> TemporaryFile {
    let path = std::env::temp_dir().join(format!(
        "syndromes-{}-{}-{name}.{extension}",
        std::process::id(),
        // Tests run in parallel inside one process, so the thread
        // distinguishes files with the same name.
        format!("{:?}", std::thread::current().id()).replace(['(', ')'], "")
    ));
    TemporaryFile(path)
}

/// Writes `contents` to a fresh file in the system temporary directory.
pub fn write_temporary(name: &str, extension: &str, contents: &str) -> TemporaryFile {
    let file = temporary_path(name, extension);
    fs::write(&file.0, contents).expect("temporary directory is writable");
    file
}
