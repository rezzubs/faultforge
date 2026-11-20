//! What can go wrong loading a pool.

use super::{Group, Regime};
use ndarray_npy::ReadNpzError;
use std::{io, path::PathBuf};

/// Failures of loading a pool from an artifact.
#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    /// The file could not be opened.
    #[error("could not read artifact {path}")]
    Read {
        /// The artifact file.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: io::Error,
    },
    /// The file is not an archive with the expected members and types.
    #[error("artifact {path} is malformed")]
    Format {
        /// The artifact file.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: ReadNpzError,
    },
    /// A member's dimensions do not fit the others.
    #[error("artifact {path}: member {member} has shape {actual}, expected {expected}")]
    Shape {
        /// The artifact file.
        path: PathBuf,
        /// The offending member.
        member: &'static str,
        /// The shape that would fit.
        expected: String,
        /// The shape found.
        actual: String,
    },
    /// An element claims more real samples than the archive holds for it.
    #[error(
        "artifact {path}: element ({row}, {column}) has fill {fill} above the capacity {capacity}"
    )]
    Fill {
        /// The artifact file.
        path: PathBuf,
        /// The element's row.
        row: usize,
        /// The element's column.
        column: usize,
        /// The number of real samples claimed.
        fill: u64,
        /// The number of samples stored per element.
        capacity: usize,
    },
    /// The group lies outside the recorded array.
    #[error("artifact {path}: {group} is outside the {rows}x{columns} array")]
    GroupOutOfRange {
        /// The artifact file.
        path: PathBuf,
        /// The requested group.
        group: Group,
        /// The number of rows recorded.
        rows: usize,
        /// The number of columns recorded.
        columns: usize,
    },
    /// No element of the group recorded any sample in the regime.
    #[error("artifact {path}: {group} has no {regime} samples")]
    EmptyGroup {
        /// The artifact file.
        path: PathBuf,
        /// The requested regime.
        regime: Regime,
        /// The requested group.
        group: Group,
    },
}
