//! Result and error types for bit buffer operations.

/// The status of a copy operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CopyIntoStatus {
    /// The destination was filled before the source could be exhausted.
    Partial,
    /// The source is exhausted.
    Exhausted,
}

/// Metrics about `copy_into*` operations on bit buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CopyIntoResult {
    /// The number of items copied of the fundamental unit of the buffer. Bits
    /// or bytes depending on the function.
    pub units_copied: usize,
    /// The status of the copy operation.
    pub status: CopyIntoStatus,
}

impl CopyIntoResult {
    /// Construct a new result with an `Exhausted` status.
    #[must_use]
    pub fn exhausted(units_copied: usize) -> Self {
        Self {
            units_copied,
            status: CopyIntoStatus::Exhausted,
        }
    }

    /// Construct a new result with a `Partial` status.
    #[must_use]
    pub fn partial(units_copied: usize) -> Self {
        Self {
            units_copied,
            status: CopyIntoStatus::Partial,
        }
    }
}

// TODO: Candidate for deletion/relocation
/// An error for cases where a bit buffer operations goes out of bounds.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
#[error("Out of bounds")]
pub struct OutOfBoundsError;
