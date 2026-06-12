//! Majority vote encoding for a single bit.
//!
//! A choosen bit is copied into an odd number of other bits (see [`Scheme`]).
//! When decoding, a majority vote will be used for the final value of the
//! protected bit.

#[cfg(test)]
mod tests;

use std::collections::HashSet;

use crate::BitBuffer;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SchemeCreationError {
    /// The were an odd number of target bits
    #[error("The were an odd number of target bits")]
    OddTargets,
    /// The list of targets was empty
    #[error("The list of targets was empty")]
    EmptyTargets,
    /// The list of targets contained duplicate entries
    #[error("Index {0} occurs more than once.")]
    DuplicateIndex(usize),
    /// An index was out of bounds for the buffer
    #[error("The index {0} is out of bounds for the buffer")]
    IndexOutOfBounds(usize),
}

/// A scheme for most significant bit encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scheme<const N: usize> {
    /// The length of the buffer that the scheme was configured for.
    buffer_length: usize,
    /// The source bit that's going to be copied into targets.
    source: usize,
    /// The duplicates of `source`.
    targets: [usize; N],
}

impl<const N: usize> Scheme<N> {
    /// Create a new scheme
    pub fn new(
        buffer_length: usize,
        source: usize,
        targets: [usize; N],
    ) -> Result<Self, SchemeCreationError> {
        if targets.is_empty() {
            return Err(SchemeCreationError::EmptyTargets);
        }

        if !targets.len().is_multiple_of(2) {
            return Err(SchemeCreationError::OddTargets);
        }

        for &i in targets.iter().chain([&source]) {
            if i >= buffer_length {
                return Err(SchemeCreationError::IndexOutOfBounds(i));
            }
        }

        if let Err(non_unique) = is_unique(targets.iter().chain([&source])) {
            return Err(SchemeCreationError::DuplicateIndex(*non_unique));
        }

        Ok(Self {
            buffer_length,
            source,
            targets,
        })
    }

    /// Create a new scheme for a given buffer.
    pub fn for_buffer<B>(
        buffer: &B,
        source: usize,
        targets: [usize; N],
    ) -> Result<Self, SchemeCreationError>
    where
        B: BitBuffer,
    {
        Self::new(buffer.bit_count(), source, targets)
    }

    pub fn is_valid_for<B>(&self, buffer: &B) -> bool
    where
        B: BitBuffer,
    {
        self.buffer_length == buffer.bit_count()
    }
}

/// The scheme is not configured for a buffer of this length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("The scheme is not configured for a buffer of this length")]
pub struct InvalidSchemeError {
    pub expected_length: usize,
    pub actual_length: usize,
}

/// Duplicate a bit into a number of other bits.
///
/// The buffer is not modified when the scheme is invalid.
///
/// See module docs for details.
pub fn encode<B, const N: usize>(
    buffer: &mut B,
    scheme: Scheme<N>,
) -> Result<(), InvalidSchemeError>
where
    B: BitBuffer,
{
    if !scheme.is_valid_for(buffer) {
        return Err(InvalidSchemeError {
            expected_length: scheme.buffer_length,
            actual_length: buffer.bit_count(),
        });
    }

    let source_is_1 = buffer.is_1(scheme.source);

    for &target in &scheme.targets {
        if source_is_1 {
            buffer.set_1(target);
        } else {
            buffer.set_0(target);
        }
    }

    Ok(())
}

/// Decode the buffer with this scheme.
///
/// The buffer is not modified when the scheme is invalid.
///
/// See module docs for details.
pub fn decode<B, const N: usize>(
    buffer: &mut B,
    scheme: Scheme<N>,
) -> Result<(), InvalidSchemeError>
where
    B: BitBuffer,
{
    if !scheme.is_valid_for(buffer) {
        return Err(InvalidSchemeError {
            expected_length: scheme.buffer_length,
            actual_length: buffer.bit_count(),
        });
    }

    let indices = scheme.targets.iter().copied().chain([scheme.source]);

    let mut is_1 = 0;
    let mut is_0 = 0;

    for i in indices {
        if buffer.is_1(i) { is_1 += 1 } else { is_0 += 1 }
    }

    // Cannot be the same for a valid scheme because the number of `indices` is always odd.
    #[cfg(debug_assertions)]
    if (is_1 != 0) && is_0 != 0 {
        debug_assert_ne!(is_1, is_0);
    }

    if is_1 > is_0 {
        buffer.set_1(scheme.source);
    } else {
        buffer.set_0(scheme.source);
    }

    Ok(())
}

fn is_unique<T>(iter: impl IntoIterator<Item = T>) -> Result<(), T>
where
    T: std::hash::Hash + std::cmp::Eq + Copy,
{
    let mut seen = HashSet::new();

    for item in iter {
        if !seen.insert(item) {
            return Err(item);
        }
    }

    Ok(())
}
