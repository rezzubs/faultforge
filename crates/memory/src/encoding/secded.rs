//! Single error correction (SEC), double error detection (DED) encoding.
//!
//! This is an extended Hamming code. In the encoded buffer:
//! - Bit 0 is the overall parity bit. It makes the number of 1 bits in the
//!   whole buffer even.
//! - Bits at power of two positions (1, 2, 4, 8, ...) are Hamming parity bits.
//! - All other positions hold the data bits in order.
//!
//! # Error semantics
//!
//! Decoding looks at two values:
//! - The syndrome: the XOR of the indices of all 1 bits. It's 0 for a valid
//!   buffer. A single flip at index `i` changes it by `i`, so a flip at bit 0
//!   doesn't change it at all.
//! - The overall parity. Every single flip, including one at bit 0, makes it
//!   odd.
//!
//! | Syndrome      | Parity | Interpretation               | Action                |
//! |---------------|--------|------------------------------|-----------------------|
//! | 0             | even   | no error                     | none                  |
//! | 0             | odd    | single error at bit 0        | corrected             |
//! | in bounds     | odd    | single error at the syndrome | corrected             |
//! | out of bounds | odd    | not a single error           | double error detected |
//! | not 0         | even   | double error                 | double error detected |
//!
//! Guarantees:
//! - Every single bit error is corrected.
//! - Every double bit error is detected. Two flips keep the parity even and
//!   two different indices never XOR to 0.
//! - If a double error is detected, the buffer is left as is and decoded in
//!   its faulty state.
//!
//! There are no guarantees for three or more errors:
//! - An odd number of errors looks like a single error. It's either
//!   "corrected" at the wrong position, which silently corrupts the data, or
//!   detected if the syndrome is out of bounds.
//! - An even number of errors is detected unless the syndrome is 0. Then the
//!   buffer looks valid, e.g. after flips at 1, 2, 4 and 7.

#[cfg(test)]
mod tests;

use crate::{BitBuffer, Limited};

/// Check if an index is reserved for parity (a power of two or 0).
#[must_use]
fn is_parity_index(i: usize) -> bool {
    if i == 0 {
        return true;
    }
    (i & (i - 1)) == 0
}

/// Get corresponding number of bits required for error correction for a buffer
/// with length `source_length`.
///
/// Returns None if `data_bit_count` is 0.
#[must_use]
fn error_correction_bit_count(data_bit_count: usize) -> Option<usize> {
    if data_bit_count == 0 {
        return None;
    }

    // NOTE: 2 is the minimum possible number of parity bits.
    let mut parity_bits = 2u32;
    loop {
        let max_data_bits_per_parity_bits = (2u32.pow(parity_bits) - parity_bits - 1) as usize;
        if data_bit_count <= max_data_bits_per_parity_bits {
            return Some(parity_bits as usize);
        }
        parity_bits += 1
    }
}

/// Get the number of total bits that are required to encode a buffer with a
/// given length.
///
/// Returns None if `data_bit_count` is 0.
#[must_use]
pub fn encoded_bit_count(data_bit_count: usize) -> Option<usize> {
    // +1 for the overall parity bit at index 0.
    Some(data_bit_count + error_correction_bit_count(data_bit_count)? + 1)
}

/// Compute the syndrome of an encoded buffer: the XOR of the indices of all 1
/// bits.
///
/// See module docs for how it's interpreted.
fn syndrome<T>(buffer: &T) -> usize
where
    T: BitBuffer,
{
    use std::ops::BitXor;

    buffer
        .bits()
        .enumerate()
        .filter_map(|(i, bit)| bit.is_1().then_some(i))
        .fold(0, |acc, x| acc.bitxor(x))
}

/// Whether decoding detected a double error.
///
/// See module docs for error semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum DoubleErrorDetection {
    /// No double error was detected. If there was a single bit error, it was
    /// corrected.
    NotDetected,
    /// A double error was detected. The buffer was left as is.
    Detected,
}

impl DoubleErrorDetection {
    /// Check if a double error was detected.
    #[must_use]
    pub fn is_detected(self) -> bool {
        self == DoubleErrorDetection::Detected
    }
}

/// Correct any single bit flip error in an encoded buffer.
///
/// See module docs for error semantics.
fn correct_error<T>(buffer: &mut T) -> DoubleErrorDetection
where
    T: BitBuffer,
{
    // An odd parity means an odd number of flips. It's assumed to be a single
    // flip, located by the syndrome. See the table in the module docs.
    match (syndrome(buffer), buffer.total_parity_is_even()) {
        (0, true) => DoubleErrorDetection::NotDetected,
        // The syndrome shows an error but the parity says the number of flips
        // is even, so it's at least two. It can't be corrected.
        (_, true) => DoubleErrorDetection::Detected,
        // A single flip always produces a syndrome within the buffer, so this
        // must be a multi-bit error. It's possible because the buffer can be
        // shorter than the largest index its parity bits can address.
        (error_index, false) if error_index >= buffer.bit_count() => DoubleErrorDetection::Detected,
        // A single flip at the syndrome. A syndrome of 0 points at the overall
        // parity bit.
        (error_index, false) => {
            buffer.flip_bit(error_index);
            DoubleErrorDetection::NotDetected
        }
    }
}

/// An error for [`encode_into`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EncodeIntoError {
    /// The source buffer is empty.
    #[error("cannot encode an empty source buffer")]
    SourceEmpty,
    /// The destination buffer does not have the correct size.
    #[error("the destination buffer should have {expected} bits based on the source, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },
}

/// Encode the `source` buffer as a hamming code inside the `destination`
/// buffer.
///
/// The destination buffer needs to have an exact size to fit the data + parity
/// bits. This can be computed using [`encoded_bit_count`]. Use [`Limited`] to
/// restrict the size of the destination buffer if necessary. Every bit of the
/// destination is overwritten, so its previous contents don't matter.
///
/// See [`encode`] for an automatically allocated destination buffer.
///
/// See module docs for error semantics.
pub fn encode_into<S, D>(source: &S, destination: &mut D) -> Result<(), EncodeIntoError>
where
    S: BitBuffer,
    D: BitBuffer,
{
    let error_correction_bit_count =
        error_correction_bit_count(source.bit_count()).ok_or(EncodeIntoError::SourceEmpty)?;
    let encoded_bit_count = encoded_bit_count(source.bit_count())
        .expect("known to be non-zero after the previous check");

    if destination.bit_count() != encoded_bit_count {
        return Err(EncodeIntoError::LengthMismatch {
            expected: encoded_bit_count,
            actual: destination.bit_count(),
        });
    }

    // Parity positions are cleared here and filled in below.
    let mut input_index = 0;
    for output_index in 0..encoded_bit_count {
        if is_parity_index(output_index) {
            destination.set_0(output_index);
            continue;
        }

        if source.is_1(input_index) {
            destination.set_1(output_index);
        } else {
            destination.set_0(output_index);
        }

        input_index += 1;
    }

    // With all parity bits at 0, the syndrome only depends on the data bits.
    // Data bits are never at power of two positions, so setting the parity bit
    // at `2^i` for every set bit `i` of the syndrome makes the syndrome 0.
    let data_syndrome = syndrome(destination);
    for i in 0..error_correction_bit_count {
        if data_syndrome & (1 << i) != 0 {
            destination.set_1(1 << i);
        }
    }

    // Bit 0 is still 0, so setting it makes the overall parity even.
    if !destination.total_parity_is_even() {
        destination.set_1(0);
    }

    Ok(())
}

/// Encode the given buffer using SECDED encoding.
///
/// Returns `None` if the source buffer is empty.
///
/// See [`encode_into`] for custom output buffers.
///
/// See module docs for error semantics.
pub fn encode<B>(buffer: &B) -> Option<Limited<Vec<u8>>>
where
    B: BitBuffer,
{
    let encoded_bit_count = encoded_bit_count(buffer.bit_count())?;

    let mut destination = Limited::bytes(encoded_bit_count);

    if let Err(err) = encode_into(buffer, &mut destination) {
        match err {
            crate::encoding::secded::EncodeIntoError::SourceEmpty => {
                unreachable!("Checked by encoded_bit_count")
            }
            crate::encoding::secded::EncodeIntoError::LengthMismatch { .. } => {
                unreachable!("We created the buffer with the correct size")
            }
        }
    };

    Some(destination)
}

/// An error for [`decode_into`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DecodeError {
    /// The destination buffer is empty.
    #[error("the destination buffer cannot be empty")]
    DestEmpty,
    /// The encoded buffer does not have the correct size.
    #[error(
        "the encoded buffer should have {expected} bits based on the destination, got {actual}"
    )]
    LengthMismatch { expected: usize, actual: usize },
}

/// Decode an encoded buffer to the original representation.
///
/// The decoding process will correct single-bit-errors which will modify the
/// source buffer. If an error is returned, neither buffer is modified.
///
/// See module docs for error semantics.
pub fn decode_into<S, D>(source: &mut S, dest: &mut D) -> Result<DoubleErrorDetection, DecodeError>
where
    S: BitBuffer + std::fmt::Debug,
    D: BitBuffer,
{
    let encoded_bit_count = encoded_bit_count(dest.bit_count()).ok_or(DecodeError::DestEmpty)?;

    if source.bit_count() != encoded_bit_count {
        return Err(DecodeError::LengthMismatch {
            expected: encoded_bit_count,
            actual: source.bit_count(),
        });
    }

    // Only after the checks, so `source` is left as is on error.
    let double_error_detection = correct_error(source);

    let mut output_index = 0;
    for input_index in 3..encoded_bit_count {
        if is_parity_index(input_index) {
            continue;
        }

        if source.is_1(input_index) {
            dest.set_1(output_index);
        } else {
            dest.set_0(output_index);
        }

        output_index += 1;
    }

    Ok(double_error_detection)
}
