//! Single error correction (SEC), double error detection (DED) encoding.
//!
//! This scheme embeds parity bits in power of two positions and shifts all
//! other bits accordingly. Bit zero is used as a global parity check for the
//! whole buffer.
//!
//! # Error semantics
//!
//! This scheme can recover from all single-bit errors and can detect any
//! multiple of two number of errors (commonly called DED). If DED is triggered
//! the buffer will not be altered and will be decoded in the faulty state as
//! is.
//!
//! Flips in bit zero will always cause a double error detection but will not
//! corrupt the data.
//!
//! Odd numbers of faults other than 1 will not trigger DED and will trigger an
//! invalid correction. There is an exception for fault patterns which appear
//! like a single out of bounds error to the algorithm; these will still trigger
//! DED because that case is only possible for multi-bit errors. This typically
//! happens if an odd number of high bits are flipped simultaneously.

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
    // +1 for the 0th double error detection bit.
    Some(data_bit_count + error_correction_bit_count(data_bit_count)? + 1)
}

/// Get the index of a flipped bit in an encoded buffer in case of a single bit
/// flip.
///
/// 0 marks a successful case. This means it's not possible to correct a flip on
/// bit 0. Flips on bit 0 will always trigger a double error detection.
fn error_index<T>(buffer: &T) -> usize
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

/// Correct any single bit flip error in an encoded buffer.
///
/// A return value of `false` signifies a double error detection. See module
/// docs for error semantics.
fn correct_error<T>(buffer: &mut T) -> bool
where
    T: BitBuffer,
{
    // The parity check provides double error detection. During encoding the
    // 0th bit is set so the parity across all bits is even. For single bit
    // errors we expect an odd parity.
    match (error_index(buffer), buffer.total_parity_is_even()) {
        // We couldn't find an error location and the total parity has not
        // changed.
        (0, true) => true,
        // We're not detecting any errors but the parity changed therefore
        // there must be multiple errors which are canceling each other out
        (0, false) => false,
        // We found an error location but the parity didn't change therefore
        // there must be two or more errors.
        (_, true) => false,
        // If only one of our protected bits flipped it will cause the error
        // index to be in our protected range.
        (e, false) if e >= buffer.bit_count() => false,
        // We found an error location and the parity changed which means we
        // either have 1 error which we will attempt to correct or an
        // undetectable odd number of errors.
        (e, false) => {
            buffer.flip_bit(e);
            true
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
/// restrict the size of the destination buffer if necessary.
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

    let mut input_index = 0;
    // Starting from 3 because 0, 1, 2 are all reserved for parity.
    for output_index in 3..encoded_bit_count {
        if is_parity_index(output_index) {
            continue;
        }

        if source.is_1(input_index) {
            destination.set_1(output_index);
        } else {
            destination.set_0(output_index);
        }

        input_index += 1;
    }

    let bits_to_toggle =
        u64::try_from(error_index(destination)).expect("error index out of bounds");

    for i in 0..error_correction_bit_count {
        let parity_bit = 1 << i;

        if bits_to_toggle.is_1(i) {
            destination.flip_bit(parity_bit);
        }
    }

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
/// source buffer.
///
/// `false` represents a double error detection.
///
/// See module docs for error semantics.
pub fn decode_into<S, D>(source: &mut S, dest: &mut D) -> Result<bool, DecodeError>
where
    S: BitBuffer + std::fmt::Debug,
    D: BitBuffer,
{
    let success = correct_error(source);

    let encoded_bit_count = encoded_bit_count(dest.bit_count()).ok_or(DecodeError::DestEmpty)?;

    if source.bit_count() != encoded_bit_count {
        return Err(DecodeError::LengthMismatch {
            expected: encoded_bit_count,
            actual: source.bit_count(),
        });
    }

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

    Ok(success)
}
