//! This is an encoding which computes the parity of some source bits and stores
//! the result in another bit, overwriting the previous value.
//!
//! During decoding the bits used for storing parity are always zeroed. If a
//! parity check fails then all mapped bits are zeroed.
//!
//! If a parity check fails during decoding then all related bits are set to zero.

#[cfg(test)]
mod tests;

use crate::BitBuffer;

#[derive(Debug, PartialEq, Eq, Clone, Copy, thiserror::Error)]

/// An error for when an invalid scheme is used during [`encode`] or [`decode`].
pub enum SchemeError {
    /// 0 source bits were specified.
    #[error("0 source bits were specified")]
    SourceEmpty,
    /// A source bit is out of bounds.
    #[error("A source bit is out of bounds")]
    SourceOutOfBounds,
    /// The destination bit is out of bounds.
    #[error("The destination bit is out of bounds")]
    DestinationOutOfBounds,
    /// The destination bit is among the source bits.
    #[error("The destination bit is among the source bits")]
    DestAmongSource,
}

fn validate_scheme<I, B>(
    source_bits: I,
    destination_bit: usize,
    buffer: &B,
) -> Result<(), SchemeError>
where
    I: IntoIterator<Item = usize>,
    B: BitBuffer,
{
    let length = buffer.bit_count();

    let mut iter_length = 0;
    for index in source_bits {
        iter_length += 1;

        if index == destination_bit {
            return Err(SchemeError::DestAmongSource);
        }

        if index >= length {
            return Err(SchemeError::SourceOutOfBounds);
        }
    }

    if iter_length == 0 {
        return Err(SchemeError::SourceEmpty);
    }

    if destination_bit >= length {
        return Err(SchemeError::DestinationOutOfBounds);
    }

    Ok(())
}

/// Set the `destination_bit` to make the total parity in `source_bits` +
/// `destination_bit` even.
///
/// See module docs for more information.
pub fn encode<I, B>(
    source_bits: I,
    destination_bit: usize,
    buffer: &mut B,
) -> Result<(), SchemeError>
where
    I: IntoIterator<Item = usize> + Clone,
    B: BitBuffer,
{
    validate_scheme(source_bits.clone(), destination_bit, buffer)?;

    let mut ones: usize = 0;
    for index in source_bits {
        if buffer.is_1(index) {
            ones += 1;
        }
    }

    if ones.is_multiple_of(2) {
        buffer.set_0(destination_bit);
    } else {
        buffer.set_1(destination_bit);
    }

    Ok(())
}

/// Decode the output of [`encode`].
///
/// See module docs for more information.
pub fn decode<I, B>(
    source_bits: I,
    destination_bit: usize,
    buffer: &mut B,
) -> Result<(), SchemeError>
where
    I: IntoIterator<Item = usize> + Clone,
    B: BitBuffer,
{
    validate_scheme(source_bits.clone(), destination_bit, buffer)?;

    let mut ones: usize = 0;
    for index in source_bits.clone() {
        if buffer.is_1(index) {
            ones += 1;
        }
    }

    if buffer.is_1(destination_bit) {
        ones += 1;
    }

    buffer.set_0(destination_bit);

    if ones.is_multiple_of(2) {
        return Ok(());
    }

    for index in source_bits {
        buffer.set_0(index);
    }

    Ok(())
}
