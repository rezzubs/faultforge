//! An encoding scheme based which embedds parity bits into a [`BitBuffer`],
//! overwriting existing bits.
//!
//! If a parity check fails during decoding then all related bits are set to zero.

use crate::prelude::*;

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
    let length = buffer.bits_count();

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

/// Set the `destination_bit` to make the total parity in `source_bits` + `destination_bit` even.
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

    if ones % 2 == 0 {
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

    if ones % 2 == 0 {
        return Ok(());
    }

    for index in source_bits {
        buffer.set_0(index);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    mod u16_tests {
        use super::*;

        #[test]
        fn encode_even_parity() {
            let source = [0, 1, 2, 3];
            let dest = 15;

            let mut buffer: u16 = 0b0000_0000_0000_0000;
            encode(source, dest, &mut buffer).unwrap();
            assert_eq!(buffer, 0b0000_0000_0000_0000);

            let mut buffer: u16 = 0b0000_0000_0000_0101;
            encode(source, dest, &mut buffer).unwrap();
            assert_eq!(buffer, 0b0000_0000_0000_0101);

            let mut buffer: u16 = 0b0000_0000_0000_1111;
            encode(source, dest, &mut buffer).unwrap();
            assert_eq!(buffer, 0b0000_0000_0000_1111);
        }

        #[test]
        fn encode_odd_parity() {
            let source = [0, 1, 2, 3];
            let dest = 15;

            let mut buffer: u16 = 0b0000_0000_0000_0001;
            encode(source, dest, &mut buffer).unwrap();
            assert_eq!(buffer, 0b1000_0000_0000_0001);

            let mut buffer: u16 = 0b0000_0000_0000_0111;
            encode(source, dest, &mut buffer).unwrap();
            assert_eq!(buffer, 0b1000_0000_0000_0111);
        }

        #[test]
        fn encode_different_source_bits() {
            let mut buffer: u16 = 0b0000_0001_0000_0001;
            encode(vec![0, 8], 15, &mut buffer).unwrap();
            assert!(buffer.is_0(15), "Parity should be even (2 ones)");

            let mut buffer: u16 = 0b0100_1000_0010_0000;
            encode(vec![5, 11, 14], 15, &mut buffer).unwrap();
            assert!(buffer.is_1(15), "Parity should be odd (3 ones)");
        }

        #[test]
        fn encode_overwrites_existing_parity_bit() {
            let source = [0, 1, 2, 3];
            let dest = 15;

            let mut buffer: u16 = 0b1000_0000_0000_0000;
            encode(source, dest, &mut buffer).unwrap();
            assert!(buffer.is_0(dest), "Parity bit should be overwritten to 0");

            let mut buffer: u16 = 0b0000_0000_0000_0001;
            encode(source, dest, &mut buffer).unwrap();
            assert!(buffer.is_1(dest), "Parity bit should be overwritten to 1");
        }

        #[test]
        fn decode_valid_parity_preserves_data() {
            let source = [0, 1, 2, 3];
            let dest = 15;

            let mut buffer: u16 = 0b0000_0000_0000_0101;
            decode(source, dest, &mut buffer).unwrap();
            assert_eq!(
                buffer, 0b0000_0000_0000_0101,
                "Valid parity should preserve data"
            );

            let mut buffer: u16 = 0b1000_0000_0000_0001;
            decode(source, dest, &mut buffer).unwrap();
            assert_eq!(
                buffer, 0b0000_0000_0000_0001,
                "Valid parity should preserve data and zero the parity bits"
            );
        }

        #[test]
        fn decode_invalid_parity_zeros_data() {
            let source = [0, 1, 2, 3];
            let dest = 15;

            // Even parity but marked as odd.
            let mut buffer: u16 = 0b1000_0000_0000_0000;
            decode(source, dest, &mut buffer).unwrap();
            assert_eq!(
                buffer, 0b0000_0000_0000_0000,
                "Parity bit should be preserved"
            );

            // Invalid parity with data bits set
            let mut buffer: u16 = 0b0000_0000_0000_0001;
            decode(source, dest, &mut buffer).unwrap();
            assert_eq!(
                buffer, 0b0000_0000_0000_0000,
                "Source bits should be zeroed"
            );

            // Invalid parity with multiple data bits
            let mut buffer: u16 = 0b1000_0000_0000_1111;
            decode(source, dest, &mut buffer).unwrap();
            assert_eq!(
                buffer, 0b0000_0000_0000_0000,
                "Source bits should be zeroed, parity preserved"
            );
        }

        #[test]
        fn decode_non_contiguous_bits() {
            let source = [0, 8];
            let dest = 15;
            let mut buffer: u16 = 0b0000_0001_0000_0001; // bits 0 and 8 set (even)
            decode(source, dest, &mut buffer).unwrap();
            assert_eq!(
                buffer, 0b0000_0001_0000_0001,
                "Valid parity should preserve data"
            );

            let mut buffer: u16 = 0b1000_0001_0000_0001; // bits 0, 8 set + parity 15 = odd
            decode(source, dest, &mut buffer).unwrap();
            assert_eq!(
                buffer, 0b0000_0000_0000_0000,
                "Invalid parity should zero source bits"
            );
        }

        #[test]
        fn encode_decode() {
            let source = [0, 1, 2, 3, 4, 5, 6, 7];
            let dest = 15;

            let mut buffer: u16 = 0b0000_0000_0101_0101;
            let original = buffer;

            encode(source, dest, &mut buffer).unwrap();

            decode(source, dest, &mut buffer).unwrap();
            assert_eq!(buffer, original, "Round trip should preserve data");

            buffer.flip_bit(dest);
            decode(source, dest, &mut buffer).unwrap();

            assert_eq!(
                buffer, 0b0000_0000_0000_0000,
                "Invalid parity should zero source bits"
            );
        }

        #[test]
        fn two_chunks() {
            let src1 = 0..3;
            let src2 = 3..6;

            let dest1 = 14;
            let dest2 = 15;

            // src1 has odd parity (all ones)
            // src2 has even parity (2/3 ones)
            let mut buffer: u16 = 0b0000_0000_0011_0111;

            encode(src1.clone(), dest1, &mut buffer).unwrap();
            encode(src2.clone(), dest2, &mut buffer).unwrap();

            assert_eq!(buffer, 0b0100_0000_0011_0111);

            let mut decoded = buffer;

            decode(src1.clone(), dest1, &mut decoded).unwrap();
            decode(src2.clone(), dest2, &mut decoded).unwrap();

            assert_eq!(decoded, 0b0000_0000_0011_0111);

            let mut decoded = buffer;

            decoded.flip_bit(0);

            decode(src1, dest1, &mut decoded).unwrap();
            decode(src2, dest2, &mut decoded).unwrap();

            println!("{:b} {:b}", buffer, decoded);

            assert_eq!(
                decoded, 0b0000_0000_0011_0000,
                "The 3 lowest bits should be zeroed due to being invalid"
            );
        }

        #[test]
        fn source_out_of_bounds() {
            let mut buffer: u16 = 0;

            let result = encode([0, 1, 16], 15, &mut buffer);
            assert!(matches!(result, Err(SchemeError::SourceOutOfBounds)));

            let result = encode([100], 15, &mut buffer);
            assert!(matches!(result, Err(SchemeError::SourceOutOfBounds)));

            let result = decode([0, 1, 16], 15, &mut buffer);
            assert!(matches!(result, Err(SchemeError::SourceOutOfBounds)));
        }

        #[test]
        fn destination_out_of_bounds() {
            let mut buffer: u16 = 0;

            let source = [0, 1, 2];

            let result = encode(source, 16, &mut buffer);
            assert!(matches!(result, Err(SchemeError::DestinationOutOfBounds)));

            let result = encode(source, 100, &mut buffer);
            assert!(matches!(result, Err(SchemeError::DestinationOutOfBounds)));

            let result = decode(source, 16, &mut buffer);
            assert!(matches!(result, Err(SchemeError::DestinationOutOfBounds)));
        }

        #[test]
        fn empty_source_bits() {
            let mut buf = 0u16;

            let result = encode([], 15, &mut buf);
            assert!(matches!(result, Err(SchemeError::SourceEmpty)));

            let result = decode([], 15, &mut buf);
            assert!(matches!(result, Err(SchemeError::SourceEmpty)));
        }

        #[test]
        fn parity_bit_in_source_bits() {
            let mut buffer: u16 = 0b0000_0000_0000_0001; // bit 0 set

            let source = [0, 7];

            assert_eq!(
                encode(source, 7, &mut buffer),
                Err(SchemeError::DestAmongSource)
            );
            assert_eq!(
                encode(source, 0, &mut buffer),
                Err(SchemeError::DestAmongSource)
            );
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn encode_decode_roundtrip(
            (mut buffer, (source_bits, target_bit)) in prop::collection::vec(any::<u8>(), 1..1024)
            .prop_flat_map(|buffer| (
                Just(buffer.clone()),
                prop::collection::hash_set(0usize..buffer.len(), 1..buffer.len())
                .prop_flat_map(move |source_bits| (
                    Just(source_bits.clone()),
                    (0..buffer.len()).prop_filter(
                        "target_bit not in source_bits",
                        move |target_bit| !source_bits.contains(target_bit)
                    )
                )),
            )),
        ) {
            let mut mapped = buffer.clone();
            mapped.set_0(target_bit);

            let mut ones = 0;
            for b in source_bits.iter() {
                if buffer.is_1(*b) {
                    ones += 1;
                }
            }

            encode(source_bits.clone(), target_bit, &mut buffer).unwrap();

            if ones % 2 == 0 {
                assert!(buffer.is_0(target_bit))
            } else {
                assert!(buffer.is_1(target_bit))
            }

            decode(source_bits, target_bit, &mut buffer).unwrap();

            assert_eq!(mapped, buffer)
        }
    }
}
