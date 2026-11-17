use super::*;
use proptest::prelude::*;
use std::ops::RangeInclusive;

#[test]
fn par_i() {
    assert!(is_parity_index(0));
    assert!(is_parity_index(1));
    assert!(is_parity_index(2));
    assert!(!is_parity_index(3));
    assert!(is_parity_index(4));
    assert!(!is_parity_index(5));
    assert!(!is_parity_index(6));
    assert!(!is_parity_index(7));
    assert!(is_parity_index(8));
    assert!(!is_parity_index(9));
    assert!(!is_parity_index(10));
    assert!(!is_parity_index(11));
    assert!(!is_parity_index(12));
    assert!(!is_parity_index(13));
    assert!(!is_parity_index(14));
    assert!(!is_parity_index(15));
    assert!(is_parity_index(16));
}

#[test]
fn bit_count() {
    assert_eq!(error_correction_bit_count(1).unwrap(), 2);
    for i in 3..=4 {
        assert_eq!(error_correction_bit_count(i).unwrap(), 3);
    }
    for i in 5..=11 {
        assert_eq!(error_correction_bit_count(i).unwrap(), 4);
    }
    for i in 12..=26 {
        assert_eq!(error_correction_bit_count(i).unwrap(), 5);
    }
    for i in 27..=57 {
        assert_eq!(error_correction_bit_count(i).unwrap(), 6);
    }
    for i in 58..=120 {
        assert_eq!(error_correction_bit_count(i).unwrap(), 7);
    }
    for i in 121..=247 {
        assert_eq!(error_correction_bit_count(i).unwrap(), 8);
    }
    for i in 248..=502 {
        assert_eq!(error_correction_bit_count(i).unwrap(), 9);
    }
    assert_eq!(error_correction_bit_count(512).unwrap(), 10);
}

/// Map an encoded bit position to the index of the data bit stored there.
///
/// Returns `None` for parity positions.
fn data_index(position: usize) -> Option<usize> {
    // Below a data position there are `ilog2(position) + 2` parity positions:
    // index 0 and every power of two.
    let parity_positions_below =
        usize::try_from(position.checked_ilog2()?).expect("the log of a usize fits in a usize") + 2;
    (!is_parity_index(position)).then(|| position - parity_positions_below)
}

/// Create a buffer of `bit_count` data bits with an irregular bit pattern.
fn data_buffer(bit_count: usize) -> Limited<Vec<u8>> {
    let mut buffer = Limited::bytes(bit_count);
    for i in 0..bit_count {
        if (i * 7 + i / 3) % 5 < 2 {
            buffer.set_1(i);
        }
    }
    buffer
}

/// Turn two random indices into two distinct positions in `0..bit_count`.
fn distinct_positions(
    first: prop::sample::Index,
    second: prop::sample::Index,
    bit_count: usize,
) -> [usize; 2] {
    let first = first.index(bit_count);
    let second = second.index(bit_count - 1);
    // Skip over `first` so the positions never collide.
    let second = if second >= first { second + 1 } else { second };
    [first, second]
}

/// Check the decoding guarantees for up to two distinct faults at encoded
/// bit positions.
///
/// - Every encoded buffer has a syndrome of 0 and an even parity.
/// - No faults: nothing is detected and the data is intact.
/// - One fault anywhere, including the overall parity bit: it's corrected in
///   the encoded buffer, nothing is detected and the data is intact.
/// - Two faults anywhere: a double error is detected, the encoded buffer is
///   left as is and the data is decoded as stored.
///
/// There are no guarantees for three or more faults.
///
/// Data is compared by bits because faults can turn floats into NaN.
fn check_decode<B>(data: &B, faults: &[usize])
where
    B: BitBuffer + Clone,
{
    let encoded = encode(data).expect("the data is never empty");
    assert_eq!(syndrome(&encoded), 0);
    assert!(encoded.total_parity_is_even());

    let mut faulty = encoded.clone();
    for &fault in faults {
        faulty.flip_bit(fault);
    }

    // Start from the inverted data so a decode that doesn't write every bit
    // can't pass by accident.
    let mut decoded = data.clone();
    for i in 0..decoded.bit_count() {
        decoded.flip_bit(i);
    }

    let mut stored = faulty.clone();
    let double_error_detection = decode_into(&mut stored, &mut decoded).unwrap();

    match faults {
        [] | [_] => {
            assert_eq!(double_error_detection, DoubleErrorDetection::NotDetected);
            assert_eq!(stored, encoded);
            assert_eq!(decoded.bit_string(), data.bit_string());
        }
        [_, _] => {
            assert_eq!(double_error_detection, DoubleErrorDetection::Detected);
            assert_eq!(stored, faulty);

            let mut expected = data.clone();
            for &fault in faults {
                if let Some(data_index) = data_index(fault) {
                    expected.flip_bit(data_index);
                }
            }
            assert_eq!(decoded.bit_string(), expected.bit_string());
        }
        _ => panic!("there are no guarantees for three or more faults"),
    }
}

#[test]
fn data_index_skips_parity_positions() {
    let data_positions = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17];
    for (expected, position) in data_positions.into_iter().enumerate() {
        assert_eq!(data_index(position), Some(expected));
    }
    for position in [0, 1, 2, 4, 8, 16, 32] {
        assert_eq!(data_index(position), None);
    }
}

#[test]
fn every_single_and_double_fault() {
    for data_bit_count in 1..=72 {
        let data = data_buffer(data_bit_count);
        let encoded_bit_count = encoded_bit_count(data_bit_count).unwrap();

        check_decode(&data, &[]);
        for first in 0..encoded_bit_count {
            check_decode(&data, &[first]);
            for second in first + 1..encoded_bit_count {
                check_decode(&data, &[first, second]);
            }
        }
    }
}

#[test]
fn encode_into_overwrites_destination() {
    for data_bit_count in 1..=72 {
        let data = data_buffer(data_bit_count);
        let encoded_bit_count = encoded_bit_count(data_bit_count).unwrap();

        let mut destination = Limited::bytes(encoded_bit_count);
        for i in 0..encoded_bit_count {
            destination.set_1(i);
        }
        encode_into(&data, &mut destination).unwrap();

        assert_eq!(destination, encode(&data).unwrap());
    }
}

#[test]
fn decode_into_leaves_buffers_as_is_on_error() {
    let data = data_buffer(16);
    let mut faulty = encode(&data).unwrap();
    // A single error that a successful decode would correct.
    faulty.flip_bit(5);

    let mut source = faulty.clone();
    let mut dest = data_buffer(17);
    assert!(matches!(
        decode_into(&mut source, &mut dest),
        Err(DecodeError::LengthMismatch { .. })
    ));
    assert_eq!(source, faulty);
    assert_eq!(dest, data_buffer(17));

    let mut dest = Limited::bytes(0);
    assert_eq!(
        decode_into(&mut source, &mut dest),
        Err(DecodeError::DestEmpty)
    );
    assert_eq!(source, faulty);
}

const RANGE: RangeInclusive<usize> = 1..=512;

proptest! {
    #[test]
    fn u8_zero_faults(buf in prop::collection::vec(any::<u8>(), RANGE)) {
        check_decode(&buf, &[]);
    }

    #[test]
    fn u8_single_fault(
        buf in prop::collection::vec(any::<u8>(), RANGE),
        fault in any::<prop::sample::Index>(),
    ) {
        let encoded_bit_count = encoded_bit_count(buf.bit_count()).unwrap();
        check_decode(&buf, &[fault.index(encoded_bit_count)]);
    }

    #[test]
    fn u8_two_faults(
        buf in prop::collection::vec(any::<u8>(), RANGE),
        first in any::<prop::sample::Index>(),
        second in any::<prop::sample::Index>(),
    ) {
        let encoded_bit_count = encoded_bit_count(buf.bit_count()).unwrap();
        check_decode(&buf, &distinct_positions(first, second, encoded_bit_count));
    }

    #[test]
    fn f32_zero_faults(buf in prop::collection::vec(any::<f32>(), RANGE)) {
        check_decode(&buf, &[]);
    }

    #[test]
    fn f32_single_fault(
        buf in prop::collection::vec(any::<f32>(), RANGE),
        fault in any::<prop::sample::Index>(),
    ) {
        let encoded_bit_count = encoded_bit_count(buf.bit_count()).unwrap();
        check_decode(&buf, &[fault.index(encoded_bit_count)]);
    }

    #[test]
    fn f32_two_faults(
        buf in prop::collection::vec(any::<f32>(), RANGE),
        first in any::<prop::sample::Index>(),
        second in any::<prop::sample::Index>(),
    ) {
        let encoded_bit_count = encoded_bit_count(buf.bit_count()).unwrap();
        check_decode(&buf, &distinct_positions(first, second, encoded_bit_count));
    }
}
