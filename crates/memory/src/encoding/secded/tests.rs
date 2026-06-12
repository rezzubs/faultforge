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

const RANGE: RangeInclusive<usize> = 1..=512;

proptest! {
    #[test]
    fn encode_decode_u8_zero_fault(
        (buf, mut decoded) in (RANGE).prop_flat_map(|len| {
            (
                prop::collection::vec(any::<u8>(), len),
                prop::collection::vec(any::<u8>(), len),
            )
        })
    ) {
        let mut encoded = encode(&buf).unwrap();

        let success = decode_into(&mut encoded, &mut decoded).unwrap();
        assert!(success);

        assert_eq!(buf, decoded);
    }

    #[test]
    fn encode_decode_u8_single_fault(
        ((buf, fault), mut decoded) in (RANGE).prop_flat_map(|len| {
            (
                prop::collection::vec(any::<u8>(), len).prop_flat_map(|v| {
                    let fault_max = 8 * v.len();
                    (Just(v), 1..=fault_max)
                }),
                prop::collection::vec(any::<u8>(), len),
            )
        })
    ) {
        let mut encoded = encode(&buf).expect("we own the output buffer");

        encoded.flip_bit(fault);

        let success = decode_into(&mut encoded, &mut decoded).unwrap();
        assert!(success);

        assert_eq!(buf, decoded);
    }

    #[test]
    fn encode_decode_u8_bit0_fault(
        (buf, mut decoded) in (RANGE).prop_flat_map(|len| {
            (
                prop::collection::vec(any::<u8>(), len),
                prop::collection::vec(any::<u8>(), len),
            )
        })
    ) {
        let mut encoded = encode(&buf).unwrap();

        encoded.flip_bit(0);

        let success = decode_into(&mut encoded, &mut decoded).unwrap();
        // Bit zero faults always trigger a double error detection.
       assert!(!success);

        assert_eq!(buf, decoded);
    }

    #[test]
    fn encode_decode_f32_zero_fault(
        (buf, mut decoded) in (RANGE).prop_flat_map(|len| {
            (
                prop::collection::vec(any::<f32>(), len),
                prop::collection::vec(any::<f32>(), len),
            )
        })
    ) {
        let mut encoded = encode(&buf).unwrap();

        let success = decode_into(&mut encoded, &mut decoded).unwrap();
        assert!(success);

        assert_eq!(buf, decoded);
    }

    #[test]
    fn encode_decode_f32_single_fault(
        ((buf, fault), mut decoded) in (RANGE).prop_flat_map(|len| {
            (
                prop::collection::vec(any::<f32>(), len).prop_flat_map(|v| {
                    let fault_max = 8 * v.len();
                    (Just(v), 1..=fault_max)
                }),
                prop::collection::vec(any::<f32>(), len),
            )
        })
    ) {
        let mut encoded = encode(&buf).unwrap();

        encoded.flip_bit(fault);

        let success = decode_into(&mut encoded, &mut decoded).unwrap();
        assert!(success);

        assert_eq!(buf, decoded);
    }

    #[test]
    fn encode_decode_f32_two_faults(
        ((buf, (fault1, fault2)), mut decoded) in (RANGE).prop_flat_map(|len| (
            prop::collection::vec(any::<f32>(), len).prop_flat_map(|v| {
                let fault_max = 8 * v.len();
                (Just(v), (1..=fault_max, 1..=fault_max).prop_filter("must differ", |(a, b)| a != b))
            }),
            prop::collection::vec(any::<f32>(), len),
        ))
    ) {
        let encoded = encode(&buf).unwrap();

        let mut faulty = encoded.clone();
        faulty.flip_bit(fault1);
        faulty.flip_bit(fault2);

        assert_ne!(faulty, encoded);

        // A two bit flip will always be marked as unsuccessful even if in
        // reality the data matches. See [`correct_error`].
        let success = decode_into(&mut faulty, &mut decoded).unwrap();
        assert!(!success);

        // If only parity bits were hit then the original data is safe. The decoding
        // algorithm must still call a failure because there's no way to tell which bits
        // were hit.
        if is_parity_index(fault1) && is_parity_index(fault2) {
            assert_eq!(buf, decoded);
        }        }
}
