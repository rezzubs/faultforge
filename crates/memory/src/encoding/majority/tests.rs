use super::*;

#[test]
fn invalid_schemes() {
    let buffer = 0u32;

    assert_eq!(
        Scheme::for_buffer(&buffer, 32, [0, 1]),
        Err(SchemeCreationError::IndexOutOfBounds(32))
    );
    assert_eq!(
        Scheme::for_buffer(&buffer, 31, [0, 32]),
        Err(SchemeCreationError::IndexOutOfBounds(32))
    );
    assert_eq!(
        Scheme::for_buffer(&buffer, 31, [0, 32]),
        Err(SchemeCreationError::IndexOutOfBounds(32))
    );

    assert_eq!(
        Scheme::for_buffer(&buffer, 31, [31, 1]),
        Err(SchemeCreationError::DuplicateIndex(31))
    );
    assert_eq!(
        Scheme::for_buffer(&buffer, 31, [0, 0]),
        Err(SchemeCreationError::DuplicateIndex(0))
    );

    assert_eq!(
        Scheme::for_buffer(&buffer, 31, [0, 1, 2]),
        Err(SchemeCreationError::OddTargets)
    );
    assert_eq!(
        Scheme::for_buffer(&buffer, 31, [0]),
        Err(SchemeCreationError::OddTargets)
    );

    assert_eq!(
        Scheme::for_buffer(&buffer, 31, []),
        Err(SchemeCreationError::EmptyTargets)
    );

    let mut buffer = 0u32;
    assert_eq!(
        encode(&mut buffer, Scheme::new(33, 31, [32, 0]).unwrap()),
        Err(InvalidSchemeError {
            expected_length: 33,
            actual_length: 32
        })
    );
}

#[test]
fn encode_decode_2copies() {
    let buffers = (0..=u16::MAX).collect::<Vec<_>>();

    let source_bit = 15;
    let scheme = Scheme::for_buffer(&0u16, source_bit, [0, 1]).unwrap();

    for original in buffers {
        let mut encoded = original;

        encode(&mut encoded, scheme).unwrap();

        let mut decoded = encoded;

        // Decode without faults
        decode(&mut decoded, scheme).unwrap();

        assert_eq!(encoded, decoded);

        for fault_index in [0, 1, source_bit] {
            let mut faulty = encoded;

            faulty.flip_bit(fault_index);
            assert_ne!(faulty, encoded);

            decode(&mut faulty, scheme).unwrap();

            assert_eq!(faulty.is_1(source_bit), encoded.is_1(source_bit));
        }
    }
}

#[test]
fn encode_decode_4copies() {
    let buffers = (0..=u16::MAX).collect::<Vec<_>>();

    let source_bit = 15;
    let scheme = Scheme::for_buffer(&0u16, source_bit, [0, 1, 2, 3]).unwrap();

    for original in buffers {
        let mut encoded = original;

        encode(&mut encoded, scheme).unwrap();

        let mut decoded = encoded;

        // Decode without faults
        decode(&mut decoded, scheme).unwrap();

        assert_eq!(encoded, decoded);

        let faults = [0, 1, 2, 3, source_bit];

        for (&fault1, fault2) in faults.iter().zip(faults) {
            if fault1 == fault2 {
                continue;
            }
            let mut faulty = encoded;

            faulty.flip_bit(fault1);
            faulty.flip_bit(fault2);

            assert_ne!(faulty, encoded);

            decode(&mut faulty, scheme).unwrap();

            assert_eq!(faulty.is_1(source_bit), encoded.is_1(source_bit));
        }
    }
}
