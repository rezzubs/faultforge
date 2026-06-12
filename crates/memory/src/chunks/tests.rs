use super::*;
use crate::CopyIntoResult;
use proptest::prelude::*;
use std::ops::RangeInclusive;

#[test]
fn dyn_chunks_creation_and_restore() {
    let source = [0xffffu16; 3];
    let chunks = DynChunks::from_buffer(&source, 7).unwrap();
    // The original is 6 bytes -> 48 bits long;
    // The chunk size is 7 bits.
    // We need 7 chunks -> 49 (7*7) bits to store the original data.
    assert_eq!(chunks.chunk_count(), 7);
    assert_eq!(chunks.bit_count(), 49);

    let mut bytes = chunks.0.clone().into_inner().into_iter();

    for _ in 0..6 {
        assert_eq!(bytes.next().unwrap().into_inner(), vec![0b01111111])
    }

    // The last byte should store 48 - 7 * 6 = 6 bits
    assert_eq!(bytes.next().unwrap().into_inner(), vec![0b00111111]);
    assert_eq!(bytes.next(), None);

    let mut restored = [0u16; 3];
    let result = chunks.copy_into(&mut restored);
    // The result will be pending because the chunks had extra padding at the end.
    assert_eq!(result, CopyIntoResult::partial(restored.bit_count()));
    assert_eq!(restored, source);

    assert_eq!(
        chunks.bits().skip(result.units_copied).count(),
        chunks.bit_count() - source.bit_count()
    );
    assert!(
        chunks
            .bits()
            .skip(result.units_copied)
            .all(|bit| bit.is_0())
    );

    let chunks = DynChunks::from_buffer(&source, 9).unwrap();
    // The original is 6 bytes -> 48 bits long;
    // The chunk size is 9 bits.
    // We need 6 bytes -> 54 (9*6) bits to store the original data.
    assert_eq!(chunks.chunk_count(), 6);
    assert_eq!(chunks.bit_count(), 54);

    let mut bytes = chunks.0.clone().into_inner().into_iter();

    for _ in 0..5 {
        assert_eq!(bytes.next().unwrap().into_inner(), vec![0xff, 0b00000001])
    }

    // The last byte should store 48 - 9 * 5 = 3 bits
    assert_eq!(bytes.next().unwrap().into_inner(), vec![0b00000111, 0]);
    assert_eq!(bytes.next(), None);

    let mut restored = [0u16; 3];
    let result = chunks.copy_into(&mut restored);
    assert_eq!(result.units_copied, restored.bit_count());
}

#[test]
fn byte_chunks_creation_and_restore() {
    let source = [0xffffu16; 3];
    let chunks = source.to_byte_chunks(1).unwrap();
    // The original is 6 bytes -> 48 bits long;
    // The chunk size is 1 byte.
    // 6 chunks are used to store the data.
    assert_eq!(chunks.chunk_count(), 6);
    assert_eq!(chunks.bit_count(), 48);

    let mut bytes = chunks.0.clone().into_inner().into_iter();

    for _ in 0..6 {
        assert_eq!(bytes.next().unwrap(), vec![0xff])
    }
    assert_eq!(bytes.next(), None);

    let mut restored = [0u16; 3];
    let result = chunks.copy_into(&mut restored);
    assert_eq!(result, CopyIntoResult::exhausted(restored.bit_count()));

    let chunks = source.to_byte_chunks(2).unwrap();
    // The original is 6 bytes -> 48 bits long;
    // The chunk size is 2 bytes.
    // 3 chunks are used to store the data.
    assert_eq!(chunks.chunk_count(), 3);
    assert_eq!(chunks.bit_count(), 48);

    let mut bytes = chunks.0.clone().into_inner().into_iter();

    for _ in 0..3 {
        assert_eq!(bytes.next().unwrap(), vec![0xff, 0xff])
    }
    assert_eq!(bytes.next(), None);

    let mut restored = [0u16; 3];
    let result = chunks.copy_into(&mut restored);
    assert_eq!(result, CopyIntoResult::exhausted(restored.bit_count()));
}

#[test]
fn byte_chunked_encoding() {
    let source = [123.123f32, std::f32::consts::PI, 0.001, 10000.123];
    let expected_byte_count = 4 * 4;
    assert_eq!(source.byte_count(), expected_byte_count);

    let chunk_size = 16;
    let chunk_count_expected = 8;
    let expected_bytes_per_chunk = chunk_size / chunk_count_expected;
    let chunks = source.to_chunks(chunk_size).unwrap();

    match chunks {
        Chunks::Byte(ref byte_chunks) => {
            assert_eq!(byte_chunks.byte_count(), expected_byte_count);
            assert_eq!(byte_chunks.chunk_count(), chunk_count_expected);
            assert_eq!(
                byte_chunks.0.inner().first().map(|x| x.len()),
                Some(expected_bytes_per_chunk)
            );
        }
        Chunks::Dyn(dyn_chunks) => panic!("Expected byte chunks, got {:?}", dyn_chunks),
    }

    let encoded = chunks.encode_chunks();
    assert_eq!(encoded.chunk_count(), chunk_count_expected);

    {
        let non_faulty = encoded.clone();
        let (raw_decoded, ded_results) = non_faulty.decode_chunks(chunk_size).unwrap();

        for success in ded_results {
            assert!(success);
        }

        let raw_decoded = match raw_decoded {
            Chunks::Byte(byte_chunks) => byte_chunks,
            Chunks::Dyn(dyn_chunks) => panic!("Expected byte chunks, got {:?}", dyn_chunks),
        };

        let mut target = [0f32; 4];
        let result = raw_decoded.copy_into_chunked(&mut target);
        assert_eq!(result.units_copied, source.byte_count());

        assert_eq!(target, source);
    }

    for fault_index in 0..chunk_size {
        let mut faulty = encoded.clone();
        for chunk in faulty.0.inner_mut() {
            chunk.flip_bit(fault_index);
        }
        assert_ne!(encoded, faulty);

        let (raw_decoded, ded_results) = faulty.decode_chunks(chunk_size).unwrap();

        let raw_decoded = match raw_decoded {
            Chunks::Byte(byte_chunks) => byte_chunks,
            Chunks::Dyn(dyn_chunks) => panic!("Expected byte chunks, got {:?}", dyn_chunks),
        };

        if fault_index == 0 {
            assert!(ded_results.into_iter().all(|x| !x));
        } else {
            assert!(ded_results.into_iter().all(|x| x));
        }

        let mut target = [0f32; 4];
        let result = raw_decoded.copy_into_chunked(&mut target);
        assert_eq!(result.units_copied, source.byte_count());

        assert_eq!(target, source, "failed on fault_index={fault_index}");
    }
}

#[test]
fn dyn_chunked_encoding() {
    fn test_dyn(chunk_size: usize) {
        let source = [123.123f32, std::f32::consts::PI, 0.001, 10000.123];

        let chunks = source.to_chunks(chunk_size).unwrap();

        match chunks {
            Chunks::Byte(byte_chunks) => {
                panic!("expected dyn chunks, got {:?}", byte_chunks)
            }
            Chunks::Dyn(ref dyn_chunks) => dyn_chunks
                .0
                .inner()
                .iter()
                .for_each(|chunk| assert_eq!(chunk.bit_count(), chunk_size)),
        }

        let encoded = chunks.encode_chunks();

        {
            let non_faulty = encoded.clone();
            let (raw_decoded, ded_results) = non_faulty.decode_chunks(chunk_size).unwrap();

            for success in ded_results {
                assert!(success);
            }

            if let Chunks::Byte(byte_chunks) = raw_decoded {
                panic!("Expected dyn chunks, got {:?}", byte_chunks)
            };

            let mut target = [0f32; 4];
            let result = raw_decoded.copy_into(&mut target);
            if source.bit_count() == chunks.bit_count() {
                assert_eq!(result, CopyIntoResult::exhausted(source.bit_count()));
            } else {
                assert_eq!(result, CopyIntoResult::partial(source.bit_count()));
            }

            assert_eq!(target, source);
        }

        for fault_index in 0..chunk_size {
            let mut faulty = encoded.clone();
            for chunk in faulty.0.inner_mut() {
                chunk.flip_bit(fault_index);
            }
            assert_ne!(encoded, faulty);

            let (raw_decoded, ded_results) = faulty.decode_chunks(chunk_size).unwrap();

            let raw_decoded = match raw_decoded {
                Chunks::Byte(byte_chunks) => {
                    panic!("Expected dyn chunks, got {:?}", byte_chunks)
                }
                Chunks::Dyn(dyn_chunks) => dyn_chunks,
            };

            if fault_index == 0 {
                assert!(ded_results.into_iter().all(|x| !x));
            } else {
                assert!(ded_results.into_iter().all(|x| x));
            }

            let mut target = [0f32; 4];
            let result = raw_decoded.copy_into(&mut target);
            if source.bit_count() == chunks.bit_count() {
                assert_eq!(result, CopyIntoResult::exhausted(source.bit_count()));
            } else {
                assert_eq!(result, CopyIntoResult::partial(source.bit_count()));
            }

            assert_eq!(target, source, "failed on fault_index={fault_index}");
        }
    }

    for i in 1usize..=129 {
        if i.is_multiple_of(2) {
            continue;
        }
        test_dyn(i);
    }
}

#[test]
fn zero() {
    let zero_buffer = [0u32; 4];

    for i in 1..=16 {
        assert_eq!(
            zero_buffer.to_chunks(i).unwrap(),
            Chunks::zero(zero_buffer.bit_count(), i).unwrap()
        );
    }
}

#[test]
fn invalid() {
    let buffer: UniformSequence<Vec<u8>> = UniformSequence::new(vec![]).unwrap();

    for size in 0..16 {
        assert_eq!(buffer.to_chunks(size), Err(ChunksCreationError::Empty));
    }

    let buffer = 0u64;

    assert_eq!(buffer.to_chunks(0), Err(ChunksCreationError::ZeroChunksize));
    assert_eq!(
        DynChunks::from_buffer(&buffer, 0),
        Err(ChunksCreationError::ZeroChunksize)
    );
    assert_eq!(
        buffer.to_byte_chunks(0),
        Err(ChunksCreationError::ZeroChunksize)
    );
}

#[test]
#[doc(alias = "chunk_size")]
fn bits_per_chunk() {
    for i in 1..=64 {
        let chunks = [0u8; 14].to_chunks(i).unwrap();
        assert_eq!(chunks.bits_per_chunk(), i);
    }
}

const RANGE: RangeInclusive<usize> = 1..=256;

proptest! {
    #[test]
    fn pack_unpack(
        buf in (RANGE).prop_flat_map(|len| {
            prop::collection::vec(any::<u32>(), len)
        }),
        chunk_size in RANGE,
    ) {

        let chunks = buf.to_chunks(chunk_size).unwrap();

        let mut output_buffer = vec![0u32; buf.len()];
        let bits_copied = match chunks {
            Chunks::Byte(byte_chunks) => {
                byte_chunks
                    .copy_into_chunked(&mut output_buffer)
                    .units_copied
                    * 8
            }
            Chunks::Dyn(dyn_chunks) => dyn_chunks.copy_into(&mut output_buffer).units_copied,
        };
        assert_eq!(bits_copied, buf.bit_count());

        assert_eq!(output_buffer, buf);
    }

    #[test]
    fn encode_decode_u32_single_fault(
        (buf, fault) in (RANGE).prop_flat_map(|len| {
            prop::collection::vec(any::<u32>(), len).prop_flat_map(|v| {
                let fault_max = 8 * v.len();
                (Just(v), 1..fault_max)
            })
        }),
        chunk_size in RANGE,
    ) {

        let chunks = buf.to_chunks(chunk_size).unwrap();

        let mut encoded_chunks = chunks.encode_chunks();
        let encoded_chunk_size = encoded_chunks.bits_per_chunk();
        let before_faults = encoded_chunks.clone();

        encoded_chunks.flip_bit(fault);
        assert_ne!(before_faults, encoded_chunks);

        let (decoded, results) = encoded_chunks.decode_chunks(chunk_size).unwrap();

        dbg!(&results);
        for (i, success) in results.into_iter().enumerate() {
            let hit_bit_0 =  fault % encoded_chunk_size == 0;
            let hit_chunk = fault / encoded_chunk_size == i;

            if hit_bit_0 && hit_chunk {
                assert!(!success);
            } else {
                assert!(success);
            }
        }

        assert_eq!(chunks, decoded);

        let mut output_buffer = vec![0u32; buf.len()];
        let bits_copied = match decoded {
            Chunks::Byte(byte_chunks) => {
                byte_chunks
                    .copy_into_chunked(&mut output_buffer)
                    .units_copied
                    * 8
            }
            Chunks::Dyn(dyn_chunks) => dyn_chunks.copy_into(&mut output_buffer).units_copied,
        };
        assert_eq!(bits_copied, buf.bit_count());

        assert_eq!(output_buffer, buf);
    }
}
