use super::*;
use crate::{
    Bit, Fault,
    encoding::secded::{decode_into, encode},
};
use proptest::prelude::*;

#[test]
fn encode_rejects_invalid_inputs() {
    assert_eq!(
        EncodedChunks::encode(&[], 8),
        Err(EncodeChunksError::SourceEmpty)
    );
    assert_eq!(
        EncodedChunks::encode(&[0u8; 4], 0),
        Err(EncodeChunksError::ZeroDataBytesPerChunk)
    );
    assert_eq!(
        EncodedChunks::encode(&[0u8; 4], usize::MAX),
        Err(EncodeChunksError::DataBytesPerChunkTooLarge)
    );
}

#[test]
fn sizes() {
    // 13 data bytes in chunks of 8 need 2 chunks of 72 encoded bits.
    let encoded = EncodedChunks::encode(&[0u8; 13], 8).unwrap();

    assert_eq!(encoded.chunk_count(), 2);
    assert_eq!(encoded.data_bytes_per_chunk(), 8);
    assert_eq!(encoded.encoded_bits_per_chunk(), 72);
    assert_eq!(encoded.encoded_bytes_per_chunk(), 9);
    assert_eq!(encoded.data_byte_count(), 13);
    assert_eq!(encoded.bit_count(), 144);
    assert_eq!(encoded.encoded_bytes.len(), 18);
}

#[test]
fn padding_is_not_addressable() {
    // 8 data bits encode into 13 bits, which leaves 3 bits of padding in the
    // second byte of every chunk.
    let mut encoded = EncodedChunks::encode(&[0u8; 3], 1).unwrap();
    assert_eq!(encoded.encoded_bits_per_chunk(), 13);

    for bit_index in 0..encoded.bit_count() {
        encoded.set_1(bit_index);
    }

    for chunk in encoded.encoded_bytes.as_chunks::<2>().0 {
        assert_eq!(*chunk, [0xff, 0x1f]);
    }
}

#[test]
#[should_panic(expected = "out of bounds")]
fn bit_access_out_of_bounds_panics() {
    let encoded = EncodedChunks::encode(&[0u8; 3], 1).unwrap();
    let _ = encoded.is_1(encoded.bit_count());
}

#[test]
fn decode_rejects_wrong_dest_length() {
    let encoded = EncodedChunks::encode(&[0u8; 3], 1).unwrap();
    let expected = DecodeChunksError::LengthMismatch {
        expected: 3,
        actual: 4,
    };

    let mut dest = [0u8; 4];
    assert_eq!(encoded.decode_into(&mut dest), Err(expected.clone()));
    assert_eq!(
        encoded.decode_into_with_double_errors(&mut dest),
        Err(expected)
    );
}

#[test]
fn decode_range_rejects_invalid_ranges() {
    let encoded = EncodedChunks::encode(&[0u8; 3], 1).unwrap();
    let invalid = |start, end| DecodeChunksError::InvalidRange {
        start,
        end,
        data_byte_count: 3,
    };

    let mut dest = [0u8; 2];
    assert_eq!(
        encoded.decode_range_into(2..4, &mut dest),
        Err(invalid(2, 4))
    );
    #[expect(clippy::reversed_empty_ranges, reason = "testing the error")]
    let reversed = 2..1;
    assert_eq!(
        encoded.decode_range_into_with_double_errors(reversed, &mut []),
        Err(invalid(2, 1))
    );
    assert_eq!(
        encoded.decode_range_into(0..1, &mut dest),
        Err(DecodeChunksError::LengthMismatch {
            expected: 1,
            actual: 2,
        })
    );

    // An empty range at the end is valid.
    assert_eq!(encoded.decode_range_into(3..3, &mut []), Ok(()));
}

/// Encode every chunk separately with [`encode`], zero-padding the last one.
///
/// A simple reference for [`EncodedChunks::encode`].
fn reference_encode(source: &[u8], data_bytes_per_chunk: usize) -> Vec<Limited<Vec<u8>>> {
    source
        .chunks(data_bytes_per_chunk)
        .map(|data| {
            let mut padded = data.to_vec();
            padded.resize(data_bytes_per_chunk, 0);
            encode(&padded).unwrap()
        })
        .collect()
}

/// The bits of all reference chunks, in order.
fn reference_bits(chunks: &[Limited<Vec<u8>>]) -> Vec<bool> {
    chunks
        .iter()
        .flat_map(|chunk| chunk.bits().map(|bit| bit.is_1()))
        .collect()
}

/// Apply a fault to the reference chunks at the global bit index of
/// [`EncodedChunks`].
fn reference_apply_fault(chunks: &mut [Limited<Vec<u8>>], fault: Fault, bit_index: usize) {
    let encoded_bits_per_chunk = chunks[0].bit_count();
    chunks[bit_index / encoded_bits_per_chunk]
        .apply_fault(fault, bit_index % encoded_bits_per_chunk);
}

/// Decode every reference chunk separately and drop the padding.
///
/// Returns the decoded bytes and the indices of the chunks with a detected
/// double error.
fn reference_decode(
    mut chunks: Vec<Limited<Vec<u8>>>,
    data_bytes_per_chunk: usize,
    byte_count: usize,
) -> (Vec<u8>, Vec<usize>) {
    let mut output = Vec::new();
    let mut double_error_chunks = Vec::new();
    for (chunk_index, chunk) in chunks.iter_mut().enumerate() {
        let mut decoded = vec![0u8; data_bytes_per_chunk];
        if decode_into(chunk, &mut decoded).unwrap().is_detected() {
            double_error_chunks.push(chunk_index);
        }
        output.extend(decoded);
    }
    output.truncate(byte_count);

    (output, double_error_chunks)
}

fn fault_strategy() -> impl Strategy<Value = Fault> {
    prop_oneof![
        Just(Fault::Flip),
        Just(Fault::StuckAt(Bit::Zero)),
        Just(Fault::StuckAt(Bit::One)),
    ]
}

/// Inputs, chunk sizes and faults. The fault targets are resolved against the
/// encoded bit count with [`prop::sample::Index`].
fn case_strategy() -> impl Strategy<Value = (Vec<u8>, usize, Vec<(Fault, prop::sample::Index)>)> {
    (
        prop::collection::vec(any::<u8>(), 1..=128),
        1..=32usize,
        prop::collection::vec((fault_strategy(), any::<prop::sample::Index>()), 0..8),
    )
}

/// Encode `source` and apply `faults`.
fn encode_with_faults(
    source: &[u8],
    data_bytes_per_chunk: usize,
    faults: &[(Fault, prop::sample::Index)],
) -> EncodedChunks {
    let mut encoded = EncodedChunks::encode(source, data_bytes_per_chunk).unwrap();
    let bit_count = encoded.bit_count();
    encoded.apply_faults(
        faults
            .iter()
            .map(|(fault, target)| (*fault, target.index(bit_count))),
    );
    encoded
}

/// Resolve two indices into a valid range of `byte_count` bytes, possibly
/// empty.
fn resolve_range(
    (first, second): (prop::sample::Index, prop::sample::Index),
    byte_count: usize,
) -> Range<usize> {
    let first = first.index(byte_count + 1);
    let second = second.index(byte_count + 1);
    first.min(second)..first.max(second)
}

proptest! {
    #[test]
    fn matches_per_chunk_reference((source, data_bytes_per_chunk, faults) in case_strategy()) {
        let mut reference = reference_encode(&source, data_bytes_per_chunk);
        let mut encoded = EncodedChunks::encode(&source, data_bytes_per_chunk).unwrap();

        let encoded_bits = |encoded: &EncodedChunks| {
            encoded.bits().map(|bit| bit.is_1()).collect::<Vec<_>>()
        };
        prop_assert_eq!(encoded_bits(&encoded), reference_bits(&reference));

        let bit_count = encoded.bit_count();
        for (fault, target) in &faults {
            let bit_index = target.index(bit_count);
            encoded.apply_fault(*fault, bit_index);
            reference_apply_fault(&mut reference, *fault, bit_index);
        }
        prop_assert_eq!(encoded_bits(&encoded), reference_bits(&reference));

        let (expected, expected_double_error_chunks) =
            reference_decode(reference, data_bytes_per_chunk, source.len());

        let mut decoded = vec![0u8; source.len()];
        let double_error_chunks = encoded
            .decode_into_with_double_errors(&mut decoded)
            .unwrap();
        prop_assert_eq!(&decoded, &expected);
        prop_assert_eq!(double_error_chunks, expected_double_error_chunks);

        let mut decoded_without_double_errors = vec![0u8; source.len()];
        encoded.decode_into(&mut decoded_without_double_errors).unwrap();
        prop_assert_eq!(decoded_without_double_errors, expected);
    }

    #[test]
    fn decode_does_not_modify_chunks((source, data_bytes_per_chunk, faults) in case_strategy()) {
        let encoded = encode_with_faults(&source, data_bytes_per_chunk, &faults);
        let before = encoded.clone();

        let mut first = vec![0u8; source.len()];
        encoded.decode_into(&mut first).unwrap();
        prop_assert_eq!(&encoded, &before);

        let mut second = vec![0u8; source.len()];
        encoded.decode_into(&mut second).unwrap();
        prop_assert_eq!(first, second);
    }

    #[test]
    fn decode_range_into_matches_decode_into(
        (source, data_bytes_per_chunk, faults) in case_strategy(),
        range_indices in any::<(prop::sample::Index, prop::sample::Index)>(),
    ) {
        let encoded = encode_with_faults(&source, data_bytes_per_chunk, &faults);
        let data_byte_range = resolve_range(range_indices, source.len());

        let mut full = vec![0u8; source.len()];
        let full_double_error_chunks = encoded
            .decode_into_with_double_errors(&mut full)
            .unwrap();

        // Start from a different value to check that every bit is written.
        let mut decoded = vec![u8::MAX; data_byte_range.len()];
        let double_error_chunks = encoded
            .decode_range_into_with_double_errors(data_byte_range.clone(), &mut decoded)
            .unwrap();
        prop_assert_eq!(&decoded, &full[data_byte_range.clone()]);

        // The chunks that hold at least one byte of the range.
        let expected_double_error_chunks = full_double_error_chunks
            .into_iter()
            .filter(|&chunk_index| {
                !data_byte_range.is_empty()
                    && chunk_index * data_bytes_per_chunk < data_byte_range.end
                    && data_byte_range.start < (chunk_index + 1) * data_bytes_per_chunk
            })
            .collect::<Vec<_>>();
        prop_assert_eq!(double_error_chunks, expected_double_error_chunks);

        let mut decoded_without_double_errors = vec![u8::MAX; data_byte_range.len()];
        encoded
            .decode_range_into(data_byte_range.clone(), &mut decoded_without_double_errors)
            .unwrap();
        prop_assert_eq!(&decoded_without_double_errors, &full[data_byte_range]);
    }
}

#[test]
fn decode_range_across_buffers() {
    // The example from the `EncodedChunks` docs: two buffers encoded together
    // in chunks of 4, so chunk 1 holds bytes of both.
    let a = [1, 2, 3, 4, 5, 6];
    let b = [7, 8, 9, 10, 11];
    let encoded = EncodedChunks::encode(&[a.as_slice(), b.as_slice()].concat(), 4).unwrap();

    let mut decoded_a = [0u8; 6];
    encoded.decode_range_into(0..6, &mut decoded_a).unwrap();
    let mut decoded_b = [0u8; 5];
    encoded.decode_range_into(6..11, &mut decoded_b).unwrap();

    assert_eq!(decoded_a, a);
    assert_eq!(decoded_b, b);
}
