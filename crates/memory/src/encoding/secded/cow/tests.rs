use super::*;
use crate::Bit;
use proptest::prelude::*;

/// 4 chunks of 8 data bytes, each 72 encoded bits in 9 bytes.
fn four_chunks() -> CowEncodedChunks {
    let source = (0..32).collect::<Vec<u8>>();
    let cow = CowEncodedChunks::encode(&source, 8).unwrap();
    assert_eq!(cow.chunk_count(), 4);
    cow
}

fn bits(buffer: &impl BitBuffer) -> Vec<bool> {
    buffer.bits().map(|bit| bit.is_1()).collect()
}

#[test]
fn clones_share_chunks() {
    let original = four_chunks();
    let clone = original.clone();

    assert!(Arc::ptr_eq(&original.pristine, &clone.pristine));
    assert_eq!(clone.overlay_chunk_count(), 0);
    assert_eq!(clone.overlay_memory_size(), 0);
}

#[test]
fn writes_copy_single_chunks() {
    let original = four_chunks();
    let mut clone = original.clone();

    clone.flip_bit(0);
    clone.flip_bit(1);
    assert_eq!(clone.overlay_chunk_count(), 1);
    // One index and one chunk.
    assert_eq!(clone.overlay_memory_size(), size_of::<usize>() + 9);

    // The last bit of the last chunk.
    clone.flip_bit(clone.bit_count() - 1);
    assert_eq!(clone.overlay_chunk_count(), 2);

    assert!(clone.is_1(0));
    assert!(!original.is_1(0));
    assert_ne!(bits(&clone), bits(&original));
    assert!(Arc::ptr_eq(&original.pristine, &clone.pristine));
}

#[test]
fn overlay_of_all_chunks_is_larger_than_a_copy() {
    let mut cow = four_chunks();
    cow.apply_faults((0..4).map(|chunk_index| (Fault::Flip, chunk_index * 72)));

    assert_eq!(cow.overlay_chunk_count(), 4);
    // Every chunk takes up 9 bytes and its index another 8.
    assert_eq!(cow.overlay_memory_size(), 4 * (size_of::<usize>() + 9));
}

#[test]
fn reads_do_not_copy() {
    let cow = four_chunks();
    let _ = bits(&cow);
    let mut dest = vec![0u8; cow.data_byte_count()];
    cow.decode_into(&mut dest).unwrap();
    assert_eq!(cow.overlay_chunk_count(), 0);
}

#[test]
fn clones_copy_the_overlay() {
    let mut original = four_chunks();
    original.flip_bit(5);
    let before = bits(&original);

    let mut clone = original.clone();
    assert_eq!(bits(&clone), before);
    clone.flip_bit(5);
    clone.flip_bit(100);

    assert_eq!(bits(&original), before);
    assert_eq!(original.overlay_chunk_count(), 1);
    assert_eq!(clone.overlay_chunk_count(), 2);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn apply_faults_checks_all_targets_first() {
    let mut cow = four_chunks();
    let bit_count = cow.bit_count();
    cow.apply_faults([(Fault::Flip, 0), (Fault::Flip, bit_count)].into_iter());
}

#[test]
fn no_changes_without_faults() {
    let cow = four_chunks();
    let changes = cow.decoded_changes(0..cow.data_byte_count()).unwrap();
    assert_eq!(changes, DecodedChanges::default());
}

#[test]
fn decoded_changes_checks_the_range() {
    let cow = four_chunks();
    let data_byte_count = cow.data_byte_count();
    let invalid_range = |start, end| DecodeChunksError::InvalidRange {
        start,
        end,
        data_byte_count,
    };

    assert_eq!(
        cow.decoded_changes(0..data_byte_count + 1),
        Err(invalid_range(0, data_byte_count + 1))
    );
    #[expect(clippy::reversed_empty_ranges, reason = "testing an invalid range")]
    let reversed = 2..1;
    assert_eq!(cow.decoded_changes(reversed), Err(invalid_range(2, 1)));
    assert_eq!(
        cow.decoded_changes(data_byte_count..data_byte_count),
        Ok(DecodedChanges::default())
    );
}

/// `decoded_changes` skips chunks with a single differing bit without decoding
/// them, which is only correct if every single flip is corrected.
#[test]
fn single_flips_decode_without_changes() {
    for data_bytes_per_chunk in [1, 2, 3, 8, 9, 32] {
        // Two full chunks and a shorter, padded last one.
        let source = (0..2 * data_bytes_per_chunk + 1)
            .map(|index| u8::try_from(index * 37 % 256).expect("the value is below 256"))
            .collect::<Vec<u8>>();
        let original = CowEncodedChunks::encode(&source, data_bytes_per_chunk).unwrap();

        for bit_index in 0..original.bit_count() {
            let mut cow = original.clone();
            cow.flip_bit(bit_index);

            let mut decoded = vec![0u8; source.len()];
            cow.decode_into(&mut decoded).unwrap();
            assert_eq!(decoded, source, "flipped bit {bit_index}");
            assert_eq!(
                cow.decoded_changes(0..source.len()).unwrap(),
                DecodedChanges::default(),
                "flipped bit {bit_index}"
            );
        }
    }
}

fn fault_strategy() -> impl Strategy<Value = Fault> {
    prop_oneof![
        Just(Fault::Flip),
        Just(Fault::StuckAt(Bit::Zero)),
        Just(Fault::StuckAt(Bit::One)),
    ]
}

/// Faults with targets that are resolved against the encoded bit count.
fn faults_strategy() -> impl Strategy<Value = Vec<(Fault, prop::sample::Index)>> {
    prop::collection::vec((fault_strategy(), any::<prop::sample::Index>()), 0..8)
}

fn resolve(
    faults: &[(Fault, prop::sample::Index)],
    bit_count: usize,
) -> impl Iterator<Item = (Fault, usize)> {
    faults
        .iter()
        .map(move |(fault, target)| (*fault, target.index(bit_count)))
}

/// Check that decoding `cow` gives the same results as decoding `expected`,
/// and that its decoded changes turn `source` into the same bytes.
///
/// The range decode is checked by splitting the data into adjacent ranges at
/// `split_points`.
fn check_decode(
    cow: &CowEncodedChunks,
    expected: &EncodedChunks,
    source: &[u8],
    split_points: &[prop::sample::Index],
) -> Result<(), TestCaseError> {
    let byte_count = expected.data_byte_count();

    let mut expected_decoded = vec![0u8; byte_count];
    let expected_double_error_chunks = expected
        .decode_into_with_double_errors(&mut expected_decoded)
        .unwrap();

    let mut decoded = vec![0u8; byte_count];
    let double_error_chunks = cow.decode_into_with_double_errors(&mut decoded).unwrap();
    prop_assert_eq!(&decoded, &expected_decoded);
    prop_assert_eq!(&double_error_chunks, &expected_double_error_chunks);

    let mut decoded = vec![0u8; byte_count];
    cow.decode_into(&mut decoded).unwrap();
    prop_assert_eq!(&decoded, &expected_decoded);

    let mut boundaries = split_points
        .iter()
        .map(|split_point| split_point.index(byte_count + 1))
        .collect::<Vec<_>>();
    boundaries.push(0);
    boundaries.push(byte_count);
    boundaries.sort_unstable();
    // Start from a different value to check that every bit is written.
    let mut decoded = vec![u8::MAX; byte_count];
    let mut double_error_chunks = Vec::new();
    for range in boundaries.windows(2) {
        let data_byte_range = range[0]..range[1];
        double_error_chunks.extend(
            cow.decode_range_into_with_double_errors(
                data_byte_range.clone(),
                &mut decoded[data_byte_range.clone()],
            )
            .unwrap(),
        );
        let mut without_double_errors = vec![u8::MAX; data_byte_range.len()];
        cow.decode_range_into(data_byte_range.clone(), &mut without_double_errors)
            .unwrap();
        prop_assert_eq!(
            &without_double_errors,
            &expected_decoded[data_byte_range.clone()]
        );

        let changes = cow.decoded_changes(data_byte_range.clone()).unwrap();
        prop_assert_eq!(changes.byte_offsets.len(), changes.byte_values.len());
        prop_assert!(
            changes
                .byte_offsets
                .is_sorted_by(|previous, next| previous < next)
        );
        let mut changed = source[data_byte_range.clone()].to_vec();
        for (&byte_offset, &byte_value) in changes.byte_offsets.iter().zip(&changes.byte_values) {
            // Only bytes that differ from the source are reported.
            prop_assert_ne!(changed[byte_offset], byte_value);
            changed[byte_offset] = byte_value;
        }
        prop_assert_eq!(&changed, &expected_decoded[data_byte_range]);
    }
    // A chunk that's split between two ranges is reported by both.
    double_error_chunks.dedup();
    prop_assert_eq!(decoded, expected_decoded);
    prop_assert_eq!(double_error_chunks, expected_double_error_chunks);

    Ok(())
}

proptest! {
    #[test]
    fn matches_encoded_chunks(
        source in prop::collection::vec(any::<u8>(), 1..=128),
        data_bytes_per_chunk in 1..=32usize,
        // Each batch is applied with `apply_faults` or one fault at a time.
        batches in prop::collection::vec((any::<bool>(), faults_strategy()), 0..4),
        // Where to split the data for range decodes.
        split_points in prop::collection::vec(any::<prop::sample::Index>(), 0..4),
    ) {
        let mut expected = EncodedChunks::encode(&source, data_bytes_per_chunk).unwrap();
        let bit_count = expected.bit_count();
        let pristine_bits = bits(&expected);

        let original = CowEncodedChunks::encode(&source, data_bytes_per_chunk).unwrap();
        let mut cow = original.clone();
        prop_assert_eq!(cow.bit_count(), bit_count);
        prop_assert_eq!(cow.chunk_count(), expected.chunk_count());
        prop_assert_eq!(cow.data_byte_count(), expected.data_byte_count());
        check_decode(&cow, &expected, &source, &split_points)?;

        for (batched, faults) in &batches {
            if *batched {
                cow.apply_faults(resolve(faults, bit_count));
            } else {
                for (fault, bit_index) in resolve(faults, bit_count) {
                    cow.apply_fault(fault, bit_index);
                }
            }
            expected.apply_faults(resolve(faults, bit_count));

            prop_assert_eq!(bits(&cow), bits(&expected));
        }

        check_decode(&cow, &expected, &source, &split_points)?;
        // The clone's writes never reach the original.
        prop_assert_eq!(bits(&original), pristine_bits);
    }
}
