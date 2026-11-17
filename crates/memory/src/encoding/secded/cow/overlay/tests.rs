use super::*;
use proptest::prelude::*;
use std::collections::BTreeMap;

/// 8 chunks of 1 data byte, each 13 encoded bits in 2 bytes.
///
/// The data bytes differ, so every chunk has different bytes.
fn pristine() -> EncodedChunks {
    let encoded = EncodedChunks::encode(&[1, 2, 3, 4, 5, 6, 7, 8], 1).unwrap();
    assert_eq!(encoded.chunk_count(), 8);
    assert_eq!(encoded.encoded_bytes_per_chunk(), 2);
    encoded
}

/// The chunk indices and bytes of `overlay` in order.
fn contents(overlay: &Overlay) -> Vec<(usize, Vec<u8>)> {
    overlay
        .iter()
        .map(|(chunk_index, encoded_chunk)| (chunk_index, encoded_chunk.to_vec()))
        .collect()
}

/// Add the chunks one at a time.
fn extend_one_by_one(overlay: &mut Overlay, chunk_indices: &[usize], pristine: &EncodedChunks) {
    for &chunk_index in chunk_indices {
        overlay.extend(&[chunk_index], pristine);
    }
}

#[test]
fn new_is_empty() {
    let overlay = Overlay::new(2);
    assert_eq!(overlay.chunk_count(), 0);
    assert_eq!(overlay.memory_size(), 0);
    assert_eq!(overlay.iter().count(), 0);
    assert_eq!(overlay.position(3), Err(0));
}

#[test]
#[should_panic]
fn new_rejects_zero_encoded_bytes_per_chunk() {
    let _ = Overlay::new(0);
}

#[test]
fn extend_keeps_order() {
    let pristine = pristine();
    let mut overlay = Overlay::new(2);

    extend_one_by_one(&mut overlay, &[5, 1, 7, 3], &pristine);

    let expected = [1, 3, 5, 7]
        .into_iter()
        .map(|chunk_index| (chunk_index, pristine.encoded_chunk(chunk_index).to_vec()))
        .collect::<Vec<_>>();
    assert_eq!(contents(&overlay), expected);
    assert_eq!(overlay.position(5), Ok(2));
    assert_eq!(overlay.encoded_chunk(2), pristine.encoded_chunk(5));
    assert_eq!(overlay.position(4), Err(2));
    assert_eq!(overlay.memory_size(), 4 * (size_of::<usize>() + 2));
}

#[test]
fn chunk_mut_writes_one_chunk() {
    let pristine = pristine();
    let mut overlay = Overlay::new(2);
    overlay.extend(&[2, 4], &pristine);

    overlay.encoded_chunk_mut(1)[0] ^= 0xff;

    assert_eq!(overlay.encoded_chunk(0), pristine.encoded_chunk(2));
    assert_ne!(overlay.encoded_chunk(1), pristine.encoded_chunk(4));
}

#[test]
#[should_panic]
fn extend_rejects_other_chunk_size() {
    let pristine = pristine();
    let mut overlay = Overlay::new(3);
    overlay.extend(&[0], &pristine);
}

#[test]
fn extend_keeps_written_chunks() {
    let pristine = pristine();
    let mut overlay = Overlay::new(2);
    overlay.extend(&[1, 5], &pristine);
    overlay.encoded_chunk_mut(0)[0] ^= 0xff;
    let written = overlay.encoded_chunk(0).to_vec();

    overlay.extend(&[0, 3, 7], &pristine);

    let expected = vec![
        (0, pristine.encoded_chunk(0).to_vec()),
        (1, written),
        (3, pristine.encoded_chunk(3).to_vec()),
        (5, pristine.encoded_chunk(5).to_vec()),
        (7, pristine.encoded_chunk(7).to_vec()),
    ];
    assert_eq!(contents(&overlay), expected);
}

#[test]
#[should_panic(expected = "already in the overlay")]
fn extend_rejects_existing_chunk() {
    let pristine = pristine();
    let mut overlay = Overlay::new(2);
    overlay.extend(&[3], &pristine);
    overlay.extend(&[1, 3], &pristine);
}

#[test]
#[should_panic(expected = "ascending order without duplicates")]
fn extend_rejects_unsorted_chunks() {
    let pristine = pristine();
    let mut overlay = Overlay::new(2);
    overlay.extend(&[3, 1], &pristine);
}

#[test]
#[should_panic(expected = "ascending order without duplicates")]
fn extend_rejects_duplicates() {
    let pristine = pristine();
    let mut overlay = Overlay::new(2);
    overlay.extend(&[1, 1], &pristine);
}

#[test]
#[should_panic]
fn extend_rejects_out_of_bounds_chunk() {
    let pristine = pristine();
    let mut overlay = Overlay::new(2);
    overlay.extend(&[pristine.chunk_count()], &pristine);
}

/// An operation on an [`Overlay`] and its reference.
#[derive(Debug, Clone)]
enum Operation {
    /// Extend with the missing chunks of a list with any order and
    /// duplicates.
    Extend(Vec<usize>),
    /// Flip the bits of a byte of the chunk if it's in the overlay.
    Write {
        chunk_index: usize,
        byte_index: usize,
    },
}

fn operation_strategy(chunk_count: usize) -> impl Strategy<Value = Operation> {
    prop_oneof![
        prop::collection::vec(0..chunk_count, 0..8).prop_map(Operation::Extend),
        (0..chunk_count, 0..2usize).prop_map(|(chunk_index, byte_index)| Operation::Write {
            chunk_index,
            byte_index,
        }),
    ]
}

proptest! {
    #[test]
    fn matches_map_reference(
        operations in prop::collection::vec(operation_strategy(8), 0..16),
    ) {
        let pristine = pristine();
        let mut overlay = Overlay::new(2);
        let mut reference = BTreeMap::<usize, Vec<u8>>::new();

        for operation in operations {
            match operation {
                Operation::Extend(mut chunk_indices) => {
                    chunk_indices.sort_unstable();
                    chunk_indices.dedup();
                    chunk_indices.retain(|chunk_index| overlay.position(*chunk_index).is_err());
                    overlay.extend(&chunk_indices, &pristine);
                    for chunk_index in chunk_indices {
                        reference.insert(chunk_index, pristine.encoded_chunk(chunk_index).to_vec());
                    }
                }
                Operation::Write { chunk_index, byte_index } => {
                    if let Ok(position) = overlay.position(chunk_index) {
                        overlay.encoded_chunk_mut(position)[byte_index] ^= 0xff;
                    }
                    if let Some(bytes) = reference.get_mut(&chunk_index) {
                        bytes[byte_index] ^= 0xff;
                    }
                }
            }

            let expected = reference
                .iter()
                .map(|(chunk_index, bytes)| (*chunk_index, bytes.clone()))
                .collect::<Vec<_>>();
            prop_assert_eq!(contents(&overlay), expected);
            prop_assert_eq!(overlay.chunk_count(), reference.len());
            prop_assert_eq!(overlay.memory_size(), reference.len() * (size_of::<usize>() + 2));
            for (position, chunk_index) in reference.keys().enumerate() {
                prop_assert_eq!(overlay.position(*chunk_index), Ok(position));
                prop_assert_eq!(overlay.encoded_chunk(position), reference[chunk_index].as_slice());
            }
        }
    }
}
