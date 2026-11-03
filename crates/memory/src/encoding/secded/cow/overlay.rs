#[cfg(test)]
mod tests;

use crate::encoding::secded::EncodedChunks;

/// Copies of some encoded chunks of an [`EncodedChunks`], kept sorted by
/// chunk index.
///
/// A sorted pair of buffers needs a lot less memory per chunk than a map with
/// a separate allocation for every chunk. Chunks are usually only a few bytes.
#[derive(Debug, Clone)]
pub(super) struct Overlay {
    /// The indices of the copied chunks in ascending order, without
    /// duplicates.
    chunk_indices: Vec<usize>,
    /// The encoded bytes of the copied chunks in the order of
    /// `chunk_indices`: the chunk at position `p` is
    /// `encoded_bytes[p * encoded_bytes_per_chunk..(p + 1) * encoded_bytes_per_chunk]`.
    encoded_bytes: Vec<u8>,
    /// The number of encoded bytes in each chunk.
    encoded_bytes_per_chunk: usize,
}

impl Overlay {
    /// Create an empty overlay for encoded chunks of `encoded_bytes_per_chunk`
    /// bytes.
    ///
    /// Panics if `encoded_bytes_per_chunk` is 0.
    pub(super) fn new(encoded_bytes_per_chunk: usize) -> Self {
        assert_ne!(encoded_bytes_per_chunk, 0);
        Self {
            chunk_indices: Vec::new(),
            encoded_bytes: Vec::new(),
            encoded_bytes_per_chunk,
        }
    }

    /// The number of copied chunks.
    pub(super) fn chunk_count(&self) -> usize {
        self.chunk_indices.len()
    }

    /// The number of bytes the encoded chunks and their indices take up.
    pub(super) fn memory_size(&self) -> usize {
        let memory_size = size_of_val(self.chunk_indices.as_slice()) + self.encoded_bytes.len();
        assert_eq!(
            memory_size,
            self.chunk_count() * Self::memory_size_per_chunk(self.encoded_bytes_per_chunk)
        );
        memory_size
    }

    /// The number of bytes a copied chunk of `encoded_bytes_per_chunk` bytes
    /// takes up with its index.
    pub(super) fn memory_size_per_chunk(encoded_bytes_per_chunk: usize) -> usize {
        size_of::<usize>() + encoded_bytes_per_chunk
    }

    /// Find the position of a chunk.
    ///
    /// Like [`slice::binary_search`], returns the position at which the chunk
    /// would be inserted if it's not in the overlay.
    pub(super) fn position(&self, chunk_index: usize) -> Result<usize, usize> {
        self.chunk_indices.binary_search(&chunk_index)
    }

    /// The encoded bytes of the chunk at `position`.
    ///
    /// # Panics
    ///
    /// Panics if `position` is out of bounds.
    pub(super) fn encoded_chunk(&self, position: usize) -> &[u8] {
        let start = position * self.encoded_bytes_per_chunk;
        &self.encoded_bytes[start..start + self.encoded_bytes_per_chunk]
    }

    /// The encoded bytes of the chunk at `position`, for writing.
    ///
    /// # Panics
    ///
    /// Panics if `position` is out of bounds.
    pub(super) fn encoded_chunk_mut(&mut self, position: usize) -> &mut [u8] {
        let start = position * self.encoded_bytes_per_chunk;
        &mut self.encoded_bytes[start..start + self.encoded_bytes_per_chunk]
    }

    /// Iterate over the chunk indices and encoded bytes in ascending order of
    /// the chunk index.
    pub(super) fn iter(&self) -> impl Iterator<Item = (usize, &[u8])> {
        let encoded_chunks = self
            .encoded_bytes
            .chunks_exact(self.encoded_bytes_per_chunk);
        assert_eq!(encoded_chunks.len(), self.chunk_indices.len());
        self.chunk_indices.iter().copied().zip(encoded_chunks)
    }

    /// Add copies of the chunks of `pristine` at `chunk_indices`.
    ///
    /// `chunk_indices` must be in ascending order without duplicates, and none
    /// of them may be in the overlay already.
    ///
    /// This rebuilds the overlay, so add many chunks at once where possible.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `pristine` has a different chunk size.
    /// - `chunk_indices` breaks any of the requirements above.
    /// - Any of `chunk_indices` is out of bounds for `pristine`.
    pub(super) fn extend(&mut self, chunk_indices: &[usize], pristine: &EncodedChunks) {
        assert_eq!(
            pristine.encoded_bytes_per_chunk(),
            self.encoded_bytes_per_chunk
        );

        // Inserting the chunks one by one would shift the rest of the overlay
        // every time. Merging into new buffers moves every chunk only once.
        let merged_count = self.chunk_count() + chunk_indices.len();
        let mut merged_indices = Vec::with_capacity(merged_count);
        let mut merged_encoded_bytes =
            Vec::with_capacity(merged_count * self.encoded_bytes_per_chunk);

        let mut existing = self.iter().peekable();
        let mut added = chunk_indices.iter().copied().peekable();
        // Every iteration takes one chunk from either side.
        for _ in 0..merged_count {
            let take_existing = match (existing.peek(), added.peek()) {
                (Some(&(existing_index, _)), Some(&added_index)) => {
                    assert_ne!(
                        existing_index, added_index,
                        "chunk {added_index} is already in the overlay"
                    );
                    existing_index < added_index
                }
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => {
                    unreachable!("both sides ran out before {merged_count} chunks were merged")
                }
            };

            if take_existing {
                let (chunk_index, encoded_chunk) = existing.next().expect("the chunk was peeked");
                merged_indices.push(chunk_index);
                merged_encoded_bytes.extend_from_slice(encoded_chunk);
            } else {
                let chunk_index = added.next().expect("the chunk was peeked");
                merged_indices.push(chunk_index);
                merged_encoded_bytes.extend_from_slice(pristine.encoded_chunk(chunk_index));
            }
        }
        assert!(
            existing.next().is_none() && added.next().is_none(),
            "all chunks are merged after {merged_count} iterations"
        );

        // `existing` borrows the old buffers until it's dropped.
        drop(existing);
        self.chunk_indices = merged_indices;
        self.encoded_bytes = merged_encoded_bytes;
        self.assert_invariants();
    }

    fn assert_invariants(&self) {
        assert!(
            self.chunk_indices
                .is_sorted_by(|previous, next| previous < next),
            "the chunk indices are in ascending order without duplicates"
        );
        assert_eq!(
            self.encoded_bytes.len(),
            self.chunk_indices.len() * self.encoded_bytes_per_chunk
        );
    }
}
