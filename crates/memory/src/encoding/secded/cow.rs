mod overlay;
#[cfg(test)]
mod tests;

use crate::{
    BitBuffer, Fault,
    encoding::secded::{DecodeChunksError, EncodeChunksError, EncodedChunks, chunks::ChunkDecoder},
};
use overlay::Overlay;
use std::{ops::Range, sync::Arc};

/// Decoded data bytes that differ from the original data.
///
/// See [`CowEncodedChunks::decoded_changes`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DecodedChanges {
    /// The offsets of the changed bytes from the start of the decoded range,
    /// in ascending order.
    pub byte_offsets: Vec<usize>,
    /// The decoded value of the byte at each offset in `byte_offsets`.
    pub byte_values: Vec<u8>,
}

/// [`EncodedChunks`] that share their unchanged chunks with their clones.
///
/// A new instance and all its clones share one copy of the encoded chunks
/// through an [`Arc`]. Writing to a chunk copies it into an overlay first, so
/// clones never see each other's changes. Cloning copies the overlay.
///
/// Apart from that, it behaves like [`EncodedChunks`] of the same source.
///
/// # Trade-offs
///
/// The overlay is made for faults at low bit error rates, which only touch a
/// small part of the chunks. Compared with cloning [`EncodedChunks`]:
/// - Every copied chunk also stores its index, see
///   [`CowEncodedChunks::overlay_memory_size`]. Once most chunks are copied,
///   the overlay takes up more memory than a full copy. At most
///   `(size_of::<usize>() + encoded_bytes_per_chunk) / encoded_bytes_per_chunk`
///   times as much, which is 1.89 for 64-bit chunks.
/// - Every write, and every chunk a decode reads, is looked up in the
///   overlay first, which gets slower as the overlay grows.
///
/// The overlay also makes [`CowEncodedChunks::decoded_changes`] possible,
/// which is much faster than a full decode at low bit error rates.
///
/// From `benches/cow.rs` (clone, apply faults and decode 1 MiB of data with
/// 64-bit chunks). The times are relative to a clone of [`EncodedChunks`] with
/// a full decode:
///
/// | Bit error rate | Overlay memory of a full copy | Time with a full decode | Time with `decoded_changes` |
/// |----------------|-------------------------------|-------------------------|-----------------------------|
/// | 1e-4           | 1.4%                          | 1.01x                   | 0.0014x                     |
/// | 1e-3           | 13%                           | 1.06x                   | 0.024x                      |
/// | 1e-2           | 97%                           | 1.33x                   | 0.62x                       |
/// | 1e-1           | 189%                          | 1.89x                   | 2.5x                        |
/// | 1              | 189%                          | 11.5x                   | 12.4x                       |
#[derive(Debug, Clone)]
pub struct CowEncodedChunks {
    /// The shared chunks. They're never written to, so they hold the encoding
    /// of the source without faults.
    pristine: Arc<EncodedChunks>,
    /// Copies of the chunks that were written to. They take precedence over
    /// the ones in `pristine`.
    overlay: Overlay,
}

impl CowEncodedChunks {
    /// Encode `source` in chunks of `data_bytes_per_chunk` bytes.
    ///
    /// See [`EncodedChunks::encode`].
    pub fn encode(source: &[u8], data_bytes_per_chunk: usize) -> Result<Self, EncodeChunksError> {
        let encoded = EncodedChunks::encode(source, data_bytes_per_chunk)?;
        Ok(Self {
            overlay: Overlay::new(encoded.encoded_bytes_per_chunk()),
            pristine: Arc::new(encoded),
        })
    }

    /// Decode all chunks into `dest`.
    ///
    /// See [`EncodedChunks::decode_into`].
    pub fn decode_into(&self, dest: &mut [u8]) -> Result<(), DecodeChunksError> {
        self.decode_range_into(0..self.data_byte_count(), dest)
    }

    /// Decode all chunks into `dest` and return the chunks in which a double
    /// error was detected.
    ///
    /// See [`EncodedChunks::decode_into_with_double_errors`].
    pub fn decode_into_with_double_errors(
        &self,
        dest: &mut [u8],
    ) -> Result<Vec<usize>, DecodeChunksError> {
        self.decode_range_into_with_double_errors(0..self.data_byte_count(), dest)
    }

    /// Decode the data bytes in `data_byte_range` into `dest`.
    ///
    /// See [`EncodedChunks::decode_range_into`].
    pub fn decode_range_into(
        &self,
        data_byte_range: Range<usize>,
        dest: &mut [u8],
    ) -> Result<(), DecodeChunksError> {
        self.decode_range_with(data_byte_range, dest, |_| {})
    }

    /// Decode the data bytes in `data_byte_range` into `dest` and return the
    /// chunks in which a double error was detected.
    ///
    /// See [`EncodedChunks::decode_range_into_with_double_errors`].
    pub fn decode_range_into_with_double_errors(
        &self,
        data_byte_range: Range<usize>,
        dest: &mut [u8],
    ) -> Result<Vec<usize>, DecodeChunksError> {
        let mut double_error_chunks = Vec::new();
        self.decode_range_with(data_byte_range, dest, |chunk_index| {
            double_error_chunks.push(chunk_index);
        })?;
        Ok(double_error_chunks)
    }

    /// Find the decoded data bytes in `data_byte_range` that differ from the
    /// original data.
    ///
    /// The original data is what was encoded, before any faults. Writing the
    /// changes into a copy of it gives the same bytes as
    /// [`CowEncodedChunks::decode_range_into`]. Only chunks that were written
    /// to can change, so this is much faster than a full decode while few
    /// chunks were written to.
    ///
    /// Earlier calls don't matter. A byte that an earlier call reported is
    /// left out once it decodes to its original value again. So writing the
    /// changes into the result of an earlier decode with faults can give wrong
    /// bytes.
    ///
    /// # Errors
    ///
    /// If `data_byte_range` ends before it starts or goes beyond the data, see
    /// [`DecodeChunksError::InvalidRange`].
    pub fn decoded_changes(
        &self,
        data_byte_range: Range<usize>,
    ) -> Result<DecodedChanges, DecodeChunksError> {
        let layout = self.pristine.layout();
        layout.check_range_bounds(&data_byte_range)?;

        let mut changes = DecodedChanges::default();
        if data_byte_range.is_empty() {
            return Ok(changes);
        }

        let data_bytes_per_chunk = self.data_bytes_per_chunk();
        let first_chunk = data_byte_range.start / data_bytes_per_chunk;
        let last_chunk = (data_byte_range.end - 1) / data_bytes_per_chunk;
        let (Ok(first_position) | Err(first_position)) = self.overlay.position(first_chunk);

        let mut decoder = ChunkDecoder::new(layout);
        let mut faulty_data_chunk = vec![0u8; data_bytes_per_chunk];
        let mut pristine_data_chunk = vec![0u8; data_bytes_per_chunk];

        // Chunks that aren't in the overlay were never written to, so they
        // decode to their data without faults.
        for (chunk_index, faulty_encoded_chunk) in self
            .overlay
            .iter()
            .skip(first_position)
            .take_while(|&(chunk_index, _)| chunk_index <= last_chunk)
        {
            let pristine_encoded_chunk = self.pristine.encoded_chunk(chunk_index);
            // `zip` below would silently compare only part of the chunk.
            assert_eq!(faulty_encoded_chunk.len(), pristine_encoded_chunk.len());

            // The pristine chunk is a valid code word. With one differing bit,
            // decoding corrects it back to the pristine data, see the decision
            // table in the `secded` module docs. Only chunks with at least two
            // differing bits can decode differently, which are a small part of
            // the written chunks at low bit error rates. Skipping the others
            // made `overlay_changes` in `benches/cow.rs` 12x faster at a bit
            // error rate of 1e-4, 7x at 1e-3 and 2x at 1e-2 (64-bit chunks).
            // The padding bits are never written, so they never differ.
            let differing_bit_count: u32 = faulty_encoded_chunk
                .iter()
                .zip(pristine_encoded_chunk)
                .map(|(faulty_byte, pristine_byte)| (faulty_byte ^ pristine_byte).count_ones())
                .sum();
            if differing_bit_count <= 1 {
                continue;
            }

            // Discarding the double error detection results because we only
            // care about the bytes.
            let _ = decoder.decode(faulty_encoded_chunk, &mut faulty_data_chunk);
            let _ = decoder.decode(pristine_encoded_chunk, &mut pristine_data_chunk);

            // The range can start and end in the middle of a chunk. The
            // padding of a shorter last chunk is never in the range because
            // the range is within the data.
            let chunk_start = chunk_index * data_bytes_per_chunk;
            let chunk_end = chunk_start + data_bytes_per_chunk;
            // The part of the chunk that's in the range, as indices into the
            // chunk.
            let first_in_chunk = data_byte_range.start.max(chunk_start) - chunk_start;
            let end_in_chunk = data_byte_range.end.min(chunk_end) - chunk_start;
            for index_in_chunk in first_in_chunk..end_in_chunk {
                let faulty_byte = faulty_data_chunk[index_in_chunk];
                if faulty_byte == pristine_data_chunk[index_in_chunk] {
                    continue;
                }
                changes
                    .byte_offsets
                    .push(chunk_start + index_in_chunk - data_byte_range.start);
                changes.byte_values.push(faulty_byte);
            }
        }

        Ok(changes)
    }

    /// Get the number of chunks that were copied into the overlay.
    #[must_use]
    pub fn overlay_chunk_count(&self) -> usize {
        self.overlay.chunk_count()
    }

    /// Get the number of bytes of memory the overlay takes up.
    ///
    /// This includes the indices of the copied chunks, but not allocator
    /// overhead or unused capacity. The shared chunks are never counted, even
    /// if nothing else shares them anymore.
    #[must_use]
    pub fn overlay_memory_size(&self) -> usize {
        self.overlay.memory_size()
    }

    /// Get the number of chunks.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.pristine.chunk_count()
    }

    /// Get the number of data bytes in each chunk.
    #[must_use]
    pub fn data_bytes_per_chunk(&self) -> usize {
        self.pristine.data_bytes_per_chunk()
    }

    /// Get the number of encoded bits in each chunk.
    #[must_use]
    pub fn encoded_bits_per_chunk(&self) -> usize {
        self.pristine.encoded_bits_per_chunk()
    }

    /// Get the number of bytes each encoded chunk takes up.
    ///
    /// See [`EncodedChunks::encoded_bytes_per_chunk`].
    #[must_use]
    pub fn encoded_bytes_per_chunk(&self) -> usize {
        self.pristine.encoded_bytes_per_chunk()
    }

    /// Get the number of data bytes that were encoded.
    ///
    /// This is the number of bytes a decode destination must have.
    #[must_use]
    pub fn data_byte_count(&self) -> usize {
        self.pristine.data_byte_count()
    }

    /// Decode the data bytes in `data_byte_range` into `dest`, calling
    /// `on_double_error` with the index of every chunk with a detected double
    /// error, in ascending order.
    fn decode_range_with(
        &self,
        data_byte_range: Range<usize>,
        dest: &mut [u8],
        on_double_error: impl FnMut(usize),
    ) -> Result<(), DecodeChunksError> {
        // The chunks are decoded in ascending order. So instead of searching
        // the overlay for every chunk, go through the overlay alongside them,
        // starting at the first chunk of the range.
        let first_chunk = data_byte_range.start / self.data_bytes_per_chunk();
        let (Ok(first_position) | Err(first_position)) = self.overlay.position(first_chunk);
        let mut overlay_chunks = self.overlay.iter().skip(first_position).peekable();

        self.pristine.layout().decode_range(
            data_byte_range,
            dest,
            |chunk_index| {
                while overlay_chunks
                    .next_if(|&(overlay_chunk_index, _)| overlay_chunk_index < chunk_index)
                    .is_some()
                {}
                match overlay_chunks.peek() {
                    Some(&(overlay_chunk_index, encoded_chunk))
                        if overlay_chunk_index == chunk_index =>
                    {
                        encoded_chunk
                    }
                    _ => self.pristine.encoded_chunk(chunk_index),
                }
            },
            on_double_error,
        )
    }

    /// The encoded bytes of chunk `chunk_index`, with the faults applied so
    /// far.
    ///
    /// They're the copy in the overlay if there is one, and the shared chunk
    /// otherwise.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_index` is out of bounds.
    fn encoded_chunk(&self, chunk_index: usize) -> &[u8] {
        match self.overlay.position(chunk_index) {
            Ok(position) => self.overlay.encoded_chunk(position),
            Err(_) => self.pristine.encoded_chunk(chunk_index),
        }
    }

    /// The encoded bytes of chunk `chunk_index`, for writing.
    ///
    /// Copies the chunk into the overlay first if it isn't there yet. The
    /// caller must never write the padding bits.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_index` is out of bounds.
    fn encoded_chunk_mut(&mut self, chunk_index: usize) -> &mut [u8] {
        let position = match self.overlay.position(chunk_index) {
            Ok(position) => position,
            Err(_) => {
                self.copy_chunks(vec![chunk_index]);
                self.overlay
                    .position(chunk_index)
                    .expect("the chunk was copied into the overlay")
            }
        };
        self.overlay.encoded_chunk_mut(position)
    }

    /// Copy the chunks at `chunk_indices` into the overlay at once.
    ///
    /// `chunk_indices` may be in any order, contain duplicates and chunks that
    /// were already copied.
    ///
    /// # Panics
    ///
    /// Panics if any of `chunk_indices` is out of bounds.
    fn copy_chunks(&mut self, mut chunk_indices: Vec<usize>) {
        if let Some(&max) = chunk_indices.iter().max() {
            assert!(
                max < self.chunk_count(),
                "chunk {max} is out of bounds for {} chunks",
                self.chunk_count()
            );
        }

        chunk_indices.sort_unstable();
        chunk_indices.dedup();
        chunk_indices.retain(|&chunk_index| self.overlay.position(chunk_index).is_err());
        self.overlay.extend(&chunk_indices, &self.pristine);
    }
}

impl BitBuffer for CowEncodedChunks {
    fn bit_count(&self) -> usize {
        self.pristine.bit_count()
    }

    fn set_1(&mut self, bit_index: usize) {
        let location = self.pristine.layout().locate(bit_index);
        self.encoded_chunk_mut(location.chunk_index)
            .set_1(location.bit_in_chunk);
    }

    fn set_0(&mut self, bit_index: usize) {
        let location = self.pristine.layout().locate(bit_index);
        self.encoded_chunk_mut(location.chunk_index)
            .set_0(location.bit_in_chunk);
    }

    fn is_1(&self, bit_index: usize) -> bool {
        let location = self.pristine.layout().locate(bit_index);
        self.encoded_chunk(location.chunk_index)
            .is_1(location.bit_in_chunk)
    }

    fn flip_bit(&mut self, bit_index: usize) {
        let location = self.pristine.layout().locate(bit_index);
        self.encoded_chunk_mut(location.chunk_index)
            .flip_bit(location.bit_in_chunk);
    }

    /// Apply many faults in sequence.
    ///
    /// # Panics
    ///
    /// If any target is out of bounds. All targets are checked before any
    /// fault is applied, so nothing is changed then.
    fn apply_faults<I>(&mut self, faults: I)
    where
        I: Iterator<Item = (Fault, usize)>,
    {
        let faults = faults.collect::<Vec<_>>();
        let chunk_indices = faults
            .iter()
            .map(|&(_, bit_index)| self.pristine.layout().locate(bit_index).chunk_index)
            .collect::<Vec<_>>();

        // Copying the chunks one by one in `chunk_mut` would shift the overlay
        // for every new chunk.
        self.copy_chunks(chunk_indices);

        for (fault, bit_index) in faults {
            self.apply_fault(fault, bit_index);
        }
    }
}
