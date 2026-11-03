mod overlay;
#[cfg(test)]
mod tests;

use crate::{
    BitBuffer, Fault,
    encoding::secded::{DecodeChunksError, EncodeChunksError, EncodedChunks},
};
use overlay::Overlay;
use std::{ops::Range, sync::Arc};

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
/// From `benches/cow.rs` (clone, apply faults and decode 1 MiB of data with
/// 64-bit chunks):
///
/// | Bit error rate | Overlay memory of a full copy | Time of a full copy |
/// |----------------|-------------------------------|---------------------|
/// | 1e-4           | 1.4%                          | 1.03x               |
/// | 1e-3           | 13%                           | 1.06x               |
/// | 1e-2           | 97%                           | 1.32x               |
/// | 1e-1           | 189%                          | 1.9x                |
/// | 1              | 189%                          | 11.6x               |
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
