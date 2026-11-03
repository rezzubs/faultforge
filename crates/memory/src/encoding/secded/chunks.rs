mod layout;
#[cfg(test)]
mod tests;

use crate::{BitBuffer, Limited, encoding::secded::encode_into};
use layout::ChunkLayout;
use std::ops::Range;

/// An error for [`EncodedChunks::encode`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EncodeChunksError {
    /// The source buffer is empty.
    #[error("cannot encode an empty source buffer")]
    SourceEmpty,
    /// The number of data bytes per chunk is 0.
    #[error("the number of data bytes per chunk must be non-zero")]
    ZeroDataBytesPerChunk,
    /// The number of data bits per chunk doesn't fit into a `usize`.
    #[error("the number of data bytes per chunk is too large")]
    DataBytesPerChunkTooLarge,
}

/// An error for decoding [`EncodedChunks`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DecodeChunksError {
    /// The destination doesn't have the number of bytes that were encoded.
    #[error("the destination should have {expected} bytes, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },
    /// The byte range ends before it starts or goes beyond the data.
    #[error("the range {start}..{end} is invalid for {data_byte_count} data bytes")]
    InvalidRange {
        start: usize,
        end: usize,
        data_byte_count: usize,
    },
}

/// A byte buffer split into chunks that are each SECDED encoded.
///
/// Chunk `i` holds the data bytes starting at `i * data_bytes_per_chunk`. If
/// the source doesn't divide evenly, the last chunk is padded with 0 bytes.
/// The padding is dropped on decode.
///
/// For example, with 4 data bytes per chunk, encoding the 6 bytes of a buffer
/// `a` followed by the 5 bytes of a buffer `b` gives 3 chunks:
///
/// ```text
/// data byte index | 0  1  2  3  | 4  5  6  7  | 8  9  10 (11)
/// data byte       | a0 a1 a2 a3 | a4 a5 b0 b1 | b2 b3 b4 (0)
/// chunk           | 0           | 1           | 2
/// ```
///
/// Chunks don't depend on where one buffer ends and the next starts, so chunk
/// 1 holds bytes of both buffers. Chunk 2 is padded with a 0 byte.
///
/// As a [`BitBuffer`], the encoded chunks are laid out one after the other:
/// bit `i` is bit `i % encoded_bits_per_chunk` of chunk
/// `i / encoded_bits_per_chunk`. Use this to inject faults.
///
/// Decoding never modifies the stored chunks. Single errors are only
/// corrected in the decoded output, so decoding again gives the same result.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EncodedChunks {
    /// The encoded chunks, each starting at a byte boundary: chunk `i` is
    /// `encoded_bytes[i * encoded_bytes_per_chunk..(i + 1) * encoded_bytes_per_chunk]`.
    ///
    /// The bits after `encoded_bits_per_chunk` in a chunk's last byte are
    /// padding. They're never written, so they stay 0 and the derived `Eq`
    /// and `Hash` only depend on the encoded bits.
    encoded_bytes: Vec<u8>,
    /// The sizes of the chunks and which data bytes each one holds.
    layout: ChunkLayout,
}

impl EncodedChunks {
    /// Encode `source` in chunks of `data_bytes_per_chunk` bytes.
    pub fn encode(source: &[u8], data_bytes_per_chunk: usize) -> Result<Self, EncodeChunksError> {
        let layout = ChunkLayout::new(data_bytes_per_chunk, source.len())?;
        let encoded_bits_per_chunk = layout.encoded_bits_per_chunk();
        let encoded_bytes_per_chunk = layout.encoded_bytes_per_chunk();

        let full_data_chunks = source.chunks_exact(data_bytes_per_chunk);
        // The bytes that don't fill a whole chunk. Empty if the source divides
        // evenly.
        let last_data = full_data_chunks.remainder();

        let full_chunk_count = full_data_chunks.len();
        let mut encoded_bytes = vec![0u8; layout.chunk_count() * encoded_bytes_per_chunk];

        let (full_encoded, last_encoded) =
            encoded_bytes.split_at_mut(full_chunk_count * encoded_bytes_per_chunk);
        let full_encoded_chunks = full_encoded.chunks_exact_mut(encoded_bytes_per_chunk);
        assert_eq!(full_data_chunks.len(), full_encoded_chunks.len());
        for (data_chunk, encoded_chunk) in full_data_chunks.zip(full_encoded_chunks) {
            encode_chunk(data_chunk, encoded_chunk, encoded_bits_per_chunk);
        }

        if last_data.is_empty() {
            assert!(last_encoded.is_empty());
        } else {
            assert_eq!(last_encoded.len(), encoded_bytes_per_chunk);
            // The encoded size is fixed per chunk, so the remaining bytes are
            // encoded from a copy padded with 0 bytes.
            let mut padded = vec![0u8; data_bytes_per_chunk];
            padded[..last_data.len()].copy_from_slice(last_data);
            encode_chunk(&padded, last_encoded, encoded_bits_per_chunk);
        }

        Ok(Self {
            encoded_bytes,
            layout,
        })
    }

    /// Decode all chunks into `dest`.
    ///
    /// `dest` must have exactly [`EncodedChunks::data_byte_count`] bytes.
    pub fn decode_into(&self, dest: &mut [u8]) -> Result<(), DecodeChunksError> {
        self.decode_range_into(0..self.data_byte_count(), dest)
    }

    /// Like [`EncodedChunks::decode_into`] but also returns the chunks in
    /// which a double error was detected.
    ///
    /// The indices are in ascending order. They're empty if no double errors
    /// were detected.
    pub fn decode_into_with_double_errors(
        &self,
        dest: &mut [u8],
    ) -> Result<Vec<usize>, DecodeChunksError> {
        self.decode_range_into_with_double_errors(0..self.data_byte_count(), dest)
    }

    /// Decode the data bytes in `data_byte_range` into `dest`.
    ///
    /// `dest` must have exactly as many bytes as `data_byte_range`. Decoding a
    /// sequence of adjacent ranges gives the same bytes as decoding all chunks
    /// at once.
    ///
    /// The range doesn't need to start or end at a chunk boundary. For
    /// example, `b` from the example in [`EncodedChunks`] is the range
    /// `6..11`. Decoding it:
    /// 1. Decodes chunk 1 and copies its last 2 bytes, `b0 b1`.
    /// 2. Decodes chunk 2 and copies its first 3 bytes, `b2 b3 b4`.
    pub fn decode_range_into(
        &self,
        data_byte_range: Range<usize>,
        dest: &mut [u8],
    ) -> Result<(), DecodeChunksError> {
        self.decode_range_with(data_byte_range, dest, |_| {})
    }

    /// Like [`EncodedChunks::decode_range_into`] but also returns the chunks
    /// in which a double error was detected.
    ///
    /// The indices are in ascending order. They include the chunks that are
    /// only partly in `data_byte_range`, so adjacent ranges can report the same
    /// chunk. For example, in the example in [`EncodedChunks`], a double error
    /// in chunk 1 is reported when decoding `a` and again when decoding `b`.
    pub fn decode_range_into_with_double_errors(
        &self,
        data_byte_range: Range<usize>,
        dest: &mut [u8],
    ) -> Result<Vec<usize>, DecodeChunksError> {
        // Chunks are decoded in ascending order, so the indices stay sorted.
        let mut double_error_chunks = Vec::new();
        self.decode_range_with(data_byte_range, dest, |chunk_index| {
            double_error_chunks.push(chunk_index);
        })?;
        Ok(double_error_chunks)
    }

    /// Get the number of chunks.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.layout.chunk_count()
    }

    /// Get the number of data bytes in each chunk.
    #[must_use]
    pub fn data_bytes_per_chunk(&self) -> usize {
        self.layout.data_bytes_per_chunk()
    }

    /// Get the number of encoded bits in each chunk.
    #[must_use]
    pub fn encoded_bits_per_chunk(&self) -> usize {
        self.layout.encoded_bits_per_chunk()
    }

    /// Get the number of data bytes that were encoded.
    ///
    /// This is the number of bytes a decode destination must have.
    #[must_use]
    pub fn data_byte_count(&self) -> usize {
        self.layout.data_byte_count()
    }

    /// Get the number of bytes each encoded chunk takes up.
    ///
    /// The bits after [`EncodedChunks::encoded_bits_per_chunk`] in a chunk's
    /// last byte are padding.
    #[must_use]
    pub fn encoded_bytes_per_chunk(&self) -> usize {
        self.layout.encoded_bytes_per_chunk()
    }

    /// The sizes of the chunks.
    pub(super) fn layout(&self) -> &ChunkLayout {
        &self.layout
    }

    /// The slice of `encoded_bytes` that holds chunk `chunk_index`.
    ///
    /// It's [`EncodedChunks::encoded_bytes_per_chunk`] bytes long, including
    /// the padding bits in its last byte.
    ///
    /// Panics if the chunk index is out of bounds.
    pub(super) fn encoded_chunk(&self, chunk_index: usize) -> &[u8] {
        let start = chunk_index * self.encoded_bytes_per_chunk();
        &self.encoded_bytes[start..start + self.encoded_bytes_per_chunk()]
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
        self.layout.decode_range(
            data_byte_range,
            dest,
            |chunk_index| self.encoded_chunk(chunk_index),
            on_double_error,
        )
    }
}

/// Encode one chunk of `data_bytes_per_chunk` bytes into `encoded_chunk`.
fn encode_chunk(data_chunk: &[u8], encoded_chunk: &mut [u8], encoded_bits_per_chunk: usize) {
    let mut destination = Limited::new(encoded_chunk, encoded_bits_per_chunk)
        .expect("a chunk has enough bytes for its encoded bits");
    encode_into(data_chunk, &mut destination)
        .expect("the chunk sizes are derived from the data size");
}

impl BitBuffer for EncodedChunks {
    fn bit_count(&self) -> usize {
        self.layout.bit_count()
    }

    fn set_1(&mut self, bit_index: usize) {
        let encoded_bytes_bit_index = self.layout.encoded_bytes_bit_index(bit_index);
        self.encoded_bytes.set_1(encoded_bytes_bit_index);
    }

    fn set_0(&mut self, bit_index: usize) {
        let encoded_bytes_bit_index = self.layout.encoded_bytes_bit_index(bit_index);
        self.encoded_bytes.set_0(encoded_bytes_bit_index);
    }

    fn is_1(&self, bit_index: usize) -> bool {
        self.encoded_bytes
            .is_1(self.layout.encoded_bytes_bit_index(bit_index))
    }

    fn flip_bit(&mut self, bit_index: usize) {
        let encoded_bytes_bit_index = self.layout.encoded_bytes_bit_index(bit_index);
        self.encoded_bytes.flip_bit(encoded_bytes_bit_index);
    }
}
