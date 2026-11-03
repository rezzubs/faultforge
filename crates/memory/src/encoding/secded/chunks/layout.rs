use super::{DecodeChunksError, EncodeChunksError};
use crate::encoding::secded::encoded_bit_count;
use std::ops::Range;

/// The sizes of [`EncodedChunks`](super::EncodedChunks).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct ChunkLayout {
    /// The number of data bytes in each chunk.
    data_bytes_per_chunk: usize,
    /// The number of encoded bits in each chunk, derived from
    /// `data_bytes_per_chunk`.
    encoded_bits_per_chunk: usize,
    /// The number of data bytes that were encoded, without padding.
    data_byte_count: usize,
}

impl ChunkLayout {
    /// The layout of `data_byte_count` data bytes in chunks of
    /// `data_bytes_per_chunk` bytes.
    pub(super) fn new(
        data_bytes_per_chunk: usize,
        data_byte_count: usize,
    ) -> Result<Self, EncodeChunksError> {
        if data_byte_count == 0 {
            return Err(EncodeChunksError::SourceEmpty);
        }
        if data_bytes_per_chunk == 0 {
            return Err(EncodeChunksError::ZeroDataBytesPerChunk);
        }
        let data_bits_per_chunk = data_bytes_per_chunk
            .checked_mul(8)
            .ok_or(EncodeChunksError::DataBytesPerChunkTooLarge)?;
        let encoded_bits_per_chunk = encoded_bit_count(data_bits_per_chunk)
            .expect("the data bit count is non-zero after the previous check");

        Ok(Self {
            data_bytes_per_chunk,
            encoded_bits_per_chunk,
            data_byte_count,
        })
    }

    /// The number of data bytes in each chunk.
    pub(super) fn data_bytes_per_chunk(&self) -> usize {
        self.data_bytes_per_chunk
    }

    /// The number of encoded bits in each chunk.
    pub(super) fn encoded_bits_per_chunk(&self) -> usize {
        self.encoded_bits_per_chunk
    }

    /// The number of data bytes that were encoded, without padding.
    pub(super) fn data_byte_count(&self) -> usize {
        self.data_byte_count
    }

    /// The number of bytes each encoded chunk takes up.
    ///
    /// The bits after `encoded_bits_per_chunk` in a chunk's last byte are
    /// padding.
    pub(super) fn encoded_bytes_per_chunk(&self) -> usize {
        self.encoded_bits_per_chunk.div_ceil(8)
    }

    /// The number of chunks, including a padded last chunk.
    pub(super) fn chunk_count(&self) -> usize {
        self.data_byte_count.div_ceil(self.data_bytes_per_chunk)
    }

    /// The number of encoded bits in all chunks, without the padding bits.
    pub(super) fn bit_count(&self) -> usize {
        self.chunk_count() * self.encoded_bits_per_chunk
    }

    /// Map a bit index of the encoded bits to a bit index in the encoded
    /// chunks stored one after the other.
    ///
    /// The two differ when a chunk has padding bits, because the encoded bit
    /// indices skip them.
    ///
    /// # Panics
    ///
    /// Panics if `bit_index` is out of bounds.
    pub(super) fn encoded_bytes_bit_index(&self, bit_index: usize) -> usize {
        assert!(
            bit_index < self.bit_count(),
            "bit {bit_index} is out of bounds for {} bits",
            self.bit_count()
        );
        let chunk_index = bit_index / self.encoded_bits_per_chunk;
        // Never reaches the padding because it's below the encoded bit count.
        let bit_in_chunk = bit_index % self.encoded_bits_per_chunk;
        chunk_index * self.encoded_bytes_per_chunk() * 8 + bit_in_chunk
    }

    /// Check that `data_byte_range` is a valid range of data bytes and that `dest`
    /// has as many bytes.
    pub(super) fn check_range(
        &self,
        data_byte_range: &Range<usize>,
        dest: &[u8],
    ) -> Result<(), DecodeChunksError> {
        if data_byte_range.start > data_byte_range.end || data_byte_range.end > self.data_byte_count
        {
            return Err(DecodeChunksError::InvalidRange {
                start: data_byte_range.start,
                end: data_byte_range.end,
                data_byte_count: self.data_byte_count,
            });
        }
        if dest.len() != data_byte_range.len() {
            return Err(DecodeChunksError::LengthMismatch {
                expected: data_byte_range.len(),
                actual: dest.len(),
            });
        }
        Ok(())
    }
}
