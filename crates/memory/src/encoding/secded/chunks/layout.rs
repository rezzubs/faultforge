use super::{DecodeChunksError, EncodeChunksError};
use crate::{
    Limited,
    encoding::secded::{DoubleErrorDetection, decode_into, encoded_bit_count},
};
use std::ops::Range;

/// Where an encoded bit is, see [`ChunkLayout::locate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::encoding::secded) struct EncodedBitLocation {
    /// The chunk that holds the bit.
    pub chunk_index: usize,
    /// The index of the bit in the chunk's encoded bytes. Always below
    /// `encoded_bits_per_chunk`, so it's never a padding bit.
    pub bit_in_chunk: usize,
}

/// The sizes of [`EncodedChunks`](super::EncodedChunks).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(in crate::encoding::secded) struct ChunkLayout {
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

    /// Find the chunk of a bit index of the encoded bits and the bit's index
    /// in that chunk.
    ///
    /// # Panics
    ///
    /// Panics if `bit_index` is out of bounds.
    pub(in crate::encoding::secded) fn locate(&self, bit_index: usize) -> EncodedBitLocation {
        assert!(
            bit_index < self.bit_count(),
            "bit {bit_index} is out of bounds for {} bits",
            self.bit_count()
        );
        EncodedBitLocation {
            chunk_index: bit_index / self.encoded_bits_per_chunk,
            // Never reaches the padding because it's below the encoded bit
            // count.
            bit_in_chunk: bit_index % self.encoded_bits_per_chunk,
        }
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
        let location = self.locate(bit_index);
        location.chunk_index * self.encoded_bytes_per_chunk() * 8 + location.bit_in_chunk
    }

    /// Check that `data_byte_range` is a valid range of data bytes.
    pub(in crate::encoding::secded) fn check_range_bounds(
        &self,
        data_byte_range: &Range<usize>,
    ) -> Result<(), DecodeChunksError> {
        if data_byte_range.start > data_byte_range.end || data_byte_range.end > self.data_byte_count
        {
            return Err(DecodeChunksError::InvalidRange {
                start: data_byte_range.start,
                end: data_byte_range.end,
                data_byte_count: self.data_byte_count,
            });
        }
        Ok(())
    }

    /// Check that `data_byte_range` is a valid range of data bytes and that `dest`
    /// has as many bytes.
    pub(super) fn check_range(
        &self,
        data_byte_range: &Range<usize>,
        dest: &[u8],
    ) -> Result<(), DecodeChunksError> {
        self.check_range_bounds(data_byte_range)?;
        if dest.len() != data_byte_range.len() {
            return Err(DecodeChunksError::LengthMismatch {
                expected: data_byte_range.len(),
                actual: dest.len(),
            });
        }
        Ok(())
    }

    /// Decode the data bytes in `data_byte_range` into `dest`, calling
    /// `on_double_error` with the index of every chunk with a detected double
    /// error, in ascending order.
    ///
    /// The layout holds no chunks, so `encoded_chunk` gives the encoded bytes
    /// of the chunk at an index. It's called once for every chunk in the
    /// range, in ascending order. The encoded bytes are not changed.
    ///
    /// # Panics
    ///
    /// Panics if `encoded_chunk` returns a slice that doesn't have
    /// [`ChunkLayout::encoded_bytes_per_chunk`] bytes.
    pub(in crate::encoding::secded) fn decode_range<'chunks>(
        &self,
        data_byte_range: Range<usize>,
        dest: &mut [u8],
        mut encoded_chunk: impl FnMut(usize) -> &'chunks [u8],
        mut on_double_error: impl FnMut(usize),
    ) -> Result<(), DecodeChunksError> {
        self.check_range(&data_byte_range, dest)?;

        let mut decoder = ChunkDecoder::new(self);
        let mut data_chunk = vec![0u8; self.data_bytes_per_chunk];

        // Go through the range one chunk at a time. The range can start and
        // end in the middle of a chunk, so only part of the first and last
        // chunk may be copied.
        let mut data_byte_index = data_byte_range.start;
        while data_byte_index < data_byte_range.end {
            let chunk_index = data_byte_index / self.data_bytes_per_chunk;
            let index_in_chunk = data_byte_index % self.data_bytes_per_chunk;
            // Up to the end of the chunk or of the range, whichever is first.
            let copy_count = (self.data_bytes_per_chunk - index_in_chunk)
                .min(data_byte_range.end - data_byte_index);

            // Panics if `encoded_chunk` returns the wrong number of bytes, see
            // the panics section.
            let detection = decoder.decode(encoded_chunk(chunk_index), &mut data_chunk);
            if detection.is_detected() {
                on_double_error(chunk_index);
            }

            let index_in_dest = data_byte_index - data_byte_range.start;
            dest[index_in_dest..index_in_dest + copy_count]
                .copy_from_slice(&data_chunk[index_in_chunk..index_in_chunk + copy_count]);
            data_byte_index += copy_count;
        }

        Ok(())
    }
}

/// Decodes single chunks of a [`ChunkLayout`] without changing them.
pub(in crate::encoding::secded) struct ChunkDecoder {
    /// A copy of the chunk being decoded. Decoding corrects single errors in
    /// its source, and the stored chunks must stay unchanged.
    encoded_chunk_scratch: Vec<u8>,
    /// The number of encoded bits in each chunk.
    encoded_bits_per_chunk: usize,
}

impl ChunkDecoder {
    /// Create a decoder for the chunks of `layout`.
    pub(in crate::encoding::secded) fn new(layout: &ChunkLayout) -> Self {
        Self {
            encoded_chunk_scratch: vec![0u8; layout.encoded_bytes_per_chunk()],
            encoded_bits_per_chunk: layout.encoded_bits_per_chunk(),
        }
    }

    /// Decode `encoded_chunk` into `data_chunk`.
    ///
    /// `encoded_chunk` is not changed.
    ///
    /// # Panics
    ///
    /// Panics if `encoded_chunk` doesn't have
    /// [`ChunkLayout::encoded_bytes_per_chunk`] bytes or `data_chunk` doesn't
    /// have [`ChunkLayout::data_bytes_per_chunk`] bytes.
    pub(in crate::encoding::secded) fn decode(
        &mut self,
        encoded_chunk: &[u8],
        mut data_chunk: &mut [u8],
    ) -> DoubleErrorDetection {
        // `decode_into` corrects single errors in its source, so decode a
        // copy to keep the stored chunk unchanged. This copy panics if
        // `encoded_chunk` has the wrong number of bytes.
        self.encoded_chunk_scratch.copy_from_slice(encoded_chunk);
        let mut source = Limited::new(
            self.encoded_chunk_scratch.as_mut_slice(),
            self.encoded_bits_per_chunk,
        )
        .expect("the buffer has enough bytes for the encoded bits");
        decode_into(&mut source, &mut data_chunk)
            .expect("the data chunk has the size the encoded chunk was made from")
    }
}
