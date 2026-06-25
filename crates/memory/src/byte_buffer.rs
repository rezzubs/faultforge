mod impls;
#[cfg(test)]
mod tests;

use crate::{
    BitBuffer, CopyIntoResult,
    chunks::{ByteChunks, Chunks, ChunksCreationError, DynChunks},
};

/// A [`BitBuffer`] that is byte addressable.
pub trait ByteBuffer: BitBuffer {
    /// Return the number of bytes in the buffer.
    fn byte_count(&self) -> usize;

    /// Get the byte at index `n`.
    fn get_byte(&self, n: usize) -> u8;

    /// Set byte `n` to `value`.
    fn set_byte(&mut self, n: usize, value: u8);

    /// [`ByteBuffer::copy_into_chunked`] with start offsets for `self` and `dest`.
    ///
    /// `self_offset` can be useful for copying into many sequential destinations.
    ///
    /// `dest_offset` can be useful for copying different sources into the same destination.
    fn copy_into_chunked_offset<D>(
        &self,
        self_offset: usize,
        dest_offset: usize,
        dest: &mut D,
    ) -> CopyIntoResult
    where
        D: ByteBuffer,
    {
        let remaining_source = self.byte_count().saturating_sub(self_offset);
        let remaining_dest = dest.byte_count().saturating_sub(dest_offset);

        if remaining_source == 0 {
            return CopyIntoResult::exhausted(0);
        }

        if remaining_dest == 0 {
            return CopyIntoResult::partial(0);
        }

        for (source_i, dest_i) in
            (self_offset..self.byte_count()).zip(dest_offset..dest.byte_count())
        {
            dest.set_byte(dest_i, self.get_byte(source_i));
        }

        if remaining_source <= remaining_dest {
            CopyIntoResult::exhausted(remaining_source)
        } else {
            CopyIntoResult::partial(remaining_dest)
        }
    }

    /// Copy all the bytes from `self` to `other`
    ///
    /// See [`ByteBuffer::copy_into_chunked_offset`] for copying from/to multiple sequential buffers.
    #[must_use]
    fn copy_into_chunked<D>(&self, dest: &mut D) -> CopyIntoResult
    where
        D: ByteBuffer,
    {
        self.copy_into_chunked_offset(0, 0, dest)
    }

    /// Convert to chunks of equal length where each chunk is a number of bytes long.
    ///
    /// If the number of bytes in the buffer isn't a multiple of the number of bytes per chunk then
    /// it will result in a number of bytes of (essentially useless) padding at the end of the final
    /// chunk.
    ///
    /// Returns [`None`] if the buffer is empty.
    fn to_byte_chunks(&self, bytes_per_chunk: usize) -> Result<ByteChunks, ChunksCreationError>
    where
        Self: std::marker::Sized,
    {
        ByteChunks::from_buffer(self, bytes_per_chunk)
    }

    /// Convert to chunks of equal length.
    ///
    /// Choose the optimal format automatically. For manual selection see:
    /// - [`ByteBuffer::to_byte_chunks`]
    /// - [`ByteBuffer::to_dyn_chunks`]
    ///
    /// If the number of bits in the buffer isn't a multiple of the number of
    /// bits per chunk then it will result in a number of bits of (essentially
    /// useless) padding at the end of the final chunk.
    fn to_chunks(&self, bits_per_chunk: usize) -> Result<Chunks, ChunksCreationError>
    where
        Self: std::marker::Sized,
    {
        Chunks::from_buffer(self, bits_per_chunk)
    }

    fn to_dyn_chunks(&self, bits_per_chunk: usize) -> Result<DynChunks, ChunksCreationError>
    where
        Self: std::marker::Sized,
    {
        DynChunks::from_buffer(self, bits_per_chunk)
    }
}
