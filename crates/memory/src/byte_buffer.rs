mod impls;
#[cfg(test)]
mod tests;

use crate::{BitBuffer, CopyIntoResult};

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
}
