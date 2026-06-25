mod bit;
mod impls;
#[cfg(feature = "numpy")]
mod impls_numpy;
mod iter;
mod result;
#[cfg(test)]
mod tests;

pub use bit::Bit;
pub use iter::Bits;
pub use result::{CopyIntoResult, CopyIntoStatus, OutOfBoundsError};

use crate::Fault;

/// A [`BitBuffer`] with a compile-time known length.
pub trait SizedBitBuffer: BitBuffer {
    /// Total number of bits in the buffer.
    ///
    /// Must be an exact match with [`BitBuffer::bit_count`].
    const BITS_COUNT: usize;
}

/// A type which supports bit operations.
///
/// # Notes for implementors
///
/// - If the number of bits for the type is known at compile time consider
///   additionally implementing [`SizedBitBuffer`].
/// - There is a default implementation for [`BitBuffer::flip_bit`] but many types should
///   probably override it for improved performance.
pub trait BitBuffer {
    /// Number of bits stored by this buffer.
    fn bit_count(&self) -> usize;

    /// Set a bit with index `bit_index` to 1.
    #[doc(alias = "set_one")]
    fn set_1(&mut self, bit_index: usize);

    /// Set a bit with index `bit_index` to 0.
    #[doc(alias = "set_zero")]
    fn set_0(&mut self, bit_index: usize);

    /// Check if the bit at index `bit_index` is 1.
    #[doc(alias = "is_one")]
    fn is_1(&self, bit_index: usize) -> bool;

    /// Check if the bit with index `bit_index` is 0.
    #[doc(alias = "is_zero")]
    fn is_0(&self, bit_index: usize) -> bool {
        !self.is_1(bit_index)
    }

    /// Flip the bit with index `bit_index`.
    fn flip_bit(&mut self, bit_index: usize) {
        if self.is_0(bit_index) {
            self.set_1(bit_index);
        } else {
            self.set_0(bit_index);
        }
    }

    /// Get the bit at index `bit_index`.
    fn bit_at(&self, bit_index: usize) -> Bit {
        if self.is_1(bit_index) {
            Bit::One
        } else {
            Bit::Zero
        }
    }

    /// Apply a fault to a single targeted bit.
    fn apply_fault(&mut self, fault: Fault, bit_index: usize) {
        match fault {
            Fault::Flip => self.flip_bit(bit_index),
            Fault::StuckAt(bit) => match bit {
                Bit::Zero => self.set_0(bit_index),
                Bit::One => self.set_1(bit_index),
            },
        }
    }

    /// Apply many faults in sequence.
    ///
    /// Takes an iterator of `(fault, bit_index)` pairs.
    fn apply_faults<I>(&mut self, faults: I)
    where
        I: Iterator<Item = (Fault, usize)>,
    {
        for (fault, bit_index) in faults {
            self.apply_fault(fault, bit_index);
        }
    }

    /// Iterate over the bits of the array.
    fn bits<'a>(&'a self) -> Bits<'a, Self> {
        Bits::new(self)
    }

    /// Count how many bits are 1.
    fn count_1_bits(&self) -> usize {
        self.bits().filter(|bit| bit.is_1()).count()
    }

    /// Count how many bits are 0.
    fn count_0_bits(&self) -> usize {
        self.bits().filter(|bit| bit.is_0()).count()
    }

    /// Return true if the number 1 bits is even.
    fn total_parity_is_even(&self) -> bool {
        self.count_1_bits().is_multiple_of(2)
    }

    /// Return a string of the bit representation.
    ///
    /// For example, `5u8` would become `0b101`.
    fn bit_string(&self) -> String {
        let bits = self
            .bits()
            .map(|bit| match bit {
                Bit::One => '1',
                Bit::Zero => '0',
            })
            // TODO: implement double ended iteration for Bits to remove the collect + rev.
            .collect::<Vec<char>>()
            .into_iter()
            .rev();

        "0b".chars().chain(bits).collect()
    }

    /// [`BitBuffer::copy_into`] with start offsets for `self` and
    /// `destination`.
    ///
    /// `self_offset` can be useful for splitting a single buffer among multiple
    /// others.
    ///
    /// `destination_offset` can be useful for filling the same destination with
    /// bits from multiple sources.
    #[must_use]
    fn copy_into_offset<D>(
        &self,
        self_offset: usize,
        destination_offset: usize,
        destination: &mut D,
    ) -> CopyIntoResult
    where
        D: BitBuffer,
    {
        let remaining_source = self.bit_count().saturating_sub(self_offset);
        let remaining_dest = destination.bit_count().saturating_sub(destination_offset);

        if remaining_source == 0 {
            return CopyIntoResult::exhausted(0);
        }

        if remaining_dest == 0 {
            return CopyIntoResult::partial(0);
        }

        for (source_i, dest_i) in
            (self_offset..self.bit_count()).zip(destination_offset..destination.bit_count())
        {
            if self.is_1(source_i) {
                destination.set_1(dest_i);
            } else {
                destination.set_0(dest_i);
            }
        }

        if remaining_source <= remaining_dest {
            CopyIntoResult::exhausted(remaining_source)
        } else {
            CopyIntoResult::partial(remaining_dest)
        }
    }

    /// Copy all the bits from `self` to `other`
    ///
    /// Copying bit by bit is pretty slow. If both the source and destination
    /// buffers also satisfy [`crate::ByteBuffer`] then
    /// [`crate::ByteBuffer::copy_into_chunked`] should be used for better
    /// performance.
    ///
    /// See [`BitBuffer::copy_into_offset`] for copying from/to multiple
    /// sequential buffers.
    #[must_use]
    fn copy_into<D>(&self, dest: &mut D) -> CopyIntoResult
    where
        D: BitBuffer,
    {
        self.copy_into_offset(0, 0, dest)
    }
}
