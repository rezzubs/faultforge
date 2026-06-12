use crate::{BitBuffer, ByteBuffer};

/// Error for [`UniformSequence::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, thiserror::Error)]
#[error("item at index {0} has a different length than the rest of the sequence")]
pub struct NonMatchingItemError(usize);

/// Gives a [`BitBuffer`] implementation to sequences where the items cannot
/// satisfy [`SizedBitBuffer`](crate::SizedBitBuffer).
///
/// All the elements of this sequence are expected to have the same number of
/// bits. If this condition cannot be upheld then
/// [`NonUniformSequence`](super::NonUniformSequence) should be used instead.
///
/// If a sequence already satisfies [`BitBuffer`] by itself then this wrapper
/// provides no value.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Default, Hash)]
pub struct UniformSequence<S> {
    item_bit_count: usize,
    item_count: usize,
    sequence: S,
}

impl<S> UniformSequence<S> {
    /// Create a new [`UniformSequence`] from the given sequence.
    ///
    /// Returns an error if the sequence is empty or if not all items have the same bit count.
    pub fn new<T>(sequence: S) -> Result<Self, NonMatchingItemError>
    where
        for<'a> &'a S: IntoIterator<Item = &'a T>,
        T: BitBuffer,
    {
        let mut item_bit_count: Option<usize> = None;
        let mut item_count = 0;
        for (i, item) in (&sequence).into_iter().enumerate() {
            match item_bit_count {
                Some(prev) => {
                    if item.bit_count() != prev {
                        return Err(NonMatchingItemError(i));
                    }
                }
                None => item_bit_count = Some(item.bit_count()),
            }
            item_count += 1;
        }

        let item_bit_count = item_bit_count.unwrap_or(0);

        Ok(Self::new_unchecked(sequence, item_bit_count, item_count))
    }

    /// Creates a new [`UniformSequence`] without checking that all items have
    /// the same bit count.
    ///
    /// If the bits do not have the same bit count then the
    /// behavior will be wrong.
    ///
    /// See also [`UniformSequence::new`].
    pub fn new_unchecked(sequence: S, item_bit_count: usize, item_count: usize) -> Self {
        Self {
            item_bit_count,
            item_count,
            sequence,
        }
    }

    pub fn item_bit_count(&self) -> usize {
        self.item_bit_count
    }

    pub fn items_count(&self) -> usize {
        self.item_count
    }

    pub fn inner(&self) -> &S {
        &self.sequence
    }

    #[cfg(test)]
    pub(crate) fn inner_mut(&mut self) -> &mut S {
        &mut self.sequence
    }

    pub fn into_inner(self) -> S {
        self.sequence
    }
}

impl<S, T> BitBuffer for UniformSequence<S>
where
    S: std::ops::IndexMut<usize, Output = T>,
    T: BitBuffer,
{
    fn bit_count(&self) -> usize {
        self.item_count * self.item_bit_count
    }

    fn set_1(&mut self, bit_index: usize) {
        debug_assert!(bit_index < self.bit_count());
        let item_index = bit_index / self.item_bit_count;
        self.sequence[item_index].set_1(bit_index % self.item_bit_count)
    }

    fn set_0(&mut self, bit_index: usize) {
        debug_assert!(bit_index < self.bit_count());
        let item_index = bit_index / self.item_bit_count;
        self.sequence[item_index].set_0(bit_index % self.item_bit_count)
    }

    fn is_1(&self, bit_index: usize) -> bool {
        debug_assert!(bit_index < self.bit_count());
        let item_index = bit_index / self.item_bit_count;
        self.sequence[item_index].is_1(bit_index % self.item_bit_count)
    }

    fn flip_bit(&mut self, bit_index: usize) {
        debug_assert!(bit_index < self.bit_count());
        let item_index = bit_index / self.item_bit_count;
        self.sequence[item_index].flip_bit(bit_index % self.item_bit_count)
    }
}

impl<S, T> ByteBuffer for UniformSequence<S>
where
    S: std::ops::IndexMut<usize, Output = T>,
    for<'a> &'a S: IntoIterator<Item = &'a T>,
    T: ByteBuffer,
{
    fn byte_count(&self) -> usize {
        let Some(first) = self.sequence.into_iter().next() else {
            debug_assert_eq!(self.item_count, 0);
            debug_assert_eq!(self.item_bit_count, 0);
            return 0;
        };

        first.byte_count() * self.item_count
    }

    fn get_byte(&self, n: usize) -> u8 {
        debug_assert!(n < self.byte_count());
        let item_index = (n * 8) / self.item_bit_count;
        let index_in_item = ((n * 8) % self.item_bit_count) / 8;
        self.sequence[item_index].get_byte(index_in_item)
    }

    fn set_byte(&mut self, n: usize, value: u8) {
        debug_assert!(n < self.byte_count());
        let item_index = (n * 8) / self.item_bit_count;
        let index_in_item = ((n * 8) % self.item_bit_count) / 8;
        self.sequence[item_index].set_byte(index_in_item, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creation_error() {
        let result = UniformSequence::new([vec![1, 2], vec![3, 4], vec![5, 6, 7], vec![8, 9, 10]]);
        assert_eq!(result, Err(NonMatchingItemError(2)));
    }

    #[test]
    fn empty() {
        let buffer: Vec<[u8; 6]> = vec![];
        let uniform = UniformSequence::new(buffer).unwrap();

        assert_eq!(uniform.items_count(), 0);
        assert_eq!(uniform.item_bit_count(), 0);
        assert_eq!(uniform.byte_count(), 0);
        assert_eq!(uniform.bit_count(), 0);
    }

    #[test]
    fn common_ops() {
        let mut buf = UniformSequence::new([0u8, 0b1010u8]).unwrap();
        assert_eq!(buf.bit_count(), 16);
        assert_eq!(buf.item_bit_count(), 8);

        for i in 0..9 {
            assert!(buf.is_0(i));
        }
        assert!(buf.is_1(9));
        assert!(buf.is_0(10));
        assert!(buf.is_1(11));
        for i in 12..16 {
            assert!(buf.is_0(i));
        }

        buf.set_0(10);
        assert!(buf.is_0(10));
        buf.set_1(10);
        assert!(buf.is_1(10));

        buf.flip_bit(10);
        assert!(buf.is_0(10));
        buf.flip_bit(10);
        assert!(buf.is_1(10));

        assert_eq!(buf.byte_count(), 2);
        assert_eq!(buf.get_byte(0), 0);
        assert_eq!(buf.get_byte(1), 0b1110);
        buf.set_byte(0, u8::MAX);
        assert_eq!(buf.get_byte(0), u8::MAX);

        let inner = buf.into_inner();
        assert_eq!(inner, [u8::MAX, 0b1110]);
    }

    #[test]
    fn multi_byte_items() {
        let mut buf = UniformSequence::new([0u32, 0xffu32]).unwrap();
        assert_eq!(buf.byte_count(), 8);
        assert_eq!(buf.item_bit_count(), 32);

        assert_eq!(buf.get_byte(4), 0xff);
        buf.set_byte(1, 0xaa);
        assert_eq!(buf.get_byte(1), 0xaa);
        buf.set_byte(5, 0x88);
        assert_eq!(buf.get_byte(5), 0x88);
        buf.set_byte(4, 0xbb);
        assert_eq!(buf.get_byte(4), 0xbb);
    }
}
