use crate::BitBuffer;

/// How many bytes does it take to store `bit_count` bits.
#[inline]
#[must_use]
fn bytes_to_store_n_bits(bit_count: usize) -> usize {
    match (bit_count / 8, bit_count % 8) {
        (0, 0) => 0,
        (a, 0) => a,
        (a, _) => a + 1,
    }
}

/// Limit the number of bits in a [`BitBuffer`].
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct Limited<T> {
    buffer: T,
    bit_count: usize,
}

impl Limited<Vec<u8>> {
    // Create an arbitrary length sequence stored in a vector of bytes.
    //
    // All bytes are initialized to zero.
    #[must_use]
    pub fn bytes(bit_count: usize) -> Self {
        let byte_count = bytes_to_store_n_bits(bit_count);
        Limited::new(vec![0u8; byte_count], bit_count)
            .expect("cannot fail as long as the function above is correct")
    }
}

impl<T> Limited<T>
where
    T: BitBuffer,
{
    /// Create a new limited bit buffer.
    pub fn new(buffer: T, bit_count: usize) -> Option<Self> {
        (bit_count <= buffer.bit_count()).then_some(Self { buffer, bit_count })
    }

    /// Extract the original bitbuffer.
    pub fn into_inner(self) -> T {
        self.buffer
    }
}

impl<T> BitBuffer for Limited<T>
where
    T: BitBuffer,
{
    fn bit_count(&self) -> usize {
        self.bit_count
    }

    fn is_1(&self, bit_index: usize) -> bool {
        if bit_index >= self.bit_count {
            panic!("{bit_index} is out of bounds");
        }
        self.buffer.is_1(bit_index)
    }

    fn set_0(&mut self, bit_index: usize) {
        if bit_index >= self.bit_count {
            panic!("{bit_index} is out of bounds");
        }
        self.buffer.set_0(bit_index)
    }

    fn set_1(&mut self, bit_index: usize) {
        if bit_index >= self.bit_count {
            panic!("{bit_index} is out of bounds");
        }
        self.buffer.set_1(bit_index)
    }

    fn flip_bit(&mut self, bit_index: usize) {
        if bit_index >= self.bit_count {
            panic!("{bit_index} is out of bounds");
        }
        self.buffer.flip_bit(bit_index)
    }
}

#[cfg(test)]
mod tests {
    use crate::CopyIntoResult;

    use super::*;

    #[test]
    fn in_bounds_is() {
        assert!(Limited::new(0b1001u8, 4).unwrap().is_1(0));
        assert!(Limited::new(0b1001u8, 4).unwrap().is_0(1));
        assert!(Limited::new(0b1001u8, 4).unwrap().is_0(2));
        assert!(Limited::new(0b1001u8, 4).unwrap().is_1(3));

        assert!(Limited::new(0b10000000u8, 8).unwrap().is_0(6));
        assert!(Limited::new(0b10000000u8, 8).unwrap().is_1(7));
    }

    #[test]
    fn in_bounds_set() {
        let mut buf = Limited::new(0xffu8, 4).unwrap();

        for i in 0..4 {
            buf.set_0(i);
        }

        assert_eq!(buf.into_inner(), 0xf0);

        for i in 0..4 {
            buf.set_1(i)
        }

        assert_eq!(buf.into_inner(), 0xff);
    }

    #[test]
    fn in_bounds_flip() {
        let mut buf = Limited::new(0xffu8, 4).unwrap();

        buf.flip_bit(0);

        assert_eq!(buf.into_inner(), 0b11111110u8);

        for i in 1..4 {
            buf.flip_bit(i);
        }

        assert_eq!(buf.into_inner(), 0xf0);

        for i in 0..4 {
            buf.flip_bit(i)
        }

        assert_eq!(buf.into_inner(), 0xff);
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_is() {
        Limited::new(0b0000, 4).unwrap().is_1(4);
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_set_11() {
        let mut buf = Limited::new(0x00, 4).unwrap();
        buf.set_1(4);
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_set_0() {
        let mut buf = Limited::new(0xff, 4).unwrap();
        buf.set_0(4);
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_flip() {
        let mut buf = Limited::new(0xffu8, 4).unwrap();
        buf.flip_bit(4);
    }

    #[test]
    fn invalid_bit_count() {
        assert!(Limited::new(0u8, 8).is_some());
        assert!(Limited::new(0u8, 9).is_none());
    }

    #[test]
    fn bit_copy() {
        let source = Limited::new([255u8; 2], 15).unwrap();
        let mut dest = [0u8; 2];

        let result = source.copy_into(&mut dest);
        assert_eq!(result, CopyIntoResult::exhausted(source.bit_count()));
        assert_eq!(dest, [255u8, 0b01111111u8]);

        let source = [255u8; 2];
        let mut dest = Limited::new([0u8; 2], 15).unwrap();

        let result = source.copy_into(&mut dest);
        assert_eq!(result, CopyIntoResult::partial(dest.bit_count()));
        assert_eq!(dest.into_inner(), [255u8, 0b01111111u8]);
    }

    #[test]
    fn bytes_per_bits() {
        assert_eq!(bytes_to_store_n_bits(0), 0);
        assert_eq!(bytes_to_store_n_bits(1), 1);
        assert_eq!(bytes_to_store_n_bits(8), 1);
        assert_eq!(bytes_to_store_n_bits(9), 2);
        assert_eq!(bytes_to_store_n_bits(16), 2);
        assert_eq!(bytes_to_store_n_bits(17), 3);
        assert_eq!(bytes_to_store_n_bits(24), 3);
        assert_eq!(bytes_to_store_n_bits(25), 4);
    }
}
