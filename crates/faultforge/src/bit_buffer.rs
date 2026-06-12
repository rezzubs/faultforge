pub mod bits;
mod byte_chunked;
pub mod chunks;

pub use bits::Bits;
pub use byte_chunked::ByteChunkedBitBuffer;
use chunks::DynChunks;
use picker::Picker;

use crate::{
    bit_buffer::chunks::InvalidChunks,
    buffers::Limited,
    encoding::secded::{encode_into, encoded_bits_count},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CopyIntoResultKind {
    /// The destination was filled before the source could be exhausted.
    Pending,
    /// The source is exhausted.
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CopyIntoResult {
    /// The number of items copied of the fundamental unit of the buffer. Bits or bytes depending on
    /// the function called.
    pub units_copied: usize,
    pub kind: CopyIntoResultKind,
}

impl CopyIntoResult {
    #[must_use]
    pub fn done(units_copied: usize) -> Self {
        Self {
            units_copied,
            kind: CopyIntoResultKind::Done,
        }
    }

    #[must_use]
    pub fn pending(units_copied: usize) -> Self {
        Self {
            units_copied,
            kind: CopyIntoResultKind::Pending,
        }
    }
}

/// An error for cases where a bit buffer operations goes out of bounds.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
#[error("Out of bounds")]
pub struct OutOfBounds;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EncodeError {
    #[error("Cannot encode an empty buffer")]
    Empty,
}

pub trait BitBuffer {
    /// Number of bits stored by this buffer.
    fn bits_count(&self) -> usize;

    /// Set a bit with index `bit_index` to 1.
    fn set_1(&mut self, bit_index: usize);

    /// Set a bit with index `bit_index` to 0.
    fn set_0(&mut self, bit_index: usize);

    /// Check if the bit at index `bit_index` is 1.
    fn is_1(&self, bit_index: usize) -> bool;

    /// Check if the bit with index `bit_index` is 0.
    fn is_0(&self, bit_index: usize) -> bool {
        !self.is_1(bit_index)
    }

    /// Flip the bit with index `bit_index`.
    fn flip_bit(&mut self, bit_index: usize) {
        // NOTE: It's likely that a custom implementation for a specific type will be faster.
        if self.is_0(bit_index) {
            self.set_1(bit_index);
        } else {
            self.set_0(bit_index);
        }
    }

    /// Iterate over the bits of the array.
    fn bits<'a>(&'a self) -> Bits<'a, Self> {
        Bits::new(self)
    }

    /// Return true if the number 1 bits is even.
    fn total_parity_is_even(&self) -> bool {
        self.bits().filter(|is_1| *is_1).count() % 2 == 0
    }

    /// Return a string of the bit representation.
    ///
    /// For example, `5u8` would become `0b00000101`.
    fn bit_string(&self) -> String {
        let bits = self
            .bits()
            .map(|bit| if bit { '1' } else { '0' })
            // FIXME: implement double ended iteration for Bits to remove the collect + rev.
            .collect::<Vec<char>>()
            .into_iter()
            .rev();

        "0b".chars().chain(bits).collect()
    }

    /// Count the number of bits which are 1.
    fn count_1_bits(&self) -> usize {
        self.bits().filter(|is_1| *is_1).count()
    }

    /// Flip exactly n bits randomly in the buffer.
    ///
    /// All bit flips will be unique.
    fn flip_n_bits(&mut self, n: usize) -> Result<(), OutOfBounds> {
        let bits_count = self.bits_count();

        if n > bits_count {
            return Err(OutOfBounds);
        }

        let mut possible_faults = Picker::new(bits_count, rand::rng());

        for _ in 0..n {
            let fault_target = possible_faults.next().expect(
                "we confirmed that n <= bits_count so RandomPicker will always have >= n elements",
            );
            self.flip_bit(fault_target);
        }

        Ok(())
    }

    /// Flip a number of bits by the given bit error rate.
    ///
    /// All bit flips will be unique.
    ///
    /// Returns the number of bits flipped
    ///
    /// # Arguments
    ///
    /// * `ber` - A bit error rate `0..=1`.
    fn flip_by_ber(&mut self, ber: f64) -> Result<usize, OutOfBounds> {
        if !(0f64..=1f64).contains(&ber) {
            return Err(OutOfBounds);
        }

        let flips_count = (self.bits_count() as f64 * ber) as usize;

        self.flip_n_bits(flips_count).unwrap_or_else(|_| {
            unreachable!("We checked the range of `ber` so `flips_count` cannot be out of bounds.")
        });

        Ok(flips_count)
    }

    /// [`BitBuffer::copy_into`] with start offsets for `self` and `dest`.
    ///
    /// `self_offset` can be useful for copying into many sequential destinations.
    ///
    /// `dest_offset` can be useful for copying different sources into the same destination.
    #[must_use]
    fn copy_into_offset<D>(
        &self,
        self_offset: usize,
        dest_offset: usize,
        dest: &mut D,
    ) -> CopyIntoResult
    where
        D: BitBuffer,
    {
        let remaining_source = self.bits_count().saturating_sub(self_offset);
        let remaining_dest = dest.bits_count().saturating_sub(dest_offset);

        if remaining_source == 0 {
            return CopyIntoResult::done(0);
        }

        if remaining_dest == 0 {
            return CopyIntoResult::pending(0);
        }

        for (source_i, dest_i) in
            (self_offset..self.bits_count()).zip(dest_offset..dest.bits_count())
        {
            if self.is_1(source_i) {
                dest.set_1(dest_i);
            } else {
                dest.set_0(dest_i);
            }
        }

        if remaining_source <= remaining_dest {
            CopyIntoResult::done(remaining_source)
        } else {
            CopyIntoResult::pending(remaining_dest)
        }
    }

    /// Copy all the bits from `self` to `other`
    ///
    /// If both the source and destination buffers also satisfy [`ByteChunkedBitBuffer`] then
    /// [`ByteChunkedBitBuffer::copy_into_chunked`] should be used for much greated performance.
    ///
    /// See [`BitBuffer::copy_into_offset`] for copying from/to multiple sequential buffers.
    #[must_use]
    fn copy_into<D>(&self, dest: &mut D) -> CopyIntoResult
    where
        D: BitBuffer,
    {
        self.copy_into_offset(0, 0, dest)
    }

    /// Convert the buffer to chunks of equal length.
    ///
    /// If the number of bits in the buffer isn't a multiple of the number of bits per chunk then
    /// it will result in a number of bits of (essentially useless) padding at the end of the final
    /// chunk.
    fn to_dyn_chunks(&self, bits_per_chunk: usize) -> Result<DynChunks, InvalidChunks>
    where
        Self: Sized,
    {
        DynChunks::from_buffer(self, bits_per_chunk)
    }

    /// Encode the buffer as a hamming code.
    ///
    /// See [`encode_into`] for usage with custom output buffers.
    fn encode(&self) -> Result<Limited<Vec<u8>>, EncodeError>
    where
        Self: std::marker::Sized,
    {
        let encoded_bits_count = encoded_bits_count(self.bits_count()).ok_or(EncodeError::Empty)?;

        let mut dest = Limited::bytes(encoded_bits_count);

        if let Err(err) = encode_into(self, &mut dest) {
            match err {
                crate::encoding::secded::EncodeError::SourceEmpty => {
                    unreachable!("Already checked")
                }
                crate::encoding::secded::EncodeError::LengthMismatch { .. } => {
                    unreachable!("Using the given buffer must be correct")
                }
            }
        };

        Ok(dest)
    }
}

/// A [`BitBuffer`] with a comptime known length.
pub trait SizedBitBuffer: BitBuffer {
    /// Total number of bits in the buffer.
    ///
    /// Must be an exact match with [`BitBuffer::bits_count`].
    const BITS_COUNT: usize;
}

macro_rules! int_impl {
    ($t:ty, $bits:expr) => {
        impl SizedBitBuffer for $t {
            const BITS_COUNT: usize = $bits;
        }

        impl BitBuffer for $t {
            fn bits_count(&self) -> usize {
                Self::BITS_COUNT
            }

            fn set_1(&mut self, bit_index: usize) {
                debug_assert!(bit_index < Self::BITS_COUNT, "{bit_index} is out of bounds");
                *self |= 1 << bit_index
            }

            fn set_0(&mut self, bit_index: usize) {
                debug_assert!(bit_index < Self::BITS_COUNT, "{bit_index} is out of bounds");
                *self &= !(1 << bit_index)
            }

            fn is_1(&self, bit_index: usize) -> bool {
                debug_assert!(bit_index < Self::BITS_COUNT, "{bit_index} is out of bounds");
                (self & (1 << bit_index)) > 0
            }

            fn flip_bit(&mut self, bit_index: usize) {
                debug_assert!(bit_index < Self::BITS_COUNT, "{bit_index} is out of bounds");
                *self ^= 1 << bit_index
            }
        }
    };
}

int_impl!(u8, 8);
int_impl!(u16, 16);
int_impl!(u32, 32);
int_impl!(u64, 64);
int_impl!(u128, 128);
int_impl!(i8, 8);
int_impl!(i16, 16);
int_impl!(i32, 32);
int_impl!(i64, 64);
int_impl!(i128, 128);

macro_rules! float_impl {
    ($t:ty, $bits:expr) => {
        impl SizedBitBuffer for $t {
            const BITS_COUNT: usize = $bits;
        }

        impl BitBuffer for $t {
            fn bits_count(&self) -> usize {
                Self::BITS_COUNT
            }

            fn set_1(&mut self, bit_index: usize) {
                let mut unsigned = self.to_bits();
                unsigned.set_1(bit_index);
                *self = <$t>::from_bits(unsigned)
            }

            fn set_0(&mut self, bit_index: usize) {
                let mut unsigned = self.to_bits();
                unsigned.set_0(bit_index);
                *self = <$t>::from_bits(unsigned)
            }

            fn is_1(&self, bit_index: usize) -> bool {
                let unsigned = self.to_bits();
                unsigned.is_1(bit_index)
            }

            fn flip_bit(&mut self, bit_index: usize) {
                let mut unsigned = self.to_bits();
                unsigned.flip_bit(bit_index);
                *self = <$t>::from_bits(unsigned)
            }
        }
    };
}

float_impl!(f32, 32);
float_impl!(f64, 64);

impl<T> BitBuffer for [T]
where
    T: SizedBitBuffer,
{
    /// Total number of bits in the buffer.
    fn bits_count(&self) -> usize {
        self.len() * T::BITS_COUNT
    }

    fn set_1(&mut self, bit_index: usize) {
        debug_assert!(bit_index < self.bits_count());
        let item_index = bit_index / T::BITS_COUNT;
        self[item_index].set_1(bit_index % T::BITS_COUNT);
    }

    fn set_0(&mut self, bit_index: usize) {
        debug_assert!(bit_index < self.bits_count());
        let item_index = bit_index / T::BITS_COUNT;
        self[item_index].set_0(bit_index % T::BITS_COUNT);
    }

    fn is_1(&self, bit_index: usize) -> bool {
        debug_assert!(bit_index < self.bits_count());
        let item_index = bit_index / T::BITS_COUNT;
        self[item_index].is_1(bit_index % T::BITS_COUNT)
    }

    fn flip_bit(&mut self, bit_index: usize) {
        debug_assert!(bit_index < self.bits_count());
        let item_index = bit_index / T::BITS_COUNT;
        self[item_index].flip_bit(bit_index % T::BITS_COUNT);
    }
}

impl<const N: usize, T> SizedBitBuffer for [T; N]
where
    T: SizedBitBuffer,
{
    const BITS_COUNT: usize = N * T::BITS_COUNT;
}

impl<const N: usize, T> BitBuffer for [T; N]
where
    T: SizedBitBuffer,
{
    fn bits_count(&self) -> usize {
        self.as_slice().bits_count()
    }

    fn set_1(&mut self, bit_index: usize) {
        self.as_mut_slice().set_1(bit_index);
    }

    fn set_0(&mut self, bit_index: usize) {
        self.as_mut_slice().set_0(bit_index);
    }

    fn is_1(&self, bit_index: usize) -> bool {
        self.as_slice().is_1(bit_index)
    }

    fn flip_bit(&mut self, bit_index: usize) {
        self.as_mut_slice().flip_bit(bit_index);
    }
}

impl<T> BitBuffer for Vec<T>
where
    T: SizedBitBuffer,
{
    fn bits_count(&self) -> usize {
        self.as_slice().bits_count()
    }

    fn set_1(&mut self, bit_index: usize) {
        self.as_mut_slice().set_1(bit_index);
    }

    fn set_0(&mut self, bit_index: usize) {
        self.as_mut_slice().set_0(bit_index);
    }

    fn is_1(&self, bit_index: usize) -> bool {
        self.as_slice().is_1(bit_index)
    }

    fn flip_bit(&mut self, bit_index: usize) {
        self.as_mut_slice().flip_bit(bit_index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod u8 {
        use super::*;

        #[test]
        fn set1() {
            let mut val = 0b0001u8;
            val.set_1(0);
            assert_eq!(val, 0b0001);

            let mut val = 0b0001u8;
            val.set_1(1);
            assert_eq!(val, 0b0011);

            let mut val = 0b0001u8;
            val.set_1(2);
            assert_eq!(val, 0b0101);

            let mut val = 0b0001u8;
            val.set_1(3);
            assert_eq!(val, 0b1001);

            let mut val = 0b0000_0001u8;
            val.set_1(7);
            assert_eq!(val, 0b1000_0001);
        }

        #[test]
        fn set0() {
            let mut val = 0b1110u8;
            val.set_0(0);
            assert_eq!(val, 0b1110);

            let mut val = 0b1110u8;
            val.set_0(1);
            assert_eq!(val, 0b1100);

            let mut val = 0b1110u8;
            val.set_0(2);
            assert_eq!(val, 0b1010);

            let mut val = 0b1110u8;
            val.set_0(3);
            assert_eq!(val, 0b0110);

            let mut val = 0b1000_0001u8;
            val.set_0(7);
            assert_eq!(val, 0b0000_0001);
        }

        #[test]
        fn is_1() {
            assert!(0b0001u8.is_1(0));
            assert!(!0b0001u8.is_1(2));
            assert!(!0b0001u8.is_1(4));
            assert!(!0b0001u8.is_1(4));
            assert!(!0b0010u8.is_1(0));
            assert!(0b0010u8.is_1(1));
        }

        #[test]
        fn flip_bit() {
            let mut val = 0b0000u8;
            val.flip_bit(0);
            assert_eq!(val, 0b0001);

            let mut val = 0b0000u8;
            val.flip_bit(1);
            assert_eq!(val, 0b0010);

            let mut val = 0b0000u8;
            val.flip_bit(3);
            assert_eq!(val, 0b1000);

            let mut val = 0b1111u8;
            val.flip_bit(0);
            assert_eq!(val, 0b1110);
        }

        #[test]
        fn total_even() {
            assert!(0b00000000u8.total_parity_is_even());
            assert!(!0b00000001u8.total_parity_is_even());
            assert!(0b00000011u8.total_parity_is_even());
            assert!(!0b00000111u8.total_parity_is_even());
            assert!(0b10000001u8.total_parity_is_even());
            assert!(!0b10010001u8.total_parity_is_even());
            assert!(0b11111111u8.total_parity_is_even());
        }

        #[test]
        fn count_1_bits() {
            assert_eq!(0b00101100u8.count_1_bits(), 3);
            assert_eq!(0b1000001u8.count_1_bits(), 2);
        }

        #[test]
        fn fault_injection() {
            let mut buf = 0u8;
            buf.flip_n_bits(1).unwrap();
            assert_eq!(buf.count_1_bits(), 1);

            let mut buf = 0u8;
            buf.flip_n_bits(2).unwrap();
            assert_eq!(buf.count_1_bits(), 2);

            let mut buf = 0u8;
            buf.flip_n_bits(3).unwrap();
            assert_eq!(buf.count_1_bits(), 3);

            let mut buf = 0u8;
            buf.flip_n_bits(4).unwrap();
            assert_eq!(buf.count_1_bits(), 4);

            let mut buf = 0u8;
            buf.flip_by_ber(1.).unwrap();
            assert_eq!(buf.count_1_bits(), 8);
        }

        #[test]
        fn invalid_flips() {
            let mut buf = 0u8;

            assert_eq!(buf.flip_n_bits(9), Err(OutOfBounds));
            assert_eq!(buf.flip_by_ber(-f64::EPSILON), Err(OutOfBounds));
            assert_eq!(buf.flip_by_ber(1. + f64::EPSILON), Err(OutOfBounds));
        }

        #[test]
        fn bit_string() {
            let bits = 0b01010101u8;

            assert_eq!("0b01010101", bits.bit_string());
        }
    }

    mod i32 {
        use super::*;

        #[test]
        fn flip() {
            let mut buf = 0i32;
            buf.flip_bit(31);
            assert_eq!(buf, -2147483648i32);
        }
    }

    mod f32 {
        use super::*;

        #[test]
        fn flip() {
            let mut buf = 1f32;
            buf.flip_bit(31);
            assert_eq!(buf, -1f32);
        }
    }

    mod f64 {
        use super::*;

        #[test]
        fn flip() {
            let mut buf = 1f64;
            buf.flip_bit(63);
            assert_eq!(buf, -1f64);
        }
    }

    mod sequence {
        use super::*;

        #[test]
        fn is_1() {
            let a = [0u8, 0b110u8];
            for i in 0..=8 {
                assert!(a.is_0(i))
            }
            assert!(a.is_1(9));
            assert!(a.is_1(10));
            assert!(a.is_0(11));
        }
    }

    #[test]
    fn copy_into_different_structure() {
        let a_actual: [u8; 8] = [123, 4, 255, 0, 2, 97, 34, 255];
        let bits_count = a_actual.bits_count();
        let b_actual: [u16; 4] = [
            u16::from_le_bytes([123, 4]),
            u16::from_le_bytes([255, 0]),
            u16::from_le_bytes([2, 97]),
            u16::from_le_bytes([34, 255]),
        ];

        let mut b: [u16; 4] = [0xabc2, 0x1234, 0x1ab2, 0x4a89];
        let result = a_actual.copy_into(&mut b);
        assert_eq!(result, CopyIntoResult::done(bits_count));
        assert_eq!(b, b_actual);

        let mut a: [u8; 8] = [0xfa, 0xab, 0x42, 0x01, 0xaa, 0x00, 0xff, 0x4c];
        let result = b_actual.copy_into(&mut a);
        assert_eq!(result, CopyIntoResult::done(bits_count));
        assert_eq!(a, a_actual);
    }

    #[test]
    fn copy_into_multiple_dest() {
        let source = [255u8, 127u8, 63u8, 31u8];
        let mut dest = 0u8;

        let mut start = 0;
        for (i, &expected) in source.iter().enumerate() {
            let CopyIntoResult {
                units_copied: bits_copied,
                kind,
            } = source.copy_into_offset(start, 0, &mut dest);
            assert_eq!(bits_copied, 8);
            start += bits_copied;
            assert_eq!(dest, expected);
            if i == source.len() - 1 {
                assert_eq!(kind, CopyIntoResultKind::Done);
            } else {
                assert_eq!(kind, CopyIntoResultKind::Pending);
            }
        }
    }

    #[test]
    fn copy_into_empty() {
        let source = 1234u16;
        let mut dest = Vec::<u8>::new();

        let result = source.copy_into(&mut dest);
        assert_eq!(result, CopyIntoResult::pending(0));

        let source = Vec::<u8>::new();
        let mut dest = 1234u16;

        let result = source.copy_into(&mut dest);
        assert_eq!(result, CopyIntoResult::done(0));
    }

    mod default_flip_bit {
        use super::*;

        #[derive(Debug, Clone, PartialEq, Eq)]
        struct CustomU8(u8);

        impl BitBuffer for CustomU8 {
            fn bits_count(&self) -> usize {
                self.0.bits_count()
            }

            fn set_1(&mut self, bit_index: usize) {
                self.0.set_1(bit_index);
            }

            fn set_0(&mut self, bit_index: usize) {
                self.0.set_0(bit_index);
            }

            fn is_1(&self, bit_index: usize) -> bool {
                self.0.is_1(bit_index)
            }
        }

        #[test]
        fn default_impl() {
            let value = CustomU8(0);

            assert_eq!(value.bits_count(), 8);

            for i in 0..8 {
                let mut copy = value.clone();
                copy.flip_bit(i);
                assert_eq!(copy.0, 1 << i);
                copy.flip_bit(i);
                assert_eq!(value, copy);
            }
        }
    }

    #[test]
    fn bits_count() {
        assert_eq!(0f32.bits_count(), 32);
        assert_eq!(0f64.bits_count(), 64);
        assert_eq!(0u8.bits_count(), 8);
        assert_eq!(0u16.bits_count(), 16);
    }

    #[test]
    fn array_flip_bit() {
        let mut arr = [0u8; 2];

        arr.flip_bit(0);
        assert_eq!(arr, [1u8, 0u8]);
        arr.flip_bit(8);
        assert_eq!(arr, [1u8, 1u8]);
        arr.flip_bit(8);
        assert_eq!(arr, [1u8, 0u8]);
    }
}
