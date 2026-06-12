use crate::SizedBitBuffer;

use super::ByteBuffer;

impl ByteBuffer for u8 {
    fn byte_count(&self) -> usize {
        1
    }

    fn get_byte(&self, n: usize) -> u8 {
        assert_eq!(n, 0, "u8 only has one byte, got index {n}");
        *self
    }

    fn set_byte(&mut self, n: usize, value: u8) {
        assert_eq!(n, 0, "u8 only has one byte, got index {n}");
        *self = value;
    }
}

macro_rules! uint_impl {
    ($t:ty, $bytes:expr) => {
        impl ByteBuffer for $t {
            fn byte_count(&self) -> usize {
                $bytes
            }

            fn get_byte(&self, n: usize) -> u8 {
                assert!(n < $bytes, "buffer has {} bytes, got index {}", $bytes, n);

                (self >> (n * 8)) as u8
            }

            fn set_byte(&mut self, n: usize, value: u8) {
                let bit_count = n * 8;
                let value_shifted = (value as $t) << (bit_count);
                let mask: $t = !(0xff << bit_count);

                *self &= mask;
                *self |= value_shifted;
            }
        }
    };
}

uint_impl!(u16, 2);
uint_impl!(u32, 4);
uint_impl!(u64, 8);
uint_impl!(u128, 16);

macro_rules! int_impl {
    ($t:ty, $ut:ty) => {
        impl ByteBuffer for $t {
            fn byte_count(&self) -> usize {
                (*self as $ut).byte_count()
            }

            fn get_byte(&self, n: usize) -> u8 {
                (*self as $ut).get_byte(n)
            }

            fn set_byte(&mut self, n: usize, value: u8) {
                let mut unsigned = *self as $ut;
                unsigned.set_byte(n, value);
                *self = unsigned as $t;
            }
        }
    };
}

int_impl!(i8, u8);
int_impl!(i16, u16);
int_impl!(i32, u32);
int_impl!(i64, u64);
int_impl!(i128, u128);

macro_rules! float_impl {
    ($ty:ty, $bytes:expr) => {
        impl ByteBuffer for $ty {
            fn byte_count(&self) -> usize {
                $bytes
            }

            fn get_byte(&self, n: usize) -> u8 {
                self.to_bits().get_byte(n)
            }

            fn set_byte(&mut self, n: usize, value: u8) {
                let mut uint = self.to_bits();
                uint.set_byte(n, value);
                *self = <$ty>::from_bits(uint);
            }
        }
    };
}

float_impl!(f32, 4);
float_impl!(f64, 8);

impl<T> ByteBuffer for [T]
where
    T: ByteBuffer + SizedBitBuffer,
{
    fn byte_count(&self) -> usize {
        self.len() * (T::BITS_COUNT / 8)
    }

    fn get_byte(&self, n: usize) -> u8 {
        let child_byte_count = T::BITS_COUNT / 8;
        let item_index = n / child_byte_count;
        let index_in_item = n % child_byte_count;
        let item = self.get(item_index).unwrap_or_else(|| panic!("out of bounds, child_byte_count: {child_byte_count}, item_index: {item_index}, index_in_item: {index_in_item}"));
        item.get_byte(index_in_item)
    }

    fn set_byte(&mut self, n: usize, value: u8) {
        let byte_count = T::BITS_COUNT / 8;
        let item_index = n / byte_count;
        let index_in_item = n % byte_count;
        let Some(item) = self.get_mut(item_index) else {
            panic!("{n} is out of bounds");
        };
        item.set_byte(index_in_item, value)
    }
}

impl<const N: usize, T> ByteBuffer for [T; N]
where
    T: ByteBuffer + SizedBitBuffer,
{
    fn byte_count(&self) -> usize {
        self.as_slice().byte_count()
    }

    fn get_byte(&self, n: usize) -> u8 {
        self.as_slice().get_byte(n)
    }

    fn set_byte(&mut self, n: usize, value: u8) {
        self.as_mut_slice().set_byte(n, value)
    }
}

impl<T> ByteBuffer for Vec<T>
where
    T: ByteBuffer + SizedBitBuffer,
{
    fn byte_count(&self) -> usize {
        self.as_slice().byte_count()
    }

    fn get_byte(&self, n: usize) -> u8 {
        self.as_slice().get_byte(n)
    }

    fn set_byte(&mut self, n: usize, value: u8) {
        self.as_mut_slice().set_byte(n, value)
    }
}
