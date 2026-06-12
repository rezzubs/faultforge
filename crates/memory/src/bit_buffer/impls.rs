use crate::{BitBuffer, SizedBitBuffer};

macro_rules! int_impl {
    ($t:ty, $bits:expr) => {
        impl SizedBitBuffer for $t {
            const BITS_COUNT: usize = $bits;
        }

        impl BitBuffer for $t {
            fn bit_count(&self) -> usize {
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
            fn bit_count(&self) -> usize {
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
    fn bit_count(&self) -> usize {
        self.len() * T::BITS_COUNT
    }

    fn set_1(&mut self, bit_index: usize) {
        debug_assert!(bit_index < self.bit_count());
        let item_index = bit_index / T::BITS_COUNT;
        self[item_index].set_1(bit_index % T::BITS_COUNT);
    }

    fn set_0(&mut self, bit_index: usize) {
        debug_assert!(bit_index < self.bit_count());
        let item_index = bit_index / T::BITS_COUNT;
        self[item_index].set_0(bit_index % T::BITS_COUNT);
    }

    fn is_1(&self, bit_index: usize) -> bool {
        debug_assert!(bit_index < self.bit_count());
        let item_index = bit_index / T::BITS_COUNT;
        self[item_index].is_1(bit_index % T::BITS_COUNT)
    }

    fn flip_bit(&mut self, bit_index: usize) {
        debug_assert!(bit_index < self.bit_count());
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
    fn bit_count(&self) -> usize {
        self.as_slice().bit_count()
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
    fn bit_count(&self) -> usize {
        self.as_slice().bit_count()
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
