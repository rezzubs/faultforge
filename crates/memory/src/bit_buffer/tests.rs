use crate::CopyIntoStatus;

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
    let bit_count = a_actual.bit_count();
    let b_actual: [u16; 4] = [
        u16::from_le_bytes([123, 4]),
        u16::from_le_bytes([255, 0]),
        u16::from_le_bytes([2, 97]),
        u16::from_le_bytes([34, 255]),
    ];

    let mut b: [u16; 4] = [0xabc2, 0x1234, 0x1ab2, 0x4a89];
    let result = a_actual.copy_into(&mut b);
    assert_eq!(result, CopyIntoResult::exhausted(bit_count));
    assert_eq!(b, b_actual);

    let mut a: [u8; 8] = [0xfa, 0xab, 0x42, 0x01, 0xaa, 0x00, 0xff, 0x4c];
    let result = b_actual.copy_into(&mut a);
    assert_eq!(result, CopyIntoResult::exhausted(bit_count));
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
            status,
        } = source.copy_into_offset(start, 0, &mut dest);
        assert_eq!(bits_copied, 8);
        start += bits_copied;
        assert_eq!(dest, expected);
        if i == source.len() - 1 {
            assert_eq!(status, CopyIntoStatus::Exhausted);
        } else {
            assert_eq!(status, CopyIntoStatus::Partial);
        }
    }
}

#[test]
fn copy_into_empty() {
    let source = 1234u16;
    let mut dest = Vec::<u8>::new();

    let result = source.copy_into(&mut dest);
    assert_eq!(result, CopyIntoResult::partial(0));

    let source = Vec::<u8>::new();
    let mut dest = 1234u16;

    let result = source.copy_into(&mut dest);
    assert_eq!(result, CopyIntoResult::exhausted(0));
}

mod default_flip_bit {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct CustomU8(u8);

    impl BitBuffer for CustomU8 {
        fn bit_count(&self) -> usize {
            self.0.bit_count()
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

        assert_eq!(value.bit_count(), 8);

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
fn bit_count() {
    assert_eq!(0f32.bit_count(), 32);
    assert_eq!(0f64.bit_count(), 64);
    assert_eq!(0u8.bit_count(), 8);
    assert_eq!(0u16.bit_count(), 16);
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
