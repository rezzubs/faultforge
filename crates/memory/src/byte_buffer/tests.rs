use super::*;

#[test]
fn get_byte() {
    let source = [450u16, 12u16, 0];
    assert_eq!(source.get_byte(0), source[0].to_le_bytes()[0]);
    assert_eq!(source.get_byte(1), source[0].to_le_bytes()[1]);
    assert_eq!(source.get_byte(2), source[1].to_le_bytes()[0]);
    assert_eq!(source.get_byte(3), source[1].to_le_bytes()[1]);
    assert_eq!(source.get_byte(4), source[2].to_le_bytes()[0]);
    assert_eq!(source.get_byte(5), source[2].to_le_bytes()[1]);
}

#[test]
fn set_byte() {
    let source = [450u16, 12u16, 0];
    let mut dest = [0u16, 0u16, 258u16];
    for i in 0..6 {
        dest.set_byte(i, source.get_byte(i));
    }
    assert_eq!(dest.get_byte(0), source[0].to_le_bytes()[0]);
    assert_eq!(dest.get_byte(1), source[0].to_le_bytes()[1]);
    assert_eq!(dest.get_byte(2), source[1].to_le_bytes()[0]);
    assert_eq!(dest.get_byte(3), source[1].to_le_bytes()[1]);
    assert_eq!(dest.get_byte(4), source[2].to_le_bytes()[0]);
    assert_eq!(dest.get_byte(5), source[2].to_le_bytes()[1]);
}

#[test]
fn copy_into_different_structure() {
    let a_actual: Vec<u8> = vec![123, 4, 255, 0, 2, 97, 34, 255];
    let byte_count = a_actual.len();
    let b_actual: Vec<u16> = vec![
        u16::from_le_bytes([123, 4]),
        u16::from_le_bytes([255, 0]),
        u16::from_le_bytes([2, 97]),
        u16::from_le_bytes([34, 255]),
    ];

    let mut b = vec![0u16; 4];
    let result = a_actual.copy_into_chunked(&mut b);
    assert_eq!(result, CopyIntoResult::exhausted(byte_count));
    assert_eq!(b, b_actual);

    let mut a = vec![0u8; 8];
    let result = b_actual.copy_into_chunked(&mut a);
    assert_eq!(result, CopyIntoResult::exhausted(byte_count));
    assert_eq!(a, a_actual);
}

#[test]
fn copy_into_partial() {
    let source = [255u8, 127u8, 63u8, 31u8];
    let mut dest = 0b01011010u8;

    let mut start = 0;
    for (i, expected) in source.into_iter().enumerate() {
        let copied = source.copy_into_chunked_offset(start, 0, &mut dest);
        if i + 1 == source.len() {
            assert_eq!(copied, CopyIntoResult::exhausted(1));
        } else {
            assert_eq!(copied, CopyIntoResult::partial(1));
        }
        start += copied.units_copied;
        assert_eq!(dest, expected);
    }

    let mut dest = [0b01010101u8, 0b10101010u8];

    let mut start = 0;
    for (i, expected) in source.chunks(2).enumerate() {
        let result = source.copy_into_chunked_offset(start, 0, &mut dest);
        if i == 1 {
            // The last chunk
            assert_eq!(result, CopyIntoResult::exhausted(2));
        } else {
            assert_eq!(result, CopyIntoResult::partial(2));
        }
        start += result.units_copied;
        assert_eq!(dest, expected);
    }
}

#[test]
fn uint_impl() {
    let source = 123456u32;
    let mut dest = 0u32;

    for (i, byte) in source.to_le_bytes().into_iter().enumerate() {
        assert_eq!(source.get_byte(i), byte);
        dest.set_byte(i, byte);
    }

    assert_eq!(source, dest);

    let source = 123456743032483000u64;
    let mut dest = 0u64;

    for (i, byte) in source.to_le_bytes().into_iter().enumerate() {
        assert_eq!(source.get_byte(i), byte);
        dest.set_byte(i, byte);
    }

    assert_eq!(source, dest);

    let source = 10456u16;
    let mut dest = 0u16;

    for (i, byte) in source.to_le_bytes().into_iter().enumerate() {
        assert_eq!(source.get_byte(i), byte);
        dest.set_byte(i, byte);
    }

    assert_eq!(source, dest);
}

#[test]
fn float_impl() {
    let source = 13456.029f32;
    let mut dest = 0f32;

    for (i, byte) in source.to_le_bytes().into_iter().enumerate() {
        assert_eq!(source.get_byte(i), byte);
        dest.set_byte(i, byte);
    }

    assert_eq!(source, dest);

    let source = 12345324789.483002f64;
    let mut dest = 0f64;

    for (i, byte) in source.to_le_bytes().into_iter().enumerate() {
        assert_eq!(source.get_byte(i), byte);
        dest.set_byte(i, byte);
    }

    assert_eq!(source, dest);
}

mod out_of_bounds {
    use super::*;

    #[test]
    #[should_panic]
    fn u8_get() {
        0u8.get_byte(1);
    }

    #[test]
    #[should_panic]
    fn u16_get() {
        0u16.get_byte(2);
    }

    #[test]
    #[should_panic]
    fn u32_get() {
        0u32.get_byte(4);
    }

    #[test]
    #[should_panic]
    fn u64_get() {
        0u64.get_byte(8);
    }

    #[test]
    #[should_panic]
    fn u8_set() {
        0u8.set_byte(1, 0);
    }

    #[test]
    #[should_panic]
    fn u16_set() {
        0u16.set_byte(2, 0);
    }

    #[test]
    #[should_panic]
    fn u32_set() {
        0u32.set_byte(4, 0);
    }

    #[test]
    #[should_panic]
    fn u64_set() {
        0u64.set_byte(8, 0);
    }

    #[test]
    #[should_panic]
    fn f32_get() {
        0f32.get_byte(4);
    }

    #[test]
    #[should_panic]
    fn f64_get() {
        0f64.get_byte(8);
    }

    #[test]
    #[should_panic]
    fn f32_set() {
        0f32.set_byte(4, 0);
    }

    #[test]
    #[should_panic]
    fn f64_set() {
        0f64.set_byte(8, 0);
    }
}
