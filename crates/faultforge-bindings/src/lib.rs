mod common;
mod comparison;
mod embedded_parity;
mod encoding_bit_pattern;
mod encoding_full;
mod generic_fault_injection;
mod single_bit_encoding;

use pyo3::pymodule;

#[pymodule]
mod _core {
    #[pymodule_export]
    use crate::generic_fault_injection::f32_array_list_fi;
    #[pymodule_export]
    use crate::generic_fault_injection::u8_array_list_fi;
    #[pymodule_export]
    use crate::generic_fault_injection::u16_array_list_fi;

    #[pymodule_export]
    use crate::comparison::compare_array_list_bitwise_f32;
    #[pymodule_export]
    use crate::comparison::compare_array_list_bitwise_u16;

    #[pymodule_export]
    use crate::encoding_full::PyFullEncoding;
    #[pymodule_export]
    use crate::encoding_full::encode_full_f32;
    #[pymodule_export]
    use crate::encoding_full::encode_full_u16;

    #[pymodule_export]
    use crate::encoding_bit_pattern::PyBitPatternEncoding;
    #[pymodule_export]
    use crate::encoding_bit_pattern::encode_bit_pattern_f32;
    #[pymodule_export]
    use crate::encoding_bit_pattern::encode_bit_pattern_u16;

    #[pymodule_export]
    use crate::single_bit_encoding::bit14_decode_u16;
    #[pymodule_export]
    use crate::single_bit_encoding::bit14_encode_u16;
    #[pymodule_export]
    use crate::single_bit_encoding::bit30_decode_f32;
    #[pymodule_export]
    use crate::single_bit_encoding::bit30_encode_f32;

    #[pymodule_export]
    use crate::embedded_parity::EpScheme;
    #[pymodule_export]
    use crate::embedded_parity::embedded_parity_decode_f32;
    #[pymodule_export]
    use crate::embedded_parity::embedded_parity_decode_u16;
    #[pymodule_export]
    use crate::embedded_parity::embedded_parity_encode_f32;
    #[pymodule_export]
    use crate::embedded_parity::embedded_parity_encode_u16;
}
