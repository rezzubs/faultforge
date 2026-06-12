use crate::common::*;
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;

#[inline]
fn compare_f32_vecs_bitwise(a: Vec<f32>, b: Vec<f32>) -> Vec<u32> {
    a.into_par_iter()
        .zip(b)
        .map(|(a_item, b_item)| a_item.to_bits() ^ b_item.to_bits())
        .filter(|x| *x > 0)
        .collect::<Vec<u32>>()
}

#[inline]
fn compare_u16_vecs_bitwise(a: Vec<u16>, b: Vec<u16>) -> Vec<u16> {
    a.into_par_iter()
        .zip(b)
        .map(|(a_item, b_item)| a_item ^ b_item)
        .filter(|x| *x > 0)
        .collect::<Vec<u16>>()
}

/// Computes bit masks for non-matching bits for all items.
#[pyfunction]
pub fn compare_array_list_bitwise_f32<'py>(
    _py: Python<'py>,
    a: Vec<InputArr<f32>>,
    b: Vec<InputArr<f32>>,
) -> PyResult<Vec<u32>> {
    let a = a
        .into_iter()
        .flat_map(|x| x.as_array().into_iter().copied().collect::<Vec<f32>>())
        .collect::<Vec<_>>();
    let b = b
        .into_iter()
        .flat_map(|x| x.as_array().into_iter().copied().collect::<Vec<f32>>())
        .collect::<Vec<_>>();

    if a.len() != b.len() {
        return Err(PyValueError::new_err(
            "The number of items in `a` and `b` don't match",
        ));
    }

    Ok(compare_f32_vecs_bitwise(a, b))
}

/// Computes bit masks for non-matching bits for all items.
#[pyfunction]
pub fn compare_array_list_bitwise_u16<'py>(
    _py: Python<'py>,
    a: Vec<InputArr<u16>>,
    b: Vec<InputArr<u16>>,
) -> PyResult<Vec<u16>> {
    let a = a
        .into_iter()
        .flat_map(|x| x.as_array().into_iter().copied().collect::<Vec<u16>>())
        .collect::<Vec<_>>();
    let b = b
        .into_iter()
        .flat_map(|x| x.as_array().into_iter().copied().collect::<Vec<u16>>())
        .collect::<Vec<_>>();

    if a.len() != b.len() {
        return Err(PyValueError::new_err(
            "The number of items in `a` and `b` don't match",
        ));
    }

    Ok(compare_u16_vecs_bitwise(a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_f32_vec() {
        let a = [0b10001101u32, 0b10000001u32]
            .into_iter()
            .map(f32::from_bits)
            .collect();
        let b = [0b10011100u32, 0b00000001u32]
            .into_iter()
            .map(f32::from_bits)
            .collect();

        let expected = vec![0b00010001u32, 0b10000000u32];

        assert_eq!(compare_f32_vecs_bitwise(a, b), expected);
    }

    #[test]
    fn compare_u16_vec() {
        let a = vec![0b10001101u16, 0b10000001u16];
        let b = vec![0b10011100u16, 0b00000001u16];

        let expected = vec![0b00010001u16, 0b10000000u16];

        assert_eq!(compare_u16_vecs_bitwise(a, b), expected);
    }
}
