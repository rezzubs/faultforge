use std::sync::LazyLock;

use memory::encoding::majority::{Scheme, decode, encode};
use numpy::PyReadwriteArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use rayon::prelude::*;

static F32_SCHEME: LazyLock<Scheme<2>> =
    LazyLock::new(|| Scheme::for_buffer(&0f32, 30, [0, 1]).expect("known to be correct for f32"));

#[pyfunction]
pub fn encode_f32(mut arr: PyReadwriteArrayDyn<f32>) -> PyResult<()> {
    arr.as_slice_mut()
        .map_err(|_| PyValueError::new_err("`arr` is not contiguous."))?
        .par_iter_mut()
        .for_each(|item| encode(item, *F32_SCHEME).expect("The scheme is known to be correct"));

    Ok(())
}

#[pyfunction]
pub fn decode_f32(mut arr: PyReadwriteArrayDyn<f32>) -> PyResult<()> {
    arr.as_slice_mut()
        .map_err(|_| PyValueError::new_err("`arr` is not contiguous."))?
        .par_iter_mut()
        .for_each(|item| decode(item, *F32_SCHEME).expect("The scheme is known to be correct"));

    Ok(())
}

static F16_SCHEME: LazyLock<Scheme<2>> =
    LazyLock::new(|| Scheme::for_buffer(&0u16, 14, [0, 1]).expect("known to be correct for f16"));

#[pyfunction]
pub fn encode_u16(mut arr: PyReadwriteArrayDyn<u16>) -> PyResult<()> {
    arr.as_slice_mut()
        .map_err(|_| PyValueError::new_err("`arr` is not contiguous."))?
        .par_iter_mut()
        .for_each(|item| encode(item, *F16_SCHEME).expect("The scheme is known to be correct"));

    Ok(())
}

#[pyfunction]
pub fn decode_u16(mut arr: PyReadwriteArrayDyn<u16>) -> PyResult<()> {
    arr.as_slice_mut()
        .map_err(|_| PyValueError::new_err("`arr` is not contiguous."))?
        .par_iter_mut()
        .for_each(|item| decode(item, *F16_SCHEME).expect("The scheme is known to be correct"));

    Ok(())
}
