use crate::common::*;
use faultforge::prelude::*;
use numpy::PyArray1;
use pyo3::{exceptions::PyValueError, prelude::*};

fn array_list_fi_generic<'py, T>(
    py: Python<'py>,
    input: Vec<InputArr<T>>,
    faults_count: usize,
) -> PyResult<Vec<OutputArr<'py, T>>>
where
    T: numpy::Element + Copy + SizedBitBuffer,
{
    let mut buffer = prep_input_array_list(input);

    let bits_count = buffer.bits_count();

    buffer.flip_n_bits(faults_count).map_err(|_| {
        PyValueError::new_err(format!(
            "Buffer has {} bits, cannot flip {}",
            bits_count, faults_count
        ))
    })?;

    Ok(buffer
        .0
        .into_iter()
        .map(|arr| PyArray1::from_vec(py, arr))
        .collect::<Vec<_>>())
}

#[pyfunction]
pub fn f32_array_list_fi<'py>(
    py: Python<'py>,
    input: Vec<InputArr<'py, f32>>,
    faults_count: usize,
) -> PyResult<Vec<OutputArr<'py, f32>>> {
    array_list_fi_generic(py, input, faults_count)
}

#[pyfunction]
pub fn u16_array_list_fi<'py>(
    py: Python<'py>,
    input: Vec<InputArr<'py, u16>>,
    faults_count: usize,
) -> PyResult<Vec<OutputArr<'py, u16>>> {
    array_list_fi_generic(py, input, faults_count)
}

#[pyfunction]
pub fn u8_array_list_fi<'py>(
    py: Python<'py>,
    input: Vec<InputArr<'py, u8>>,
    faults_count: usize,
) -> PyResult<Vec<OutputArr<'py, u8>>> {
    array_list_fi_generic(py, input, faults_count)
}
