use crate::fault::PyFault;
use memory::{BitBuffer, SizedBitBuffer, sequence::NonUniformSequence};
use numpy::PyArrayDyn;
use pyo3::{exceptions::PyIndexError, prelude::*};

type InputArr<'py, T> = Bound<'py, PyArrayDyn<T>>;

fn list_of_array_inject_fault_generic<'py, T>(
    _py: Python<'py>,
    input: Vec<InputArr<T>>,
    fault: PyFault,
    target_bit: usize,
) -> PyResult<()>
where
    T: numpy::Element + Copy + SizedBitBuffer,
{
    let mut buffer = NonUniformSequence(input);

    let bit_count = buffer.bit_count();
    if target_bit >= bit_count {
        return Err(PyIndexError::new_err("target_bit is out of bounds"));
    }

    buffer.apply_fault(fault.0, target_bit);

    Ok(())
}

/// Applies a batch of faults to a list of arrays.
///
/// The caller-side numpy conversion (see `tensor_list_faults` in
/// `faultforge._internal.tensor`) is only paid once for the whole batch instead
/// of once per fault. Bit lookups are also batched via `NonUniformSequence`'s
/// `BitBuffer::apply_faults` override, which looks up each target array once
/// via a precomputed offset table rather than rescanning the whole array list
/// per fault.
fn list_of_array_inject_faults_generic<'py, T>(
    _py: Python<'py>,
    input: Vec<InputArr<T>>,
    faults: Vec<(PyFault, usize)>,
) -> PyResult<()>
where
    T: numpy::Element + Copy + SizedBitBuffer,
{
    let mut buffer = NonUniformSequence(input);
    let bit_count = buffer.bit_count();

    for (_, target_bit) in &faults {
        if *target_bit >= bit_count {
            return Err(PyIndexError::new_err("target_bit is out of bounds"));
        }
    }

    buffer.apply_faults(
        faults
            .into_iter()
            .map(|(fault, bit_index)| (fault.0, bit_index)),
    );

    Ok(())
}

#[pyfunction]
pub fn list_of_array_fault_f32<'py>(
    py: Python<'py>,
    input: Vec<InputArr<'py, f32>>,
    fault: PyFault,
    target_bit: usize,
) -> PyResult<()> {
    list_of_array_inject_fault_generic(py, input, fault, target_bit)
}

#[pyfunction]
pub fn list_of_array_fault_u16<'py>(
    py: Python<'py>,
    input: Vec<InputArr<'py, u16>>,
    fault: PyFault,
    target_bit: usize,
) -> PyResult<()> {
    list_of_array_inject_fault_generic(py, input, fault, target_bit)
}

#[pyfunction]
pub fn list_of_array_fault_u8<'py>(
    py: Python<'py>,
    input: Vec<InputArr<'py, u8>>,
    fault: PyFault,
    target_bit: usize,
) -> PyResult<()> {
    list_of_array_inject_fault_generic(py, input, fault, target_bit)
}

#[pyfunction]
pub fn list_of_array_faults_f32<'py>(
    py: Python<'py>,
    input: Vec<InputArr<'py, f32>>,
    faults: Vec<(PyFault, usize)>,
) -> PyResult<()> {
    list_of_array_inject_faults_generic(py, input, faults)
}

#[pyfunction]
pub fn list_of_array_faults_u16<'py>(
    py: Python<'py>,
    input: Vec<InputArr<'py, u16>>,
    faults: Vec<(PyFault, usize)>,
) -> PyResult<()> {
    list_of_array_inject_faults_generic(py, input, faults)
}

#[pyfunction]
pub fn list_of_array_faults_u8<'py>(
    py: Python<'py>,
    input: Vec<InputArr<'py, u8>>,
    faults: Vec<(PyFault, usize)>,
) -> PyResult<()> {
    list_of_array_inject_faults_generic(py, input, faults)
}
