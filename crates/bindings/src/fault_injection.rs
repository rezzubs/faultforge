use crate::fault::PyFault;
use memory::{BitBuffer, SizedBitBuffer, sequence::NonUniformSequence};
use numpy::PyArrayDyn;
use pyo3::{exceptions::PyValueError, prelude::*};

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
        return Err(PyValueError::new_err("target_bit is out of bounds"));
    }

    buffer.apply_fault(fault.0, target_bit);

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
