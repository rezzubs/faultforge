use faultforge::buffers::NonUniformSequence;
use numpy::{PyArray1, PyReadonlyArrayDyn};
use pyo3::prelude::*;

pub(crate) type OutputArr<'py, T> = Bound<'py, PyArray1<T>>;
pub(crate) type InputArr<'py, T> = PyReadonlyArrayDyn<'py, T>;

/// Helper for transforming numpy arrays received from python.
pub(crate) fn prep_input_array<'py, T>(input: InputArr<'py, T>) -> Vec<T>
where
    T: numpy::Element + Copy,
{
    input.as_array().into_iter().copied().collect::<Vec<T>>()
}

/// Helper for transforming lists of numpy arrays received from python.
pub(crate) fn prep_input_array_list<'py, T>(
    input: Vec<InputArr<'py, T>>,
) -> NonUniformSequence<Vec<Vec<T>>>
where
    T: numpy::Element + Copy,
{
    NonUniformSequence(
        input
            .into_iter()
            .map(prep_input_array)
            .collect::<Vec<Vec<T>>>(),
    )
}
