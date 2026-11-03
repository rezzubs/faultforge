use numpy::{PyArray1, PyReadonlyArrayDyn};
use pyo3::prelude::*;

pub type OutputArr<'py, T> = Bound<'py, PyArray1<T>>;
pub type InputArr<'py, T> = PyReadonlyArrayDyn<'py, T>;
