use picker::Picker;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::collections::HashSet;

/// An iterator which returns numbers from 0..n in a random order until all
/// values are consumed.
///
/// Every returned value is unique.
///
/// This is based on the [Fisher-Yates
/// shuffle](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle) but
/// instead of shuffling the whole sequence we just return the target index for
/// the swap.
#[pyclass(name = "Picker")]
#[derive(Debug)]
pub struct PyPicker {
    picker: Picker<SmallRng>,
}

#[pymethods]
impl PyPicker {
    /// Create a new picker that returns elements from `0..size`.
    ///
    /// If `seed` is given the order is deterministic, otherwise it is seeded
    /// from the operating system entropy source.
    #[new]
    #[pyo3(signature = (size, seed=None))]
    fn new(size: usize, seed: Option<u64>) -> Self {
        Self {
            picker: Picker::new(size, make_rng(seed)),
        }
    }

    /// Reconstruct a picker that will not return any of the `already_returned`
    /// values.
    ///
    /// The remaining values returned will be a valid permutation of
    /// `0..initial_size` excluding `already_returned`. Raises `ValueError` if
    /// any returned value is outside `0..initial_size`.
    #[staticmethod]
    #[pyo3(signature = (initial_size, already_returned, seed=None))]
    fn from_returned(
        initial_size: usize,
        already_returned: HashSet<usize>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        Picker::from_returned(initial_size, &already_returned, make_rng(seed))
            .map(|picker| Self { picker })
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Reset the picker to its initial state.
    fn reset(&mut self) {
        self.picker.reset();
    }

    /// The initial size of the picker, as passed to the constructor.
    #[getter]
    fn initial_size(&self) -> usize {
        self.picker.initial_size()
    }

    /// The number of remaining values.
    #[getter]
    fn size(&self) -> usize {
        self.picker.size()
    }

    fn __len__(&self) -> usize {
        self.picker.size()
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<usize> {
        self.picker.next()
    }
}

fn make_rng(seed: Option<u64>) -> SmallRng {
    match seed {
        Some(seed) => SmallRng::seed_from_u64(seed),
        None => rand::make_rng(),
    }
}
