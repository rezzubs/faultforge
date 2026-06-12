use memory::{Bit, Fault};
use pyo3::prelude::*;

#[pyclass(from_py_object, name = "Fault")]
#[derive(Clone)]
pub struct PyFault(pub Fault);

#[pymethods]
impl PyFault {
    #[staticmethod]
    fn stuck_at_0() -> PyFault {
        PyFault(Fault::StuckAt(Bit::Zero))
    }

    #[staticmethod]
    fn stuck_at_1() -> PyFault {
        PyFault(Fault::StuckAt(Bit::One))
    }

    #[staticmethod]
    fn flip() -> PyFault {
        PyFault(Fault::Flip)
    }
}
