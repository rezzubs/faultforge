use crate::common::*;
use faultforge::{
    bit_buffer::chunks::DynChunks,
    buffers::{Limited, NonUniformSequence},
    encoding::{
        bit_patterns::{self, BitPattern, BitPatternEncoding, BitPatternEncodingData},
        secded::encoded_bits_count,
    },
    prelude::*,
};
use numpy::PyArray1;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBytes, PyList},
};
use std::collections::HashSet;

#[pyclass(name = "BitPatternEncoding")]
pub struct PyBitPatternEncoding {
    unprotected: Py<PyBytes>,
    unprotected_bits_count: usize,
    // Storing PyBytes elements
    protected: Py<PyList>,
    bits_per_chunk: usize,
    pattern_bits: HashSet<usize>,
    pattern_length: usize,
    total_bits_count: usize,
}

impl PyBitPatternEncoding {
    fn to_rust<'py>(&self, py: Python<'py>) -> BitPatternEncoding {
        let unprotected = Vec::from(self.unprotected.as_bytes(py));
        let unprotected_bits_count_actual = unprotected.bits_count();
        let unprotected = Limited::new(unprotected, self.unprotected_bits_count).unwrap_or_else(|| {
            panic!("`unprotected_bits_count` ({}) cannot be larger than the number bits in of `unprotected` ({})", self.unprotected_bits_count, unprotected_bits_count_actual);
        });

        let encoded_bits_per_chunk = encoded_bits_count(self.bits_per_chunk ).expect("PyBitPatternEncoding is the result of a successful encoding, therefore this cannot fail.");

        let protected =
            self.protected
                .bind(py)
                .iter()
                .map(|chunk| {
                    let chunk = chunk
                        .downcast_into::<PyBytes>()
                        .expect("failure should not be possible as the encoding type is immutable");

                    Limited::new(Vec::from(chunk.as_bytes()), encoded_bits_per_chunk)
                        .unwrap_or_else(|| {
                            panic!(
                                "`bits_per_chunk` ({}) doesn't match the encoded chunks",
                                self.bits_per_chunk
                            )
                        })
                })
                .collect();

        let protected = DynChunks::from_raw(protected)
            .expect("Should pass given the immutable nature of PyBitPatternEncoding");

        BitPatternEncoding {
            data: BitPatternEncodingData {
                unprotected,
                protected,
            },
            bits_per_chunk: self.bits_per_chunk,
            pattern: BitPattern::new(self.pattern_bits.clone(), self.pattern_length)
                .expect("The pattern is expected to be correct because the python side is opaque"),
        }
    }

    fn from_rust<'py>(py: Python<'py>, encoding: BitPatternEncoding) -> PyResult<Self> {
        let unprotected_bits_count = encoding.data.unprotected.bits_count();

        let total_bits_count = encoding.data.bits_count();
        let protected = encoding
            .data
            .protected
            .into_raw()
            .into_iter()
            .map(|chunk| PyBytes::new(py, &chunk.into_inner()).unbind());

        let protected = PyList::new(py, protected)?.unbind();

        Ok(PyBitPatternEncoding {
            unprotected: PyBytes::new(py, &encoding.data.unprotected.into_inner()).unbind(),
            unprotected_bits_count,
            protected,
            bits_per_chunk: encoding.bits_per_chunk,
            pattern_bits: encoding.pattern.mask().clone(),
            pattern_length: encoding.pattern.length(),
            total_bits_count,
        })
    }

    pub fn decode_bit_pattern_generic<'py, T>(
        &self,
        py: Python<'py>,
        mut output_buffer: NonUniformSequence<Vec<Vec<T>>>,
    ) -> (Vec<OutputArr<'py, T>>, Vec<bool>)
    where
        T: numpy::Element + BitBuffer + SizedBitBuffer,
    {
        let encoding = self.to_rust(py);

        let decoding_results = encoding.decode_into(&mut output_buffer);

        (
            output_buffer
                .0
                .into_iter()
                .map(|vec| PyArray1::from_vec(py, vec))
                .collect(),
            decoding_results,
        )
    }
}

#[pymethods]
impl PyBitPatternEncoding {
    fn flip_n_bits<'py>(&mut self, py: Python<'py>, n: usize) -> PyResult<()> {
        let mut encoding = self.to_rust(py);

        encoding.data.flip_n_bits(n).map_err(|_| {
            PyValueError::new_err(format!(
                "invalid number of bits {n}, only 0 <= {} is possible",
                encoding.data.bits_count()
            ))
        })?;

        *self = PyBitPatternEncoding::from_rust(py, encoding)?;

        Ok(())
    }

    pub fn decode_bit_pattern_f32<'py>(
        &self,
        py: Python<'py>,
        decoded_array_element_counts: Vec<usize>,
    ) -> (Vec<OutputArr<'py, f32>>, Vec<bool>) {
        let output_buffer = NonUniformSequence(
            decoded_array_element_counts
                .iter()
                .map(|&numel| vec![0f32; numel])
                .collect::<Vec<_>>(),
        );

        self.decode_bit_pattern_generic(py, output_buffer)
    }

    pub fn decode_bit_pattern_u16<'py>(
        &self,
        py: Python<'py>,
        decoded_array_element_counts: Vec<usize>,
    ) -> (Vec<OutputArr<'py, u16>>, Vec<bool>) {
        let output_buffer = NonUniformSequence(
            decoded_array_element_counts
                .iter()
                .map(|&numel| vec![0u16; numel])
                .collect::<Vec<_>>(),
        );

        self.decode_bit_pattern_generic(py, output_buffer)
    }

    /// Return a new instance with cloned data.
    pub fn clone(&self, py: Python) -> Self {
        let unprotected = self.unprotected.as_bytes(py).to_owned();
        let unprotected = PyBytes::new(py, &unprotected).unbind();

        let protected = self.protected.bind(py).iter().map(|chunk| {
            let chunk = chunk
                .downcast_into::<PyBytes>()
                .expect("Cannot fail because the values cannot be modified from python");

            let cloned = chunk.as_bytes().to_owned();

            PyBytes::new(py, &cloned)
        });
        let protected = PyList::new(py, protected)
            .expect("list creation failed")
            .unbind();

        Self {
            unprotected,
            unprotected_bits_count: self.unprotected_bits_count,
            protected,
            bits_per_chunk: self.bits_per_chunk,
            pattern_bits: self.pattern_bits.clone(),
            pattern_length: self.pattern_length,
            total_bits_count: self.total_bits_count,
        }
    }

    /// Return the number of bits in the encoded buffer
    pub fn bits_count(&self) -> usize {
        self.total_bits_count
    }
}

fn py_bit_pattern(
    protected_bits: HashSet<usize>,
    bit_pattern_length: usize,
) -> PyResult<BitPattern> {
    BitPattern::new(protected_bits.clone(), bit_pattern_length).map_err(|err| {
        PyValueError::new_err(match err {
            bit_patterns::CreationError::InvalidLength { expected_at_least } => {
                format!(
                    "Invalid length ({}) for bit pattern ({:?}), expected at least {}",
                    bit_pattern_length, protected_bits, expected_at_least
                )
            }
            bit_patterns::CreationError::AllProtected => format!(
                "The pattern {:?} with length {} covers all bits, use some other technique",
                protected_bits, bit_pattern_length,
            ),
            bit_patterns::CreationError::Empty => bit_patterns::CreationError::Empty.to_string(),
        })
    })
}

fn encode_bit_pattern_generic<'py, T>(
    py: Python<'py>,
    buffer: NonUniformSequence<Vec<Vec<T>>>,
    bit_pattern_bits: HashSet<usize>,
    bit_pattern_length: usize,
    bits_per_chunk: usize,
) -> PyResult<PyBitPatternEncoding>
where
    T: SizedBitBuffer,
{
    let pattern = py_bit_pattern(bit_pattern_bits, bit_pattern_length)?;

    let encoding = pattern
        .encode(&buffer, bits_per_chunk)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    PyBitPatternEncoding::from_rust(py, encoding)
}

#[pyfunction]
pub fn encode_bit_pattern_f32<'py>(
    py: Python<'py>,
    input: Vec<InputArr<f32>>,
    bit_pattern_bits: HashSet<usize>,
    bit_pattern_length: usize,
    bits_per_chunk: usize,
) -> PyResult<PyBitPatternEncoding> {
    let buffer = prep_input_array_list(input);

    encode_bit_pattern_generic(
        py,
        buffer,
        bit_pattern_bits,
        bit_pattern_length,
        bits_per_chunk,
    )
}

#[pyfunction]
pub fn encode_bit_pattern_u16<'py>(
    py: Python<'py>,
    input: Vec<InputArr<u16>>,
    bit_pattern_bits: HashSet<usize>,
    bit_pattern_length: usize,
    bits_per_chunk: usize,
) -> PyResult<PyBitPatternEncoding> {
    let buffer = prep_input_array_list(input);

    encode_bit_pattern_generic(
        py,
        buffer,
        bit_pattern_bits,
        bit_pattern_length,
        bits_per_chunk,
    )
}
