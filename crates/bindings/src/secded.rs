use crate::{common::*, fault::PyFault};
use memory::{
    BitBuffer,
    encoding::secded::{CowEncodedChunks, DecodeChunksError},
};
use numpy::{PyArray1, PyReadwriteArrayDyn};
use pyo3::{
    exceptions::{PyIndexError, PyValueError},
    prelude::*,
};
use std::ops::Range;

#[pyclass(name = "Encoding")]
pub struct PyEncoding {
    /// The encoding of the bytes of all input arrays, concatenated in input
    /// order.
    encoded: CowEncodedChunks,
    /// The range of each input array's bytes in the data bytes of `encoded`.
    array_byte_ranges: Vec<Range<usize>>,
}

#[pymethods]
impl PyEncoding {
    /// Decode the bytes of one input array into `dest`.
    ///
    /// `dest` must be contiguous and have as many bytes as the input array.
    ///
    /// The double error detection results are only collected (and returned)
    /// if `collect_double_errors` is set. They're the indices of the chunks
    /// that hold part of the array and in which a double error was detected,
    /// in ascending order. A chunk can hold the end of one array and the start
    /// of the next, so decoding both reports it twice.
    #[pyo3(signature = (array_index, dest, *, collect_double_errors = false))]
    pub fn decode_into<'py>(
        &self,
        py: Python<'py>,
        array_index: usize,
        mut dest: PyReadwriteArrayDyn<'py, u8>,
        collect_double_errors: bool,
    ) -> PyResult<Option<OutputArr<'py, u64>>> {
        let array_byte_range = self.array_byte_range(array_index)?;
        let dest = dest
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("`dest` is not contiguous"))?;

        let double_error_chunks = if collect_double_errors {
            self.encoded
                .decode_range_into_with_double_errors(array_byte_range, dest)
                .map(Some)
        } else {
            self.encoded
                .decode_range_into(array_byte_range, dest)
                .map(|()| None)
        }
        .map_err(|err| match err {
            DecodeChunksError::LengthMismatch { .. } => PyValueError::new_err(err.to_string()),
            DecodeChunksError::InvalidRange { .. } => {
                unreachable!("the array ranges are within the data bytes: {err}")
            }
        })?;

        Ok(double_error_chunks.map(|double_error_chunks| {
            PyArray1::from_iter(
                py,
                double_error_chunks.into_iter().map(|chunk_index| {
                    u64::try_from(chunk_index)
                        .expect("usize is at most 64 bits on supported targets")
                }),
            )
        }))
    }

    /// Find the decoded bytes of one input array that differ from the
    /// original data.
    ///
    /// Returns the offsets of the bytes from the start of the array in
    /// ascending order, and the decoded value of each byte. Writing them into
    /// a copy of the original array gives the same bytes as `decode_into`.
    ///
    /// The changes are always relative to the original data. So writing them
    /// into the result of an earlier decode with faults can give wrong bytes.
    pub fn decoded_changes<'py>(
        &self,
        py: Python<'py>,
        array_index: usize,
    ) -> PyResult<(OutputArr<'py, i64>, OutputArr<'py, u8>)> {
        let array_byte_range = self.array_byte_range(array_index)?;
        let changes = self
            .encoded
            .decoded_changes(array_byte_range)
            .unwrap_or_else(|err| {
                unreachable!("the array ranges are within the data bytes: {err}")
            });

        // Torch only indexes with int64 tensors.
        let byte_offsets = changes
            .byte_offsets
            .into_iter()
            .map(|byte_offset| {
                i64::try_from(byte_offset).expect("an array has fewer than 2^63 bytes")
            })
            .collect::<Vec<_>>();
        Ok((
            PyArray1::from_vec(py, byte_offsets),
            PyArray1::from_vec(py, changes.byte_values),
        ))
    }

    /// Return the number of bytes in each input array.
    pub fn array_byte_counts(&self) -> Vec<usize> {
        self.array_byte_ranges.iter().map(Range::len).collect()
    }

    pub fn apply_fault(&mut self, fault: PyFault, target_bit: usize) -> PyResult<()> {
        self.apply_faults(vec![(fault, target_bit)])
    }

    /// Apply multiple faults at once.
    ///
    /// All targets are checked before any fault is applied, so an out of
    /// bounds target leaves the encoding unchanged.
    pub fn apply_faults(&mut self, faults: Vec<(PyFault, usize)>) -> PyResult<()> {
        let bit_count = self.bit_count();
        for (_, target_bit) in &faults {
            if *target_bit >= bit_count {
                return Err(PyIndexError::new_err(format!(
                    "target_bit {} is out of bounds",
                    target_bit
                )));
            }
        }

        self.encoded.apply_faults(
            faults
                .into_iter()
                .map(|(fault, target_bit)| (fault.0, target_bit)),
        );

        Ok(())
    }

    /// Return an independent copy.
    ///
    /// Copies of an instance without faults share their data until faults are
    /// applied, so they're cheap to make.
    pub fn clone(&self) -> PyEncoding {
        PyEncoding {
            encoded: self.encoded.clone(),
            array_byte_ranges: self.array_byte_ranges.clone(),
        }
    }

    pub fn bit_count(&self) -> usize {
        self.encoded.bit_count()
    }
}

impl PyEncoding {
    /// The range of an input array's bytes in the data bytes of `encoded`.
    fn array_byte_range(&self, array_index: usize) -> PyResult<Range<usize>> {
        self.array_byte_ranges
            .get(array_index)
            .cloned()
            .ok_or_else(|| {
                PyIndexError::new_err(format!(
                    "array {array_index} is out of bounds for {} arrays",
                    self.array_byte_ranges.len()
                ))
            })
    }
}

/// Encode the bytes of a list of arrays.
///
/// All arrays are encoded as one buffer, so a chunk can hold bytes of two
/// arrays. The arrays must be contiguous and `bits_per_chunk` must be a
/// positive multiple of 8.
#[pyfunction]
pub fn encode(input: Vec<InputArr<u8>>, bits_per_chunk: usize) -> PyResult<PyEncoding> {
    // `EncodedChunks` only supports whole bytes per chunk.
    if !bits_per_chunk.is_multiple_of(8) {
        return Err(PyValueError::new_err(format!(
            "bits_per_chunk must be a multiple of 8, got {bits_per_chunk}"
        )));
    }

    let arrays = input
        .iter()
        .map(|array| {
            array
                .as_slice()
                .map_err(|_| PyValueError::new_err("an input array is not contiguous"))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let mut array_byte_ranges = Vec::with_capacity(arrays.len());
    let mut data_bytes = Vec::with_capacity(arrays.iter().map(|array| array.len()).sum());
    for array in arrays {
        let start = data_bytes.len();
        data_bytes.extend_from_slice(array);
        array_byte_ranges.push(start..data_bytes.len());
    }

    let encoded = CowEncodedChunks::encode(&data_bytes, bits_per_chunk / 8)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    Ok(PyEncoding {
        encoded,
        array_byte_ranges,
    })
}
