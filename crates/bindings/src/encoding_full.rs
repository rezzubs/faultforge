use crate::common::*;
use faultforge::{
    bit_buffer::chunks::{Chunks, DynChunks},
    buffers::{Limited, NonUniformSequence},
    encoding::secded::encoded_bits_count,
    prelude::*,
};
use numpy::PyArray1;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBytes, PyList},
};

#[pyclass(name = "FullEncoding")]
pub struct PyFullEncoding {
    /// A list of the encoded chunks. Guaranteed to store [`PyBytes`] elements.
    encoded_chunks: Py<PyList>,
    /// The number of data bits per chunk, used for bounds in `Limited`.
    data_bits_count: usize,
    /// The number of items in each array in the input list for the encoding
    /// function.
    item_counts: Vec<usize>,
}

impl PyFullEncoding {
    pub fn decode_full_generic<'py, T>(
        &self,
        py: Python<'py>,
        mut output_buffer: NonUniformSequence<Vec<Vec<T>>>,
    ) -> PyResult<(Vec<OutputArr<'py, T>>, Vec<bool>)>
    where
        T: SizedBitBuffer + numpy::Element + ByteChunkedBitBuffer,
    {
        let input_chunks = self.to_rust(py)?;

        let (output_chunks, decoding_results) = input_chunks
            .decode_chunks(self.data_bits_count)
            .unwrap_or_else(|err| match err {
                faultforge::bit_buffer::chunks::DecodeError::InvalidDataBitsCount(_) => {
                    panic!("The data bits count is immutable from the python side and should be correct yet: {}", err);
                }
            });

        let bits_copied = match output_chunks {
            Chunks::Byte(byte_chunks) => {
                byte_chunks
                    .copy_into_chunked(&mut output_buffer)
                    .units_copied
                    * 8
            }
            Chunks::Dyn(dyn_chunks) => dyn_chunks.copy_into(&mut output_buffer).units_copied,
        };
        assert_eq!(
            bits_copied,
            output_buffer.bits_count(),
            "these must match because the chunked buffer can potentially only have more bits, not less.",
        );

        Ok((
            output_buffer
                .0
                .into_iter()
                .map(|vec| PyArray1::from_vec(py, vec))
                .collect(),
            decoding_results,
        ))
    }

    pub fn from_rust<'py>(
        py: Python<'py>,
        encoded_chunks: DynChunks,
        data_bits_count: usize,
        item_counts: Vec<usize>,
    ) -> PyResult<Self> {
        let bytes = encoded_chunks.into_raw().into_iter().map(|chunk| {
            debug_assert_eq!(chunk.bits_count(), data_bits_count);
            let chunk_bytes = chunk.into_inner();
            PyBytes::new(py, &chunk_bytes).unbind()
        });

        let encoded_chunks = PyList::new(py, bytes)?.unbind();

        Ok(PyFullEncoding {
            encoded_chunks,
            data_bits_count,
            item_counts,
        })
    }

    pub fn to_rust<'py>(&self, py: Python<'py>) -> PyResult<DynChunks> {
        let encoded_bits_count = encoded_bits_count(self.data_bits_count)
            .expect("Known to be correct. Checked during initialization.");

        let raw_chunks = self
            .encoded_chunks
            .bind(py)
            .iter()
            .map(|chunk| {
                let py_bytes = chunk.downcast_into::<PyBytes>().expect(
                    "It was created as `PyBytes` and it should not be possible to modify it after.",
                );

                Limited::new(Vec::from(py_bytes.as_bytes()), encoded_bits_count).unwrap_or_else(
                    || {
                        panic!(
                            "`bits_per_chunk` {} doesn't match the encoded chunks.",
                            encoded_bits_count
                        )
                    },
                )
            })
            .collect::<Vec<_>>();

        Ok(
            DynChunks::from_raw(raw_chunks).expect(
                "There's no reason recreating the chunks should fail assuming the data remains immutable."
            )
        )
    }
}

#[pymethods]
impl PyFullEncoding {
    /// Decode a list of float32 values.
    pub fn decode_full_f32<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Vec<OutputArr<'py, f32>>, Vec<bool>)> {
        let output_buffer = NonUniformSequence(
            self.item_counts
                .iter()
                .map(|&numel| vec![0f32; numel])
                .collect::<Vec<_>>(),
        );

        self.decode_full_generic(py, output_buffer)
    }

    pub fn decode_full_u16<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Vec<OutputArr<'py, u16>>, Vec<bool>)> {
        let output_buffer = NonUniformSequence(
            self.item_counts
                .iter()
                .map(|&numel| vec![0u16; numel])
                .collect::<Vec<_>>(),
        );

        self.decode_full_generic(py, output_buffer)
    }

    pub fn flip_n_bits<'py>(&mut self, py: Python<'py>, n: usize) -> PyResult<()> {
        let mut encoding = self.to_rust(py)?;

        encoding.flip_n_bits(n).map_err(|_| {
            PyValueError::new_err(format!(
                "invalid number of bits {n}, only 0 <= {} is possible",
                encoding.bits_count()
            ))
        })?;

        // PERF: The `item_counts` clone could be avoided if the conversion of
        // `encoding` is extracted from the `from_rust` method. Then we would
        // just need to reassign `bits_per_chunk` instead of the whole struct.
        *self = PyFullEncoding::from_rust(
            py,
            encoding,
            self.data_bits_count,
            self.item_counts.clone(),
        )?;

        Ok(())
    }

    /// Return a new instance with cloned data.
    pub fn clone<'py>(&self, py: Python<'py>) -> PyResult<PyFullEncoding> {
        let encoding_clone = self.encoded_chunks.bind(py).iter().map(|chunk| {
            let bytes = chunk
                .downcast_into::<PyBytes>()
                .expect("Was constructed as PyBytes and is immutable.");

            let cloned = bytes.as_bytes().to_owned();

            PyBytes::new(py, &cloned).unbind()
        });

        Ok(PyFullEncoding {
            encoded_chunks: PyList::new(py, encoding_clone)?.unbind(),
            data_bits_count: self.data_bits_count,
            item_counts: self.item_counts.clone(),
        })
    }

    pub fn bits_count<'py>(&self, py: Python<'py>) -> usize {
        self.encoded_chunks.bind(py).len()
            * encoded_bits_count(self.data_bits_count)
                .expect("the data bits count must be checked during initialization")
    }
}

fn encode_full_generic<'py, T>(
    py: Python<'py>,
    buffer: NonUniformSequence<Vec<Vec<T>>>,
    bits_per_chunk: usize,
) -> PyResult<PyFullEncoding>
where
    T: ByteChunkedBitBuffer + SizedBitBuffer,
{
    let item_counts = buffer.0.iter().map(|chunk| chunk.len()).collect::<Vec<_>>();

    let encoded_chunks = buffer
        .to_chunks(bits_per_chunk)
        .map_err(|err| match err {
            faultforge::bit_buffer::chunks::InvalidChunks::Empty
            | faultforge::bit_buffer::chunks::InvalidChunks::ZeroChunksize => {
                PyValueError::new_err(err.to_string())
            }
        })?
        .encode_chunks();

    PyFullEncoding::from_rust(py, encoded_chunks, bits_per_chunk, item_counts)
}

/// Encode a all bits of a buffer of 32 bit floats.
///
/// Returns a tuple containing:
/// - A uint8 array with the encoded bits
/// - The number of useful bits in the encoded buffer
///
/// These need to be given to the decoding function to restore the original representation.
#[pyfunction]
pub fn encode_full_f32<'py>(
    py: Python<'py>,
    input: Vec<InputArr<f32>>,
    bits_per_chunk: usize,
) -> PyResult<PyFullEncoding> {
    let buffer = prep_input_array_list(input);

    encode_full_generic(py, buffer, bits_per_chunk)
}

/// Encode a all bits of a buffer of 16 bit unsigned integers.
///
/// Returns a tuple containing:
/// - A uint8 array with the encoded bits
/// - The number of useful bits in the encoded buffer
///
/// These need to be given to the decoding function to restore the original representation.
#[pyfunction]
pub fn encode_full_u16<'py>(
    py: Python<'py>,
    input: Vec<InputArr<u16>>,
    bits_per_chunk: usize,
) -> PyResult<PyFullEncoding> {
    let buffer = prep_input_array_list(input);

    encode_full_generic(py, buffer, bits_per_chunk)
}
