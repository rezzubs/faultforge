//! Arbitrary width chunks that support parallel secded encoding and decoding.

#[cfg(test)]
mod tests;

use crate::encoding::secded::encode;
use rayon::prelude::*;

use crate::{
    BitBuffer, ByteBuffer, Limited,
    encoding::secded::decode_into,
    sequence::{NonMatchingItemError, UniformSequence},
};

/// A single chunk in [`ByteChunks`].
pub type ByteChunk = Vec<u8>;

/// A single chunk in [`DynChunks`].
pub type DynChunk = Limited<Vec<u8>>;

/// The given configuration would result in invalid chunks.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ChunksCreationError {
    #[error("Cannot chunk an empty buffer")]
    Empty,
    #[error("The chunk size has to be non-zero")]
    ZeroChunksize,
}

/// How many chunks are required to store a buffer with `chunk_size` bits per chunk.
#[inline]
fn chunk_count(buffer_size: usize, chunk_size: usize) -> Result<usize, ChunksCreationError> {
    if buffer_size == 0 {
        return Err(ChunksCreationError::Empty);
    }

    if chunk_size == 0 {
        return Err(ChunksCreationError::ZeroChunksize);
    }

    Ok(buffer_size / chunk_size
        + if buffer_size.is_multiple_of(chunk_size) {
            0
        } else {
            1
        })
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DecodeError {
    #[error(
        "The data bits count must be invalid because a length mismatch was detected during decoding: {0}"
    )]
    InvalidDataBitsCount(#[source] crate::encoding::secded::DecodeError),
}

/// A [`BitBuffer`] that's chunked into chunks that are multiples of 8 bits.
///
/// Should be prefered over [`DynChunks`] (if possible) due to performance reasons.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ByteChunks(UniformSequence<Vec<ByteChunk>>);

impl ByteChunks {
    /// Create new chunks with all bits initialized to zero.
    ///
    /// Returns [`None`] if `byte_count` is 0.
    pub fn zero(byte_count: usize, bytes_per_chunk: usize) -> Result<Self, ChunksCreationError> {
        let chunk_count = chunk_count(byte_count, bytes_per_chunk)?;

        Ok(Self(UniformSequence::new_unchecked(
            vec![vec![0u8; bytes_per_chunk]; chunk_count],
            bytes_per_chunk * 8,
            chunk_count,
        )))
    }

    /// Create chunks from the `buffer`.
    ///
    /// If the number of bytes in the buffer isn't a multiple of the number of bytes per chunk then
    /// it will result in a number of bytes of (essentially useless) padding at the end of the final
    /// chunk.
    ///
    /// Returns [`None`] for empty buffers.
    pub fn from_buffer<T>(buffer: &T, bytes_per_chunk: usize) -> Result<Self, ChunksCreationError>
    where
        T: ByteBuffer,
    {
        let byte_count = buffer.byte_count();
        let mut output_buffer = Self::zero(byte_count, bytes_per_chunk)?;

        let result = buffer.copy_into_chunked(&mut output_buffer);
        assert_eq!(result.units_copied, byte_count);

        Ok(output_buffer)
    }

    /// Encode all chunks in parallel.
    #[must_use]
    pub fn encode_chunks(&self) -> DynChunks {
        let output_buffer = self
            .0
            .inner()
            .par_iter()
            .map(|chunk| encode(chunk).expect("chunks cannot be empty"))
            .collect::<Vec<_>>();

        DynChunks(UniformSequence::new(output_buffer).unwrap_or_else(|err| {
            unreachable!(
                "UniformSequence creation shouldn't fail as the chunk sizes are known to be the \
same unless they have been tampered with after creation. Got error {err}",
            );
        }))
    }

    /// Get the number of chunks.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.0.items_count()
    }

    /// Get the number of bytes per chunk.
    #[must_use]
    pub fn bytes_per_chunk(&self) -> usize {
        self.0
            .inner()
            .iter()
            .next()
            .map(|chunk| chunk.len())
            .unwrap_or(0)
    }
}

/// A [`BitBuffer`] that's chunked into chunks of any size.
///
/// [`ByteChunks`] should be prefered (if possible) due to performance reasons.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DynChunks(UniformSequence<Vec<DynChunk>>);

impl DynChunks {
    pub fn zero(bit_count: usize, chunk_bit_count: usize) -> Result<Self, ChunksCreationError> {
        let chunk_count = chunk_count(bit_count, chunk_bit_count)?;

        Ok(Self(UniformSequence::new_unchecked(
            vec![Limited::bytes(chunk_bit_count); chunk_count],
            chunk_bit_count,
            chunk_count,
        )))
    }
    /// Create new dynamic chunks from the `buffer`.
    ///
    /// If the number of bits in the buffer isn't a multiple of the number of
    /// bits per chunk then it will result in a number of bits of (essentially
    /// useless) padding at the end of the final chunk.
    pub fn from_buffer<T>(buffer: &T, bits_per_chunk: usize) -> Result<Self, ChunksCreationError>
    where
        T: BitBuffer,
    {
        let input_size = buffer.bit_count();
        let mut output_buffer = Self::zero(input_size, bits_per_chunk)?;
        let result = buffer.copy_into(&mut output_buffer);
        assert_eq!(result.units_copied, input_size);

        Ok(output_buffer)
    }

    /// Encode all chunks in parallel.
    #[must_use]
    pub fn encode_chunks(&self) -> DynChunks {
        let output_buffer = self
            .0
            .inner()
            .par_iter()
            .map(|chunk| encode(chunk).expect("chunks cannot be empty"))
            .collect::<Vec<_>>();

        DynChunks(UniformSequence::new(output_buffer).unwrap_or_else(|err| {
            unreachable!(
                "UniformSequence creation shouldn't fail as the chunk sizes are known to be the \
same unless they have been tampered with after creation. Got error {err}",
            );
        }))
    }

    /// Decode the chunks in parallel.
    ///
    /// The output is guaranteed to be [`DynChunks`].
    ///
    /// The second `Vec` records double error detections.
    ///
    /// See also:
    /// - `DynChunks::decode_chunks_byte` (private, for byte-aligned output)
    /// - [`DynChunks::decode_chunks`]
    pub fn decode_chunks_dyn(
        self,
        chunk_data_bit_count: usize,
    ) -> Result<(DynChunks, Vec<bool>), DecodeError> {
        let chunk_count = self.chunk_count();

        let output_buffer = vec![Limited::bytes(chunk_data_bit_count); chunk_count];
        let (decoded_output, double_error_detections) = self
            .0
            .into_inner()
            .into_par_iter()
            .zip(output_buffer)
            .map(|(mut source, mut dest)| {
                let double_error_detected =
                    decode_into(&mut source, &mut dest).map_err(|err| match err {
                        crate::encoding::secded::DecodeError::DestEmpty => {
                            unreachable!("There is always at least one chunk")
                        }
                        crate::encoding::secded::DecodeError::LengthMismatch { .. } => {
                            DecodeError::InvalidDataBitsCount(err)
                        }
                    })?;

                Ok((dest, double_error_detected))
            })
            .collect::<Result<(Vec<_>, Vec<_>), DecodeError>>()?;

        Ok((
            DynChunks(UniformSequence::new_unchecked(
                decoded_output,
                chunk_data_bit_count,
                chunk_count,
            )),
            double_error_detections,
        ))
    }

    /// Decode the chunks in parallel.
    ///
    /// The output is guaranteed to be [`ByteChunks`].
    ///
    /// The second `Vec` records double error detections.
    ///
    /// See also:
    /// - [`DynChunks::decode_chunks_dyn`]
    /// - [`DynChunks::decode_chunks`]
    fn decode_chunks_byte(
        self,
        chunk_data_byte_count: usize,
    ) -> Result<(ByteChunks, Vec<bool>), DecodeError> {
        let chunk_count = self.chunk_count();
        let output_buffer = vec![vec![0u8; chunk_data_byte_count]; chunk_count];
        let (decoded_output, double_error_detections) = self
            .0
            .into_inner()
            .into_par_iter()
            .zip(output_buffer)
            .map(|(mut source, mut dest)| {
                let double_error_detected =
                    decode_into(&mut source, &mut dest).map_err(|err| match err {
                        crate::encoding::secded::DecodeError::DestEmpty => {
                            unreachable!("There is always at least one chunk")
                        }
                        crate::encoding::secded::DecodeError::LengthMismatch { .. } => {
                            DecodeError::InvalidDataBitsCount(err)
                        }
                    })?;

                Ok((dest, double_error_detected))
            })
            .collect::<Result<(Vec<_>, Vec<_>), DecodeError>>()?;

        Ok((
            ByteChunks(UniformSequence::new_unchecked(
                decoded_output,
                chunk_data_byte_count * 8,
                chunk_count,
            )),
            double_error_detections,
        ))
    }

    /// Decode all chunks in parallel.
    ///
    /// Automatically determines the appropriate output format. For manual selection see:
    /// - [`DynChunks::decode_chunks_dyn`]
    /// - `DynChunks::decode_chunks_byte` (private, for byte-aligned output)
    ///
    /// The second `Vec` records double error detections.
    ///
    /// While it's simple to compute the number of required parity bits to protect a number of data
    /// bits. There is no straightforward way to compute the number of data bits from the number
    /// of encoded bits. Approximations or a brute force method will need to be used. That's why
    /// `data_bits` is given again instead.
    pub fn decode_chunks(
        self,
        chunk_data_bit_count: usize,
    ) -> Result<(Chunks, Vec<bool>), DecodeError> {
        Ok(if chunk_data_bit_count.is_multiple_of(8) {
            let data_byte_count = chunk_data_bit_count / 8;
            let (chunks, double_error_detections) = self.decode_chunks_byte(data_byte_count)?;
            (Chunks::Byte(chunks), double_error_detections)
        } else {
            let (chunks, double_error_detections) = self.decode_chunks_dyn(chunk_data_bit_count)?;
            (Chunks::Dyn(chunks), double_error_detections)
        })
    }

    /// Get the number of chunks.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        let chunks = self.0.items_count();
        assert!(chunks > 0);
        chunks
    }

    /// Get the number of bits per chunk.
    #[must_use]
    #[doc(alias = "chunk_size")]
    pub fn bits_per_chunk(&self) -> usize {
        self.0
            .inner()
            .iter()
            .next()
            .map(|chunk| chunk.bit_count())
            .unwrap_or(0)
    }

    #[must_use]
    pub fn into_raw(self) -> Vec<DynChunk> {
        self.0.into_inner()
    }

    pub fn from_raw(raw: Vec<DynChunk>) -> Result<Self, NonMatchingItemError> {
        UniformSequence::new(raw).map(Self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Chunks {
    Byte(ByteChunks),
    Dyn(DynChunks),
}

impl Chunks {
    /// Encode all chunks in parallel.
    #[must_use]
    pub fn encode_chunks(&self) -> DynChunks {
        match self {
            Chunks::Byte(byte_chunks) => byte_chunks.encode_chunks(),
            Chunks::Dyn(dyn_chunks) => dyn_chunks.encode_chunks(),
        }
    }

    /// Create chunks from the `buffer`.
    ///
    /// If the number of bits in the buffer isn't a multiple of the number of
    /// bits per chunk then it will result in a number of bits of (essentially
    /// useless) padding at the end of the final chunk.
    pub fn from_buffer<T>(buffer: &T, bits_per_chunk: usize) -> Result<Self, ChunksCreationError>
    where
        T: ByteBuffer,
    {
        Ok(if bits_per_chunk.is_multiple_of(8) {
            Chunks::Byte(buffer.to_byte_chunks(bits_per_chunk / 8)?)
        } else {
            Chunks::Dyn(DynChunks::from_buffer(buffer, bits_per_chunk)?)
        })
    }

    /// Create new chunks with all bits initialized to zero.
    ///
    /// Returns [`None`] if `bit_count` is 0.
    pub fn zero(bit_count: usize, chunk_bit_count: usize) -> Result<Self, ChunksCreationError> {
        Ok(
            if chunk_bit_count.is_multiple_of(8) && bit_count.is_multiple_of(8) {
                Chunks::Byte(ByteChunks::zero(bit_count / 8, chunk_bit_count / 8)?)
            } else {
                Chunks::Dyn(DynChunks::zero(bit_count, chunk_bit_count)?)
            },
        )
    }

    /// Get the number of chunks.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        match self {
            Chunks::Byte(byte_chunks) => byte_chunks.chunk_count(),
            Chunks::Dyn(dyn_chunks) => dyn_chunks.chunk_count(),
        }
    }

    #[must_use]
    #[doc(alias = "chunk_size")]
    pub fn bits_per_chunk(&self) -> usize {
        match self {
            Chunks::Byte(byte_chunks) => byte_chunks.bytes_per_chunk() * 8,
            Chunks::Dyn(dyn_chunks) => dyn_chunks.bits_per_chunk(),
        }
    }
}

impl BitBuffer for ByteChunks {
    fn bit_count(&self) -> usize {
        self.0.bit_count()
    }

    fn set_1(&mut self, bit_index: usize) {
        self.0.set_1(bit_index)
    }

    fn set_0(&mut self, bit_index: usize) {
        self.0.set_0(bit_index)
    }

    fn is_1(&self, bit_index: usize) -> bool {
        self.0.is_1(bit_index)
    }

    fn flip_bit(&mut self, bit_index: usize) {
        self.0.flip_bit(bit_index)
    }
}

impl BitBuffer for DynChunks {
    fn bit_count(&self) -> usize {
        self.0.bit_count()
    }

    fn set_1(&mut self, bit_index: usize) {
        self.0.set_1(bit_index)
    }

    fn set_0(&mut self, bit_index: usize) {
        self.0.set_0(bit_index)
    }

    fn is_1(&self, bit_index: usize) -> bool {
        self.0.is_1(bit_index)
    }

    fn flip_bit(&mut self, bit_index: usize) {
        self.0.flip_bit(bit_index)
    }
}

impl BitBuffer for Chunks {
    fn bit_count(&self) -> usize {
        match self {
            Chunks::Byte(chunks) => chunks.bit_count(),
            Chunks::Dyn(chunks) => chunks.bit_count(),
        }
    }

    fn set_1(&mut self, bit_index: usize) {
        match self {
            Chunks::Byte(chunks) => chunks.set_1(bit_index),
            Chunks::Dyn(chunks) => chunks.set_1(bit_index),
        }
    }

    fn set_0(&mut self, bit_index: usize) {
        match self {
            Chunks::Byte(chunks) => chunks.set_0(bit_index),
            Chunks::Dyn(chunks) => chunks.set_0(bit_index),
        }
    }

    fn is_1(&self, bit_index: usize) -> bool {
        match self {
            Chunks::Byte(chunks) => chunks.is_1(bit_index),
            Chunks::Dyn(chunks) => chunks.is_1(bit_index),
        }
    }

    fn flip_bit(&mut self, bit_index: usize) {
        match self {
            Chunks::Byte(chunks) => chunks.flip_bit(bit_index),
            Chunks::Dyn(chunks) => chunks.flip_bit(bit_index),
        }
    }
}

impl ByteBuffer for ByteChunks {
    fn byte_count(&self) -> usize {
        self.0.byte_count()
    }

    fn get_byte(&self, n: usize) -> u8 {
        self.0.get_byte(n)
    }

    fn set_byte(&mut self, n: usize, value: u8) {
        self.0.set_byte(n, value)
    }
}
