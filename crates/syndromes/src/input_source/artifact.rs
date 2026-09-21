//! Triples recorded from a simulated systolic array.
//!
//! A profiling artifact holds, per processing element and per regime, a
//! bounded sample of the inputs that element saw while a model ran. A
//! [`Pool`] is the part of it that one run samples from: one regime and one
//! group of elements.

mod error;
mod group;
mod joint;
mod marginals;
mod pool;
mod position;
mod regime;
mod regime_samples;

pub use error::ArtifactError;
pub use group::Group;
pub use joint::Joint;
pub use marginals::Marginals;
pub use pool::Pool;
pub use regime::Regime;

#[cfg(test)]
mod test_archives {
    //! Archives with a small, traceable artifact, written to temporary files.

    use super::*;
    use crate::{
        Triple,
        test_files::{TemporaryFile, temporary_path},
    };
    use ndarray::{Array2, Array3, Array4};
    use ndarray_npy::NpzWriter;
    use std::fs::File;

    pub const ROWS: usize = 2;
    pub const COLUMNS: usize = 3;
    pub const CAPACITY: usize = 4;

    /// The number of real samples of element `(row, column)`, chosen so the
    /// fills differ and two elements are empty.
    pub fn fill_of(row: usize, column: usize) -> usize {
        (row * COLUMNS + column) % (CAPACITY + 1)
    }

    /// A value unique to its position, so pooled triples can be traced back.
    /// Nonzero past the fill too, so padding is caught if it leaks.
    pub fn value(row: usize, column: usize, index: usize, field: usize) -> f32 {
        (row * 1000 + column * 100 + index * 10 + field) as f32
    }

    pub fn fixture_fill() -> Array2<u64> {
        Array2::from_shape_fn((ROWS, COLUMNS), |(row, column)| {
            u64::try_from(fill_of(row, column)).expect("small")
        })
    }

    pub fn fixture_triples() -> Array4<f32> {
        Array4::from_shape_fn(
            (ROWS, COLUMNS, CAPACITY, 3),
            |(row, column, index, field)| value(row, column, index, field),
        )
    }

    pub fn fixture_partial_sums() -> Array3<f32> {
        Array3::from_shape_fn((ROWS, COLUMNS, CAPACITY), |(row, column, index)| {
            value(row, column, index, 2)
        })
    }

    /// The triples the fixture records for one element, in stored order.
    pub fn expected_triples(row: usize, column: usize) -> Vec<Triple> {
        (0..fill_of(row, column))
            .map(|index| Triple {
                activation: value(row, column, index, 0),
                weight: value(row, column, index, 1),
                partial_sum: value(row, column, index, 2),
            })
            .collect()
    }

    pub fn write_archive(name: &str, add: impl FnOnce(&mut NpzWriter<File>)) -> TemporaryFile {
        let file = temporary_path(name, "npz");
        let mut writer = NpzWriter::new_compressed(
            File::create(&file.0).expect("temporary directory is writable"),
        );
        add(&mut writer);
        writer.finish().expect("archive is written");
        file
    }

    /// An archive with every member the profiler writes.
    pub fn fixture_archive(name: &str) -> TemporaryFile {
        write_archive(name, |writer| {
            writer
                .add_array("first_triples", &fixture_triples())
                .expect("write");
            writer
                .add_array("first_fill", &fixture_fill())
                .expect("write");
            writer
                .add_array("active_triples", &fixture_triples())
                .expect("write");
            writer
                .add_array("active_fill", &fixture_fill())
                .expect("write");
            writer
                .add_array("drain_partial_sums", &fixture_partial_sums())
                .expect("write");
            writer
                .add_array("drain_fill", &fixture_fill())
                .expect("write");
            writer
                .add_array(
                    "metadata_json",
                    &ndarray::Array1::<u8>::from_vec(b"{}".to_vec()),
                )
                .expect("write");
        })
    }

    pub fn sorted(mut triples: Vec<Triple>) -> Vec<Triple> {
        let key = |triple: &Triple| {
            (
                triple.activation.to_bits(),
                triple.weight.to_bits(),
                triple.partial_sum.to_bits(),
            )
        };
        triples.sort_by_key(key);
        triples
    }

    pub fn load(file: &TemporaryFile, regime: Regime, group: Group) -> Result<Pool, ArtifactError> {
        Pool::load(&file.0, regime, group)
    }
}
