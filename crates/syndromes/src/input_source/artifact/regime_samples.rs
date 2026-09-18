//! One regime's samples, read from an artifact and checked for consistency.

use super::{ArtifactError, Regime, position::Position};
use crate::Triple;
use ndarray::{Array2, Array3, Array4};
use ndarray_npy::NpzReader;
use std::{fs::File, path::Path};

// Where a regime's data lives in the archive. Kept here rather than on
// `Regime` itself so the file layout is known to the reader only.
impl Regime {
    /// The archive member holding the recorded samples.
    fn samples_member(self) -> &'static str {
        match self {
            Self::First => "first_triples",
            Self::Active => "active_triples",
            Self::Drain => "drain_partial_sums",
        }
    }

    /// The archive member holding the number of real samples per element.
    fn fill_member(self) -> &'static str {
        match self {
            Self::First => "first_fill",
            Self::Active => "active_fill",
            Self::Drain => "drain_fill",
        }
    }
}

/// The sizes of a regime's sample array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dimensions {
    /// The number of rows recorded.
    pub rows: usize,
    /// The number of columns recorded.
    pub columns: usize,
    /// The number of samples stored per element, real or padding.
    pub capacity: usize,
}

/// The recorded samples of one regime, as stored.
enum SampleArray {
    /// `(rows, columns, capacity, 3)` with the fields in triple order.
    Triples(Array4<f32>),
    /// `(rows, columns, capacity)` partial sums; activation and weight are
    /// zero in the drain regime.
    PartialSums(Array3<f32>),
}

impl SampleArray {
    fn dimensions(&self) -> Dimensions {
        let (rows, columns, capacity) = match self {
            Self::Triples(triples) => {
                let (rows, columns, capacity, _fields) = triples.dim();
                (rows, columns, capacity)
            }
            Self::PartialSums(partial_sums) => partial_sums.dim(),
        };
        Dimensions {
            rows,
            columns,
            capacity,
        }
    }

    fn triple(&self, position: Position, index: usize) -> Triple {
        let Position { row, column } = position;
        match self {
            Self::Triples(triples) => Triple {
                activation: triples[[row, column, index, 0]],
                weight: triples[[row, column, index, 1]],
                partial_sum: triples[[row, column, index, 2]],
            },
            Self::PartialSums(partial_sums) => Triple {
                activation: 0.0,
                weight: 0.0,
                partial_sum: partial_sums[[row, column, index]],
            },
        }
    }
}

/// The samples and fill counts of one regime for every element.
pub struct RegimeSamples {
    samples: SampleArray,
    /// The number of real samples per element, each at most the capacity.
    fill: Array2<usize>,
}

impl RegimeSamples {
    /// Reads `regime`'s members from the artifact at `path`.
    ///
    /// Fails if the file is missing or malformed, if the members' shapes do
    /// not fit together, or if an element claims more samples than are
    /// stored.
    pub fn read(path: &Path, regime: Regime) -> Result<Self, ArtifactError> {
        let file = File::open(path).map_err(|source| ArtifactError::Read {
            path: path.to_owned(),
            source,
        })?;
        let map_format_error = |source| ArtifactError::Format {
            path: path.to_owned(),
            source,
        };
        let mut reader = NpzReader::new(file).map_err(map_format_error)?;
        let fill: Array2<u64> = reader
            .by_name(regime.fill_member())
            .map_err(map_format_error)?;
        let samples = match regime {
            Regime::First | Regime::Active => SampleArray::Triples(
                reader
                    .by_name(regime.samples_member())
                    .map_err(map_format_error)?,
            ),
            Regime::Drain => SampleArray::PartialSums(
                reader
                    .by_name(regime.samples_member())
                    .map_err(map_format_error)?,
            ),
        };

        let Dimensions {
            rows,
            columns,
            capacity,
        } = samples.dimensions();
        if let SampleArray::Triples(triples) = &samples
            && triples.dim().3 != 3
        {
            return Err(ArtifactError::Shape {
                path: path.to_owned(),
                member: regime.samples_member(),
                expected: format!("[{rows}, {columns}, {capacity}, 3]"),
                actual: format!("{:?}", triples.shape()),
            });
        }
        if fill.dim() != (rows, columns) {
            return Err(ArtifactError::Shape {
                path: path.to_owned(),
                member: regime.fill_member(),
                expected: format!("[{rows}, {columns}]"),
                actual: format!("{:?}", fill.shape()),
            });
        }

        // The file stores fills as fixed-width integers; past this point a
        // fill is a valid length along the sample axis.
        let mut checked_fill = Array2::zeros((rows, columns));
        for ((row, column), &fill) in fill.indexed_iter() {
            checked_fill[[row, column]] = usize::try_from(fill)
                .ok()
                .filter(|&fill| fill <= capacity)
                .ok_or(ArtifactError::Fill {
                    path: path.to_owned(),
                    row,
                    column,
                    fill,
                    capacity,
                })?;
        }

        Ok(Self {
            samples,
            fill: checked_fill,
        })
    }

    /// The sizes of the sample array.
    pub fn dimensions(&self) -> Dimensions {
        self.samples.dimensions()
    }

    /// The real samples of one element.
    pub fn element_triples(&self, position: Position) -> impl Iterator<Item = Triple> + '_ {
        let Position { row, column } = position;
        // Entries past the fill are padding, not samples.
        (0..self.fill[[row, column]]).map(move |index| self.samples.triple(position, index))
    }
}

#[cfg(test)]
mod tests {
    use super::{super::test_archives::*, *};
    use crate::test_files::{TemporaryFile, temporary_path};

    fn read(file: &TemporaryFile, regime: Regime) -> Result<RegimeSamples, ArtifactError> {
        RegimeSamples::read(&file.0, regime)
    }

    #[test]
    fn reads_dimensions_and_real_samples() {
        let file = fixture_archive("read");
        let samples = read(&file, Regime::Active).expect("reads");
        assert_eq!(
            samples.dimensions(),
            Dimensions {
                rows: ROWS,
                columns: COLUMNS,
                capacity: CAPACITY,
            }
        );
        for row in 0..ROWS {
            for column in 0..COLUMNS {
                let triples: Vec<Triple> =
                    samples.element_triples(Position { row, column }).collect();
                assert_eq!(triples, expected_triples(row, column));
            }
        }
    }

    #[test]
    fn drain_samples_are_partial_sums_only() {
        let file = fixture_archive("drain");
        let samples = read(&file, Regime::Drain).expect("reads");
        let expected: Vec<Triple> = expected_triples(1, 0)
            .into_iter()
            .map(|triple| Triple {
                activation: 0.0,
                weight: 0.0,
                partial_sum: triple.partial_sum,
            })
            .collect();
        let triples: Vec<Triple> = samples
            .element_triples(Position { row: 1, column: 0 })
            .collect();
        assert_eq!(triples, expected);
    }

    #[test]
    fn missing_file_is_a_read_error() {
        let file = temporary_path("missing", "npz");
        assert!(matches!(
            read(&file, Regime::Active),
            Err(ArtifactError::Read { .. })
        ));
    }

    #[test]
    fn missing_member_is_a_format_error() {
        let file = write_archive("member", |writer| {
            writer
                .add_array("active_fill", &fixture_fill())
                .expect("write");
        });
        assert!(matches!(
            read(&file, Regime::Active),
            Err(ArtifactError::Format { .. })
        ));
    }

    #[test]
    fn fill_above_capacity_is_an_error() {
        let file = write_archive("fill", |writer| {
            let mut fill = fixture_fill();
            fill[[0, 2]] = 5;
            writer
                .add_array("active_triples", &fixture_triples())
                .expect("write");
            writer.add_array("active_fill", &fill).expect("write");
        });
        assert!(matches!(
            read(&file, Regime::Active),
            Err(ArtifactError::Fill {
                row: 0,
                column: 2,
                fill: 5,
                capacity: CAPACITY,
                ..
            })
        ));
    }

    #[test]
    fn mismatched_fill_shape_is_an_error() {
        let file = write_archive("fill-shape", |writer| {
            writer
                .add_array("active_triples", &fixture_triples())
                .expect("write");
            writer
                .add_array("active_fill", &Array2::<u64>::zeros((ROWS, COLUMNS + 1)))
                .expect("write");
        });
        assert!(matches!(
            read(&file, Regime::Active),
            Err(ArtifactError::Shape {
                member: "active_fill",
                ..
            })
        ));
    }

    #[test]
    fn wrong_field_count_is_an_error() {
        let file = write_archive("fields", |writer| {
            writer
                .add_array(
                    "first_triples",
                    &Array4::<f32>::zeros((ROWS, COLUMNS, CAPACITY, 2)),
                )
                .expect("write");
            writer
                .add_array("first_fill", &fixture_fill())
                .expect("write");
        });
        assert!(matches!(
            read(&file, Regime::First),
            Err(ArtifactError::Shape {
                member: "first_triples",
                ..
            })
        ));
    }
}
