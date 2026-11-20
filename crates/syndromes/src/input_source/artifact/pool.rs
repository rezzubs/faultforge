//! The recorded triples one run samples from.

use super::{
    ArtifactError, Group, Regime,
    position::Position,
    regime_samples::{Dimensions, RegimeSamples},
};
use crate::Triple;
use rand::{Rng, RngExt};
use std::path::Path;

/// The positions in `group`, or `None` if the group lies outside a
/// `rows x columns` array.
fn elements(group: Group, rows: usize, columns: usize) -> Option<Vec<Position>> {
    let position = |row, column| Position { row, column };
    match group {
        Group::Array => Some(
            (0..rows)
                .flat_map(|row| (0..columns).map(move |column| position(row, column)))
                .collect(),
        ),
        Group::Row(row) => {
            (row < rows).then(|| (0..columns).map(|column| position(row, column)).collect())
        }
        Group::Column(column) => {
            (column < columns).then(|| (0..rows).map(|row| position(row, column)).collect())
        }
        Group::Element { row, column } => {
            (row < rows && column < columns).then(|| vec![position(row, column)])
        }
    }
}
/// The recorded triples of one regime and one group, pooled.
///
/// Every real sample of every element in the group is in the pool, so an
/// element weighs in by its number of samples.
#[derive(Debug, Clone, PartialEq)]
pub struct Pool {
    triples: Vec<Triple>,
}

impl Pool {
    /// Reads the samples of `regime` for the elements in `group` from the
    /// artifact at `path`.
    ///
    /// Fails if the file is missing or malformed, if `group` lies outside
    /// the array, or if no element in the group has a sample.
    pub fn load(path: &Path, regime: Regime, group: Group) -> Result<Self, ArtifactError> {
        let regime_samples = RegimeSamples::read(path, regime)?;
        let Dimensions { rows, columns, .. } = regime_samples.dimensions();
        let elements = elements(group, rows, columns).ok_or(ArtifactError::GroupOutOfRange {
            path: path.to_owned(),
            group,
            rows,
            columns,
        })?;

        let mut triples = Vec::new();
        for position in elements {
            triples.extend(regime_samples.element_triples(position));
        }
        if triples.is_empty() {
            return Err(ArtifactError::EmptyGroup {
                path: path.to_owned(),
                regime,
                group,
            });
        }
        Ok(Self { triples })
    }

    /// The number of triples in the pool.
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Whether the pool has no triples. Never true for a loaded pool.
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// The pooled triples, in no particular order.
    pub fn triples(&self) -> &[Triple] {
        &self.triples
    }

    /// A uniformly chosen triple.
    pub fn pick(&self, rng: &mut impl Rng) -> &Triple {
        &self.triples[rng.random_range(0..self.triples.len())]
    }
}

#[cfg(test)]
mod tests {
    use super::{super::test_archives::*, *};

    #[test]
    fn array_pools_every_real_sample() {
        let file = fixture_archive("array");
        let pool = load(&file, Regime::Active, Group::Array).expect("loads");
        let expected: Vec<Triple> = (0..ROWS)
            .flat_map(|row| (0..COLUMNS).flat_map(move |column| expected_triples(row, column)))
            .collect();
        assert_eq!(pool.len(), 10);
        assert_eq!(sorted(pool.triples().to_vec()), sorted(expected));
    }

    #[test]
    fn row_column_and_element_select_their_samples() {
        let file = fixture_archive("groups");

        let row = load(&file, Regime::First, Group::Row(1)).expect("loads");
        let expected: Vec<Triple> = (0..COLUMNS)
            .flat_map(|column| expected_triples(1, column))
            .collect();
        assert_eq!(sorted(row.triples().to_vec()), sorted(expected));

        let column = load(&file, Regime::First, Group::Column(2)).expect("loads");
        let expected: Vec<Triple> = (0..ROWS).flat_map(|row| expected_triples(row, 2)).collect();
        assert_eq!(sorted(column.triples().to_vec()), sorted(expected));

        let element =
            load(&file, Regime::First, Group::Element { row: 1, column: 1 }).expect("loads");
        assert_eq!(element.triples(), expected_triples(1, 1));
    }

    #[test]
    fn group_outside_the_array_is_an_error() {
        let file = fixture_archive("range");
        for group in [
            Group::Row(ROWS),
            Group::Column(COLUMNS),
            Group::Element {
                row: 0,
                column: COLUMNS,
            },
        ] {
            assert!(matches!(
                load(&file, Regime::Active, group),
                Err(ArtifactError::GroupOutOfRange {
                    rows: ROWS,
                    columns: COLUMNS,
                    ..
                })
            ));
        }
    }

    #[test]
    fn group_without_samples_is_an_error() {
        let file = fixture_archive("empty");
        assert!(matches!(
            load(&file, Regime::Drain, Group::Element { row: 0, column: 0 }),
            Err(ArtifactError::EmptyGroup {
                regime: Regime::Drain,
                ..
            })
        ));
    }
}
