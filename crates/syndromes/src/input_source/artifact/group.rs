//! Which elements of the array are pooled.

use std::fmt;

/// The elements of the array whose samples are pooled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Group {
    /// Every element.
    Array,
    /// Every element in one row.
    Row(usize),
    /// Every element in one column.
    Column(usize),
    /// A single element.
    Element {
        /// The element's row.
        row: usize,
        /// The element's column.
        column: usize,
    },
}

impl fmt::Display for Group {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Array => formatter.write_str("the whole array"),
            Self::Row(row) => write!(formatter, "row {row}"),
            Self::Column(column) => write!(formatter, "column {column}"),
            Self::Element { row, column } => write!(formatter, "element ({row}, {column})"),
        }
    }
}
