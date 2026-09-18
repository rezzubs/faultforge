//! Where an element sits in the array.

/// The position of an element in the array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    /// The element's row.
    pub row: usize,
    /// The element's column.
    pub column: usize,
}
