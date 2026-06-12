use std::ops::{BitAnd, BitOr, BitXor};

/// A single bit in a [`BitBuffer`](super::BitBuffer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Bit {
    /// A logical zero.
    Zero = 0,
    /// A logical one.
    One = 1,
}

impl Bit {
    /// Check if the bit is one.
    #[must_use]
    #[doc(alias = "is_one")]
    pub fn is_1(&self) -> bool {
        matches!(self, Bit::One)
    }

    /// Check if the bit is one.
    #[must_use]
    #[doc(alias = "is_zero")]
    pub fn is_0(&self) -> bool {
        matches!(self, Bit::Zero)
    }

    #[must_use]
    pub fn and(&self, other: Bit) -> Bit {
        match (self, other) {
            (Bit::One, Bit::One) => Bit::One,
            _ => Bit::Zero,
        }
    }

    #[must_use]
    pub fn or(&self, other: Bit) -> Bit {
        match (self, other) {
            (Bit::Zero, Bit::Zero) => Bit::Zero,
            _ => Bit::One,
        }
    }

    #[must_use]
    pub fn xor(&self, other: Bit) -> Bit {
        match (self, other) {
            (Bit::One, Bit::One) => Bit::Zero,
            (Bit::Zero, Bit::Zero) => Bit::Zero,
            _ => Bit::One,
        }
    }

    #[must_use]
    pub fn not(&self) -> Bit {
        match self {
            Bit::One => Bit::Zero,
            Bit::Zero => Bit::One,
        }
    }
}

impl BitAnd for Bit {
    type Output = Bit;

    fn bitand(self, rhs: Self) -> Self::Output {
        self.and(rhs)
    }
}

impl BitOr for Bit {
    type Output = Bit;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.or(rhs)
    }
}

impl BitXor for Bit {
    type Output = Bit;

    fn bitxor(self, rhs: Self) -> Self::Output {
        self.xor(rhs)
    }
}
