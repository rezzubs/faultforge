//! Two-valued bits and 32-bit words at the netlist boundary.

use logic_simulation::Signal;

/// The number of bits in a word.
pub const WORD_BITS: usize = 32;

/// A bit that is known to be `0` or `1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Bit {
    /// A logical `0`.
    Zero,
    /// A logical `1`.
    One,
}

/// A netlist bit that has no defined value.
///
/// Either the bit is unknown (`X`) or the bus index has nothing connected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
#[error("bit {index} has no defined value")]
pub struct UnknownBitError {
    /// The little-endian position of the bit in the word.
    pub index: usize,
}

impl TryFrom<Signal> for Bit {
    type Error = UnknownBitError;

    /// Converts a signal, failing on `Unknown`.
    ///
    /// The error's index is always `0`; callers that know the position fill
    /// it in.
    fn try_from(signal: Signal) -> Result<Self, Self::Error> {
        match signal {
            Signal::Low => Ok(Self::Zero),
            Signal::High => Ok(Self::One),
            Signal::Unknown => Err(UnknownBitError { index: 0 }),
        }
    }
}

impl From<Bit> for Signal {
    fn from(bit: Bit) -> Self {
        match bit {
            Bit::Zero => Self::Low,
            Bit::One => Self::High,
        }
    }
}

/// Splits a word into signals, least significant bit first.
pub fn signals_from_word(word: u32) -> [Signal; WORD_BITS] {
    let mut signals = [Signal::Low; WORD_BITS];
    for (index, signal) in signals.iter_mut().enumerate() {
        if (word >> index) & 1 == 1 {
            *signal = Signal::High;
        }
    }
    signals
}

/// Assembles a word from signals given a little-endian sequence of bits.
///
/// Fails on the first bit that is disconnected (`None`) or unknown.
///
/// # Panics
///
/// Panics if `signals` does not hold exactly [`WORD_BITS`] entries.
pub fn word_from_signals(signals: &[Option<Signal>]) -> Result<u32, UnknownBitError> {
    assert_eq!(
        signals.len(),
        WORD_BITS,
        "a word is assembled from exactly {WORD_BITS} signals"
    );

    let mut word = 0;
    for (index, signal) in signals.iter().enumerate() {
        let bit = signal
            .and_then(|signal| Bit::try_from(signal).ok())
            .ok_or(UnknownBitError { index })?;
        if bit == Bit::One {
            word |= 1 << index;
        }
    }
    Ok(word)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_round_trip() {
        for word in [0, 1, 0x8000_0000, 0xDEAD_BEEF, u32::MAX] {
            let signals = signals_from_word(word).map(Some);
            assert_eq!(word_from_signals(&signals), Ok(word));
        }
    }

    #[test]
    fn unknown_bit_reports_index() {
        let mut signals = signals_from_word(u32::MAX).map(Some);
        signals[7] = Some(Signal::Unknown);
        assert_eq!(
            word_from_signals(&signals),
            Err(UnknownBitError { index: 7 })
        );
    }

    #[test]
    fn disconnected_bit_reports_index() {
        let mut signals = signals_from_word(0).map(Some);
        signals[31] = None;
        assert_eq!(
            word_from_signals(&signals),
            Err(UnknownBitError { index: 31 })
        );
    }
}
