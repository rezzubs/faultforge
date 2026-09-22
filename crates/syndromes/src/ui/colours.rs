//! The colours the bars are drawn in.

use egui::Color32;

/// Which of the two loaded histograms a bar belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Series {
    /// The histogram named on the command line.
    Primary,
    /// The histogram compared against it.
    Secondary,
}

// Blue against orange rather than red against green, so the two series stay
// apart under the common colour vision deficiencies.

/// The primary histogram.
pub const PRIMARY: Color32 = Color32::from_rgb(0x4c, 0x7d, 0xd9);
/// The primary histogram's zero syndrome.
pub const PRIMARY_ZERO: Color32 = Color32::from_rgb(0x6b, 0x7d, 0x9c);
/// A primary bar whose syndrome matches the secondary at the same rank.
pub const PRIMARY_HIGHLIGHT: Color32 = Color32::from_rgb(0x8a, 0xb6, 0xff);
/// Both at once.
pub const PRIMARY_ZERO_HIGHLIGHT: Color32 = Color32::from_rgb(0x9d, 0xad, 0xc7);

/// The compared histogram.
pub const SECONDARY: Color32 = Color32::from_rgb(0xd9, 0x84, 0x3b);
/// The compared histogram's zero syndrome.
pub const SECONDARY_ZERO: Color32 = Color32::from_rgb(0x9c, 0x81, 0x62);
/// A secondary bar whose syndrome matches the primary at the same rank.
pub const SECONDARY_HIGHLIGHT: Color32 = Color32::from_rgb(0xff, 0xb4, 0x6b);
/// Both at once.
pub const SECONDARY_ZERO_HIGHLIGHT: Color32 = Color32::from_rgb(0xc7, 0xad, 0x8e);

/// The hovered syndrome, and the same syndrome in the other histogram.
///
/// One colour for both bars, since the point is that they are the same
/// syndrome. Magenta is far from blue and from orange on the blue to yellow
/// axis, which red-green colour vision deficiencies leave intact, and it is
/// lighter than either, so the pair stands out however it is seen.
pub const MATCH: Color32 = Color32::from_rgb(0xe0, 0x6c, 0xc8);

/// The colour of one bar.
///
/// The zero syndrome gets its own shade because it is usually the tallest bar
/// by a wide margin and means the fault changed nothing, which is not a
/// corruption pattern like the others.
pub fn bar(series: Series, zero_syndrome: bool, highlighted: bool) -> Color32 {
    match (series, zero_syndrome, highlighted) {
        (Series::Primary, false, false) => PRIMARY,
        (Series::Primary, true, false) => PRIMARY_ZERO,
        (Series::Primary, false, true) => PRIMARY_HIGHLIGHT,
        (Series::Primary, true, true) => PRIMARY_ZERO_HIGHLIGHT,
        (Series::Secondary, false, false) => SECONDARY,
        (Series::Secondary, true, false) => SECONDARY_ZERO,
        (Series::Secondary, false, true) => SECONDARY_HIGHLIGHT,
        (Series::Secondary, true, true) => SECONDARY_ZERO_HIGHLIGHT,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn every_combination_has_its_own_colour() {
        let mut seen = HashSet::new();
        for series in [Series::Primary, Series::Secondary] {
            for zero_syndrome in [false, true] {
                for highlighted in [false, true] {
                    let colour = bar(series, zero_syndrome, highlighted);
                    assert!(
                        seen.insert(colour),
                        "{series:?} zero={zero_syndrome} highlighted={highlighted} repeats a colour"
                    );
                }
            }
        }
        assert_eq!(seen.len(), 8);
        assert!(seen.insert(MATCH), "the match colour repeats a bar colour");
    }
}
