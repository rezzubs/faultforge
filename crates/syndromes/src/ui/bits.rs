//! How often each output bit flipped.

use crate::{
    format::group_digits,
    ui::{
        bar::{Bar, Layers},
        colours::{self, Series},
        ratio,
        snapshot::{BIT_COUNT, Snapshot},
    },
};

/// What the bar heights mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scale {
    /// The number of evaluations in which the bit flipped.
    Counts,
    /// That over every evaluation.
    OfEvaluations,
    /// That over the evaluations whose syndrome was nonzero.
    OfCorrupted,
}

impl Scale {
    /// Every scale, in the order the menu offers them.
    pub const ALL: [Self; 3] = [Self::Counts, Self::OfEvaluations, Self::OfCorrupted];

    /// What the menu calls this scale.
    pub fn label(self) -> &'static str {
        match self {
            Self::Counts => "counts",
            Self::OfEvaluations => "% of evaluations",
            Self::OfCorrupted => "% of corrupted",
        }
    }
}

/// The bars of the per-bit view.
///
/// One slot per output bit, and when a second histogram is loaded the two
/// share the slot with the shorter in front. No highlight rule: bit 7 is bit
/// 7 in both.
pub fn bars(primary: &Snapshot, secondary: Option<&Snapshot>, scale: Scale) -> Layers {
    let mut layers = Layers::with_capacity(BIT_COUNT, if secondary.is_some() { 2 } else { 1 });
    for index in 0..BIT_COUNT {
        let make = |series, snapshot: &Snapshot| {
            let flips = snapshot.bit_flips()[index];
            Bar {
                position: Bar::position(index),
                base: 0.0,
                value: value(snapshot, flips, scale),
                // A bit is not a syndrome, so the zero shade never applies.
                colour: colours::bar(series, false, false),
                name: hover(snapshot, secondary.is_some(), index, flips, scale),
                // A slot is a bit here, not a syndrome.
                syndrome: None,
            }
        };
        let mut slot_bars = vec![make(Series::Primary, primary)];
        if let Some(snapshot) = secondary {
            slot_bars.push(make(Series::Secondary, snapshot));
        }
        layers.push_slot(slot_bars);
    }
    layers
}

/// The height of one bar.
///
/// Neither fraction divides by the total number of flips across bits. Bits are
/// not mutually exclusive, so that total is not a whole anything is a share
/// of. `OfEvaluations` is the probability that an evaluation flips this bit,
/// which is what an injection campaign sees. `OfCorrupted` conditions on the
/// fault having changed the output at all, so it describes the shape of a
/// corruption without moving when the zero syndrome's share does.
fn value(snapshot: &Snapshot, flips: u64, scale: Scale) -> f64 {
    match scale {
        // Counts stay far below 2^53, where the conversion is exact.
        Scale::Counts => flips as f64,
        Scale::OfEvaluations => ratio(flips, snapshot.evaluations()),
        Scale::OfCorrupted => ratio(flips, snapshot.corrupted()),
    }
}

fn hover(snapshot: &Snapshot, comparing: bool, index: usize, flips: u64, scale: Scale) -> String {
    let share = match scale {
        Scale::OfCorrupted => ratio(flips, snapshot.corrupted()),
        _ => ratio(flips, snapshot.evaluations()),
    } * 100.0;
    let of = match scale {
        Scale::OfCorrupted => "of corrupted",
        _ => "of evaluations",
    };
    let prefix = if comparing {
        format!("{}  ", snapshot.name())
    } else {
        String::new()
    };
    format!(
        "{prefix}bit {index} ({})  {}  {share:.3}% {of}",
        field(index),
        group_digits(flips),
    )
}

/// The `f32` field a bit belongs to.
fn field(index: usize) -> &'static str {
    match index {
        31 => "sign",
        23..=30 => "exponent",
        _ => "mantissa",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::snapshot::tests::snapshot;

    /// Ten evaluations: bit 0 flips in six of them, bit 2 in one, and three
    /// evaluations were not corrupted at all.
    fn primary() -> Snapshot {
        snapshot("primary", &[(0b001, 6), (0, 3), (0b100, 1)])
    }

    /// Where each bar reaches, whatever band it was cut down to.
    fn tops(bars: &[Bar]) -> Vec<f64> {
        bars.iter().map(Bar::top).collect()
    }

    /// The bars of a single histogram, which all sit in the front layer.
    fn alone(primary: &Snapshot, scale: Scale) -> Vec<Bar> {
        let layers = bars(primary, None, scale);
        assert!(layers.back.is_empty());
        layers.front
    }

    #[test]
    fn there_is_one_slot_per_bit() {
        let single = alone(&primary(), Scale::Counts);
        assert_eq!(single.len(), BIT_COUNT);
        assert_eq!(single[0].position, 0.0);
        assert_eq!(single[1].position, 1.0);
    }

    #[test]
    fn a_pair_overlays_in_one_slot_shortest_in_front() {
        // Bit 0 flips six times in the primary and never in the secondary,
        // bit 1 the other way about.
        let secondary = snapshot("secondary", &[(0b010, 4)]);
        let layers = bars(&primary(), Some(&secondary), Scale::Counts);
        assert_eq!(layers.back.len(), BIT_COUNT);
        assert_eq!(layers.front.len(), BIT_COUNT);
        assert_eq!(layers.back[0].position, 0.0);
        assert_eq!(layers.front[0].position, 0.0);
        assert_eq!(layers.back[1].position, 1.0);
        // Bit 0's band above the secondary's nothing is the whole bar; bit
        // 1's is the secondary's four above the primary's nothing.
        assert_eq!((layers.back[0].base, layers.back[0].top()), (0.0, 6.0));
        assert_eq!((layers.front[0].base, layers.front[0].top()), (0.0, 0.0));
        assert_eq!((layers.back[1].base, layers.back[1].top()), (0.0, 4.0));

        // Which histogram is in front follows the heights, bit by bit.
        assert_eq!(layers.back[0].colour, colours::PRIMARY);
        assert_eq!(layers.front[0].colour, colours::SECONDARY);
        assert_eq!(layers.back[1].colour, colours::SECONDARY);
        assert_eq!(layers.front[1].colour, colours::PRIMARY);
    }

    #[test]
    fn the_three_scales_share_a_numerator() {
        let primary = primary();
        let counts = tops(&alone(&primary, Scale::Counts));
        assert_eq!((counts[0], counts[1], counts[2]), (6.0, 0.0, 1.0));

        // Six of ten evaluations, but six of the seven that were corrupted.
        let of_evaluations = tops(&alone(&primary, Scale::OfEvaluations));
        assert_eq!(of_evaluations[0], 0.6);
        let of_corrupted = tops(&alone(&primary, Scale::OfCorrupted));
        assert!((of_corrupted[0] - 6.0 / 7.0).abs() < 1e-12);
        assert!(of_corrupted[0] > of_evaluations[0]);
    }

    #[test]
    fn the_two_scales_agree_when_every_evaluation_was_corrupted() {
        let primary = snapshot("primary", &[(0b001, 6), (0b100, 4)]);
        let of_evaluations = tops(&alone(&primary, Scale::OfEvaluations));
        let of_corrupted = tops(&alone(&primary, Scale::OfCorrupted));
        assert_eq!(of_evaluations, of_corrupted);
    }

    #[test]
    fn nothing_corrupted_is_zero_rather_than_a_division_by_zero() {
        let primary = snapshot("primary", &[(0, 10)]);
        let tops = tops(&alone(&primary, Scale::OfCorrupted));
        assert!(tops.iter().all(|&top| top == 0.0), "{tops:?}");
    }

    #[test]
    fn hover_text_names_the_float_field() {
        let primary = primary();
        let bars = alone(&primary, Scale::OfEvaluations);
        assert_eq!(bars[0].name, "bit 0 (mantissa)  6  60.000% of evaluations");
        assert!(bars[23].name.starts_with("bit 23 (exponent)"));
        assert!(bars[30].name.starts_with("bit 30 (exponent)"));
        assert!(bars[31].name.starts_with("bit 31 (sign)"));

        let corrupted = alone(&primary, Scale::OfCorrupted);
        assert_eq!(
            corrupted[0].name,
            "bit 0 (mantissa)  6  85.714% of corrupted"
        );
    }
}
