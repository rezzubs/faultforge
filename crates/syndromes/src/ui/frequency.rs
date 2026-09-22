//! Syndromes as bars, most frequent first.

use crate::{
    format::group_digits,
    ui::{
        bar::{Bar, Layers},
        colours::{self, Series},
        ratio,
        snapshot::Snapshot,
    },
};
use egui_plot::PlotPoint;
use syndromes::Syndrome;

/// What the bar heights mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scale {
    /// The number of evaluations that produced the syndrome.
    Counts,
    /// That over every evaluation.
    OfEvaluations,
}

impl Scale {
    /// Every scale, in the order the menu offers them.
    pub const ALL: [Self; 2] = [Self::Counts, Self::OfEvaluations];

    /// What the menu calls this scale.
    pub fn label(self) -> &'static str {
        match self {
            Self::Counts => "counts",
            Self::OfEvaluations => "% of evaluations",
        }
    }
}

/// The bars of the ranked view.
///
/// `skip` and `limit` are the user's values and are clamped here, so a
/// snapshot that shrinks under them is not an error.
pub fn bars(
    primary: &Snapshot,
    secondary: Option<&Snapshot>,
    scale: Scale,
    skip: usize,
    limit: usize,
) -> Layers {
    let longest = primary
        .entries()
        .len()
        .max(secondary.map_or(0, |snapshot| snapshot.entries().len()));
    let slots = longest.saturating_sub(skip).min(limit);

    let mut layers = Layers::with_capacity(slots, if secondary.is_some() { 2 } else { 1 });
    for slot in 0..slots {
        let rank = skip + slot;
        let entry = |snapshot: &Snapshot| snapshot.entries().get(rank).copied();
        let mine = entry(primary);
        let theirs = secondary.and_then(entry);
        // Rank by rank: the two agree here when the same syndrome sits at the
        // same place in both.
        let highlighted =
            matches!((mine, theirs), (Some((left, _)), Some((right, _))) if left == right);

        let make = |series, snapshot: &Snapshot, (syndrome, count): (Syndrome, u64)| Bar {
            position: Bar::position(slot),
            base: 0.0,
            value: match scale {
                Scale::Counts => count_as_f64(count),
                Scale::OfEvaluations => ratio(count, snapshot.evaluations()),
            },
            colour: colours::bar(series, syndrome == 0, highlighted),
            name: hover(snapshot, secondary.is_some(), rank, syndrome, count),
            syndrome: Some(syndrome),
        };
        let mut slot_bars = Vec::with_capacity(2);
        if let Some(entry) = mine {
            slot_bars.push(make(Series::Primary, primary, entry));
        }
        if let (Some(snapshot), Some(entry)) = (secondary, theirs) {
            slot_bars.push(make(Series::Secondary, snapshot, entry));
        }
        layers.push_slot(slot_bars);
    }
    layers
}

/// The syndrome to mark in both histograms, given where the pointer is.
///
/// `None` unless two histograms are compared and the pointer is on a bar whose
/// syndrome sits at a different rank in the other one. A pair that shares a
/// rank already carries the highlight colour, so marking it would say nothing
/// new.
pub fn marked(layers: &Layers, comparing: bool, pointer: Option<PlotPoint>) -> Option<Syndrome> {
    if !comparing {
        return None;
    }
    let hovered = layers.at(pointer?)?;
    let syndrome = hovered.syndrome?;
    // One slot holds at most one bar per histogram, so a second bar carrying
    // the syndrome means both have it at this rank.
    let in_slot = layers
        .slot(hovered.position)
        .filter(|bar| bar.syndrome == Some(syndrome))
        .count();
    (in_slot == 1).then_some(syndrome)
}

fn hover(snapshot: &Snapshot, comparing: bool, rank: usize, syndrome: u32, count: u64) -> String {
    let share = ratio(count, snapshot.evaluations()) * 100.0;
    let prefix = if comparing {
        format!("{}  ", snapshot.name())
    } else {
        String::new()
    };
    format!(
        "{prefix}rank {rank}  {syndrome:#010x}  {}  {share:.3}%",
        group_digits(count)
    )
}

/// Counts stay far below 2^53, where the conversion is exact.
fn count_as_f64(count: u64) -> f64 {
    count as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::snapshot::tests::snapshot;

    /// Ten evaluations: syndrome 4 six times, syndrome 0 three times,
    /// syndrome 1 once.
    fn primary() -> Snapshot {
        snapshot("primary", &[(4, 6), (0, 3), (1, 1)])
    }

    /// The same ranking, but rank 1 holds a different syndrome and the counts
    /// at ranks 0 and 1 are the other way about.
    fn secondary() -> Snapshot {
        snapshot("secondary", &[(4, 5), (2, 4), (1, 1)])
    }

    /// Where each bar reaches, whatever band it was cut down to.
    fn tops(bars: &[Bar]) -> Vec<f64> {
        bars.iter().map(Bar::top).collect()
    }

    fn positions(bars: &[Bar]) -> Vec<f64> {
        bars.iter().map(|bar| bar.position).collect()
    }

    fn compared() -> Layers {
        bars(&primary(), Some(&secondary()), Scale::Counts, 0, 10)
    }

    #[test]
    fn one_histogram_puts_every_bar_in_front() {
        let layers = bars(&primary(), None, Scale::Counts, 0, 10);
        assert!(layers.back.is_empty());
        assert_eq!(tops(&layers.front), [6.0, 3.0, 1.0]);
        assert_eq!(positions(&layers.front), [0.0, 1.0, 2.0]);

        let fractions = bars(&primary(), None, Scale::OfEvaluations, 0, 10);
        assert_eq!(tops(&fractions.front), [0.6, 0.3, 0.1]);
    }

    #[test]
    fn skip_and_limit_window_the_ranks() {
        let primary = primary();
        let window = |skip, limit| tops(&bars(&primary, None, Scale::Counts, skip, limit).front);
        assert_eq!(window(1, 1), [3.0]);
        assert_eq!(window(1, 99), [3.0, 1.0]);
        // Past the end, rather than a panic or a wrapped window.
        assert!(window(99, 10).is_empty());
        assert!(window(0, 0).is_empty());
    }

    #[test]
    fn a_pair_overlays_in_one_slot_shortest_in_front() {
        let layers = compared();
        // Both bars of a slot sit in the middle of it.
        assert_eq!(positions(&layers.back), [0.0, 1.0, 2.0]);
        assert_eq!(positions(&layers.front), [0.0, 1.0, 2.0]);
        // Rank 0 is 6 against 5 and rank 1 is 3 against 4, so which histogram
        // ends up in front follows the heights, not the series.
        assert_eq!(tops(&layers.back), [6.0, 4.0, 1.0]);
        assert_eq!(tops(&layers.front), [5.0, 3.0, 1.0]);
        // The back bar only covers the band above the front one.
        assert_eq!(
            layers.back.iter().map(|bar| bar.base).collect::<Vec<_>>(),
            [5.0, 3.0, 1.0]
        );
        assert!(layers.front.iter().all(|bar| bar.base == 0.0));
    }

    #[test]
    fn a_rank_highlights_when_both_hold_the_same_syndrome() {
        let layers = compared();
        // Rank 0 is syndrome 4 in both and rank 2 is syndrome 1 in both, so
        // those highlight. Rank 1 is 0 against 2, so neither does.
        assert_eq!(layers.back[0].colour, colours::PRIMARY_HIGHLIGHT);
        assert_eq!(layers.front[0].colour, colours::SECONDARY_HIGHLIGHT);
        assert_eq!(layers.back[1].colour, colours::SECONDARY);
        assert_eq!(layers.front[1].colour, colours::PRIMARY_ZERO);
        // Equal heights leave the primary behind.
        assert_eq!(layers.back[2].colour, colours::PRIMARY_HIGHLIGHT);
        assert_eq!(layers.front[2].colour, colours::SECONDARY_HIGHLIGHT);
    }

    #[test]
    fn hovering_marks_a_syndrome_that_sits_at_another_rank() {
        let layers = compared();
        let mark = |x, y| marked(&layers, true, Some(PlotPoint::new(x, y)));
        // Rank 1 is syndrome 0 at height 3 in the primary and syndrome 2 at
        // height 4 in the secondary, so the bands are 0 to 3 and 3 to 4.
        assert_eq!(mark(1.0, 1.0), Some(0));
        assert_eq!(mark(1.0, 3.5), Some(2));
        // Rank 0 holds syndrome 4 in both, which the rank highlight covers.
        assert_eq!(mark(0.0, 1.0), None);
        // Nothing under the pointer, and nothing to compare against.
        assert_eq!(mark(1.0, 99.0), None);
        assert_eq!(marked(&layers, true, None), None);
        assert_eq!(marked(&layers, false, Some(PlotPoint::new(1.0, 1.0))), None);
    }

    #[test]
    fn the_longer_snapshot_decides_how_many_slots_there_are() {
        let primary = snapshot("primary", &[(4, 6)]);
        let secondary = secondary();
        let layers = bars(&primary, Some(&secondary), Scale::Counts, 0, 10);
        // One slot holds a pair; the two the secondary holds alone have
        // nothing to sit behind.
        assert_eq!(positions(&layers.back), [0.0]);
        assert_eq!(positions(&layers.front), [0.0, 1.0, 2.0]);
    }

    #[test]
    fn hover_text_names_the_rank_and_the_mask() {
        let layers = bars(&primary(), None, Scale::Counts, 0, 10);
        assert_eq!(layers.front[0].name, "rank 0  0x00000004  6  60.000%");

        // Comparing, the name says which histogram the bar belongs to.
        let layers = compared();
        assert!(
            layers.back[0].name.starts_with("primary  rank 0"),
            "{}",
            layers.back[0].name
        );
        assert!(
            layers.front[0].name.starts_with("secondary  rank 0"),
            "{}",
            layers.front[0].name
        );
    }
}
