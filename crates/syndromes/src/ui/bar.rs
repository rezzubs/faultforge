//! One bar, as plain data, and the call that draws a row of them.

use egui::Color32;
use egui_plot::PlotPoint;
use syndromes::Syndrome;

/// The width of a bar.
///
/// A bar fills its slot, so neighbours touch with no gap between them.
pub const WIDTH: f64 = 1.0;

/// One bar, before it becomes an `egui_plot::Bar`.
///
/// Building these is separate from drawing them so the views can be tested
/// without a window.
#[derive(Debug, Clone, PartialEq)]
pub struct Bar {
    /// Where on the x axis the bar sits.
    pub position: f64,
    /// Where the bar starts on the value axis.
    pub base: f64,
    /// How far it reaches above `base`.
    pub value: f64,
    /// What colour it is drawn in.
    pub colour: Color32,
    /// The hover text.
    pub name: String,
    /// The syndrome the bar counts, where a slot stands for one.
    ///
    /// Empty in a view whose slots are not syndromes.
    pub syndrome: Option<Syndrome>,
}

impl Bar {
    /// The middle of a slot.
    pub fn position(slot: usize) -> f64 {
        // A slot index far beyond f64's exact integers would need more bars
        // than a screen can show.
        slot as f64
    }

    /// Where the bar ends on the value axis.
    pub fn top(&self) -> f64 {
        self.base + self.value
    }

    /// Whether a plot coordinate falls on the bar.
    pub fn contains(&self, point: PlotPoint) -> bool {
        (point.x - self.position).abs() <= WIDTH / 2.0
            && (self.base..=self.top()).contains(&point.y)
    }
}

/// A chart's bars, split into the order they are drawn in.
///
/// The bars of one slot are cut into disjoint bands by [`Layers::push_slot`],
/// so which one a pointer is over is never ambiguous. Sharing an edge still
/// is, and a chart resolves that towards the bar it drew first, so the front
/// layer is a chart of its own drawn last.
#[derive(Debug, Default, PartialEq)]
pub struct Layers {
    /// Drawn first, so furthest back.
    pub back: Vec<Bar>,
    /// Drawn last, so in front, and hovered ahead of the back layer.
    pub front: Vec<Bar>,
}

impl Layers {
    /// Room for `slots` slots of `per_slot` bars each.
    pub fn with_capacity(slots: usize, per_slot: usize) -> Self {
        Self {
            back: Vec::with_capacity(slots * per_slot.saturating_sub(1)),
            front: Vec::with_capacity(slots),
        }
    }

    /// Adds the bars of one slot, tallest first.
    ///
    /// They share a position, so each is cut down to the band it alone
    /// occupies: the shortest keeps its whole height and every taller one
    /// starts where the next shorter ends. The slot still reads as one bar
    /// with the difference showing above it, but no bar covers another, so
    /// the hover highlight cannot hide the rest of the slot.
    ///
    /// The sort is stable, so equal heights leave the primary behind the
    /// secondary. A slot holding one bar puts it in front, where nothing can
    /// cover it.
    pub fn push_slot(&mut self, mut slot_bars: Vec<Bar>) {
        slot_bars.sort_by(|left, right| right.value.total_cmp(&left.value));
        // Forwards, so each bar reads the next one's height before that one
        // is cut down in its turn.
        for index in 0..slot_bars.len().saturating_sub(1) {
            let base = slot_bars[index + 1].top();
            slot_bars[index].base = base;
            slot_bars[index].value -= base;
        }
        let front = slot_bars.pop();
        self.back.append(&mut slot_bars);
        self.front.extend(front);
    }

    /// The bar a plot coordinate falls on.
    ///
    /// The bands of a slot are disjoint, so at most one bar holds the point.
    /// The front layer is searched first, so the edge two bands share resolves
    /// the way the drawn slot does.
    pub fn at(&self, point: PlotPoint) -> Option<&Bar> {
        self.bars().find(|bar| bar.contains(point))
    }

    /// The bars of the slot at `position`.
    pub fn slot(&self, position: f64) -> impl Iterator<Item = &Bar> {
        self.bars().filter(move |bar| bar.position == position)
    }

    /// Repaints every bar counting `syndrome`.
    pub fn recolour(&mut self, syndrome: Syndrome, colour: Color32) {
        for bar in self.front.iter_mut().chain(&mut self.back) {
            if bar.syndrome == Some(syndrome) {
                bar.colour = colour;
            }
        }
    }

    /// Every bar, front layer first.
    fn bars(&self) -> impl Iterator<Item = &Bar> {
        self.front.iter().chain(&self.back)
    }
}

/// Draws a chart's layers, back first.
pub fn draw(layers: Layers, plot_ui: &mut egui_plot::PlotUi<'_>) {
    for (name, bars) in [("back", layers.back), ("front", layers.front)] {
        if bars.is_empty() {
            continue;
        }
        let bars = bars
            .into_iter()
            .map(|bar| {
                egui_plot::Bar::new(bar.position, bar.value)
                    .base_offset(bar.base)
                    .fill(bar.colour)
                    .name(bar.name)
            })
            .collect();
        // The chart name only ever reaches a legend, and there is none. A
        // tooltip shows the hovered bar's own name.
        plot_ui.bar_chart(egui_plot::BarChart::new(name, bars).width(WIDTH));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(value: f64, colour: Color32) -> Bar {
        Bar {
            position: 0.0,
            base: 0.0,
            value,
            colour,
            name: String::new(),
            syndrome: None,
        }
    }

    fn slot(bars: Vec<Bar>) -> Layers {
        let mut layers = Layers::default();
        layers.push_slot(bars);
        layers
    }

    #[test]
    fn the_shortest_of_a_slot_ends_up_in_front() {
        let layers = slot(vec![
            bar(3.0, Color32::RED),
            bar(7.0, Color32::BLUE),
            bar(5.0, Color32::GREEN),
        ]);
        assert_eq!(
            layers.back.iter().map(Bar::top).collect::<Vec<_>>(),
            [7.0, 5.0]
        );
        assert_eq!(layers.front[0].top(), 3.0);
    }

    #[test]
    fn a_slot_is_cut_into_bands_that_do_not_overlap() {
        let layers = slot(vec![
            bar(3.0, Color32::RED),
            bar(7.0, Color32::BLUE),
            bar(5.0, Color32::GREEN),
        ]);
        // 5 to 7, 3 to 5, 0 to 3: each band starts where the one below ends.
        let bands = |bars: &[Bar]| {
            bars.iter()
                .map(|bar| (bar.base, bar.top()))
                .collect::<Vec<_>>()
        };
        assert_eq!(bands(&layers.back), [(5.0, 7.0), (3.0, 5.0)]);
        assert_eq!(bands(&layers.front), [(0.0, 3.0)]);
    }

    #[test]
    fn a_lone_bar_is_in_front_and_keeps_its_height() {
        let layers = slot(vec![bar(3.0, Color32::RED)]);
        assert!(layers.back.is_empty());
        assert_eq!(layers.front.len(), 1);
        assert_eq!((layers.front[0].base, layers.front[0].value), (0.0, 3.0));
    }

    #[test]
    fn a_point_finds_the_band_it_is_in() {
        let layers = slot(vec![bar(3.0, Color32::RED), bar(7.0, Color32::BLUE)]);
        let at = |x, y| layers.at(PlotPoint::new(x, y)).map(|bar| bar.colour);
        // The front band is 0 to 3 and the back band 3 to 7.
        assert_eq!(at(0.0, 1.0), Some(Color32::RED));
        assert_eq!(at(0.0, 5.0), Some(Color32::BLUE));
        // The shared edge goes to the front bar, as the drawn slot does.
        assert_eq!(at(0.0, 3.0), Some(Color32::RED));
        // A bar fills its slot, and nothing reaches past the top.
        assert_eq!(at(0.49, 1.0), Some(Color32::RED));
        assert_eq!(at(0.51, 1.0), None);
        assert_eq!(at(0.0, 7.5), None);
        assert_eq!(at(0.0, -0.5), None);
    }

    #[test]
    fn recolouring_only_touches_the_named_syndrome() {
        let mut layers = slot(vec![
            Bar {
                syndrome: Some(4),
                ..bar(3.0, Color32::RED)
            },
            Bar {
                syndrome: Some(9),
                ..bar(7.0, Color32::BLUE)
            },
        ]);
        layers.recolour(4, Color32::GREEN);
        assert_eq!(layers.front[0].colour, Color32::GREEN);
        assert_eq!(layers.back[0].colour, Color32::BLUE);
    }

    #[test]
    fn equal_heights_leave_the_first_behind_with_nothing_to_show() {
        let layers = slot(vec![bar(3.0, Color32::RED), bar(3.0, Color32::BLUE)]);
        assert_eq!(layers.back[0].colour, Color32::RED);
        assert_eq!(layers.front[0].colour, Color32::BLUE);
        // The band above the other bar is empty, so the one behind draws
        // nothing and the one in front is the whole slot.
        assert_eq!(layers.back[0].value, 0.0);
        assert_eq!(layers.front[0].top(), 3.0);
    }
}
