//! The window: one histogram on screen, or two side by side.

mod bar;
mod bits;
mod colours;
mod feed;
mod frequency;
mod snapshot;

use crate::format::group_digits;

pub use feed::Feed;
pub use snapshot::Snapshot;

/// How many bars the ranked view starts on.
///
/// The number of distinct syndromes is not limited, only the number of bars
/// drawn at once.
const DEFAULT_LIMIT: usize = 64;

/// Which view is on screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum View {
    /// Syndromes as bars, most frequent first.
    Frequency,
    /// How often each output bit flipped.
    Bits,
}

impl View {
    const ALL: [Self; 2] = [Self::Frequency, Self::Bits];

    fn label(self) -> &'static str {
        match self {
            Self::Frequency => "syndromes",
            Self::Bits => "bits",
        }
    }
}

/// The window's state.
pub struct App {
    feed: Feed,
    primary: Snapshot,
    secondary: Option<Snapshot>,
    /// The distance from `primary` to `secondary`, recomputed whenever
    /// `primary` is replaced.
    distance: Option<f64>,
    view: View,
    frequency_scale: frequency::Scale,
    bits_scale: bits::Scale,
    /// The user's window into the ranks. Clamped where it is used, since the
    /// snapshot it addresses can be replaced under it.
    skip: usize,
    limit: usize,
}

impl App {
    /// A window over `primary`, comparing it against `secondary` if given.
    pub fn new(feed: Feed, primary: Snapshot, secondary: Option<Snapshot>) -> Self {
        // Comparing, counts are the wrong scale to start on: the two runs
        // stopped at different evaluation counts.
        let comparing = secondary.is_some();
        let (frequency_scale, bits_scale) = if comparing {
            (frequency::Scale::OfEvaluations, bits::Scale::OfEvaluations)
        } else {
            (frequency::Scale::Counts, bits::Scale::Counts)
        };
        let mut app = Self {
            feed,
            primary,
            secondary,
            distance: None,
            view: View::Frequency,
            frequency_scale,
            bits_scale,
            skip: 0,
            limit: DEFAULT_LIMIT,
        };
        app.measure();
        app
    }

    /// Recomputes the distance between the two loaded histograms.
    fn measure(&mut self) {
        self.distance = self
            .secondary
            .as_ref()
            .map(|secondary| self.primary.total_variation_distance(secondary));
    }

    fn statistics_panel(&self, ui: &mut egui::Ui) {
        for snapshot in std::iter::once(&self.primary).chain(self.secondary.as_ref()) {
            let statistics = snapshot.statistics();
            ui.horizontal(|ui| {
                ui.strong(snapshot.name());
                ui.label(format!("seed {}", snapshot.seed()));
                ui.label(format!(
                    "{} evaluations",
                    group_digits(statistics.evaluations)
                ));
                ui.label(format!(
                    "{} syndromes",
                    group_digits(statistics.distinct_syndromes)
                ));
                ui.label(format!("missing mass {:.1e}", statistics.missing_mass));
                ui.label(format!("self-split {:.1e}", statistics.self_split_distance));
            });
        }
        if let Some(distance) = self.distance {
            ui.horizontal(|ui| {
                ui.strong("distance");
                ui.label(format!("{distance:.4}"));
            });
        }
    }

    fn controls_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("view")
                .selected_text(self.view.label())
                .show_ui(ui, |ui| {
                    for view in View::ALL {
                        ui.selectable_value(&mut self.view, view, view.label());
                    }
                });
            match self.view {
                View::Frequency => {
                    scale_selector(ui, &mut self.frequency_scale, frequency::Scale::ALL);
                    self.window_sliders(ui);
                }
                View::Bits => scale_selector(ui, &mut self.bits_scale, bits::Scale::ALL),
            }
        });
    }

    /// The sliders over the ranks, bounded by the snapshot in hand.
    fn window_sliders(&mut self, ui: &mut egui::Ui) {
        let entries = self.primary.entries().len();
        if entries == 0 {
            return;
        }
        ui.add(
            egui::Slider::new(&mut self.skip, 0..=entries - 1)
                .text("skip")
                .logarithmic(true),
        );
        ui.add(
            egui::Slider::new(&mut self.limit, 1..=entries)
                .text("bars")
                .logarithmic(true),
        );
    }

    /// The ranked view's bars, with the hovered syndrome marked wherever else
    /// it appears.
    fn frequency_layers(&self, plot_ui: &egui_plot::PlotUi<'_>) -> bar::Layers {
        let secondary = self.secondary.as_ref();
        let mut layers = frequency::bars(
            &self.primary,
            secondary,
            self.frequency_scale,
            self.skip,
            self.limit,
        );
        // Read before the bars go in, so the mark shows on the same frame as
        // the hover. `pointer_coordinate` follows the pointer anywhere on
        // screen, so the plot has to confirm it is the thing under it.
        let pointer = plot_ui
            .response()
            .hovered()
            .then(|| plot_ui.pointer_coordinate())
            .flatten();
        if let Some(syndrome) = frequency::marked(&layers, secondary.is_some(), pointer) {
            layers.recolour(syndrome, colours::MATCH);
        }
        layers
    }

    fn plot(&self, ui: &mut egui::Ui) {
        let secondary = self.secondary.as_ref();
        // Each view keeps its own plot, so switching does not carry the
        // other one's zoom across.
        let (id, offset) = match self.view {
            View::Frequency => ("frequency", self.skip),
            View::Bits => ("bits", 0),
        };
        egui_plot::Plot::new(id)
            .x_axis_formatter(move |mark, _| slot_label(mark.value, offset))
            .show(ui, |plot_ui| {
                let layers = match self.view {
                    View::Frequency => self.frequency_layers(plot_ui),
                    View::Bits => bits::bars(&self.primary, secondary, self.bits_scale),
                };
                bar::draw(layers, plot_ui);
            });
    }
}

impl eframe::App for App {
    fn logic(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        _ = frame;
        // Nothing is computed here. A snapshot arrives already sorted and
        // summarised, so a frame only ever reads one.
        if let Some(snapshot) = self.feed.poll() {
            self.primary = snapshot;
            self.measure();
        }
        // A file is drawn once and then only on input.
        if self.feed.is_live() {
            ctx.request_repaint();
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        _ = frame;
        egui::Panel::top("statistics").show(ui, |ui| self.statistics_panel(ui));
        egui::Panel::top("controls").show(ui, |ui| self.controls_panel(ui));
        egui::CentralPanel::default().show(ui, |ui| self.plot(ui));
    }
}

/// A combo box over one view's scales.
fn scale_selector<S>(ui: &mut egui::Ui, current: &mut S, all: impl IntoIterator<Item = S>)
where
    S: Copy + PartialEq + Scale,
{
    egui::ComboBox::from_id_salt("scale")
        .selected_text(current.label())
        .show_ui(ui, |ui| {
            for scale in all {
                ui.selectable_value(current, scale, scale.label());
            }
        });
}

/// What the two views' scale enums have in common.
trait Scale {
    fn label(self) -> &'static str;
}

impl Scale for frequency::Scale {
    fn label(self) -> &'static str {
        self.label()
    }
}

impl Scale for bits::Scale {
    fn label(self) -> &'static str {
        self.label()
    }
}

/// An axis mark as a slot number, or nothing between the slots.
fn slot_label(value: f64, offset: usize) -> String {
    if value < 0.0 || value.fract() != 0.0 {
        return String::new();
    }
    // The mark is an integer at this point, and the axis only spans the slots
    // that were drawn.
    let slot = value as usize;
    (slot + offset).to_string()
}

/// `numerator / denominator`, or zero when the denominator is zero.
///
/// Counts stay far below 2^53, where the conversion to `f64` is exact.
fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        return 0.0;
    }
    numerator as f64 / denominator as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_zero_denominator_is_zero_rather_than_a_nan() {
        assert_eq!(ratio(0, 0), 0.0);
        assert_eq!(ratio(5, 0), 0.0);
        assert_eq!(ratio(3, 4), 0.75);
    }

    #[test]
    fn axis_marks_only_label_the_slots() {
        assert_eq!(slot_label(0.0, 0), "0");
        assert_eq!(slot_label(3.0, 10), "13");
        assert_eq!(slot_label(2.5, 0), "");
        assert_eq!(slot_label(-1.0, 0), "");
    }
}
