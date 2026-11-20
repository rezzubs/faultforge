//! The status line a run prints while it goes.

use std::{
    io::{self, IsTerminal, Write},
    time::{Duration, Instant},
};
use syndromes::{generation::StoppingRule, histogram::Statistics};

/// How often the line is redrawn in place.
const REDRAW_INTERVAL: Duration = Duration::from_millis(100);
/// How often a line is printed when stderr is not a terminal.
const LOG_INTERVAL: Duration = Duration::from_secs(10);

/// A status line on stderr, redrawn in place when stderr is a terminal.
pub struct Status {
    rule: StoppingRule,
    started: Instant,
    last_drawn: Option<Instant>,
    interactive: bool,
}

impl Status {
    /// A line that starts timing now.
    pub fn new(rule: StoppingRule) -> Self {
        Self {
            rule,
            started: Instant::now(),
            last_drawn: None,
            interactive: io::stderr().is_terminal(),
        }
    }

    /// Redraws the line if enough time has passed since the last draw.
    pub fn update(&mut self, statistics: &Statistics) {
        let interval = if self.interactive {
            REDRAW_INTERVAL
        } else {
            LOG_INTERVAL
        };
        let now = Instant::now();
        if self
            .last_drawn
            .is_some_and(|last_drawn| now.duration_since(last_drawn) < interval)
        {
            return;
        }
        self.last_drawn = Some(now);
        self.draw(statistics, now);
    }

    /// Draws the final line and ends it.
    pub fn finish(&mut self, statistics: &Statistics) {
        self.draw(statistics, Instant::now());
        // Only the interactive line is still open.
        if self.interactive {
            eprintln!();
        }
    }

    fn draw(&self, statistics: &Statistics, now: Instant) {
        let line = self.line(statistics, now.duration_since(self.started));
        let mut stderr = io::stderr().lock();
        // Progress output is best effort; a closed stderr must not end the run.
        let result = if self.interactive {
            // Carriage return, then clear to the end of the line.
            write!(stderr, "\r\x1b[K{line}").and_then(|()| stderr.flush())
        } else {
            writeln!(stderr, "{line}")
        };
        drop(result);
    }

    fn line(&self, statistics: &Statistics, elapsed: Duration) -> String {
        // Guard against a zero duration on the first draw.
        let rate = statistics.evaluations as f64 / elapsed.as_secs_f64().max(1e-9);
        format!(
            "{} evaluations  {} syndromes  missing mass {:.1e} / {:.1e}  self-split {:.1e} / {:.1e}  {}/s",
            group_digits(statistics.evaluations),
            group_digits(statistics.distinct_syndromes),
            statistics.missing_mass,
            self.rule.missing_mass.get(),
            statistics.self_split_distance,
            self.rule.self_split_distance.get(),
            short_rate(rate),
        )
    }
}

/// `1234567` as `1,234,567`.
fn group_digits(value: u64) -> String {
    let digits = value.to_string();
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(digit);
    }
    grouped
}

/// A rate with a `k` or `M` suffix once it has one.
fn short_rate(rate: f64) -> String {
    if rate >= 1e6 {
        format!("{:.1}M", rate / 1e6)
    } else if rate >= 1e3 {
        format!("{:.1}k", rate / 1e3)
    } else {
        format!("{rate:.0}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digits_are_grouped_in_threes() {
        assert_eq!(group_digits(0), "0");
        assert_eq!(group_digits(999), "999");
        assert_eq!(group_digits(1000), "1,000");
        assert_eq!(group_digits(1234567), "1,234,567");
    }

    #[test]
    fn rates_are_shortened() {
        assert_eq!(short_rate(0.0), "0");
        assert_eq!(short_rate(999.4), "999");
        assert_eq!(short_rate(52_300.0), "52.3k");
        assert_eq!(short_rate(2_500_000.0), "2.5M");
    }
}
