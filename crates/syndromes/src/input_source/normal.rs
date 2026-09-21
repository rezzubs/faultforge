//! Independent normal draws per field.

use crate::Triple;
use rand::Rng;
use rand_distr::Distribution;

/// A standard deviation: a finite, non-negative number.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct StandardDeviation(f32);

/// A number that is not a [`StandardDeviation`].
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error("{0} is not a finite non-negative number")]
pub struct InvalidStandardDeviation(pub f32);

impl StandardDeviation {
    /// The scale that makes every draw zero.
    pub const ZERO: Self = Self(0.0);

    /// The scale as a number.
    pub fn get(self) -> f32 {
        self.0
    }
}

impl TryFrom<f32> for StandardDeviation {
    type Error = InvalidStandardDeviation;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        // A negative scale would only mirror the draws, and `rand_distr`
        // accepts it; it is rejected anyway so a scale means one thing.
        if value.is_finite() && value >= 0.0 {
            Ok(Self(value))
        } else {
            Err(InvalidStandardDeviation(value))
        }
    }
}

/// The standard deviations of each field of a [`Normal`] source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scales {
    /// The scale of the activation.
    pub activation: StandardDeviation,
    /// The scale of the weight.
    pub weight: StandardDeviation,
    /// The scale of the partial sum.
    pub partial_sum: StandardDeviation,
}

/// Independent draws from a zero-mean normal distribution per field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normal {
    activation: rand_distr::Normal<f32>,
    weight: rand_distr::Normal<f32>,
    partial_sum: rand_distr::Normal<f32>,
}

impl Normal {
    /// A source with the given scales.
    pub fn new(scales: Scales) -> Self {
        let distribution = |scale: StandardDeviation| {
            rand_distr::Normal::new(0.0, scale.get()).expect("a scale is finite")
        };
        Self {
            activation: distribution(scales.activation),
            weight: distribution(scales.weight),
            partial_sum: distribution(scales.partial_sum),
        }
    }
}

impl Normal {
    /// Draws one triple.
    pub fn triple(&self, rng: &mut impl Rng) -> Triple {
        Triple {
            activation: self.activation.sample(rng),
            weight: self.weight.sample(rng),
            partial_sum: self.partial_sum.sample(rng),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input_source::{InputSource, test_draws::draws};

    fn scale(value: f32) -> StandardDeviation {
        StandardDeviation::try_from(value).expect("valid scale")
    }

    #[test]
    fn normal_follows_its_scales() {
        let source = Normal::new(Scales {
            activation: scale(1.0),
            weight: StandardDeviation::ZERO,
            partial_sum: scale(4.0),
        });
        let samples = draws(&InputSource::Normal(source), 7, 4000);
        let count = samples.len() as f64;

        let statistics = |field: fn(&Triple) -> f32| {
            let mean = samples
                .iter()
                .map(|triple| f64::from(field(triple)))
                .sum::<f64>()
                / count;
            let variance = samples
                .iter()
                .map(|triple| (f64::from(field(triple)) - mean).powi(2))
                .sum::<f64>()
                / count;
            (mean, variance.sqrt())
        };

        let (activation_mean, activation_deviation) = statistics(|triple| triple.activation);
        assert!(activation_mean.abs() < 0.1, "{activation_mean}");
        assert!(
            (activation_deviation - 1.0).abs() < 0.1,
            "{activation_deviation}"
        );

        assert!(samples.iter().all(|triple| triple.weight == 0.0));

        let (partial_sum_mean, partial_sum_deviation) = statistics(|triple| triple.partial_sum);
        assert!(partial_sum_mean.abs() < 0.4, "{partial_sum_mean}");
        assert!(
            (partial_sum_deviation - 4.0).abs() < 0.4,
            "{partial_sum_deviation}"
        );
    }

    #[test]
    fn scales_are_finite_and_non_negative() {
        assert_eq!(scale(0.0), StandardDeviation::ZERO);
        assert_eq!(scale(2.5).get(), 2.5);
        assert_eq!(
            StandardDeviation::try_from(-1.0),
            Err(InvalidStandardDeviation(-1.0))
        );
        assert!(StandardDeviation::try_from(f32::NAN).is_err());
        assert!(StandardDeviation::try_from(f32::INFINITY).is_err());
    }
}
