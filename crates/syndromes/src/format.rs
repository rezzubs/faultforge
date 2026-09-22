//! Turning numbers into the short strings the status line and the UI show.

/// `1234567` as `1,234,567`.
pub fn group_digits(value: u64) -> String {
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
pub fn short_rate(rate: f64) -> String {
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
