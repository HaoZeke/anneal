// Copyright (c) anneal developers
//! Bracketed Illinois false-position root finding for example probes.

/// Find a root between a negative value at `lo` and a positive value at `hi`.
///
/// Illinois halves the retained endpoint's value after consecutive updates
/// on the other side. That prevents ordinary false position from stalling
/// when one endpoint has a much larger magnitude than the other.
pub(crate) fn find(
    mut lo: f64,
    mut hi: f64,
    mut value_lo: f64,
    mut value_hi: f64,
    relative_tolerance: f64,
    max_evaluations: usize,
    mut evaluate: impl FnMut(f64) -> Option<f64>,
) -> Option<f64> {
    if !(lo < hi
        && value_lo.is_finite()
        && value_hi.is_finite()
        && value_lo < 0.0
        && value_hi > 0.0
        && relative_tolerance > 0.0)
    {
        return None;
    }
    let mut kept = 0_i32;
    for _ in 0..max_evaluations {
        if hi - lo <= relative_tolerance * hi.abs() {
            return Some(0.5 * (lo + hi));
        }
        let at = (lo * value_hi - hi * value_lo) / (value_hi - value_lo);
        if !(lo < at && at < hi) {
            return None;
        }
        let value = evaluate(at)?;
        if !value.is_finite() {
            return None;
        }
        if value < 0.0 {
            lo = at;
            value_lo = value;
            kept = kept.min(0) - 1;
            if kept <= -2 {
                value_hi *= 0.5;
            }
        } else if value > 0.0 {
            hi = at;
            value_hi = value;
            kept = kept.max(0) + 1;
            if kept >= 2 {
                value_lo *= 0.5;
            }
        } else {
            return Some(at);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::find;

    #[test]
    fn closes_a_skewed_bracket_without_stalling() {
        let root = find(
            0.5,
            2.0,
            0.5_f64.powi(10) - 1.0,
            2.0_f64.powi(10) - 1.0,
            1.0e-10,
            100,
            |x| Some(x.powi(10) - 1.0),
        )
        .expect("Illinois should close the bracket");
        assert!((root - 1.0).abs() < 1.0e-9);
    }

    #[test]
    fn rejects_a_non_bracket() {
        assert_eq!(find(0.0, 1.0, 1.0, 2.0, 1.0e-10, 100, Some), None);
    }
}
