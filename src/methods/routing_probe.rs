//! Probe-based multimodality evidence for portfolio regime routing.
//!
//! The static mean-width rule cannot separate a Styblinski-class multi-basin
//! box from a least-squares valley of the same width: both present a
//! moderate design box, but one rewards global heavy-tailed visiting while
//! the other rewards deep local descent. This module spends a small,
//! ledger-charged probe budget on short projected-gradient descents from
//! spread starts and reports how many distinct basin depths the descents
//! reach. Routing then commits to the global regime only when the probe
//! shows at least two competitive basins.

use eindir_core::{Bounds, Gradient, Objective};
use ndarray::Array1;

use crate::methods::local_polish::projected_gradient_polish;

/// Evidence gathered by [`multimodality_probe`].
#[derive(Clone, Debug)]
pub struct RoutingProbe {
    /// Final objective values of each short descent, sorted ascending.
    pub descent_values: Vec<f64>,
    /// Number of distinct basin depth classes among the descents.
    pub distinct_basins: usize,
    /// Best raw (un-descended) start value observed.
    pub best_start_value: f64,
    /// Worst raw (un-descended) start value observed.
    pub worst_start_value: f64,
    /// Best descended value.
    pub best_descent_value: f64,
}

impl RoutingProbe {
    /// Whether the probe found evidence of two or more competitive basins.
    pub fn multimodal(&self) -> bool {
        self.distinct_basins >= 2
    }
}

/// Relative tolerance for grouping two descent endpoints into one basin
/// depth class. The scale is the probe's own observed descent range from
/// the worst start value, so the classification is invariant to objective
/// offset and scaling.
pub const BASIN_DEPTH_REL_TOL: f64 = 1e-2;

/// Group sorted descent values into depth classes.
///
/// Two values belong to the same class when their gap is at most
/// `BASIN_DEPTH_REL_TOL` of the probe's depth scale
/// `max(worst_start - best_descent, |best_descent|, 1)`.
pub fn distinct_depth_classes(sorted_values: &[f64], depth_scale: f64) -> usize {
    let scale = depth_scale.abs().max(1.0);
    let tol = BASIN_DEPTH_REL_TOL * scale;
    let mut classes = 0usize;
    let mut last = f64::NEG_INFINITY;
    for &v in sorted_values {
        if !v.is_finite() {
            continue;
        }
        if classes == 0 || (v - last) > tol {
            classes += 1;
            last = v;
        }
    }
    classes
}

/// Run `n_starts` short projected-gradient descents from spread starts and
/// classify the endpoint depths.
///
/// Every evaluation and gradient call is charged to `obj` / `grad` by the
/// caller's budgeted wrappers; `max_fevals_per_start` bounds each descent.
/// Returns `None` when fewer than two descents produced finite values.
#[allow(clippy::too_many_arguments)]
pub fn multimodality_probe<O, G>(
    obj: &O,
    grad: &G,
    bounds: &Bounds<f64>,
    starts: &[Array1<f64>],
    max_fevals_per_start: usize,
) -> Option<RoutingProbe>
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    if starts.len() < 2 || max_fevals_per_start == 0 {
        return None;
    }
    let mut start_values = Vec::with_capacity(starts.len());
    let mut descent_values = Vec::with_capacity(starts.len());
    for start in starts {
        let clipped = bounds.clip(start.view());
        let v0 = obj.eval(clipped.view());
        if v0.is_finite() {
            start_values.push(v0);
        }
        let polish = projected_gradient_polish(
            obj,
            grad,
            clipped,
            max_fevals_per_start,
            0.1,
            0.0,
        );
        if polish.best_val.is_finite() {
            descent_values.push(polish.best_val);
        }
    }
    if descent_values.len() < 2 {
        return None;
    }
    descent_values.sort_by(|a, b| a.partial_cmp(b).expect("finite values"));
    let best_descent_value = descent_values[0];
    let best_start_value = start_values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let worst_start = start_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let depth_scale = (worst_start - best_descent_value)
        .abs()
        .max(best_descent_value.abs());
    let distinct_basins = distinct_depth_classes(&descent_values, depth_scale);
    Some(RoutingProbe {
        descent_values,
        distinct_basins,
        best_start_value,
        worst_start_value: worst_start,
        best_descent_value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Quadratic {
        bounds: Bounds<f64>,
    }

    impl Objective<f64> for Quadratic {
        fn dim(&self) -> usize {
            self.bounds.dims
        }
        fn eval(&self, x: ndarray::ArrayView1<'_, f64>) -> f64 {
            x.iter().map(|v| (v - 0.3) * (v - 0.3)).sum()
        }
        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }
    }

    impl Gradient<f64> for Quadratic {
        fn dim(&self) -> usize {
            self.bounds.dims
        }
        fn grad(&self, x: ndarray::ArrayView1<'_, f64>) -> Array1<f64> {
            x.iter().map(|v| 2.0 * (v - 0.3)).collect()
        }
    }

    /// Two separated wells of clearly different depth along coordinate 0.
    struct TwoWell {
        bounds: Bounds<f64>,
    }

    impl TwoWell {
        fn f1(v: f64) -> f64 {
            // wells at -2 (depth 0) and +2 (depth +1.5)
            let a = (v + 2.0) * (v + 2.0);
            let b = (v - 2.0) * (v - 2.0) + 1.5;
            a.min(b)
        }
    }

    impl Objective<f64> for TwoWell {
        fn dim(&self) -> usize {
            self.bounds.dims
        }
        fn eval(&self, x: ndarray::ArrayView1<'_, f64>) -> f64 {
            Self::f1(x[0])
        }
        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }
    }

    impl Gradient<f64> for TwoWell {
        fn dim(&self) -> usize {
            self.bounds.dims
        }
        fn grad(&self, x: ndarray::ArrayView1<'_, f64>) -> Array1<f64> {
            let v = x[0];
            let a = (v + 2.0) * (v + 2.0);
            let b = (v - 2.0) * (v - 2.0) + 1.5;
            let g = if a <= b { 2.0 * (v + 2.0) } else { 2.0 * (v - 2.0) };
            let mut out = Array1::zeros(x.len());
            out[0] = g;
            out
        }
    }

    fn box2(width: f64) -> Bounds<f64> {
        let half = width / 2.0;
        Bounds::new(
            Array1::from_elem(2, -half),
            Array1::from_elem(2, half),
            0.0,
        )
    }

    fn starts2(bounds: &Bounds<f64>) -> Vec<Array1<f64>> {
        let center = (&bounds.low + &bounds.high) * 0.5;
        let quarter = &bounds.low + &((&bounds.high - &bounds.low) * 0.25);
        let three_quarter = &bounds.low + &((&bounds.high - &bounds.low) * 0.75);
        vec![center, quarter, three_quarter]
    }

    #[test]
    fn quadratic_probe_reports_single_basin() {
        let obj = Quadratic { bounds: box2(10.0) };
        let starts = starts2(&obj.bounds);
        let probe = multimodality_probe(&obj, &obj, &obj.bounds, &starts, 60)
            .expect("probe runs");
        assert_eq!(probe.distinct_basins, 1, "{probe:?}");
        assert!(!probe.multimodal());
    }

    #[test]
    fn two_well_probe_reports_two_basins() {
        let obj = TwoWell { bounds: box2(10.0) };
        let starts = starts2(&obj.bounds);
        let probe = multimodality_probe(&obj, &obj, &obj.bounds, &starts, 60)
            .expect("probe runs");
        assert!(probe.distinct_basins >= 2, "{probe:?}");
        assert!(probe.multimodal());
    }

    #[test]
    fn depth_classes_group_within_tolerance() {
        assert_eq!(distinct_depth_classes(&[0.0, 0.001, 0.002], 1.0), 1);
        assert_eq!(distinct_depth_classes(&[0.0, 0.5, 0.501], 1.0), 2);
        assert_eq!(distinct_depth_classes(&[], 1.0), 0);
    }
}
