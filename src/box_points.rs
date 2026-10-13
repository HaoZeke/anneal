//! Start points on boxes whose width overflows or whose walls are infinite.
//!
//! `Bounds::mkpoint` and eindir's low-discrepancy designs scale by
//! `high - low`, which is infinite on a box such as `[-1e308, 1e308]` and on
//! one with an infinite wall: the draw panics and the design's points are
//! infinite or NaN. On a box whose every width is finite these are the draws
//! and designs eindir makes, so seeded runs there are unchanged. Elsewhere an
//! axis with finite walls is scaled at half width and doubled, which is exact
//! away from subnormals, and an axis with an infinite wall, which has no
//! uniform measure, takes its finite wall, or zero when both walls are
//! infinite, as the multistart starts of `run_rs_qmc_variant` do.

use eindir_core::Bounds;
use ndarray::{Array1, Array2};
use rand::Rng;

/// Panics, naming the coordinate, unless every wall of `bounds` is a number.
/// `Bounds::new` lets a NaN wall through, and no point is inside such a box.
pub(crate) fn assert_no_nan_wall(bounds: &Bounds<f64>) {
    if let Some(i) = (0..bounds.dims).find(|&i| bounds.low[i].is_nan() || bounds.high[i].is_nan()) {
        panic!("bounds must not be NaN at dimension {i}");
    }
}

/// The point a fraction `u` of the way from `lo` to `hi`.
fn at_fraction(lo: f64, hi: f64, u: f64) -> f64 {
    match (lo.is_finite(), hi.is_finite()) {
        (true, true) => {
            let width = hi - lo;
            if width.is_finite() {
                lo + width * u
            } else {
                ((0.5 * lo + (0.5 * hi - 0.5 * lo) * u) * 2.0).clamp(lo, hi)
            }
        }
        (true, false) => lo,
        (false, true) => hi,
        (false, false) => 0.0,
    }
}

/// A uniform draw on `bounds`: `Bounds::mkpoint` when every axis has a
/// positive finite width, and otherwise one axis at a time, an axis whose
/// ends are equal at that value without a draw.
pub(crate) fn uniform_point<R: Rng>(bounds: &Bounds<f64>, rng: &mut R) -> Array1<f64> {
    assert_no_nan_wall(bounds);
    let spans = (0..bounds.dims).all(|i| {
        let width = bounds.high[i] - bounds.low[i];
        width.is_finite() && width > 0.0
    });
    if spans {
        return bounds.mkpoint(rng);
    }
    Array1::from_iter(bounds.low.iter().zip(bounds.high.iter()).map(|(&lo, &hi)| {
        if lo == hi {
            lo
        } else if (hi - lo).is_finite() {
            rng.random_range(lo..hi)
        } else if lo.is_finite() && hi.is_finite() {
            (rng.random_range(0.5 * lo..0.5 * hi) * 2.0).clamp(lo, hi)
        } else {
            at_fraction(lo, hi, 0.0)
        }
    }))
}

/// [`uniform_point`] for HMC and NUTS, whose steps move every coordinate and
/// so cannot hold an axis whose ends are equal: on such a box it panics in
/// `Bounds::mkpoint`.
pub(crate) fn spanning_uniform_point<R: Rng>(bounds: &Bounds<f64>, rng: &mut R) -> Array1<f64> {
    assert_no_nan_wall(bounds);
    if bounds
        .low
        .iter()
        .zip(bounds.high.iter())
        .any(|(lo, hi)| lo == hi)
    {
        return bounds.mkpoint(rng);
    }
    uniform_point(bounds, rng)
}

/// `eindir_core::shifted_low_discrepancy_points` on `bounds`, finite on every
/// axis. The unit cube's points are eindir's fractions exactly, since eindir
/// scales them as `0 + 1 * u`.
pub(crate) fn shifted_low_discrepancy_points(
    bounds: &Bounds<f64>,
    n: usize,
    skip: u64,
    seed: u64,
) -> Array2<f64> {
    assert_no_nan_wall(bounds);
    let widths_finite = (0..bounds.dims).all(|i| (bounds.high[i] - bounds.low[i]).is_finite());
    if widths_finite {
        return eindir_core::shifted_low_discrepancy_points(bounds, n, skip, seed);
    }
    let unit = Bounds::new(Array1::zeros(bounds.dims), Array1::ones(bounds.dims), 0.0);
    let mut points = eindir_core::shifted_low_discrepancy_points(&unit, n, skip, seed);
    for mut row in points.rows_mut() {
        for (i, u) in row.iter_mut().enumerate() {
            *u = at_fraction(bounds.low[i], bounds.high[i], *u);
        }
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn bounds(low: Array1<f64>, high: Array1<f64>) -> Bounds<f64> {
        Bounds::new(low, high, 0.0)
    }

    fn assert_spread_inside(points: &Array2<f64>, b: &Bounds<f64>) {
        for row in points.rows() {
            for (i, &x) in row.iter().enumerate() {
                assert!(x >= b.low[i] && x <= b.high[i], "{x} outside axis {i}");
            }
        }
    }

    #[test]
    fn on_finite_boxes_the_points_are_eindirs() {
        let b = bounds(array![-2.0, 0.0, 1e300], array![3.0, 0.0, 1.5e300]);
        for seed in 0..4 {
            let ours = shifted_low_discrepancy_points(&b, 9, seed + 1, seed);
            let theirs = eindir_core::shifted_low_discrepancy_points(&b, 9, seed + 1, seed);
            assert_eq!(ours.mapv(f64::to_bits), theirs.mapv(f64::to_bits));
        }
        let b = bounds(array![-2.0, 1e300], array![3.0, 1.5e300]);
        let (mut ours, mut theirs) = (StdRng::seed_from_u64(5), StdRng::seed_from_u64(5));
        for _ in 0..8 {
            let x = uniform_point(&b, &mut ours);
            let y = b.mkpoint(&mut theirs);
            assert_eq!(x.mapv(f64::to_bits), y.mapv(f64::to_bits));
        }
    }

    #[test]
    fn a_finite_axis_of_a_box_that_overflows_keeps_eindirs_points() {
        let wide = bounds(array![-2.0, -1e308], array![3.0, 1e308]);
        let narrow = bounds(array![-2.0, -1.0], array![3.0, 1.0]);
        let ours = shifted_low_discrepancy_points(&wide, 9, 2, 7);
        let theirs = eindir_core::shifted_low_discrepancy_points(&narrow, 9, 2, 7);
        assert_eq!(
            ours.column(0).mapv(f64::to_bits),
            theirs.column(0).mapv(f64::to_bits)
        );
    }

    #[test]
    fn points_on_a_box_whose_width_overflows_spread_across_it() {
        let m = f64::MAX;
        for b in [
            bounds(array![-1e308, -1e308], array![1e308, 1e308]),
            bounds(array![-m], array![m]),
            bounds(array![-m], array![0.5 * m]),
        ] {
            let points = shifted_low_discrepancy_points(&b, 16, 1, 3);
            assert_spread_inside(&points, &b);
            let mut column: Vec<f64> = points.column(0).to_vec();
            column.sort_by(f64::total_cmp);
            column.dedup();
            assert_eq!(column.len(), 16, "{column:?}");
            assert!(column[0] < 0.0 && column[15] > 0.0, "{column:?}");
            let mut rng = StdRng::seed_from_u64(3);
            let draws: Vec<f64> = (0..64).map(|_| uniform_point(&b, &mut rng)[0]).collect();
            assert!(draws.iter().all(|&x| x >= b.low[0] && x <= b.high[0]));
            assert!(draws.iter().any(|&x| x < -0.25 * m) && draws.iter().any(|&x| x > 0.25 * m));
        }
    }

    #[test]
    fn an_axis_with_an_infinite_wall_takes_its_finite_wall_or_zero() {
        let inf = f64::INFINITY;
        let b = bounds(array![2.0, -inf, -inf, -1.0], array![inf, -3.0, inf, 1.0]);
        let mut rng = StdRng::seed_from_u64(1);
        for points in [
            shifted_low_discrepancy_points(&b, 5, 1, 4),
            uniform_point(&b, &mut rng).insert_axis(ndarray::Axis(0)),
        ] {
            for row in points.rows() {
                assert_eq!(&row.to_vec()[..3], &[2.0, -3.0, 0.0]);
                assert!(row[3] >= -1.0 && row[3] < 1.0);
            }
        }
    }

    #[test]
    fn a_uniform_draw_holds_an_axis_whose_ends_are_equal() {
        let b = bounds(array![0.5, -1e308, 0.0], array![0.5, 1e308, 1.0]);
        let mut rng = StdRng::seed_from_u64(2);
        for _ in 0..8 {
            let x = uniform_point(&b, &mut rng);
            assert_eq!(x[0], 0.5);
            assert!(x[1].is_finite() && x[2] >= 0.0 && x[2] < 1.0);
        }
    }

    #[test]
    fn a_spanning_draw_is_the_uniform_draw_where_every_axis_spans() {
        let b = bounds(array![-1e308, 2.0, 0.0], array![1e308, f64::INFINITY, 1.0]);
        let (mut ours, mut theirs) = (StdRng::seed_from_u64(4), StdRng::seed_from_u64(4));
        for _ in 0..8 {
            let x = spanning_uniform_point(&b, &mut ours);
            let y = uniform_point(&b, &mut theirs);
            assert_eq!(x.mapv(f64::to_bits), y.mapv(f64::to_bits));
        }
    }

    #[test]
    #[should_panic(expected = "low < high required for Uniform sampling: EmptyRange")]
    fn a_spanning_draw_refuses_an_axis_whose_ends_are_equal() {
        let b = bounds(array![0.5, -1e308], array![0.5, 1e308]);
        spanning_uniform_point(&b, &mut StdRng::seed_from_u64(0));
    }

    #[test]
    #[should_panic(expected = "bounds must not be NaN at dimension 1")]
    fn a_nan_wall_is_refused_by_the_design() {
        let b = bounds(array![0.0, 0.0], array![1.0, f64::NAN]);
        shifted_low_discrepancy_points(&b, 4, 1, 0);
    }

    #[test]
    #[should_panic(expected = "bounds must not be NaN at dimension 0")]
    fn a_nan_wall_is_refused_by_the_draw() {
        let b = bounds(array![f64::NAN, 0.5], array![1.0, 0.5]);
        uniform_point(&b, &mut StdRng::seed_from_u64(0));
    }
}
