//! Starts and designs on boxes whose width `high - low` overflows `f64`,
//! such as `[-1e308, 1e308]`, and on boxes with an infinite wall where the
//! method takes one. The uniform start of the HMC, NUTS and classical
//! samplers is drawn inside such a box, spread across it, and the
//! low-discrepancy designs of the portfolio, the local polishers and the DMC
//! population spread across it off its walls. An axis with an infinite wall
//! starts on its finite wall, or at zero when both walls are infinite, and the
//! classical run, the polishers and the DMC population evaluate no infinite
//! coordinate there. A NaN wall is refused, naming its coordinate, before
//! anything is evaluated.

use std::sync::{Arc, Mutex};

use anneal_core::cool::LogCool;
use anneal_core::hmc::{GaussianMomentum, HmcSaSampler, LeapfrogIntegrator, NutsSaSampler};
use anneal_core::methods::dmc_population::run_dmc_population_seeded;
use anneal_core::methods::local_polish::{
    qmc_best1bin_scout, qmc_gsa_global_search_from, qmc_trust_region_poll,
    shifted_qmc_projected_gradient_polish,
};
use anneal_core::methods::{PortfolioPolicy, portfolio_optimize_from};
use anneal_core::run_rs;
use anneal_core::run_rs_variant_from;
use anneal_core::variant::{boltzmann, boltzmann_in_box};
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

const WIDE: f64 = 1e308;
const SCALE: f64 = 1e307;

type Seen = Arc<Mutex<Vec<Array1<f64>>>>;

/// Records every point it evaluates. Its value, the sum of
/// `tanh(x / 1e307)^2`, is finite and smooth over every finite `f64`.
#[derive(Clone)]
struct Recorder {
    bounds: Bounds<f64>,
    seen: Seen,
}

impl Recorder {
    fn on(low: &[f64], high: &[f64]) -> Self {
        Self {
            bounds: Bounds::new(Array1::from(low.to_vec()), Array1::from(high.to_vec()), 0.0),
            seen: Seen::default(),
        }
    }

    fn points(&self) -> Vec<Array1<f64>> {
        self.seen.lock().unwrap().clone()
    }
}

impl Objective<f64> for Recorder {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.seen.lock().unwrap().push(x.to_owned());
        x.iter().map(|v| (v / SCALE).tanh().powi(2)).sum()
    }

    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
}

#[derive(Clone)]
struct Slope(usize);

impl Gradient<f64> for Slope {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        x.mapv(|v| {
            let t = (v / SCALE).tanh();
            2.0 * t * (1.0 - t * t) / SCALE
        })
    }

    fn dim(&self) -> usize {
        self.0
    }
}

/// Boxes whose width overflows on some axis, every wall finite; on the
/// first two it overflows on every axis.
fn overflowing_boxes() -> Vec<Recorder> {
    let m = f64::MAX;
    vec![
        Recorder::on(&[-WIDE, -WIDE], &[WIDE, WIDE]),
        Recorder::on(&[-m, -m, -m], &[m, m, m]),
        Recorder::on(&[-WIDE, 0.0, -1.0], &[WIDE, 1.0, 2.0]),
    ]
}

/// Boxes with an infinite wall, and the start each axis takes on them. The
/// last also has an axis whose ends are equal, which HMC and NUTS refuse.
fn infinite_boxes() -> Vec<(Recorder, Vec<f64>)> {
    let inf = f64::INFINITY;
    vec![
        (Recorder::on(&[0.0, 0.0], &[inf, inf]), vec![0.0, 0.0]),
        (Recorder::on(&[-inf], &[-2.0]), vec![-2.0]),
        (Recorder::on(&[-inf], &[inf]), vec![0.0]),
        (Recorder::on(&[-inf, 3.0], &[inf, 3.0]), vec![0.0, 3.0]),
    ]
}

fn inside(x: &Array1<f64>, b: &Bounds<f64>) -> bool {
    x.iter()
        .enumerate()
        .all(|(i, &v)| v >= b.low[i] && v <= b.high[i])
}

fn on_a_wall(x: &Array1<f64>, b: &Bounds<f64>) -> bool {
    x.iter()
        .enumerate()
        .any(|(i, &v)| b.low[i] < b.high[i] && (v == b.low[i] || v == b.high[i]))
}

/// The distinct points of `seen` that lie off every wall of `b`.
fn distinct_off_the_walls(seen: &[Array1<f64>], b: &Bounds<f64>) -> usize {
    let mut keys: Vec<Vec<u64>> = seen
        .iter()
        .filter(|x| !on_a_wall(x, b))
        .map(|x| x.iter().map(|v| v.to_bits()).collect())
        .collect();
    keys.sort();
    keys.dedup();
    keys.len()
}

fn assert_all_inside(seen: &[Array1<f64>], b: &Bounds<f64>, what: &str) {
    assert!(!seen.is_empty(), "{what}: nothing evaluated");
    for x in seen {
        assert!(inside(x, b), "{what}: {x} outside the box");
    }
}

/// The first point each seed's run evaluates, its start. With `held`, every
/// point the run evaluates must lie in the box; NUTS's steps are not held.
fn starts(rec: &Recorder, run: impl Fn(Recorder, u64), held: bool) -> Vec<Array1<f64>> {
    (0..12)
        .map(|seed| {
            let fresh = Recorder::on(
                rec.bounds.low.as_slice().unwrap(),
                rec.bounds.high.as_slice().unwrap(),
            );
            run(fresh.clone(), seed);
            let seen = fresh.points();
            if held {
                assert_all_inside(&seen, &fresh.bounds, "a run");
            }
            seen[0].clone()
        })
        .collect()
}

/// Starts drawn across an overflowing box: inside it, off its walls, and on
/// both sides of zero by more than a quarter of the wall on its first axis.
fn assert_spread(starts: &[Array1<f64>], b: &Bounds<f64>) {
    for x in starts {
        assert!(
            inside(x, b) && !on_a_wall(x, b),
            "start {x} on a wall or outside"
        );
    }
    let quarter = 0.25 * b.high[0];
    assert!(starts.iter().any(|x| x[0] < -quarter), "{starts:?}");
    assert!(starts.iter().any(|x| x[0] > quarter), "{starts:?}");
}

fn hmc_run(rec: Recorder, seed: u64) {
    let dim = rec.bounds.dims;
    let cool = LogCool::new(1.0, 2.0);
    let sampler = HmcSaSampler::new(
        rec,
        Slope(dim),
        cool.clone(),
        LeapfrogIntegrator::new(0.1, 3, 1.0),
    );
    run_rs(sampler, &cool, 2, 3, seed);
}

fn nuts_run(rec: Recorder, seed: u64) {
    let dim = rec.bounds.dims;
    let cool = LogCool::new(1.0, 2.0);
    let sampler = NutsSaSampler::new(rec, Slope(dim), cool.clone(), GaussianMomentum, 0.1, 1.0, 3);
    run_rs(sampler, &cool, 2, 3, seed);
}

fn classical_run(rec: Recorder, seed: u64) {
    let variant = boltzmann_in_box(rec, 1.0, 1e306).expect("a valid preset");
    run_rs_variant_from(variant, 2, 3, seed, None, None);
}

#[test]
fn an_hmc_run_draws_its_start_across_a_box_whose_width_overflows() {
    for rec in overflowing_boxes() {
        assert_spread(&starts(&rec, hmc_run, true), &rec.bounds);
    }
    for (rec, start) in infinite_boxes().into_iter().take(3) {
        hmc_run(rec.clone(), 1);
        assert_eq!(rec.points()[0].to_vec(), start);
    }
}

#[test]
fn a_nuts_run_draws_its_start_across_a_box_whose_width_overflows() {
    for rec in overflowing_boxes() {
        assert_spread(&starts(&rec, nuts_run, false), &rec.bounds);
    }
    for (rec, start) in infinite_boxes().into_iter().take(3) {
        nuts_run(rec.clone(), 1);
        assert_eq!(rec.points()[0].to_vec(), start);
    }
}

#[test]
fn a_classical_run_draws_its_start_across_a_box_whose_width_overflows() {
    for rec in overflowing_boxes() {
        assert_spread(&starts(&rec, classical_run, true), &rec.bounds);
    }
    for (rec, start) in infinite_boxes() {
        classical_run(rec.clone(), 1);
        let seen = rec.points();
        assert_eq!(seen[0].to_vec(), start);
        assert_all_inside(&seen, &rec.bounds, "a classical run");
        assert!(seen.iter().all(|x| x.iter().all(|v| v.is_finite())));
    }
}

/// Runs with a gradient place most of their points by descent steps, not by
/// the designs, so only the box is checked for them.
#[test]
fn the_portfolio_designs_spread_across_a_box_whose_width_overflows() {
    for rec in overflowing_boxes() {
        let dim = rec.bounds.dims;
        for (policy, gradient, noise) in [
            (PortfolioPolicy::Auto, false, None),
            (PortfolioPolicy::Auto, false, Some(0.01)),
            (PortfolioPolicy::Legacy, false, None),
            (PortfolioPolicy::Auto, true, None),
            (PortfolioPolicy::Legacy, true, None),
        ] {
            let fresh = Recorder::on(
                rec.bounds.low.as_slice().unwrap(),
                rec.bounds.high.as_slice().unwrap(),
            );
            let slope = Slope(dim);
            let result = portfolio_optimize_from(
                &fresh,
                gradient.then_some(&slope),
                300,
                3,
                noise,
                policy,
                None,
            );
            let seen = fresh.points();
            let what = format!("{policy:?}, gradient {gradient}, noise {noise:?}");
            assert_all_inside(&seen, &fresh.bounds, &what);
            assert!(
                inside(&Array1::from(result.best_pos), &fresh.bounds),
                "{what}"
            );
            let spread = distinct_off_the_walls(&seen, &fresh.bounds);
            assert!(
                gradient || spread >= 12,
                "{what}: {spread} distinct points off the walls"
            );
        }
    }
}

#[test]
fn the_local_polishers_spread_across_a_box_whose_width_overflows() {
    for rec in overflowing_boxes() {
        let dim = rec.bounds.dims;
        let fresh = || {
            Recorder::on(
                rec.bounds.low.as_slice().unwrap(),
                rec.bounds.high.as_slice().unwrap(),
            )
        };
        // Each run with the size of its design: the population, the chains,
        // the points of one poll level and the starts.
        let runs: Vec<(&str, usize, Recorder)> = vec![
            ("best1bin", 8, {
                let r = fresh();
                qmc_best1bin_scout(&r, 60, 3, 8, 0.5, 0.3, 0.9);
                r
            }),
            ("gsa", 4, {
                let r = fresh();
                qmc_gsa_global_search_from(&r, 60, 3, 4, 1.0, 2.6, -5.0, None);
                r
            }),
            ("trust region", 10, {
                let r = fresh();
                qmc_trust_region_poll(&r, Array1::zeros(dim), 60, 3, 0.1, 3, 10);
                r
            }),
            ("projected gradient", 4, {
                let r = fresh();
                shifted_qmc_projected_gradient_polish(&r, &Slope(dim), 4, 10, 3, 1, 0.1, 1e-8, 2);
                r
            }),
        ];
        for (what, design, r) in runs {
            let seen = r.points();
            assert_all_inside(&seen, &r.bounds, what);
            let spread = distinct_off_the_walls(&seen, &r.bounds);
            assert!(
                spread >= design,
                "{what}: {spread} distinct points off the walls"
            );
        }
    }
}

#[test]
fn the_local_polishers_start_on_the_finite_wall_of_an_infinite_axis() {
    for (rec, start) in infinite_boxes() {
        let dim = rec.bounds.dims;
        let fresh = || {
            Recorder::on(
                rec.bounds.low.as_slice().unwrap(),
                rec.bounds.high.as_slice().unwrap(),
            )
        };
        let r = fresh();
        qmc_best1bin_scout(&r, 30, 3, 8, 0.5, 0.3, 0.9);
        let s = fresh();
        shifted_qmc_projected_gradient_polish(&s, &Slope(dim), 2, 5, 3, 1, 0.1, 1e-8, 1);
        for (what, r) in [("best1bin", r), ("projected gradient", s)] {
            let seen = r.points();
            assert_all_inside(&seen, &r.bounds, what);
            assert_eq!(seen[0].to_vec(), start, "{what}");
            assert!(
                seen.iter().all(|x| x.iter().all(|v| v.is_finite())),
                "{what}"
            );
        }
    }
}

#[test]
fn the_dmc_population_spreads_across_a_box_whose_width_overflows() {
    for rec in overflowing_boxes() {
        let dim = rec.bounds.dims;
        let mut rng = StdRng::seed_from_u64(3);
        run_dmc_population_seeded(&rec, Some(&Slope(dim)), 60, 3, 8, 2, 1.0, None, &mut rng);
        let seen = rec.points();
        assert_all_inside(&seen, &rec.bounds, "dmc");
        let spread = distinct_off_the_walls(&seen, &rec.bounds);
        assert!(spread >= 8, "dmc: {spread} distinct points off the walls");
    }
    for (rec, start) in infinite_boxes() {
        let dim = rec.bounds.dims;
        let mut rng = StdRng::seed_from_u64(3);
        run_dmc_population_seeded(&rec, Some(&Slope(dim)), 30, 3, 4, 2, 1.0, None, &mut rng);
        let seen = rec.points();
        assert_all_inside(&seen, &rec.bounds, "dmc");
        assert_eq!(seen[0].to_vec(), start);
        assert!(seen.iter().all(|x| x.iter().all(|v| v.is_finite())));
    }
}

fn nan_wall() -> Recorder {
    Recorder::on(&[0.0, 0.0], &[1.0, f64::NAN])
}

#[test]
#[should_panic(expected = "bounds must not be NaN at dimension 1")]
fn a_classical_run_refuses_a_nan_wall_before_drawing_its_start() {
    let variant = boltzmann(nan_wall(), 1.0, 0.1).expect("a valid preset");
    run_rs_variant_from(variant, 1, 1, 0, None, None);
}

#[test]
#[should_panic(expected = "bounds must not be NaN at dimension 1")]
fn an_hmc_run_refuses_a_nan_wall_before_drawing_its_start() {
    hmc_run(nan_wall(), 0);
}

#[test]
#[should_panic(expected = "bounds must not be NaN at dimension 1")]
fn a_nuts_run_refuses_a_nan_wall_before_drawing_its_start() {
    nuts_run(nan_wall(), 0);
}

#[test]
#[should_panic(expected = "bounds must not be NaN at dimension 1")]
fn a_low_discrepancy_design_refuses_a_nan_wall() {
    qmc_best1bin_scout(&nan_wall(), 10, 0, 4, 0.5, 0.3, 0.9);
}

#[test]
#[should_panic(expected = "bounds must not be NaN at dimension 1")]
fn the_trust_region_poll_refuses_a_nan_wall() {
    let rec = nan_wall();
    qmc_trust_region_poll(&rec, Array1::zeros(2), 10, 0, 0.1, 2, 4);
}
