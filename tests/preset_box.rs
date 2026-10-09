//! The boxed presets keep the chain in the objective's box: every
//! evaluated point and `History.best` lie in the closed box, an axis with
//! `low == high` never moves, a supplied start is the first point evaluated,
//! and inside the box the chain consumes the RNG exactly as the
//! unconstrained composition did. A NaN objective value at the start neither
//! freezes the chain nor survives as its best value, and a start that meets
//! only NaN never wins a multistart.

use std::sync::{Arc, Mutex};

use anneal_core::accept::Metropolis;
use anneal_core::cool::LogCool;
use anneal_core::movekernel::Gaussian;
use anneal_core::neigh::ContinuousR_n;
use anneal_core::sampler::Sampler;
use anneal_core::variant::{SaVariant, boltzmann_in_box, fast_in_box, gsa_in_box};
use anneal_core::{
    History, qmc_skip_from_seed, run_rs_qmc_variant_from, run_rs_variant, run_rs_variant_from,
    run_rs_variant_resumed,
};
use eindir_core::{Bounds, Objective};
use ndarray::{Array1, ArrayView1, array};
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

const N_EPOCHS: usize = 4;
const STEPS_PER_EPOCH: usize = 50;

type Seen = Arc<Mutex<Vec<Array1<f64>>>>;

/// Records every point it evaluates. The objective falls toward the upper
/// corner, so an annealing chain presses on the walls; divided by a large
/// `scale`, it falls gently enough for the chain to wander the box.
struct Recorder {
    bounds: Bounds<f64>,
    scale: f64,
    seen: Seen,
}

impl Objective<f64> for Recorder {
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.seen.lock().unwrap().push(x.to_owned());
        -x.sum() / self.scale
    }
}

/// A recorder on the box `[low, high]` with no membership slack.
fn recorder(low: &Array1<f64>, high: &Array1<f64>) -> (Recorder, Seen) {
    scaled_recorder(low, high, 1.0)
}

/// [`recorder`] with its objective divided by `scale`.
fn scaled_recorder(low: &Array1<f64>, high: &Array1<f64>, scale: f64) -> (Recorder, Seen) {
    let seen = Seen::default();
    let obj = Recorder {
        bounds: Bounds::new(low.clone(), high.clone(), 0.0),
        scale,
        seen: Arc::clone(&seen),
    };
    (obj, seen)
}

/// The squared norm, except NaN at the first `n_nan` points evaluated, which
/// begin with the start of the chain, or of the first chain of a multistart.
struct NanFirst {
    bounds: Bounds<f64>,
    n_nan: usize,
    vals: Arc<Mutex<Vec<f64>>>,
}

impl Objective<f64> for NanFirst {
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let mut vals = self.vals.lock().unwrap();
        let val = if vals.len() < self.n_nan {
            f64::NAN
        } else {
            x.dot(&x)
        };
        vals.push(val);
        val
    }
}

/// Runs preset `which` (0 Boltzmann, 1 Fast, 2 GSA) with step scale `scale`
/// (the GSA scale is its initial temperature), from a uniform start or from
/// `n_starts` low-discrepancy starts, with `x0` as the (first) start if given.
fn run_preset<O: Objective<f64> + Send + Sync>(
    which: usize,
    obj: O,
    scale: f64,
    q_v: f64,
    n_starts: Option<usize>,
    seed: u64,
    x0: Option<ArrayView1<f64>>,
) -> History {
    let x0 = x0.map(|view| view.to_owned());
    macro_rules! drive {
        ($variant:expr) => {
            match n_starts {
                Some(n) => {
                    run_rs_qmc_variant_from($variant, n, N_EPOCHS, STEPS_PER_EPOCH, seed, x0, None)
                }
                None => run_rs_variant_from($variant, N_EPOCHS, STEPS_PER_EPOCH, seed, x0, None),
            }
        };
    }
    match which {
        0 => drive!(boltzmann_in_box(obj, 1.0, scale).unwrap()),
        1 => drive!(fast_in_box(obj, 1.0, scale).unwrap()),
        _ => drive!(gsa_in_box(obj, scale, q_v, 1.7).unwrap()),
    }
}

fn assert_in_box(x: &Array1<f64>, low: &Array1<f64>, high: &Array1<f64>) {
    for k in 0..x.len() {
        assert!(
            low[k] <= x[k] && x[k] <= high[k],
            "x[{k}] = {} outside [{}, {}]",
            x[k],
            low[k],
            high[k]
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    #[test]
    fn presets_evaluate_only_inside_the_box(
        which in 0usize..3,
        seed in any::<u64>(),
        axes in prop::collection::vec((-100.0_f64..100.0, 1e-3_f64..50.0), 1..=6),
        factor in 1.0_f64..1e3,
        q_v in 1.1_f64..2.9,
        qmc in any::<bool>(),
        x0_at in prop::option::of(prop::collection::vec(
            prop_oneof![Just(0.0), Just(1.0), 0.0_f64..1.0],
            6,
        )),
    ) {
        let low = Array1::from_iter(axes.iter().map(|&(lo, _)| lo));
        let high = Array1::from_iter(axes.iter().map(|&(lo, width)| lo + width));
        let scale = factor * axes.iter().map(|&(_, width)| width).fold(0.0, f64::max);
        let n_starts = qmc.then_some(3);
        let x0 = x0_at.map(|at| {
            Array1::from_iter(axes.iter().zip(at).map(|(&(lo, width), t)| lo + t * width))
        });
        let start = x0.as_ref().map(|x| x.view());
        let (obj, seen) = recorder(&low, &high);
        let history = run_preset(which, obj, scale, q_v, n_starts, seed, start);

        let seen = seen.lock().unwrap();
        // Reflection never drops a proposal, so every step is evaluated.
        prop_assert_eq!(
            seen.len(),
            n_starts.unwrap_or(1) * (1 + N_EPOCHS * STEPS_PER_EPOCH)
        );
        if let Some(x0) = &x0 {
            prop_assert_eq!(&seen[0], x0);
        }
        for x in seen.iter().chain(std::iter::once(&history.best.pos)) {
            assert_in_box(x, &low, &high);
        }
    }
}

#[test]
fn equal_endpoints_hold_the_coordinate() {
    let low = array![-1.0, 0.25, -2.0];
    let high = array![1.0, 0.25, 2.0];
    for which in 0..3 {
        for n_starts in [None, Some(3)] {
            let (obj, seen) = recorder(&low, &high);
            let history = run_preset(which, obj, 10.0, 2.62, n_starts, 5, None);
            let seen = seen.lock().unwrap();
            assert!(
                seen.iter().all(|x| x[1] == 0.25),
                "preset {which} moved the fixed coordinate"
            );
            assert_eq!(history.best.pos[1], 0.25);
        }
    }
}

#[test]
fn presets_are_deterministic_per_seed() {
    let low = array![-1.0, 0.5, -3.0];
    let high = array![2.0, 0.5, 3.0];
    for x0 in [None, Some(array![0.0, 0.5, 1.0])] {
        for which in 0..3 {
            for n_starts in [None, Some(3)] {
                let mut runs = (0..2).map(|_| {
                    let (obj, seen) = recorder(&low, &high);
                    let x0 = x0.as_ref().map(|x| x.view());
                    let history = run_preset(which, obj, 10.0, 2.62, n_starts, 99, x0);
                    (format!("{history:?}"), seen.lock().unwrap().clone())
                });
                assert_eq!(runs.next(), runs.next());
            }
        }
    }
}

#[test]
fn x0_is_the_first_evaluation() {
    let low = array![-3.0, -1.0, 0.5];
    let high = array![3.0, 2.0, 0.75];
    // An interior point and a corner, which clipping must leave untouched.
    for x0 in [array![0.1, -0.7, 0.6], array![-3.0, 2.0, 0.75]] {
        for which in 0..3 {
            for n_starts in [None, Some(3)] {
                let (obj, seen) = recorder(&low, &high);
                run_preset(which, obj, 10.0, 2.62, n_starts, 7, Some(x0.view()));
                let seen = seen.lock().unwrap();
                assert_eq!(
                    seen.len(),
                    n_starts.unwrap_or(1) * (1 + N_EPOCHS * STEPS_PER_EPOCH)
                );
                assert_eq!(seen[0], x0, "preset {which} did not start at x0");
            }
        }
    }
}

#[test]
fn x0_replaces_only_the_first_qmc_start() {
    let low = array![-3.0, -1.0, 0.5];
    let high = array![3.0, 2.0, 0.75];
    let x0 = array![0.1, -0.7, 0.6];
    let per_chain = 1 + N_EPOCHS * STEPS_PER_EPOCH;
    for which in 0..3 {
        let (obj, plain) = recorder(&low, &high);
        run_preset(which, obj, 10.0, 2.62, Some(3), 11, None);
        let (obj, started) = recorder(&low, &high);
        run_preset(which, obj, 10.0, 2.62, Some(3), 11, Some(x0.view()));
        let (plain, started) = (plain.lock().unwrap(), started.lock().unwrap());
        assert_ne!(plain[0], x0);
        assert_eq!(started[0], x0);
        assert_eq!(plain[per_chain..], started[per_chain..]);
    }
}

#[test]
fn boxes_wider_than_half_the_f64_range_mirror_off_the_walls() {
    // Unit temperature is far below the objective's steps here, so the chain
    // climbs onto the upper wall and keeps overshooting it. Gaussian steps of
    // this size stay finite, so a mirrored overshoot never lands on a wall.
    let low = array![-6e307];
    let high = array![6e307];
    let (obj, seen) = recorder(&low, &high);
    run_preset(0, obj, 1e307, 2.62, None, 1, Some(array![3e307].view()));
    let seen = seen.lock().unwrap();
    assert!(
        seen.iter().all(|x| low[0] < x[0] && x[0] < high[0]),
        "an evaluation reached a wall"
    );
}

#[test]
fn a_step_whose_offset_from_low_overflows_mirrors_off_high() {
    // Each box is narrower than half the f64 range but lies far from zero,
    // so `x - low` overflows for a step past `high`, and near f64::MAX the
    // step itself overflows to +inf. The chain starts on `high`, the best
    // point, and every step is drawn from there.
    let m = f64::MAX;
    for (low, high, scale, n_starts) in [
        (array![-0.6 * m], array![-0.1 * m], m / 4.0, None),
        (array![-0.6 * m], array![-0.1 * m], m / 4.0, Some(2)),
        (array![1e308], array![1.7e308], 1e307, None),
        (array![1e308], array![1.7e308], 3e307, None),
    ] {
        let (obj, seen) = recorder(&low, &high);
        run_preset(0, obj, scale, 2.62, n_starts, 3, Some(high.view()));
        let seen = seen.lock().unwrap();
        assert!(
            seen.iter().all(|x| low[0] < x[0] && x[0] <= high[0]),
            "a step past {:e} landed on {:e}",
            high[0],
            low[0]
        );
    }
}

/// The seeds among `0..40` on which preset `which`, started at `x0` with step
/// scale `step` on the objective divided by `scale`, evaluates a point
/// outside the open box `(low, high)`, where an infinite wall stands at
/// `+-f64::MAX`.
fn seeds_leaving_the_open_box(
    which: usize,
    (low, high): (&Array1<f64>, &Array1<f64>),
    x0: &Array1<f64>,
    step: f64,
    scale: f64,
) -> Vec<u64> {
    let m = f64::MAX;
    (0..40)
        .filter(|&seed| {
            let (obj, seen) = scaled_recorder(low, high, scale);
            run_preset(which, obj, step, 2.62, None, seed, Some(x0.view()));
            let seen = seen.lock().unwrap();
            !seen
                .iter()
                .all(|x| low[0].max(-m) < x[0] && x[0] < high[0].min(m))
        })
        .collect()
}

#[test]
fn a_step_whose_sum_overflows_mirrors_instead_of_stopping_on_a_wall() {
    // Each box reaches within a few steps of f64::MAX, where `x + step`
    // overflows, and the chain climbs there. An infinite wall then acts as a
    // wall at f64::MAX.
    let m = f64::MAX;
    for (low, high, x0, step) in [
        (array![0.0], array![m], array![0.99 * m], 0.01 * m),
        (array![1e308], array![1.7e308], array![1.6e308], 3e307),
        (array![1e308], array![f64::INFINITY], array![1.6e308], 3e307),
        (array![-6e307], array![6e307], array![3e307], 1e307),
    ] {
        for which in 0..2 {
            let seeds = seeds_leaving_the_open_box(which, (&low, &high), &x0, step, 1.0);
            assert!(
                seeds.is_empty(),
                "preset {which} evaluated a wall of [{:e}, {:e}] on seeds {seeds:?}",
                low[0],
                high[0]
            );
        }
    }
}

#[test]
fn a_half_infinite_box_far_from_zero_is_never_evaluated_at_infinity() {
    // Across the finite wall of [1e308, inf) or (-inf, -1e308] the image of a
    // step can lie past +-f64::MAX, and its distance from the wall can
    // overflow. Divided by 1e308, the objective lets the chain wander down to
    // the finite wall.
    let inf = f64::INFINITY;
    for (low, high, x0) in [
        (array![1e308], array![inf], array![1.6e308]),
        (array![-inf], array![-1e308], array![-1.6e308]),
    ] {
        for (which, scale) in [(0, 1.0), (1, 1.0), (0, 1e308), (1, 1e308)] {
            let seeds = seeds_leaving_the_open_box(which, (&low, &high), &x0, 5e307, scale);
            assert!(
                seeds.is_empty(),
                "preset {which} on -x / {scale:e} evaluated a wall or infinity of \
                 [{:e}, {:e}] on seeds {seeds:?}",
                low[0],
                high[0]
            );
        }
    }
}

#[test]
fn half_infinite_boxes_mirror_across_the_finite_wall() {
    let inf = f64::INFINITY;
    for (low, high, x0) in [
        (array![0.0], array![inf], array![0.1]),
        (array![-inf], array![0.0], array![-0.1]),
    ] {
        for which in 0..3 {
            let (obj, seen) = recorder(&low, &high);
            let history = run_preset(which, obj, 1.0, 2.62, None, 3, Some(x0.view()));
            let seen = seen.lock().unwrap();
            for x in seen.iter().chain(std::iter::once(&history.best.pos)) {
                assert!(
                    x[0].is_finite(),
                    "preset {which} evaluated {x} in [{low}, {high}]"
                );
                assert_in_box(x, &low, &high);
            }
        }
    }
}

#[test]
fn an_infinite_wall_starts_the_chain_on_its_finite_wall() {
    // Without x0 an axis with an infinite wall has no uniform draw. It starts
    // where a pinned axis on its finite wall would, or at 0 when both walls
    // are infinite, and the other axes draw the same numbers.
    let inf = f64::INFINITY;
    let start = |low: &Array1<f64>, high: &Array1<f64>, seed: u64| {
        let (obj, _) = recorder(low, high);
        let mut rng = StdRng::seed_from_u64(seed);
        let state = boltzmann_in_box(obj, 1.0, 1.0)
            .unwrap()
            .initial_state(&mut rng);
        (state.cur.pos, rng.next_u64())
    };
    for (low, high, wall) in [
        (array![2.0, -1.0], array![inf, 1.0], 2.0),
        (array![-inf, -1.0], array![-2.0, 1.0], -2.0),
        (array![-inf, -1.0], array![inf, 1.0], 0.0),
    ] {
        let (pinned_low, pinned_high) = (array![wall, -1.0], array![wall, 1.0]);
        for seed in [0, 1, 99] {
            assert_eq!(
                start(&low, &high, seed),
                start(&pinned_low, &pinned_high, seed)
            );
        }
        for which in 0..3 {
            let (obj, seen) = recorder(&low, &high);
            let history = run_preset(which, obj, 1.0, 2.62, None, 5, None);
            let seen = seen.lock().unwrap();
            assert_eq!(seen[0][0], wall, "preset {which}");
            for x in seen.iter().chain(std::iter::once(&history.best.pos)) {
                assert!(
                    x.iter().all(|v| v.is_finite()),
                    "preset {which} evaluated {x}"
                );
                assert_in_box(x, &low, &high);
            }
        }
        let (obj, seen) = recorder(&low, &high);
        run_rs_variant_resumed(boltzmann_in_box(obj, 1.0, 1.0).unwrap(), 0, 1, 1, 5, None);
        assert_eq!(seen.lock().unwrap()[0][0], wall);
    }
}

#[test]
fn qmc_starts_on_an_infinite_wall_lie_on_its_finite_wall() {
    // The Halton points of an axis with an infinite wall are infinite or
    // NaN, so every start but x0 lies on the finite wall instead, or at 0
    // when both walls are infinite, and no chain reaches infinity.
    let inf = f64::INFINITY;
    let per_chain = 1 + N_EPOCHS * STEPS_PER_EPOCH;
    for (low, high, wall, x0) in [
        (array![2.0, -1.0], array![inf, 1.0], 2.0, array![3.0, 0.5]),
        (
            array![-inf, -1.0],
            array![-2.0, 1.0],
            -2.0,
            array![-3.0, 0.5],
        ),
        (array![-inf, -1.0], array![inf, 1.0], 0.0, array![1.0, 0.5]),
    ] {
        for which in 0..3 {
            for start in [None, Some(x0.view())] {
                let (obj, seen) = recorder(&low, &high);
                let history = run_preset(which, obj, 1.0, 2.62, Some(3), 5, start);
                let seen = seen.lock().unwrap();
                for idx in usize::from(start.is_some())..3 {
                    assert_eq!(seen[idx * per_chain][0], wall, "preset {which}");
                }
                for x in seen.iter().chain(std::iter::once(&history.best.pos)) {
                    assert!(
                        x.iter().all(|v| v.is_finite()),
                        "preset {which} evaluated {x}"
                    );
                    assert_in_box(x, &low, &high);
                }
            }
        }
    }
}

#[test]
fn a_box_whose_width_overflows_draws_its_starts_at_half_scale() {
    // `high - low` overflows on the first axis, so the uniform start and the
    // Halton starts are drawn on the halved box and doubled. The pinned last
    // axis sends the halved box down the same per-axis draw.
    let m = f64::MAX;
    let n_starts = 6;
    for (low, high) in [
        (array![-1e308, -1.0, 0.25], array![1e308, 1.0, 0.25]),
        (array![-m, -1.0, 0.25], array![m, 1.0, 0.25]),
        (array![-m, -1.0, 0.25], array![0.5 * m, 1.0, 0.25]),
    ] {
        let (half_low, half_high) = (&low * 0.5, &high * 0.5);
        for seed in [0, 1, 99] {
            let draw = |low: &Array1<f64>, high: &Array1<f64>| {
                let (obj, _) = recorder(low, high);
                let mut rng = StdRng::seed_from_u64(seed);
                let state = boltzmann_in_box(obj, 1.0, 1.0)
                    .unwrap()
                    .initial_state(&mut rng);
                (state.cur.pos, rng.next_u64())
            };
            let (pos, next) = draw(&low, &high);
            let (half_pos, half_next) = draw(&half_low, &half_high);
            assert_eq!((pos, next), (&half_pos * 2.0, half_next));
        }
        let halton = eindir_core::low_discrepancy_points(
            &Bounds::new(half_low, half_high, 0.0),
            n_starts,
            qmc_skip_from_seed(1),
        ) * 2.0;
        let x0 = array![0.0, 0.5, 0.25];
        for start in [None, Some(x0.clone())] {
            let (obj, seen) = recorder(&low, &high);
            let variant = boltzmann_in_box(obj, 1.0, 1e300).unwrap();
            run_rs_qmc_variant_from(variant, n_starts, 1, 0, 1, start.clone(), None);
            let seen = seen.lock().unwrap();
            assert_eq!(seen.len(), n_starts);
            assert_eq!(seen[0], start.unwrap_or_else(|| halton.row(0).to_owned()));
            for (x, expected) in seen.iter().zip(halton.outer_iter()).skip(1) {
                assert_eq!(x, &expected);
            }
            for (i, x) in seen.iter().enumerate() {
                assert!(low[0] < x[0] && x[0] < high[0], "a start on a wall: {x}");
                assert!(seen[..i].iter().all(|y| y[0] != x[0]), "repeated start {x}");
            }
        }
    }
}

#[test]
fn a_nan_start_is_left_and_never_kept_as_the_best() {
    let low = array![-1.0, -1.0];
    let high = array![1.0, 1.0];
    let x0 = array![0.5, 0.5];
    for which in 0..3 {
        for n_starts in [None, Some(3)] {
            for start in [None, Some(x0.view())] {
                let vals = Arc::<Mutex<Vec<f64>>>::default();
                let obj = NanFirst {
                    bounds: Bounds::new(low.clone(), high.clone(), 0.0),
                    n_nan: 1,
                    vals: Arc::clone(&vals),
                };
                let history = run_preset(which, obj, 0.5, 2.62, n_starts, 1, start);
                let vals = vals.lock().unwrap();
                let lowest = vals.iter().skip(1).copied().fold(f64::INFINITY, f64::min);
                assert!(vals[0].is_nan());
                assert!(history.total_accepted() > 0);
                assert_eq!(
                    history.best.val, lowest,
                    "preset {which}, {n_starts:?} starts, x0 {start:?}"
                );
            }
        }
    }
}

#[test]
fn a_start_that_meets_only_nan_never_wins_the_multistart() {
    // The first chain evaluates nothing but NaN, which counts as +inf, so its
    // best value is +inf, and the later starts, which meet numbers, must beat
    // it.
    let low = array![-1.0, -1.0];
    let high = array![1.0, 1.0];
    let x0 = array![0.5, 0.5];
    let first_chain = 1 + N_EPOCHS * STEPS_PER_EPOCH;
    for which in 0..3 {
        for start in [None, Some(x0.view())] {
            let vals = Arc::<Mutex<Vec<f64>>>::default();
            let obj = NanFirst {
                bounds: Bounds::new(low.clone(), high.clone(), 0.0),
                n_nan: first_chain,
                vals: Arc::clone(&vals),
            };
            let history = run_preset(which, obj, 0.5, 2.62, Some(3), 1, start);
            let vals = vals.lock().unwrap();
            let lowest = vals[first_chain..]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            assert_eq!(history.best.val, lowest, "preset {which}, x0 {start:?}");
        }
    }
}

#[test]
fn inside_the_box_the_preset_follows_the_unconstrained_chain() {
    // Steps far smaller than the box never reach a wall, so the reflected
    // chain must match the old unconstrained composition draw for draw, and
    // its start must be the `mkpoint` draw from the run seed.
    let low = array![-1e3, -1e3, -1e3];
    let high = array![1e3, 1e3, 1e3];
    for seed in 0..8 {
        let (obj, boxed) = recorder(&low, &high);
        let h_box = run_rs_variant(
            boltzmann_in_box(obj, 1.0, 1e-2).unwrap(),
            N_EPOCHS,
            STEPS_PER_EPOCH,
            seed,
        );
        let (obj, free) = recorder(&low, &high);
        let unconstrained = SaVariant::checked(
            obj,
            LogCool::new(1.0, 2.0),
            ContinuousR_n::new(3),
            Gaussian::new(1e-2),
            Metropolis,
        )
        .unwrap();
        let h_free = run_rs_variant(unconstrained, N_EPOCHS, STEPS_PER_EPOCH, seed);

        let boxed = boxed.lock().unwrap();
        let start =
            Bounds::new(low.clone(), high.clone(), 0.0).mkpoint(&mut StdRng::seed_from_u64(seed));
        assert_eq!(boxed[0], start);
        assert_eq!(*boxed, *free.lock().unwrap());
        assert_eq!(format!("{h_box:?}"), format!("{h_free:?}"));
    }
}
