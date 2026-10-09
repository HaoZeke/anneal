//! The classical presets keep the chain in the objective's box: every
//! evaluated point and `History.best` lie in the closed box, an axis with
//! `low == high` never moves, and inside the box the chain consumes the RNG
//! exactly as the unconstrained composition did.

use std::sync::{Arc, Mutex};

use anneal_core::accept::Metropolis;
use anneal_core::cool::LogCool;
use anneal_core::movekernel::Gaussian;
use anneal_core::neigh::ContinuousR_n;
use anneal_core::variant::{SaVariant, boltzmann, fast, gsa};
use anneal_core::{History, run_rs_qmc_variant, run_rs_variant};
use eindir_core::{Bounds, Objective};
use ndarray::{Array1, ArrayView1, array};
use proptest::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

const N_EPOCHS: usize = 4;
const STEPS_PER_EPOCH: usize = 50;

type Seen = Arc<Mutex<Vec<Array1<f64>>>>;

/// Records every point it evaluates. The objective falls toward the upper
/// corner, so an annealing chain presses on the walls.
struct Recorder {
    bounds: Bounds<f64>,
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
        -x.sum()
    }
}

/// A recorder on the box `[low, high]` with no membership slack.
fn recorder(low: &Array1<f64>, high: &Array1<f64>) -> (Recorder, Seen) {
    let seen = Seen::default();
    let obj = Recorder {
        bounds: Bounds::new(low.clone(), high.clone(), 0.0),
        seen: Arc::clone(&seen),
    };
    (obj, seen)
}

/// Runs preset `which` (0 Boltzmann, 1 Fast, 2 GSA) with step scale `scale`
/// (the GSA scale is its initial temperature), from a uniform start or from
/// `n_starts` low-discrepancy starts.
fn run_preset(
    which: usize,
    obj: Recorder,
    scale: f64,
    q_v: f64,
    n_starts: Option<usize>,
    seed: u64,
) -> History {
    macro_rules! drive {
        ($variant:expr) => {
            match n_starts {
                Some(n) => run_rs_qmc_variant($variant, n, N_EPOCHS, STEPS_PER_EPOCH, seed),
                None => run_rs_variant($variant, N_EPOCHS, STEPS_PER_EPOCH, seed),
            }
        };
    }
    match which {
        0 => drive!(boltzmann(obj, 1.0, scale).unwrap()),
        1 => drive!(fast(obj, 1.0, scale).unwrap()),
        _ => drive!(gsa(obj, scale, q_v, 1.7).unwrap()),
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
    ) {
        let low = Array1::from_iter(axes.iter().map(|&(lo, _)| lo));
        let high = Array1::from_iter(axes.iter().map(|&(lo, width)| lo + width));
        let scale = factor * axes.iter().map(|&(_, width)| width).fold(0.0, f64::max);
        let n_starts = qmc.then_some(3);
        let (obj, seen) = recorder(&low, &high);
        let history = run_preset(which, obj, scale, q_v, n_starts, seed);

        let seen = seen.lock().unwrap();
        // Reflection never drops a proposal, so every step is evaluated.
        prop_assert_eq!(
            seen.len(),
            n_starts.unwrap_or(1) * (1 + N_EPOCHS * STEPS_PER_EPOCH)
        );
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
            let history = run_preset(which, obj, 10.0, 2.62, n_starts, 5);
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
    for which in 0..3 {
        for n_starts in [None, Some(3)] {
            let mut runs = (0..2).map(|_| {
                let (obj, seen) = recorder(&low, &high);
                let history = run_preset(which, obj, 10.0, 2.62, n_starts, 99);
                (format!("{history:?}"), seen.lock().unwrap().clone())
            });
            assert_eq!(runs.next(), runs.next());
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
            boltzmann(obj, 1.0, 1e-2).unwrap(),
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
