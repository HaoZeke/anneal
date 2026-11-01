//! Binary Lennard-Jones and rigid TIP4P on the stock cluster presets.
//!
//! [`Config::recommended`] and [`Config::communicating`] stay numerically
//! unchanged. Species labels and a rigid chart are applied afterwards.
//! A wide diameter ratio escapes by one softened constant-energy (NVE)
//! step across a barrier and tries one downhill unlike-pair exchange on
//! a side generator. That escape needs the `rgsaddle` feature. Without
//! it the search stops and [`StockSearch::nve_unavailable`] is set; the
//! library kick is not substituted. A narrow ratio exchanges mismatched
//! labels. Water takes
//! an all-body orientation step. A chart with more than ten bodies holds
//! the acceptance factor at its cap and scales that step toward an
//! acceptance ratio of one half. Ten bodies or fewer keep the factor
//! window and the fixed orientation cap. The rigid quench is the same
//! L-BFGS descent, and a rigid record accepts the root-mean-square
//! force. After 1000 hops with no new lowest minimum the walker
//! returns to that minimum when it lies within one hopping temperature
//! of the run target. Farther above the target the walker is drawn
//! again, and the lowest minimum is kept. The preset temperature
//! is left where it is.

use ndarray::{Array1, ArrayView1};

use crate::methods::cluster_hopping::{Config, Ledger};
use crate::methods::cluster_search::search_from;
use crate::potentials::{BinaryLennardJones, GminTip4p, gmin_water_geometry};

/// Two-species Lennard-Jones parameters. Atoms of species A come first.
#[derive(Debug, Clone, Copy)]
pub struct BinaryLennardJonesProblem {
    /// Atoms of species A.
    pub n_a: usize,
    /// Atoms of species B.
    pub n_b: usize,
    /// Well depth of an A-A pair.
    pub epsilon_aa: f64,
    /// Well depth of an A-B pair.
    pub epsilon_ab: f64,
    /// Well depth of a B-B pair.
    pub epsilon_bb: f64,
    /// Diameter of an A-A pair.
    pub sigma_aa: f64,
    /// Diameter of an A-B pair.
    pub sigma_ab: f64,
    /// Diameter of a B-B pair.
    pub sigma_bb: f64,
}

/// One stock-preset search.
#[derive(Debug, Clone)]
pub struct StockSearch {
    /// Lowest energy the ledger recorded.
    pub energy: f64,
    /// Charged evaluations, one value-and-gradient at a time.
    pub charged: usize,
    /// State that attained [`StockSearch::energy`].
    pub state: Vec<f64>,
    /// The wide-ratio escape was requested and the `rgsaddle` feature is off.
    pub nve_unavailable: bool,
}

impl StockSearch {
    /// Whether the recorded energy is at or below `target`.
    pub fn success(&self, target: f64) -> bool {
        self.energy.is_finite() && self.energy <= target
    }
}

/// `Config::communicating` when `communicating` is set, otherwise `Config::recommended`.
pub fn stock_preset(n_points: usize, communicating: bool) -> Config {
    if communicating {
        Config::communicating(n_points)
    } else {
        Config::recommended(n_points)
    }
}

/// Binary Lennard-Jones from `start` (`3 n` Cartesian coordinates).
///
/// `target` stops the hop once a recorded energy reaches it. The preset is
/// [`stock_preset`], then [`Config::with_species`] with per-atom diameters.
/// A caller stop bound on this thread ends the search at once and keeps the
/// best state found. A flag that stays down does not change how many
/// evaluations are charged.
pub fn search_binary(
    problem: &BinaryLennardJonesProblem,
    start: ArrayView1<f64>,
    budget: usize,
    target: f64,
    seed: u64,
    communicating: bool,
) -> StockSearch {
    let n = problem.n_a + problem.n_b;
    assert_eq!(start.len(), 3 * n, "binary start is 3 coordinates per atom");
    let mut species = vec![1u32; problem.n_a];
    species.extend(std::iter::repeat_n(2u32, problem.n_b));
    let mut sigma = vec![problem.sigma_aa; problem.n_a];
    sigma.extend(std::iter::repeat_n(problem.sigma_bb, problem.n_b));
    let cfg = stock_preset(n, communicating).with_species(species, Some(sigma));
    let potential = BinaryLennardJones::new(
        problem.n_a,
        problem.n_b,
        problem.epsilon_aa,
        problem.epsilon_ab,
        problem.epsilon_bb,
        problem.sigma_aa,
        problem.sigma_ab,
        problem.sigma_bb,
    );
    run_stock(&potential, &cfg, start, budget, target, seed)
}

/// Rigid TIP4P from a block Rodrigues start (`3 n` centres, then `3 n` rotations).
///
/// A caller stop bound on this thread ends the search at once and keeps the
/// best state found. A flag that stays down does not change how many
/// evaluations are charged.
pub fn search_gmin_tip4p(
    n_molecules: usize,
    start: ArrayView1<f64>,
    budget: usize,
    target: f64,
    seed: u64,
    communicating: bool,
) -> StockSearch {
    assert_eq!(
        start.len(),
        6 * n_molecules,
        "TIP4P start is 6 coordinates per molecule"
    );
    let cfg = stock_preset(n_molecules, communicating).with_rigid_bodies(gmin_water_geometry());
    let potential = GminTip4p::new(n_molecules);
    run_stock(&potential, &cfg, start, budget, target, seed)
}

fn run_stock<O>(
    potential: &O,
    cfg: &Config,
    start: ArrayView1<f64>,
    budget: usize,
    target: f64,
    seed: u64,
) -> StockSearch
where
    O: eindir_core::gradient::DifferentiableObjective<f64> + ?Sized,
{
    // A caller that has bound a stop flag is not a different search. The
    // ledger carries that flag so the hop returns with the best state
    // instead of walking the rest of the budget. Nothing bound keeps the
    // plain ledger, so a normal run charges the same evaluations.
    let mut ledger = match crate::caller_stop::bound() {
        Some(flag) => Ledger::with_stop(budget, flag),
        None => Ledger::new(budget),
    };
    if target.is_finite() {
        ledger.stop_below(target);
    }
    let (out, _) = search_from(potential, cfg, &mut ledger, start, seed).expect("finite start");
    let state = ledger
        .best_state
        .clone()
        .map(|value| value.to_vec())
        .unwrap_or_else(|| start.to_vec());
    StockSearch {
        energy: ledger.best,
        charged: ledger.spent(),
        state,
        nve_unavailable: out.nve_unavailable,
    }
}

/// Same search over many starts. Each start keeps its own ledger.
pub fn search_binary_batch(
    problem: &BinaryLennardJonesProblem,
    starts: &[Vec<f64>],
    budget: usize,
    target: f64,
    seeds: &[u64],
    communicating: bool,
) -> Vec<StockSearch> {
    use rayon::prelude::*;
    assert_eq!(starts.len(), seeds.len());
    let caller_stop = crate::caller_stop::bound();
    starts
        .par_iter()
        .zip(seeds.par_iter())
        .map(|(start, seed)| {
            let _caller_stop = caller_stop.clone().map(crate::caller_stop::Bind::new);
            search_binary(
                problem,
                Array1::from_vec(start.clone()).view(),
                budget,
                target,
                *seed,
                communicating,
            )
        })
        .collect()
}

/// Rigid TIP4P over many block-chart starts.
pub fn search_gmin_tip4p_batch(
    n_molecules: usize,
    starts: &[Vec<f64>],
    budget: usize,
    target: f64,
    seeds: &[u64],
    communicating: bool,
) -> Vec<StockSearch> {
    use rayon::prelude::*;
    assert_eq!(starts.len(), seeds.len());
    let caller_stop = crate::caller_stop::bound();
    starts
        .par_iter()
        .zip(seeds.par_iter())
        .map(|(start, seed)| {
            let _caller_stop = caller_stop.clone().map(crate::caller_stop::Bind::new);
            search_gmin_tip4p(
                n_molecules,
                Array1::from_vec(start.clone()).view(),
                budget,
                target,
                *seed,
                communicating,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::cluster_hopping::random_cluster_in_radius;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn a_lennard_jones_target_stops_before_the_budget() {
        let n = 6;
        let mut rng = StdRng::seed_from_u64(11);
        let start = random_cluster_in_radius(n, 2.0, 0.9, &mut rng);
        let problem = BinaryLennardJonesProblem {
            n_a: n,
            n_b: 0,
            epsilon_aa: 1.0,
            epsilon_ab: 1.0,
            epsilon_bb: 1.0,
            sigma_aa: 1.0,
            sigma_ab: 1.0,
            sigma_bb: 1.0,
        };
        // Equal diameters still carry two labels only when both counts are
        // positive. Here every atom is A, so the swap arm stays off and the
        // stop is the ledger target alone.
        let hit = search_binary(&problem, start.view(), 20_000, -1.0, 11, false);
        assert!(hit.success(-1.0), "energy {}", hit.energy);
        assert!(hit.charged < 20_000, "charged {}", hit.charged);
    }

    #[test]
    fn a_wide_binary_search_records_a_finite_minimum() {
        let n_a = 6;
        let n_b = 7;
        let mut rng = StdRng::seed_from_u64(19);
        let start = random_cluster_in_radius(n_a + n_b, 2.2, 0.9, &mut rng);
        let problem = BinaryLennardJonesProblem {
            n_a,
            n_b,
            epsilon_aa: 1.0,
            epsilon_ab: 1.0,
            epsilon_bb: 1.0,
            sigma_aa: 1.0,
            sigma_ab: 1.15,
            sigma_bb: 1.3,
        };
        let found = search_binary(&problem, start.view(), 6_000, f64::NEG_INFINITY, 19, false);
        assert!(
            found.energy.is_finite(),
            "quenched energy was not recorded ({})",
            found.energy
        );
        if cfg!(feature = "rgsaddle") {
            assert!(
                !found.nve_unavailable,
                "the feature build compiles the constant-energy escape"
            );
            assert!(
                found.energy < -15.0,
                "energy {} never left the separated limit",
                found.energy
            );
        } else {
            assert!(
                found.nve_unavailable,
                "a default build must report the missing escape instead of kicking"
            );
        }
        assert!(found.charged <= 6_000, "charged {}", found.charged);
    }

    #[test]
    fn stock_presets_keep_their_temperature_after_adaptation() {
        let recommended = Config::recommended(10);
        let water = recommended.clone().with_rigid_bodies(gmin_water_geometry());
        assert_eq!(recommended.temperature, water.temperature);
        assert!(recommended.allocate_moves && water.allocate_moves);
        assert!(water.surfaces.is_empty());
        assert!(water.move_library.is_rigid_body());
        assert!(!water.orbit_complete_on_new);

        let communicating = Config::communicating(10);
        assert_eq!(communicating.surfaces.len(), 1);
        let adapted = communicating.with_rigid_bodies(gmin_water_geometry());
        assert!(adapted.surfaces.is_empty());
        assert_eq!(adapted.temperature, Config::communicating(10).temperature);
        assert!(adapted.return_screen);
        assert!(!adapted.depth_reward);
    }

    /// A callback that raises the caller stop after a fixed number of
    /// potential calls has to get the binary search and the rigid TIP4P
    /// search back then, with the best state in hand and with no further
    /// calls. The public entries share that path, so a flag that is already
    /// raised returns before the budget is spent.
    #[test]
    fn caller_stop_ends_binary_and_tip4p() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::sync::{Arc, Mutex};
        use std::time::{Duration, Instant};

        use eindir_core::Objective;
        use eindir_core::gradient::{DifferentiableObjective, Gradient};
        use ndarray::{Array1, ArrayView1};

        const AFTER: usize = 40;
        const BUDGET: usize = 80_000;

        struct Halt<P> {
            inner: P,
            stop: Arc<AtomicBool>,
            calls: AtomicUsize,
            best: Mutex<f64>,
        }

        impl<P: DifferentiableObjective<f64>> Objective<f64> for Halt<P> {
            fn dim(&self) -> usize {
                Objective::dim(&self.inner)
            }
            fn bounds(&self) -> &eindir_core::Bounds<f64> {
                self.inner.bounds()
            }
            fn eval(&self, x: ArrayView1<f64>) -> f64 {
                self.value_and_gradient(x).0
            }
        }

        impl<P: DifferentiableObjective<f64>> Gradient<f64> for Halt<P> {
            fn dim(&self) -> usize {
                Objective::dim(self)
            }
            fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
                self.value_and_gradient(x).1
            }
        }

        impl<P: DifferentiableObjective<f64>> DifferentiableObjective<f64> for Halt<P> {
            fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
                let n = self.calls.fetch_add(1, Ordering::SeqCst);
                let (energy, gradient) = self.inner.value_and_gradient(x);
                let mut best = self.best.lock().expect("best");
                if energy < *best {
                    *best = energy;
                }
                if n + 1 >= AFTER {
                    self.stop.store(true, Ordering::SeqCst);
                }
                assert!(
                    n < AFTER + 8,
                    "the search kept calling the potential after the stop ({n} calls)"
                );
                (energy, gradient)
            }
        }

        let binary = BinaryLennardJonesProblem {
            n_a: 3,
            n_b: 3,
            epsilon_aa: 1.0,
            epsilon_ab: 1.0,
            epsilon_bb: 1.0,
            sigma_aa: 1.0,
            sigma_ab: 1.15,
            sigma_bb: 1.3,
        };
        let mut rng = StdRng::seed_from_u64(5);
        let binary_start = random_cluster_in_radius(6, 2.2, 0.9, &mut rng);
        let mut species = vec![1u32; binary.n_a];
        species.extend(std::iter::repeat_n(2u32, binary.n_b));
        let mut sigma = vec![binary.sigma_aa; binary.n_a];
        sigma.extend(std::iter::repeat_n(binary.sigma_bb, binary.n_b));
        let binary_cfg = stock_preset(6, false).with_species(species, Some(sigma));
        let binary_stop = Arc::new(AtomicBool::new(false));
        let binary_objective = Halt {
            inner: BinaryLennardJones::new(
                binary.n_a,
                binary.n_b,
                binary.epsilon_aa,
                binary.epsilon_ab,
                binary.epsilon_bb,
                binary.sigma_aa,
                binary.sigma_ab,
                binary.sigma_bb,
            ),
            stop: Arc::clone(&binary_stop),
            calls: AtomicUsize::new(0),
            best: Mutex::new(f64::INFINITY),
        };
        let binary_started = Instant::now();
        let binary_found = {
            let _caller_stop = crate::caller_stop::Bind::new(Arc::clone(&binary_stop));
            run_stock(
                &binary_objective,
                &binary_cfg,
                binary_start.view(),
                BUDGET,
                f64::NEG_INFINITY,
                5,
            )
        };
        let binary_calls = binary_objective.calls.load(Ordering::SeqCst);
        assert!(
            binary_calls <= AFTER + 8,
            "binary made {binary_calls} calls after a stop at {AFTER}"
        );
        assert!(
            binary_started.elapsed() < Duration::from_secs(2),
            "binary returned in {:?} with {binary_calls} calls and {} charged",
            binary_started.elapsed(),
            binary_found.charged
        );
        assert_eq!(binary_found.state.len(), binary_start.len());
        if binary_found.energy.is_finite() {
            let best_seen = *binary_objective.best.lock().expect("best");
            assert!(
                binary_found.energy <= best_seen + 1e-8,
                "binary returned {} above the best energy seen {best_seen}",
                binary_found.energy
            );
        }

        let mut water_rng = StdRng::seed_from_u64(7);
        let centres = random_cluster_in_radius(2, 4.0, 2.5, &mut water_rng);
        let mut water_start = Array1::zeros(12);
        for (i, value) in centres.iter().enumerate() {
            water_start[i] = *value;
        }
        let water_cfg = stock_preset(2, true).with_rigid_bodies(gmin_water_geometry());
        let water_stop = Arc::new(AtomicBool::new(false));
        let water_objective = Halt {
            inner: GminTip4p::new(2),
            stop: Arc::clone(&water_stop),
            calls: AtomicUsize::new(0),
            best: Mutex::new(f64::INFINITY),
        };
        let water_started = Instant::now();
        let water_found = {
            let _caller_stop = crate::caller_stop::Bind::new(Arc::clone(&water_stop));
            run_stock(
                &water_objective,
                &water_cfg,
                water_start.view(),
                BUDGET,
                f64::NEG_INFINITY,
                7,
            )
        };
        let water_calls = water_objective.calls.load(Ordering::SeqCst);
        assert!(
            water_calls <= AFTER + 8,
            "TIP4P made {water_calls} calls after a stop at {AFTER}"
        );
        assert!(
            water_started.elapsed() < Duration::from_secs(2),
            "TIP4P returned in {:?} with {water_calls} calls and {} charged",
            water_started.elapsed(),
            water_found.charged
        );
        assert_eq!(water_found.state.len(), water_start.len());
        if water_found.energy.is_finite() {
            let best_seen = *water_objective.best.lock().expect("best");
            assert!(
                water_found.energy <= best_seen + 1e-8,
                "TIP4P returned {} above the best energy seen {best_seen}",
                water_found.energy
            );
        }

        let raised = Arc::new(AtomicBool::new(true));
        let _caller_stop = crate::caller_stop::Bind::new(Arc::clone(&raised));
        let raised_started = Instant::now();
        let binary_entry = search_binary(
            &binary,
            binary_start.view(),
            BUDGET,
            f64::NEG_INFINITY,
            5,
            false,
        );
        let water_entry =
            search_gmin_tip4p(2, water_start.view(), BUDGET, f64::NEG_INFINITY, 7, true);
        assert!(
            raised_started.elapsed() < Duration::from_secs(2),
            "raised stop returned in {:?}",
            raised_started.elapsed()
        );
        assert_eq!(binary_entry.charged, 0, "binary charged after the stop");
        assert_eq!(water_entry.charged, 0, "TIP4P charged after the stop");
        assert_eq!(binary_entry.state.len(), binary_start.len());
        assert_eq!(water_entry.state.len(), water_start.len());
    }

    /// A stop flag that stays down must not change a stock search.
    #[test]
    fn an_idle_stop_does_not_change_stock_calls() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;

        use ndarray::Array1;

        let binary = BinaryLennardJonesProblem {
            n_a: 3,
            n_b: 2,
            epsilon_aa: 1.0,
            epsilon_ab: 1.0,
            epsilon_bb: 1.0,
            sigma_aa: 1.0,
            sigma_ab: 1.0,
            sigma_bb: 1.0,
        };
        let mut rng = StdRng::seed_from_u64(9);
        let start = random_cluster_in_radius(5, 2.0, 0.9, &mut rng);
        let plain = search_binary(&binary, start.view(), 80, f64::NEG_INFINITY, 9, false);
        let flagged = {
            let stop = Arc::new(AtomicBool::new(false));
            let _caller_stop = crate::caller_stop::Bind::new(stop);
            search_binary(&binary, start.view(), 80, f64::NEG_INFINITY, 9, false)
        };
        assert_eq!(plain.charged, flagged.charged);
        assert_eq!(plain.energy, flagged.energy);
        assert_eq!(plain.state, flagged.state);

        let mut water_rng = StdRng::seed_from_u64(4);
        let centres = random_cluster_in_radius(2, 4.0, 2.5, &mut water_rng);
        let mut water_start = Array1::zeros(12);
        for (i, value) in centres.iter().enumerate() {
            water_start[i] = *value;
        }
        let plain_water = search_gmin_tip4p(2, water_start.view(), 40, f64::NEG_INFINITY, 4, false);
        let flagged_water = {
            let stop = Arc::new(AtomicBool::new(false));
            let _caller_stop = crate::caller_stop::Bind::new(stop);
            search_gmin_tip4p(2, water_start.view(), 40, f64::NEG_INFINITY, 4, false)
        };
        assert_eq!(plain_water.charged, flagged_water.charged);
        assert_eq!(plain_water.energy, flagged_water.energy);
        assert_eq!(plain_water.state, flagged_water.state);
    }
}
