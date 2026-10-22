//! Binary Lennard-Jones and rigid TIP4P on the stock cluster presets.
//!
//! [`Config::recommended`] and [`Config::communicating`] stay numerically
//! unchanged. Species labels and a rigid chart are applied afterwards.
//! A wide diameter ratio escapes by one softened microcanonical step across
//! a barrier and tries one downhill unlike-pair exchange on a side
//! generator. A narrow ratio exchanges mismatched labels. Water takes
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
    let mut ledger = Ledger::new(budget);
    if target.is_finite() {
        ledger.stop_below(target);
    }
    let _ = search_from(potential, cfg, &mut ledger, start, seed);
    let state = ledger
        .best_state
        .clone()
        .map(|value| value.to_vec())
        .unwrap_or_else(|| start.to_vec());
    StockSearch {
        energy: ledger.best,
        charged: ledger.spent(),
        state,
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
    starts
        .par_iter()
        .zip(seeds.par_iter())
        .map(|(start, seed)| {
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
    starts
        .par_iter()
        .zip(seeds.par_iter())
        .map(|(start, seed)| {
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
        assert!(
            found.energy < -15.0,
            "energy {} never left the separated limit",
            found.energy
        );
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
}
