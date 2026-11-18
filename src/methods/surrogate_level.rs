//! Two-level search: hop on a cheap rigid water model, polish on the caller.
//!
//! CREST screens with GFN-FF before a GFN2 optimisation. This crate does
//! not ship that Hamiltonian. For a detected water cluster the cheap model
//! is TIP4P, which has the same rigid cage and costs no GFN2 evaluation.
//! The caller's ledger is charged only for the polish.

use std::time::Instant;

use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::anc_quench::anc_minimize;
use super::cluster_hopping::Ledger;
use super::warm_lbfgs::WarmLbfgs;
use crate::potentials::Tip4pCluster;
use eindir_core::gradient::DifferentiableObjective;

/// What the cheap level did, kept apart from the charged polish.
#[derive(Debug, Clone)]
pub struct SurrogatePolish {
    /// Evaluations of the cheap model.
    pub calls: usize,
    /// Wall time of the cheap hop. The charged polish is on the caller's clock.
    pub wall_s: f64,
    /// Surrogate minima that received a charged polish.
    pub polished: usize,
    /// Lowest polished energy, when one finished.
    pub best_energy: f64,
    /// Cartesian geometry of that energy.
    pub best_state: Option<Array1<f64>>,
}

/// Hop on TIP4P from `start`, then polish the `keep` lowest distinct cages.
///
/// `start` is Cartesian, three sites per molecule, oxygen first. Surrogate
/// evaluations are not charged to `ledger`. The polish is, and it stops
/// when `ledger` does.
pub fn polish_tip4p_minima<O>(
    objective: &O,
    ledger: &mut Ledger,
    start: ArrayView1<f64>,
    n_molecules: usize,
    keep: usize,
    seed: u64,
    gtol: f64,
    polish_steps: usize,
) -> SurrogatePolish
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let wall = Instant::now();
    let (mut pool, calls) = tip4p_hop(start, n_molecules, seed, keep.max(1));
    let surrogate_wall = wall.elapsed().as_secs_f64();
    pool.sort_by(|left, right| left.0.total_cmp(&right.0));
    pool.truncate(keep.max(1));
    let z: Vec<i32> = (0..n_molecules).flat_map(|_| [8, 1, 1]).collect();
    let mut best_energy = f64::INFINITY;
    let mut best_state = None;
    let mut polished = 0usize;
    for (_, cartesian) in pool {
        if ledger.remaining() == 0 {
            break;
        }
        let (energy, state, _) = anc_minimize(cartesian.view(), &z, polish_steps, gtol, |point| {
            if !ledger.charge() {
                return None;
            }
            let (value, gradient) = objective.value_and_gradient(point);
            value.is_finite().then_some((value, gradient))
        });
        polished += 1;
        if energy < best_energy && state.iter().all(|value| value.is_finite()) {
            best_energy = energy;
            best_state = Some(state);
        }
    }
    SurrogatePolish {
        calls,
        wall_s: surrogate_wall,
        polished,
        best_energy,
        best_state,
    }
}

fn tip4p_hop(
    start: ArrayView1<f64>,
    n: usize,
    seed: u64,
    keep: usize,
) -> (Vec<(f64, Array1<f64>)>, usize) {
    let pot = Tip4pCluster::new(n);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = rigid_from_oxygens(start, n);
    let mut calls = 0usize;
    let mut pool: Vec<(f64, Array1<f64>)> = Vec::new();
    for trial in 0..16 {
        let (energy, quenched, used) = quench_tip4p(&pot, &state);
        calls += used;
        if energy.is_finite() {
            let cartesian = cartesian_from_tip4p(&pot, quenched.view());
            remember(&mut pool, energy, cartesian, keep);
            state = quenched;
        }
        if trial + 1 == 16 {
            break;
        }
        let molecule = rng.random_range(0..n);
        let mut centre = [0.0; 3];
        for other in 0..n {
            for axis in 0..3 {
                centre[axis] += state[3 * other + axis];
            }
        }
        let scale = n as f64;
        for axis in 0..3 {
            centre[axis] /= scale;
            state[3 * molecule + axis] = centre[axis] + 2.6 * rng.random_range(-1.0..1.0);
            state[3 * n + 3 * molecule + axis] = rng.random_range(-1.5..1.5);
        }
        if let Some(slice) = state.as_slice_mut() {
            pot.fold_rotations(slice);
        }
    }
    (pool, calls)
}

fn quench_tip4p(pot: &Tip4pCluster, start: &Array1<f64>) -> (f64, Array1<f64>, usize) {
    let mut opt = WarmLbfgs::default();
    opt.gtol = 1.0e-3;
    opt.max_line_evals = 8;
    let (energy, mut state, calls) = opt.minimize(start.view(), 30, |point| {
        let (value, gradient) = pot.value_and_gradient(point);
        value.is_finite().then_some((value, gradient))
    });
    if let Some(slice) = state.as_slice_mut() {
        pot.fold_rotations(slice);
    }
    (energy, state, calls)
}

fn remember(pool: &mut Vec<(f64, Array1<f64>)>, energy: f64, cartesian: Array1<f64>, keep: usize) {
    let oxygen = oxygen_key(cartesian.view());
    if pool.iter().any(|(_, other)| {
        let delta = oxygen_distance(&oxygen, &oxygen_key(other.view()));
        delta < 0.35
    }) {
        return;
    }
    pool.push((energy, cartesian));
    pool.sort_by(|left, right| left.0.total_cmp(&right.0));
    pool.truncate(keep.max(1));
}

fn rigid_from_oxygens(start: ArrayView1<f64>, n: usize) -> Array1<f64> {
    let mut state = Array1::zeros(6 * n);
    for molecule in 0..n {
        let oxygen = 9 * molecule;
        if oxygen + 2 >= start.len() {
            break;
        }
        for axis in 0..3 {
            state[3 * molecule + axis] = start[oxygen + axis];
        }
    }
    state
}

fn cartesian_from_tip4p(pot: &Tip4pCluster, state: ArrayView1<f64>) -> Array1<f64> {
    let n = pot.n_molecules;
    let mut coords = Array1::zeros(9 * n);
    for molecule in 0..n {
        let sites = pot.molecule_sites(state, molecule);
        for site in 0..3 {
            for axis in 0..3 {
                coords[9 * molecule + 3 * site + axis] = sites[site][axis];
            }
        }
    }
    coords
}

fn oxygen_key(coords: ArrayView1<f64>) -> Vec<f64> {
    let n = coords.len() / 9;
    let mut distances = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let mut square = 0.0;
            for axis in 0..3 {
                let delta = coords[9 * i + axis] - coords[9 * j + axis];
                square += delta * delta;
            }
            distances.push(square.sqrt());
        }
    }
    distances.sort_by(|left, right| left.total_cmp(right));
    distances
}

fn oxygen_distance(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.is_empty() {
        return f64::INFINITY;
    }
    let mut acc = 0.0;
    for (a, b) in left.iter().zip(right) {
        let d = a - b;
        acc += d * d;
    }
    (acc / left.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tip4p_hop_returns_one_cage_per_kept_minimum() {
        let n = 8;
        let start = Array1::zeros(9 * n);
        let (pool, calls) = tip4p_hop(start.view(), n, 3, 3);
        assert!(calls > 0, "the surrogate spent nothing");
        assert!(!pool.is_empty());
        assert!(pool.len() <= 3);
        for (_, coords) in &pool {
            assert_eq!(coords.len(), 9 * n);
            assert!(coords.iter().all(|value| value.is_finite()));
        }
    }
}
