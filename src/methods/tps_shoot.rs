//! Transition-path / shooting moves adapted from pnastps (TYCeTS / PNAS 2019).
//!
//! Source algorithm (`pnastps-master/code/tps-only_pnas/main.cpp`):
//! 1. Maintain a path of `K+1` frames connecting reactant basin A to product
//!    basin B, distinguished by an order parameter with thresholds
//!    (`structOP_a`, `structOP_b`).
//! 2. **Shooting**: pick a random interior frame, re-propagate forward or
//!    backward (backward uses reversed velocities), accept if the trial path
//!    remains *reactive* (still connects A to B).
//! 3. **Shifting**: extend one end and drop the other while preserving
//!    reactivity.
//!
//! Continuous-box analogue for black-box optimization (no LAMMPS / MD):
//! - Path frames are points in the box; the path is a linear (or noisy)
//!   interpolation between two archived endpoints from distinct basins.
//! - Order parameter defaults to true objective value along the path
//!   (lower is better): basin A is "trapped / high", basin B is "good / low".
//!   Geometric reactivity is also available: endpoints stay near the
//!   archived A/B seeds.
//! - Shooting re-propagates half-paths by isotropic noise + optional local
//!   descent; accept iff the trial path is reactive (uniform prior on the
//!   reactive ensemble, matching pnastps).
//! - Every evaluated frame is charged to the true objective; the best true-F
//!   point found on accepted or trial paths is available to the caller for
//!   incumbent updates.
//!
//! Pure functions below have no I/O and are unit-tested without CUTEst.

use ndarray::{Array1, ArrayView1};
use std::cmp::Ordering;

use rand::Rng;

/// Linear path of `n_frames` points from `x_a` to `x_b` (inclusive).
pub fn linear_path(
    x_a: ArrayView1<f64>,
    x_b: ArrayView1<f64>,
    n_frames: usize,
) -> Vec<Array1<f64>> {
    assert!(n_frames >= 2, "path needs at least two frames");
    assert_eq!(x_a.len(), x_b.len());
    let mut path = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let t = i as f64 / (n_frames - 1) as f64;
        let mut x = Array1::zeros(x_a.len());
        for d in 0..x_a.len() {
            x[d] = (1.0 - t) * x_a[d] + t * x_b[d];
        }
        path.push(x);
    }
    path
}

/// Classical TPS reactivity on a scalar order-parameter series.
///
/// Matches pnastps: start in reactant (`op[0] <= a`) and end in product
/// (`op[last] >= b`) with `a < b`. For objective-as-OP where lower is better,
/// pass negated values or use [`path_reactive_objective`].
pub fn path_is_reactive(ops: &[f64], basin_a_max: f64, basin_b_min: f64) -> bool {
    if ops.len() < 2 || basin_a_max.partial_cmp(&basin_b_min) != Some(Ordering::Less) {
        return false;
    }
    let first = ops[0];
    let last = ops[ops.len() - 1];
    first.is_finite() && last.is_finite() && first <= basin_a_max && last >= basin_b_min
}

/// Reactivity when the order parameter is the objective (lower is better).
///
/// Path starts "bad" (`ops[0] >= a`) and ends "good" (`ops[last] <= b`)
/// with `a > b` (high-to-low).
pub fn path_reactive_objective(ops: &[f64], high_threshold: f64, low_threshold: f64) -> bool {
    if ops.len() < 2 || high_threshold.partial_cmp(&low_threshold) != Some(Ordering::Greater) {
        return false;
    }
    let first = ops[0];
    let last = ops[ops.len() - 1];
    first.is_finite() && last.is_finite() && first >= high_threshold && last <= low_threshold
}

/// Geometric reactivity: endpoints stay near the archived seeds.
pub fn path_reactive_geometric(
    path: &[Array1<f64>],
    x_a: ArrayView1<f64>,
    x_b: ArrayView1<f64>,
    tol: f64,
) -> bool {
    if path.len() < 2 || tol.partial_cmp(&0.0) != Some(Ordering::Greater) {
        return false;
    }
    let d0 = l2_dist(path[0].view(), x_a);
    let d1 = l2_dist(path[path.len() - 1].view(), x_b);
    d0 <= tol && d1 <= tol
}

fn l2_dist(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(u, v)| (u - v).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Pick a uniform interior shooting index in `1..K-1` for a path of length
/// `K+1` frames (pnastps `random_shootpoint(1, K-1)`).
pub fn pick_shoot_index<R: Rng + ?Sized>(n_frames: usize, rng: &mut R) -> usize {
    assert!(n_frames >= 3, "shooting needs an interior frame");
    rng.random_range(1..n_frames - 1)
}

/// Shooting direction: forward keeps the left half and re-propagates right;
/// backward keeps the right half and re-propagates left (pnastps velocity
/// reversal is represented as re-propagating the opposite segment).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShootDirection {
    /// Re-propagate frames `shoot..end`.
    Forward,
    /// Re-propagate frames `0..shoot`.
    Backward,
}

/// Uniform random shoot direction.
pub fn pick_shoot_direction<R: Rng + ?Sized>(rng: &mut R) -> ShootDirection {
    if rng.random::<bool>() {
        ShootDirection::Forward
    } else {
        ShootDirection::Backward
    }
}

/// Apply a shooting move: keep one half of the path, replace the other half
/// by linear segments from a noisy shoot point toward the free endpoint seed.
///
/// `noise_scale` is absolute coordinate noise at the shoot point. `reflect`
/// maps a point into the feasible box (caller supplies reflection).
#[allow(clippy::too_many_arguments)]
pub fn apply_shoot<R, F>(
    path: &[Array1<f64>],
    shoot: usize,
    direction: ShootDirection,
    x_a: ArrayView1<f64>,
    x_b: ArrayView1<f64>,
    noise_scale: f64,
    rng: &mut R,
    mut reflect: F,
) -> Vec<Array1<f64>>
where
    R: Rng + ?Sized,
    F: FnMut(Array1<f64>) -> Array1<f64>,
{
    let n = path.len();
    assert!(shoot > 0 && shoot < n - 1);
    let dim = path[0].len();
    let mut trial = path.to_vec();
    // Noisy shoot configuration.
    let mut x_s = path[shoot].clone();
    for d in 0..dim {
        let z: f64 = rng.sample(rand_distr::StandardNormal);
        x_s[d] += noise_scale * z;
    }
    x_s = reflect(x_s);
    trial[shoot] = x_s.clone();

    match direction {
        ShootDirection::Forward => {
            // Keep [0..shoot], re-propagate (shoot+1)..end toward x_b.
            let steps = n - 1 - shoot;
            for j in 1..=steps {
                let t = j as f64 / steps as f64;
                let mut x = Array1::zeros(dim);
                for d in 0..dim {
                    x[d] = (1.0 - t) * x_s[d] + t * x_b[d];
                }
                trial[shoot + j] = reflect(x);
            }
        }
        ShootDirection::Backward => {
            // Keep [shoot..end], re-propagate 0..shoot-1 toward x_a.
            let steps = shoot;
            for j in 1..=steps {
                let t = j as f64 / steps as f64;
                let mut x = Array1::zeros(dim);
                for d in 0..dim {
                    x[d] = (1.0 - t) * x_s[d] + t * x_a[d];
                }
                trial[shoot - j] = reflect(x);
            }
        }
    }
    trial
}

/// TPS accept rule for shooting under a flat prior on reactive paths:
/// accept the trial if and only if it is reactive (pnastps forward/backward
/// shoot). Non-reactive trials are rejected with probability 1.
pub fn accept_reactive_shoot(trial_reactive: bool) -> bool {
    trial_reactive
}

/// Index of the best (lowest) finite order-parameter frame.
pub fn best_frame_index(ops: &[f64]) -> Option<usize> {
    ops.iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn linear_path_endpoints_match_seeds() {
        let a = array![0.0, 0.0];
        let b = array![1.0, 2.0];
        let path = linear_path(a.view(), b.view(), 5);
        assert_eq!(path.len(), 5);
        assert!((path[0][0] - 0.0).abs() < 1e-15);
        assert!((path[4][1] - 2.0).abs() < 1e-15);
    }

    #[test]
    fn classical_reactivity_matches_pnastps_thresholds() {
        // op rises from reactant (< a) to product (> b).
        let ops = vec![0.1, 0.5, 0.9, 1.5];
        assert!(path_is_reactive(&ops, 0.2, 1.0));
        assert!(!path_is_reactive(&ops, 0.05, 1.0)); // start not in A
        assert!(!path_is_reactive(&ops, 0.2, 2.0)); // end not in B
    }

    #[test]
    fn objective_reactivity_high_to_low() {
        let ops = vec![10.0, 5.0, 1.0, 0.1];
        assert!(path_reactive_objective(&ops, 8.0, 0.5));
        assert!(!path_reactive_objective(&ops, 12.0, 0.5));
    }

    #[test]
    fn shoot_accepts_only_reactive_trials() {
        assert!(accept_reactive_shoot(true));
        assert!(!accept_reactive_shoot(false));
    }

    #[test]
    fn shooting_preserves_length_and_can_stay_reactive() {
        let a = array![-2.0, 0.0];
        let b = array![2.0, 0.0];
        let path = linear_path(a.view(), b.view(), 9);
        let mut rng = StdRng::seed_from_u64(42);
        let shoot = pick_shoot_index(path.len(), &mut rng);
        let dir = ShootDirection::Forward;
        let reflect = |x: Array1<f64>| Array1::from_iter(x.iter().map(|v| v.clamp(-5.0, 5.0)));
        let trial = apply_shoot(
            &path,
            shoot,
            dir,
            a.view(),
            b.view(),
            0.05,
            &mut rng,
            reflect,
        );
        assert_eq!(trial.len(), path.len());
        // Geometric reactivity with generous tol: endpoints near seeds.
        assert!(path_reactive_geometric(&trial, a.view(), b.view(), 1.0));
    }

    #[test]
    fn best_frame_finds_minimum_op() {
        let ops = vec![3.0, 1.5, 0.2, 4.0];
        assert_eq!(best_frame_index(&ops), Some(2));
    }
}
