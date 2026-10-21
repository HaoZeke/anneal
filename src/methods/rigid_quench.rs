//! Rigid-body L-BFGS with the step GMIN takes inside a basin hop.
//!
//! The hop quench stops on the root-mean-square force. Four pairs are
//! kept. The first diagonal of the inverse Hessian is 0.1. The Euclidean
//! step is capped. A trial whose energy rises by more than the allowed
//! amount is shortened by ten and tried again, and there is no Wolfe
//! line search. A second pass uses the tight tolerance and runs only
//! when the first pass missed it and the energy is a new lowest minimum.
//!
//! The ranked TIP4P keyword file sets `SLOPPYCONV` and `TIGHTCONV` both
//! to `1e-3`, `MAXBFGS` to `0.5`, and `MAXERISE` to `1e-4`. It does not
//! set `MUPDATE` or `DGUESS`, so those stay at the source defaults of 4
//! and 0.1. The source default `MAXBFGS` is 0.4 and the source default
//! `MAXERISE` is `1e-10` when the keyword file leaves them unset. The
//! source applies its default tight tolerance of `1e-10` only in the
//! final quench of the saved list. The ranked file sets the two
//! tolerances equal, so the second pass is the same RMS and exists to
//! finish a quench that stopped on the iteration cap.

use ndarray::{Array1, ArrayView1};
use std::collections::VecDeque;

/// Pairs retained. Source `MUPDATE`.
pub(crate) const MEMORY: usize = 4;
/// Initial diagonal of the inverse Hessian. Source `DGUESS`.
pub(crate) const DGUESS: f64 = 0.1;
/// Longest Euclidean step. Ranked `MAXBFGS`.
pub(crate) const MAX_STEP: f64 = 0.5;
/// Largest energy rise that still accepts a step. Ranked `MAXERISE`.
pub(crate) const MAX_RISE: f64 = 1.0e-4;
/// RMS force for a Metropolis hop. Ranked `SLOPPYCONV` and source `BQMAX`.
pub(crate) const SLOPPY_RMS: f64 = 1.0e-3;
/// RMS force for a second pass on a new lowest minimum. Ranked `TIGHTCONV`.
pub(crate) const TIGHT_RMS: f64 = 1.0e-3;
/// Accepted steps a full quench may take. Ranked `MAXIT`.
pub(crate) const MAX_ITERATIONS: usize = 10_000;
/// Hops without a new lowest minimum before a rigid walker is replaced.
///
/// GMIN's reseed predicate is `J1 - JBEST > NRELAX`. The compiled
/// `NRELAX` is 0, which would reseed on every hop that does not lower
/// the best, and the ranked TIP4P file leaves `NEWRESTART` off. The
/// only positive MC-cycle count initialised beside a restart in
/// `keywords.f90` is `DUMPINT = 1000`: `mc.f` writes a restart file
/// every 1000 cycles. The reseed waits that many hops. It is not
/// `NRELAX`.
pub(crate) const STAGNATION_HOPS: usize = 1000;
/// Failed reseeds tried before the walker is left where it is.
/// `NEWRESATTEMPTS` stops after ten.
pub(crate) const RESEED_ATTEMPTS: usize = 10;

/// True when `hops` since the last new lowest minimum exceeds [`STAGNATION_HOPS`].
pub(crate) fn stagnation_restart(hops: usize, hops_at_lowest: usize) -> bool {
    hops.saturating_sub(hops_at_lowest) > STAGNATION_HOPS
}

/// Root-mean-square force, `sqrt(g · g / n)`.
pub(crate) fn rms_force(gradient: ArrayView1<f64>) -> f64 {
    let n = gradient.len();
    if n == 0 || gradient.iter().any(|value| !value.is_finite()) {
        return f64::INFINITY;
    }
    (gradient.dot(&gradient) / n as f64).sqrt()
}

/// Whether a rigid quench is tight enough to record.
pub(crate) fn rms_is_converged(
    gradient: ArrayView1<f64>,
    dimensions: usize,
    tolerance: f64,
) -> bool {
    gradient.len() == dimensions && dimensions > 0 && rms_force(gradient) <= tolerance
}

/// Shortens `stp` so `stp * direction_length` does not exceed `max_step`.
pub(crate) fn cap_step(stp: f64, direction_length: f64, max_step: f64) -> f64 {
    if direction_length > 0.0 && stp * direction_length > max_step {
        max_step / direction_length
    } else {
        stp
    }
}

/// Point reached by one rigid quench.
#[derive(Debug, Clone)]
pub(crate) struct RigidDescent {
    /// Energy at `point`.
    pub energy: f64,
    /// Coordinates at the end of the quench.
    pub point: Array1<f64>,
    /// Whether the RMS force is at or below the tolerance.
    pub converged: bool,
    /// Accepted steps. Rejections are potential calls that are not steps.
    ///
    /// Read by the quench tests. A campaign charges the ledger instead.
    #[allow(dead_code)]
    pub accepted: usize,
}

struct Pair {
    s: Array1<f64>,
    y: Array1<f64>,
    rho: f64,
}

fn euclidean(v: ArrayView1<f64>) -> f64 {
    v.dot(&v).sqrt()
}

fn search_direction(pairs: &VecDeque<Pair>, diag: f64, gradient: &Array1<f64>) -> Array1<f64> {
    if pairs.is_empty() {
        return gradient.mapv(|value| -diag * value);
    }
    let mut q = gradient.mapv(|value| -value);
    let mut alpha = vec![0.0; pairs.len()];
    for (index, pair) in pairs.iter().enumerate().rev() {
        let a = pair.rho * pair.s.dot(&q);
        alpha[index] = a;
        q.scaled_add(-a, &pair.y);
    }
    q.mapv_inplace(|value| {
        let scaled = diag * value;
        if scaled.abs() > 100.0 {
            scaled.signum() * 100.0
        } else {
            scaled
        }
    });
    for (index, pair) in pairs.iter().enumerate() {
        let beta = alpha[index] - pair.rho * pair.y.dot(&q);
        q.scaled_add(beta, &pair.s);
    }
    q
}

fn store_pair(pairs: &mut VecDeque<Pair>, diag: &mut f64, step: Array1<f64>, mut y: Array1<f64>) {
    for value in y.iter_mut() {
        if value.abs() > 1.0e10 {
            *value = value.signum() * 1.0e10;
        }
    }
    let mut ys = step.dot(&y);
    let mut yy = y.dot(&y);
    if !ys.is_finite() || !yy.is_finite() || ys <= 0.0 || yy <= 0.0 {
        return;
    }
    if ys.abs() < 1.0e-50 {
        ys = ys.signum() * 1.0e-50;
    }
    if yy.abs() < 1.0e-50 {
        yy = yy.signum() * 1.0e-50;
    }
    *diag = ys / yy;
    pairs.push_back(Pair {
        s: step,
        y,
        rho: 1.0 / ys,
    });
    while pairs.len() > MEMORY {
        pairs.pop_front();
    }
}

/// Minimises from `x0` with the rigid step.
///
/// `fg` returns `None` when the caller is out of budget. Every call `fg`
/// makes is one potential evaluation. `max_iter` counts accepted steps.
pub(crate) fn rigid_lbfgs<F>(
    x0: ArrayView1<f64>,
    max_iter: usize,
    rms_tol: f64,
    mut fg: F,
) -> RigidDescent
where
    F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
{
    let mut x = x0.to_owned();
    let Some((mut energy, mut gradient)) = fg(x.view()) else {
        return RigidDescent {
            energy: f64::INFINITY,
            point: x,
            converged: false,
            accepted: 0,
        };
    };
    let converged_now = rms_force(gradient.view()) <= rms_tol;
    if !energy.is_finite()
        || gradient.iter().any(|value| !value.is_finite())
        || converged_now
        || max_iter == 0
    {
        return RigidDescent {
            energy,
            point: x,
            converged: converged_now && energy.is_finite(),
            accepted: 0,
        };
    }

    let mut pairs: VecDeque<Pair> = VecDeque::new();
    let mut diag = DGUESS;
    let mut accepted = 0usize;
    while accepted < max_iter {
        if rms_force(gradient.view()) <= rms_tol {
            return RigidDescent {
                energy,
                point: x,
                converged: true,
                accepted,
            };
        }
        let mut direction = search_direction(&pairs, diag, &gradient);
        if direction.dot(&gradient) > 0.0 {
            direction.mapv_inplace(|value| -value);
        }
        let length = euclidean(direction.view());
        if !length.is_finite() || length < 1.0e-30 {
            break;
        }
        let gnorm = euclidean(gradient.view());
        let mut scale = if pairs.is_empty() {
            if !gnorm.is_finite() || gnorm < 1.0e-30 {
                break;
            }
            gnorm.min(1.0 / gnorm)
        } else {
            1.0
        };
        scale = cap_step(scale, length, MAX_STEP);
        let mut taken = false;
        // The first trial uses the proposed scale. Each rejection
        // shortens it by ten. The source gives up once the rejection
        // count passes ten, which is twelve evaluations of this step.
        for _ in 0..12 {
            let mut trial = x.clone();
            trial.scaled_add(scale, &direction);
            let Some((next_energy, next_gradient)) = fg(trial.view()) else {
                return RigidDescent {
                    energy,
                    point: x,
                    converged: rms_force(gradient.view()) <= rms_tol,
                    accepted,
                };
            };
            let delta = next_energy - energy;
            let usable = next_energy.is_finite()
                && delta.is_finite()
                && next_gradient.iter().all(|value| value.is_finite())
                && delta <= MAX_RISE;
            if usable {
                let mut step = direction.clone();
                step *= scale;
                let mut y = next_gradient.clone();
                y -= &gradient;
                store_pair(&mut pairs, &mut diag, step, y);
                x = trial;
                energy = next_energy;
                gradient = next_gradient;
                accepted += 1;
                taken = true;
                break;
            }
            scale *= 0.1;
        }
        if !taken {
            break;
        }
    }
    RigidDescent {
        energy,
        point: x,
        converged: rms_force(gradient.view()) <= rms_tol && energy.is_finite(),
        accepted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn quadratic(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let mut energy = 0.0;
        let mut gradient = Array1::zeros(x.len());
        for i in 0..x.len() {
            energy += 0.5 * x[i] * x[i];
            gradient[i] = x[i];
        }
        (energy, gradient)
    }

    #[test]
    fn a_zero_iteration_call_evaluates_once_and_does_not_step() {
        let x0 = Array1::from_vec(vec![0.4, -0.2]);
        let mut calls = 0usize;
        let descent = rigid_lbfgs(x0.view(), 0, SLOPPY_RMS, |v| {
            calls += 1;
            assert_eq!(v, x0.view());
            Some(quadratic(v))
        });
        assert_eq!(calls, 1);
        assert!(!descent.converged);
        assert!((descent.energy - 0.1).abs() < 1e-12);
    }

    #[test]
    fn the_second_quadratic_step_uses_the_curvature() {
        let x0 = Array1::from_vec(vec![1.0, -2.0, 0.5]);
        let mut norms = Vec::new();
        let _ = rigid_lbfgs(x0.view(), 3, 1.0e-12, |v| {
            norms.push(euclidean(v));
            Some(quadratic(v))
        });
        assert!(norms.len() >= 3, "saw {norms:?}");
        let first = norms[0] - norms[1];
        let second = norms[1] - norms[2];
        assert!(
            second > first * 2.0,
            "curvature did not lengthen the step: first {first}, second {second}, norms {norms:?}"
        );
    }

    #[test]
    fn the_sum_of_squares_stops_on_the_rms_force() {
        let x0 = Array1::from_elem(8, 1.0);
        let mut calls = 0usize;
        let descent = rigid_lbfgs(x0.view(), 80, SLOPPY_RMS, |v| {
            calls += 1;
            Some(quadratic(v))
        });
        assert!(descent.converged, "energy {}", descent.energy);
        assert!(rms_force(descent.point.view()) <= SLOPPY_RMS);
        assert!(calls < 80, "spent {calls} evaluations");
    }

    #[test]
    fn a_flat_quadratic_cannot_step_past_the_cap() {
        let x0 = Array1::from_vec(vec![1.0e6]);
        let mut previous = x0.clone();
        let mut longest = 0.0_f64;
        let _ = rigid_lbfgs(x0.view(), 2, SLOPPY_RMS, |v| {
            let moved = euclidean((&v.to_owned() - &previous).view());
            longest = longest.max(moved);
            previous = v.to_owned();
            let x = v[0];
            Some((0.5e-6 * x * x, Array1::from_vec(vec![1.0e-6 * x])))
        });
        assert!(
            longest <= MAX_STEP + 1.0e-8,
            "a trial moved {longest}, past {MAX_STEP}"
        );
        assert!(
            longest > 0.1,
            "the cap was never asked to bind, longest {longest}"
        );
    }

    #[test]
    fn the_first_step_uses_the_initial_diagonal() {
        let x0 = Array1::from_vec(vec![1.0]);
        let mut seen = Vec::new();
        let _ = rigid_lbfgs(x0.view(), 1, 1.0e-12, |v| {
            seen.push(v[0]);
            Some(quadratic(v))
        });
        assert_eq!(seen.len(), 2, "the step made {seen:?}");
        assert!(
            (seen[1] - 0.9).abs() < 1e-12,
            "first step landed at {}",
            seen[1]
        );
    }

    #[test]
    fn stagnation_waits_for_the_dump_interval() {
        assert_eq!(STAGNATION_HOPS, 1000);
        assert!(!stagnation_restart(1000, 0));
        assert!(stagnation_restart(1001, 0));
        assert!(!stagnation_restart(1500, 500));
        assert!(stagnation_restart(1501, 500));
    }

    #[test]
    fn the_step_cap_scales_a_long_direction() {
        let capped = cap_step(1.0, 4.0, MAX_STEP);
        assert!((capped - 0.125).abs() < 1e-12);
        assert_eq!(cap_step(0.2, 1.0, MAX_STEP), 0.2);
    }

    #[test]
    fn a_fifteen_water_draw_reaches_the_hop_tolerance() {
        use crate::potentials::{GminTip4p, random_tip4p_cluster as random_cluster};
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let pot = GminTip4p::new(15);
        let mut rng = StdRng::seed_from_u64(1);
        let x0 = random_cluster(15, &mut rng);
        let mut calls = 0usize;
        let descent = rigid_lbfgs(x0.view(), MAX_ITERATIONS, SLOPPY_RMS, |v| {
            calls += 1;
            Some(pot.value_and_gradient(v))
        });
        assert!(
            descent.converged,
            "cold start energy {} after {calls} evaluations",
            descent.energy
        );
        assert!(calls < 800, "cold start spent {calls} evaluations");
        let n = 15usize;
        for kick in [0.05_f64, 0.2, 0.5] {
            let mut kicked = descent.point.clone();
            for i in 0..n {
                kicked[3 * n + 3 * i] += kick;
            }
            let mut hop_calls = 0usize;
            let hopped = rigid_lbfgs(kicked.view(), MAX_ITERATIONS, SLOPPY_RMS, |v| {
                hop_calls += 1;
                Some(pot.value_and_gradient(v))
            });
            assert!(
                hopped.converged,
                "kick {kick} energy {} after {hop_calls}",
                hopped.energy
            );
            let limit = if kick < 0.1 {
                120
            } else if kick < 0.3 {
                160
            } else {
                600
            };
            assert!(
                hop_calls < limit,
                "kick {kick} spent {hop_calls} evaluations"
            );
        }
        assert!(descent.accepted > 0);
    }
}
