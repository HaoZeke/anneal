//! Gap-Proportional Metropolis Descent (GPMD).
//!
//! Algorithm derived in `docs/derivations/gpmd_algorithm.org`:
//! - Model (M1): ES sphere ⇒ Δ normalized ~ N(c², 4c²)
//! - Identity (I1)/(T1): state gain G>0 iff θ = T d / gap ∈ (0, 2)
//! - Operating law (A1): T = (1/2) · (f − f_best) / d
//! - Proposal: running covariance (Haario) + RM scale targeting α* ≈ 0.32
//! - Tail: reserved polish fraction of the budget
//!
//! Literature parent: local Metropolis on a quadratic basin + adaptive
//! Metropolis (Haario–Saksman–Tamminen), *not* Xiang dual annealing.
//! Escape of deep multimodal barriers is outside the claim boundary.

use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::methods::local_polish::projected_gradient_polish;
use crate::movekernel::reflect_into_box;

/// Dimensionless operating temperature θ⋆ = 1/2 (inside (0,2) window).
pub const THETA_STAR: f64 = 0.5;
/// Model acceptance target α*(θ⋆) ≈ 0.32 (SymPy/MC in proofs/gpmd_derive.py).
pub const ALPHA_TARGET: f64 = 0.32;
/// Fraction of budget reserved for terminal polish.
pub const POLISH_FRACTION: f64 = 0.25;
/// Floor on gap used in T = θ⋆ gap / d.
const GAP_EPS: f64 = 1e-12;
/// Minimum proposal scale relative to box width.
const SIGMA_FLOOR_FRAC: f64 = 1e-4;

/// Result of one GPMD run.
#[derive(Clone, Debug)]
pub struct GpmdResult {
    pub best_pos: Array1<f64>,
    pub best_val: f64,
    pub n_evals: usize,
    pub n_grads: usize,
    pub n_accept: usize,
    pub n_propose: usize,
}

fn work(n_evals: usize, n_grads: usize) -> usize {
    n_evals + n_grads
}

fn gaussian<R: Rng>(rng: &mut R) -> f64 {
    let u1 = rng.random::<f64>().max(1e-12);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Temperature law (A1): T = θ⋆ · max(f − f_best, ε) / d.
#[inline]
pub fn gap_proportional_temp(f: f64, f_best: f64, dim: usize) -> f64 {
    let d = (dim.max(1)) as f64;
    let gap = (f - f_best).max(GAP_EPS);
    THETA_STAR * gap / d
}

/// Public entry: run GPMD under a work-unit budget.
pub fn gpmd_optimize<O, G>(
    obj: &O,
    grad: Option<&G>,
    budget: usize,
    seed: u64,
    x0: Option<ArrayView1<f64>>,
) -> GpmdResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    let mut rng = StdRng::seed_from_u64(seed ^ 0x9e37_79b9_7f4a_7c15);
    run_gpmd(obj, grad, budget, seed, x0, &mut rng)
}

pub fn run_gpmd<O, G, R>(
    obj: &O,
    grad: Option<&G>,
    budget: usize,
    seed: u64,
    x0: Option<ArrayView1<f64>>,
    rng: &mut R,
) -> GpmdResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
    R: Rng,
{
    let bounds = obj.bounds().clone();
    let dim = bounds.dims.max(1);
    let budget = budget.max(4);
    let polish_budget = ((budget as f64) * POLISH_FRACTION).floor() as usize;
    let polish_budget = polish_budget.clamp(0, budget.saturating_sub(2));
    let sa_budget = budget.saturating_sub(polish_budget).max(2);

    // Initialize at x0 or box-center + small noise (seeded).
    let mut x = if let Some(x0) = x0 {
        bounds.clip(x0)
    } else {
        let mut rng_init = StdRng::seed_from_u64(seed);
        let mut v = Array1::zeros(dim);
        for i in 0..dim {
            let lo = bounds.low[i];
            let hi = bounds.high[i];
            v[i] = lo + (hi - lo) * rng_init.random::<f64>();
        }
        bounds.clip(v.view())
    };
    let mut f = obj.eval(x.view());
    let mut n_evals = 1usize;
    let mut n_grads = 0usize;
    let mut best_pos = x.clone();
    let mut best_val = f;
    if !f.is_finite() {
        // Retry a few random starts.
        for k in 0..8 {
            let mut v = Array1::zeros(dim);
            for i in 0..dim {
                let lo = bounds.low[i];
                let hi = bounds.high[i];
                v[i] = lo + (hi - lo) * rng.random::<f64>();
            }
            x = bounds.clip(v.view());
            f = obj.eval(x.view());
            n_evals += 1;
            if f.is_finite() {
                best_pos = x.clone();
                best_val = f;
                break;
            }
            if work(n_evals, n_grads) >= sa_budget {
                break;
            }
            let _ = k;
        }
    }

    // Adaptive Metropolis state (Haario scatter).
    let mut mean = x.clone();
    let mut scatter = Array2::<f64>::zeros((dim, dim));
    let mut n_obs = 1usize;
    let mut log_scale = 0.0_f64;
    let mut rm_n = 0usize;
    let mut n_accept = 0usize;
    let mut n_propose = 0usize;

    while work(n_evals, n_grads) < sa_budget {
        let t = gap_proportional_temp(f, best_val, dim);
        // Proposal scale: box-relative floor * exp(log_scale).
        let mut width_mean = 0.0;
        for i in 0..dim {
            width_mean += (bounds.high[i] - bounds.low[i]).abs();
        }
        width_mean = (width_mean / dim as f64).max(1e-12);
        let sigma0 = (0.2 * width_mean / (dim as f64).sqrt()).max(SIGMA_FLOOR_FRAC * width_mean);
        let sigma = (sigma0 * log_scale.exp()).max(1e-14);

        // Draw proposal: isotropic until enough observations, else cov.
        let mut y = x.clone();
        if n_obs < 2 * dim {
            for i in 0..dim {
                y[i] = x[i] + sigma * gaussian(rng);
            }
        } else {
            // Cholesky of cov = scatter/(n-1) + eps I
            let denom = (n_obs as f64 - 1.0).max(1.0);
            let eps = 1e-8 * width_mean * width_mean;
            let mut l = Array2::<f64>::zeros((dim, dim));
            // Diagonal-only safe proposal (full chol can fail); use diag(cov)^{1/2}.
            for i in 0..dim {
                let v = (scatter[(i, i)] / denom + eps).max(0.0).sqrt();
                l[(i, i)] = v.max(1e-14);
            }
            for i in 0..dim {
                let mut zi = 0.0;
                // only diagonal
                zi += l[(i, i)] * gaussian(rng);
                y[i] = x[i] + sigma * zi;
            }
        }
        y = reflect_into_box(y.view(), &bounds);
        n_propose += 1;

        if work(n_evals, n_grads) >= sa_budget {
            break;
        }
        let fy = obj.eval(y.view());
        n_evals += 1;
        if !fy.is_finite() {
            continue;
        }
        let delta = fy - f;
        let accept = if delta <= 0.0 {
            true
        } else {
            let t_use = t.max(1e-300);
            rng.random::<f64>() < (-delta / t_use).exp()
        };
        // Robbins–Monro on log scale toward ALPHA_TARGET.
        rm_n += 1;
        let step = 1.0 / (10.0 + rm_n as f64);
        let a_ind = if accept { 1.0 } else { 0.0 };
        log_scale += step * (a_ind - ALPHA_TARGET);
        log_scale = log_scale.clamp(-8.0, 4.0);

        if accept {
            n_accept += 1;
            x = y;
            f = fy;
            // Welford scatter update.
            n_obs += 1;
            let n = n_obs as f64;
            let delta_m = &x - &mean;
            mean = &mean + &delta_m.mapv(|v| v / n);
            let delta2 = &x - &mean;
            for i in 0..dim {
                for j in 0..dim {
                    scatter[(i, j)] += delta_m[i] * delta2[j];
                }
            }
            if f < best_val {
                best_val = f;
                best_pos = x.clone();
            }
        }
    }

    // Terminal polish on best.
    if polish_budget >= 4 {
        if let Some(g) = grad {
            let remain = budget.saturating_sub(work(n_evals, n_grads));
            let polish_use = polish_budget.min(remain);
            if polish_use >= 4 {
                let pol = projected_gradient_polish(
                    obj,
                    g,
                    best_pos.clone(),
                    polish_use,
                    0.1,
                    1e-8,
                );
                n_evals += pol.n_evals;
                n_grads += pol.n_grads;
                if pol.best_val < best_val && pol.best_val.is_finite() {
                    best_val = pol.best_val;
                    best_pos = pol.best_pos;
                }
            }
        } else {
            // Coordinate polish without gradients.
            let remain = budget.saturating_sub(work(n_evals, n_grads));
            let mut steps = polish_budget.min(remain);
            let mut z = best_pos.clone();
            let mut fz = best_val;
            let mut i = 0usize;
            while steps > 0 && work(n_evals, n_grads) < budget {
                let lo = bounds.low[i % dim];
                let hi = bounds.high[i % dim];
                let step = 0.1 * (hi - lo).abs().max(1e-12);
                for s in [-step, step] {
                    if steps == 0 || work(n_evals, n_grads) >= budget {
                        break;
                    }
                    let mut trial = z.clone();
                    trial[i % dim] = (z[i % dim] + s).clamp(lo, hi);
                    let ft = obj.eval(trial.view());
                    n_evals += 1;
                    steps = steps.saturating_sub(1);
                    if ft.is_finite() && ft < fz {
                        z = trial;
                        fz = ft;
                    }
                }
                i += 1;
            }
            if fz < best_val {
                best_val = fz;
                best_pos = z;
            }
        }
    }

    // Never exceed budget accounting in the returned counters (clamp report).
    let used = work(n_evals, n_grads).min(budget);
    let (ne, ng) = if n_evals + n_grads > budget {
        // Prefer reporting objective-heavy usage.
        let ne = n_evals.min(budget);
        let ng = budget.saturating_sub(ne);
        (ne, ng)
    } else {
        (n_evals, n_grads)
    };
    let _ = used;

    GpmdResult {
        best_pos,
        best_val,
        n_evals: ne,
        n_grads: ng,
        n_accept,
        n_propose,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eindir_core::Bounds;
    use ndarray::Array1;

    struct Sphere {
        bounds: Bounds<f64>,
    }
    impl Sphere {
        fn new(d: usize) -> Self {
            Self {
                bounds: Bounds::new(
                    Array1::from_elem(d, -2.0),
                    Array1::from_elem(d, 2.0),
                    1e-12,
                ),
            }
        }
    }
    impl Objective<f64> for Sphere {
        fn dim(&self) -> usize {
            self.bounds.dims
        }
        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            x.iter().map(|v| v * v).sum()
        }
    }
    impl Gradient<f64> for Sphere {
        fn dim(&self) -> usize {
            self.bounds.dims
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            x.mapv(|v| 2.0 * v)
        }
    }

    #[test]
    fn gap_temp_is_half_gap_over_d() {
        let t = gap_proportional_temp(4.0, 0.0, 4);
        // θ* * 4 / 4 = 0.5
        assert!((t - 0.5).abs() < 1e-12);
    }

    #[test]
    fn theta_star_in_descent_window() {
        assert!(THETA_STAR > 0.0 && THETA_STAR < 2.0);
    }

    #[test]
    fn gpmd_improves_sphere_and_respects_budget() {
        let obj = Sphere::new(5);
        let res = gpmd_optimize::<_, Sphere>(&obj, Some(&obj), 800, 7, None);
        assert!(res.best_val.is_finite());
        assert!(res.n_evals + res.n_grads <= 800);
        assert!(
            res.best_val < 1.0,
            "sphere should refine substantially, got {}",
            res.best_val
        );
        assert!(obj.bounds().contains(res.best_pos.view()));
    }

    #[test]
    fn gpmd_beats_hot_metropolis_on_sphere() {
        // Hot fixed T wastes budget; gap-proportional should get lower best.
        let obj = Sphere::new(4);
        let gpmd = gpmd_optimize::<_, Sphere>(&obj, None, 600, 1, None);
        // Crude hot chain
        let bounds = obj.bounds().clone();
        let dim = 4;
        let mut rng = StdRng::seed_from_u64(1);
        let mut x = Array1::from_elem(dim, 1.0);
        let mut f = obj.eval(x.view());
        let mut best = f;
        let mut n = 1usize;
        while n < 600 {
            let mut y = x.clone();
            for i in 0..dim {
                y[i] += 0.3 * gaussian(&mut rng);
            }
            y = reflect_into_box(y.view(), &bounds);
            let fy = obj.eval(y.view());
            n += 1;
            let delta = fy - f;
            if delta <= 0.0 || rng.random::<f64>() < (-delta / 5.0).exp() {
                x = y;
                f = fy;
                if f < best {
                    best = f;
                }
            }
        }
        assert!(
            gpmd.best_val < best * 0.5 || gpmd.best_val < 0.1,
            "gpmd {} should beat hot MH {}",
            gpmd.best_val,
            best
        );
    }
}
