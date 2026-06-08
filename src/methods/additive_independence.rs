//! Rank-1 (mean-field) independence-sampler simulated annealing.
//!
//! This is the unified MCMC+SA point that the separable surrogate enables. A
//! pilot fits an [`AdditiveSurrogate`] over all `d` coordinates; the remaining
//! budget is spent in geometric-temperature epochs. Each epoch draws a block of
//! proposals from the tempered surrogate density by independent per-coordinate
//! inverse-CDF sampling (the global *independence* Move), mixed with a
//! `local_frac` Gaussian random walk around the incumbent (the local Move), and
//! every proposal is accepted by a Metropolis rule against the *true* objective
//! (the Accept slot). Because the surrogate density factorises across
//! coordinates for a separable objective, one global draw places every
//! coordinate at its own tempered optimum at once -- the regime a local
//! random-walk proposal, whose efficiency decays like `1/d`, cannot reach. The
//! Metropolis accept against the true objective removes the mean-field bias.

use eindir_core::{AdditiveSurrogate, Objective};
use ndarray::{Array1, Array2};
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Result of a rank-1 independence-sampler run.
#[derive(Clone, Debug)]
pub struct AdditiveIndependenceResult {
    /// Best-seen position.
    pub best_pos: Vec<f64>,
    /// Best-seen objective value.
    pub best_val: f64,
    /// True-objective evaluations consumed (pilot + main loop).
    pub n_evals: usize,
}

/// Run rank-1 independence-sampler SA on `obj` under a shared work-unit budget.
///
/// `max_fevals` bounds the total true-objective evaluations (pilot included),
/// so the driver runs at parity with every other point of the algebra.
/// `degree` is the per-coordinate Chebyshev degree, `grid_m` the inverse-CDF
/// grid resolution, `local_frac` the fraction of each epoch's proposals spent
/// on the local random walk, and `n_epochs` the number of temperature levels.
#[allow(clippy::too_many_arguments)]
pub fn additive_independence_sa<O: Objective<f64>>(
    obj: &O,
    seed: u64,
    max_fevals: usize,
    degree: usize,
    grid_m: usize,
    local_frac: f64,
    n_epochs: usize,
    n_pilot: usize,
) -> AdditiveIndependenceResult {
    assert!(max_fevals > 0, "max_fevals must be positive");
    let bounds = obj.bounds().clone();
    let dim = bounds.dims;
    let mut rng = StdRng::seed_from_u64(seed);

    // Pilot evaluated on the true objective. A uniform random design gives
    // clean per-coordinate 1D coverage, which is what the separable additive
    // fit needs; a Halton design degrades on the high-prime axes in high
    // dimension and would starve the surrogate on those coordinates.
    let n_pilot = n_pilot.max(degree + 2).min(max_fevals);
    let mut pilot = Array2::<f64>::zeros((n_pilot, dim));
    for i in 0..n_pilot {
        for j in 0..dim {
            pilot[[i, j]] = bounds.low[j] + rng.random::<f64>() * (bounds.high[j] - bounds.low[j]);
        }
    }
    let mut y = Array1::<f64>::zeros(n_pilot);
    for i in 0..n_pilot {
        y[i] = obj.eval(pilot.row(i));
    }
    let mut n_evals = n_pilot;

    // Best pilot point seeds the incumbent.
    let mut best_idx = 0usize;
    for i in 1..n_pilot {
        if y[i] < y[best_idx] {
            best_idx = i;
        }
    }
    let mut x: Array1<f64> = pilot.row(best_idx).to_owned();
    let mut fx = y[best_idx];
    let mut best_pos = x.clone();
    let mut best_val = fx;

    if n_evals >= max_fevals {
        return AdditiveIndependenceResult {
            best_pos: best_pos.to_vec(),
            best_val,
            n_evals,
        };
    }

    let surrogate = AdditiveSurrogate::fit(pilot.view(), y.view(), bounds.clone(), degree);

    let remaining = max_fevals - n_evals;
    let n_epochs = n_epochs.clamp(1, remaining);
    let block = (remaining / n_epochs).max(1);
    let t_hi = fx.abs().max(1.0);
    let t_lo = 1e-3 * t_hi;
    let step: Array1<f64> = (0..dim).map(|j| 0.1 * (bounds.high[j] - bounds.low[j])).collect();

    'epochs: for epoch in 0..n_epochs {
        if n_evals >= max_fevals {
            break;
        }
        let frac = epoch as f64 / (n_epochs.max(2) - 1) as f64;
        let temperature = t_hi * (t_lo / t_hi).powf(frac);
        let n_block = block.min(max_fevals - n_evals);
        let n_local = (local_frac * n_block as f64).round() as usize;
        let n_global = n_block.saturating_sub(n_local);

        // Global independence proposals from the tempered surrogate, plus a
        // few local random-walk proposals around the incumbent. Collect both
        // streams as candidate rows, then Metropolis-accept each against truth.
        let global = surrogate.sample(n_global, temperature, grid_m, &mut rng);
        let mut candidates: Vec<Array1<f64>> = Vec::with_capacity(n_global + n_local);
        for s in 0..n_global {
            candidates.push(global.row(s).to_owned());
        }
        let rw_scale = 0.2 + 0.8 * (1.0 - frac);
        for _ in 0..n_local {
            let mut cand = x.clone();
            for j in 0..dim {
                let g: f64 = rng.sample(rand_distr::StandardNormal);
                cand[j] = (x[j] + g * step[j] * rw_scale).clamp(bounds.low[j], bounds.high[j]);
            }
            candidates.push(cand);
        }

        for cand in candidates {
            let fy = obj.eval(cand.view());
            n_evals += 1;
            let accept = fy <= fx || rng.random::<f64>() < (-(fy - fx) / temperature).exp();
            if accept {
                x = cand.clone();
                fx = fy;
            }
            if fy < best_val {
                best_val = fy;
                best_pos = cand;
            }
            if n_evals >= max_fevals {
                break 'epochs;
            }
        }
    }

    AdditiveIndependenceResult {
        best_pos: best_pos.to_vec(),
        best_val,
        n_evals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eindir_core::Bounds;
    use ndarray::ArrayView1;

    struct Styb {
        bounds: Bounds<f64>,
    }
    impl Objective<f64> for Styb {
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            0.5 * x.iter().map(|&v| v.powi(4) - 16.0 * v * v + 5.0 * v).sum::<f64>()
        }
        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }
        fn dim(&self) -> usize {
            self.bounds.dims
        }
    }

    #[test]
    fn solves_separable_styblinski_tang_in_high_dim() {
        // The separable d=20 case that RWM SA and an active-subspace surrogate
        // both miss; the rank-1 independence sampler should reach the global
        // basin (optimum -39.16599 * d) within 1% at a 4000-eval budget.
        let dim = 20;
        let bounds = Bounds::new(
            Array1::from_elem(dim, -5.0),
            Array1::from_elem(dim, 5.0),
            0.0,
        );
        let obj = Styb { bounds };
        let opt = -39.16599 * dim as f64;
        let mut hits = 0;
        for seed in 0..5 {
            let res = additive_independence_sa(&obj, seed, 4000, 8, 65, 0.2, 40, 320);
            assert_eq!(res.n_evals <= 4000, true);
            if res.best_val <= opt + 0.01 * opt.abs() {
                hits += 1;
            }
        }
        assert!(hits >= 4, "expected >=4/5 global-basin hits, got {hits}");
    }
}
