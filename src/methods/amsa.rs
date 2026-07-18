//! Standalone whitened BFWT annealed descent (AmSa).
//!
//! The portfolio's `am_sa` arm pairs the budget-feasible window temperature
//! (D11) with Haario covariance whitening, the anisotropic critical
//! temperature result mandating the latter: on a general quadratic the
//! descent window collapses by a factor of order d/kappa along soft
//! eigendirections, and a whitened proposal restores theta_c = 2. This
//! module owns that machinery and exposes it as a standalone solver so the
//! algorithm can be measured on its own, outside the bandit.

use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Distribution;

use crate::methods::local_polish::projected_gradient_polish;
use crate::movekernel::reflect_into_box;

pub(crate) struct AmSaState {
    pub(crate) x: Array1<f64>,
    pub(crate) f_x: f64,
    pub(crate) mean: Array1<f64>,
    /// Unnormalized scatter matrix sum (x - mean) outer products.
    pub(crate) scatter: Array2<f64>,
    pub(crate) n_obs: usize,
    /// Robbins-Monro log proposal-scale multiplier.
    pub(crate) log_scale: f64,
    /// Robbins-Monro step counter (diminishing adaptation).
    pub(crate) rm_n: usize,
    /// EMA of rejected uphill energy deltas (D7 barrier proxy for BFWT).
    pub(crate) barrier_hat: f64,
    /// Consecutive AmSa slices without ledger incumbent improvement.
    pub(crate) stagnant_slices: usize,
    /// IPOP-style reseed generation (grows proposal scale after reseed).
    pub(crate) reseed_gen: usize,
}

/// D6 optimal acceptance at θ⋆ = 1/2 (design interior of BFWT).
pub(crate) const AM_ALPHA_TARGET: f64 = 0.32;
/// EMA rate for barrier_hat from rejected uphill moves.
pub(crate) const AM_BARRIER_EMA: f64 = 0.15;
/// Reseed AmSa after this many non-improving slices (IPOP-style).
pub(crate) const AM_STAGNANT_RESEED: usize = 3;

impl AmSaState {
    pub(crate) fn new(x: Array1<f64>, f_x: f64) -> Self {
        let d = x.len();
        Self {
            mean: x.clone(),
            scatter: Array2::zeros((d, d)),
            n_obs: 1,
            x,
            f_x,
            log_scale: 0.0,
            rm_n: 0,
            barrier_hat: 0.0,
            stagnant_slices: 0,
            reseed_gen: 0,
        }
    }

    /// Hard reseed: new point, wipe scatter, keep barrier_hat warm.
    pub(crate) fn reseed(&mut self, x: Array1<f64>, f_x: f64) {
        let d = x.len();
        self.mean = x.clone();
        self.scatter = Array2::zeros((d, d));
        self.n_obs = 1;
        self.x = x;
        self.f_x = f_x;
        self.log_scale = 0.0;
        self.rm_n = 0;
        self.stagnant_slices = 0;
        self.reseed_gen = self.reseed_gen.saturating_add(1);
        // Mildly inflate barrier so the next incarnation keeps escape heat.
        self.barrier_hat = (self.barrier_hat * 1.5).max(1e-6);
    }

    /// Welford-style running mean/scatter update over chain states.
    pub(crate) fn observe(&mut self, x: ArrayView1<f64>) {
        self.n_obs += 1;
        let n = self.n_obs as f64;
        let delta = &x.to_owned() - &self.mean;
        self.mean = &self.mean + &delta.mapv(|v| v / n);
        let delta2 = &x.to_owned() - &self.mean;
        for i in 0..delta.len() {
            for j in 0..delta.len() {
                self.scatter[(i, j)] += delta[i] * delta2[j];
            }
        }
    }

    /// Covariance Cholesky factor with Haario-style regularization:
    /// Sigma_hat + eps diag(width^2). Falls back to the diagonal on
    /// factorization failure.
    pub(crate) fn proposal_chol(&self, bounds: &Bounds<f64>) -> Array2<f64> {
        let d = self.x.len();
        let mut cov = Array2::zeros((d, d));
        // Haario burn-in: until the chain has 2d accepted observations
        // the scatter is uninformative, so propose from a box-scaled
        // identity ((0.1 width_i)^2) rather than the near-zero scatter;
        // a frozen initial chain wastes its whole slice inside one basin.
        if self.n_obs < 2 * d {
            let mut l = Array2::zeros((d, d));
            for i in 0..d {
                let w = (bounds.high[i] - bounds.low[i]).abs().max(1e-12);
                l[(i, i)] = 0.1 * w;
            }
            return l;
        }
        let denom = (self.n_obs as f64 - 1.0).max(1.0);
        for i in 0..d {
            for j in 0..d {
                cov[(i, j)] = self.scatter[(i, j)] / denom;
            }
        }
        for i in 0..d {
            let w = (bounds.high[i] - bounds.low[i]).abs().max(1e-12);
            cov[(i, i)] += 1e-6 * w * w + 1e-12;
        }
        cholesky_lower(&cov).unwrap_or_else(|| {
            let mut l = Array2::zeros((d, d));
            for i in 0..d {
                l[(i, i)] = cov[(i, i)].sqrt();
            }
            l
        })
    }
}

/// Dense lower Cholesky for the small proposal covariances (d <= ~64).
pub(crate) fn cholesky_lower(a: &Array2<f64>) -> Option<Array2<f64>> {
    let d = a.nrows();
    let mut l: Array2<f64> = Array2::zeros((d, d));
    for i in 0..d {
        for j in 0..=i {
            let mut s = a[(i, j)];
            for k in 0..j {
                s -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    return None;
                }
                l[(i, j)] = s.sqrt();
            } else {
                l[(i, j)] = s / l[(j, j)];
            }
        }
    }
    Some(l)
}


/// Result of one standalone AmSa run.
#[derive(Clone, Debug)]
pub struct AmsaResult {
    /// Best feasible position found.
    pub best_pos: Array1<f64>,
    /// Objective at `best_pos`.
    pub best_val: f64,
    /// Objective evaluations charged.
    pub n_evals: usize,
    /// Gradient evaluations charged (polish tail).
    pub n_grads: usize,
    /// IPOP-style reseeds performed.
    pub n_reseeds: usize,
}

const POLISH_FRACTION: f64 = 0.25;
/// Probability of a heavy-tailed (Cauchy radial) jump in the whitened
/// metric: the descent analysis governs the small-step regime, while
/// occasional long jumps priced by the same Metropolis rule supply the
/// well-to-well escape that Gaussian tails cannot (Schwefel-class).
const TAIL_JUMP_PROB: f64 = 0.15;
/// Cap on the Cauchy radial factor (reflection handles the rest).
const TAIL_JUMP_CAP: f64 = 1e3;
/// Polish burst on stagnation, as a fraction of the remaining SA budget.
const STAGNATION_POLISH_FRAC: f64 = 0.125;

/// Run whitened BFWT annealed descent under a work-unit budget.
///
/// One work unit is one objective or one gradient evaluation. The SA phase
/// spends three quarters of the budget on a single adaptive chain with
/// BFWT temperature, Haario covariance whitening, Robbins--Monro scale
/// control, an online barrier estimate from rejected uphill moves, and
/// IPOP-style reseeds on stagnation; the final quarter runs the
/// stall-recovering projected quasi-Newton polish from the incumbent when
/// a gradient is available.
pub fn amsa_optimize<O, G>(
    obj: &O,
    grad: Option<&G>,
    budget: usize,
    seed: u64,
    x0: Option<ArrayView1<f64>>,
) -> AmsaResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    let bounds = obj.bounds().clone();
    let d = bounds.dims.max(1);
    let budget = budget.max(4);
    let mut rng = StdRng::seed_from_u64(seed ^ 0xa35a_11d6_0b5e_55e5);
    let polish_budget = if grad.is_some() {
        (((budget as f64) * POLISH_FRACTION).floor() as usize).min(budget.saturating_sub(2))
    } else {
        0
    };
    let sa_budget = budget.saturating_sub(polish_budget).max(2);

    let start = if let Some(x0) = x0 {
        bounds.clip(x0)
    } else {
        let mut v = Array1::zeros(d);
        for i in 0..d {
            v[i] = bounds.low[i] + (bounds.high[i] - bounds.low[i]) * rng.random::<f64>();
        }
        bounds.clip(v.view())
    };
    let f_start = obj.eval(start.view());
    let mut n_evals = 1usize;
    let mut n_grads = 0usize;
    if !f_start.is_finite() {
        return AmsaResult {
            best_pos: start,
            best_val: f_start,
            n_evals,
            n_grads: 0,
            n_reseeds: 0,
        };
    }
    let mut st = AmSaState::new(start.clone(), f_start);
    let mut best_pos = start;
    let mut best_val = f_start;
    let mut n_reseeds = 0usize;
    let base = 2.38 / (d as f64).sqrt();
    // Epoch length ties stagnation detection to dimension.
    let epoch = (8 * (d + 1)).clamp(32, 256);

    while n_evals + n_grads < sa_budget {
        let epoch_best = best_val;
        let reseed_boost = 1.5_f64.powi(st.reseed_gen.min(6) as i32);
        let l = st.proposal_chol(&bounds);
        for _ in 0..epoch {
            if n_evals + n_grads >= sa_budget {
                break;
            }
            let rem = (sa_budget - n_evals - n_grads) as f64;
            let (temp, _mode) = crate::methods::bfwt::budget_feasible_temp(
                st.f_x,
                best_val,
                d,
                rem,
                st.barrier_hat,
            );
            let temp = temp.max(if st.barrier_hat > 0.0 {
                0.0
            } else {
                1e-6 * st.f_x.abs().max(1.0) / d as f64
            });
            let z: Vec<f64> = (0..d)
                .map(|_| rand_distr::StandardNormal.sample(&mut rng))
                .collect();
            let scale = st.log_scale.exp() * base * reseed_boost;
            let mut y = st.x.clone();
            if rng.random::<f64>() < TAIL_JUMP_PROB {
                // Per-coordinate Cauchy displacements in raw coordinates:
                // separable multi-well landscapes (Schwefel class) need
                // coordinate-wise well flips that no single radial jump in
                // the whitened metric can produce.
                for i in 0..d {
                    let w = (bounds.high[i] - bounds.low[i]).abs().max(1e-12);
                    let u: f64 = rng.random::<f64>();
                    let c = (std::f64::consts::PI * (u - 0.5)).tan();
                    y[i] += 0.1 * w * c.clamp(-TAIL_JUMP_CAP, TAIL_JUMP_CAP);
                }
            } else {
                for i in 0..d {
                    let mut acc = 0.0;
                    for j in 0..=i {
                        acc += l[(i, j)] * z[j];
                    }
                    y[i] += scale * acc;
                }
            }
            let y = reflect_into_box(y.view(), &bounds);
            let f_y = obj.eval(y.view());
            n_evals += 1;
            if !f_y.is_finite() {
                continue;
            }
            let delta = f_y - st.f_x;
            let accepted =
                delta <= 0.0 || (temp > 0.0 && rng.random::<f64>() < (-delta / temp).exp());
            st.rm_n += 1;
            let a = if accepted { 1.0 } else { 0.0 };
            st.log_scale += (a - AM_ALPHA_TARGET) / (st.rm_n as f64).sqrt();
            st.log_scale = st.log_scale.clamp(-12.0, 6.0);
            if accepted {
                st.x = y.to_owned();
                st.f_x = f_y;
                let x_obs = st.x.clone();
                st.observe(x_obs.view());
                if f_y < best_val {
                    best_val = f_y;
                    best_pos = st.x.clone();
                }
            } else if delta > 0.0 {
                st.barrier_hat = if st.barrier_hat <= 0.0 {
                    delta
                } else {
                    (1.0 - AM_BARRIER_EMA) * st.barrier_hat + AM_BARRIER_EMA * delta
                };
            }
        }
        if best_val < epoch_best - 1e-15 * epoch_best.abs().max(1.0) {
            st.stagnant_slices = 0;
            st.barrier_hat *= 0.5;
        } else {
            st.stagnant_slices = st.stagnant_slices.saturating_add(1);
        }
        if st.stagnant_slices >= AM_STAGNANT_RESEED
            && let Some(g) = grad
            && n_evals + n_grads + 8 < sa_budget
        {
            // Polish burst before abandoning the basin: dual_annealing runs
            // local search throughout; matching that closes near-best cells
            // the chain alone leaves shallow.
            let rem = sa_budget - n_evals - n_grads;
            let burst = (((rem as f64) * STAGNATION_POLISH_FRAC) as usize).clamp(8, rem);
            let pol = projected_gradient_polish(obj, g, best_pos.clone(), burst / 2, 0.1, 1e-10);
            n_evals += pol.n_evals;
            n_grads += pol.n_grads;
            if pol.best_val.is_finite() && pol.best_val < best_val {
                best_val = pol.best_val;
                best_pos = pol.best_pos;
                st.stagnant_slices = 0;
            }
        }
        if st.stagnant_slices >= AM_STAGNANT_RESEED && n_evals + n_grads + 2 < sa_budget {
            let mut x = Array1::zeros(d);
            for i in 0..d {
                x[i] = bounds.low[i] + (bounds.high[i] - bounds.low[i]) * rng.random::<f64>();
            }
            let x = bounds.clip(x.view());
            let f_x = obj.eval(x.view());
            n_evals += 1;
            if f_x.is_finite() {
                if f_x < best_val {
                    best_val = f_x;
                    best_pos = x.clone();
                }
                st.reseed(x, f_x);
                n_reseeds += 1;
            } else {
                st.stagnant_slices = 0;
            }
        }
    }

    if let Some(g) = grad {
        let remain = budget.saturating_sub(n_evals + n_grads);
        let polish_use = polish_budget.min(remain);
        if polish_use >= 4 {
            let pol = projected_gradient_polish(obj, g, best_pos.clone(), polish_use, 0.1, 1e-10);
            n_evals += pol.n_evals;
            n_grads += pol.n_grads;
            if pol.best_val.is_finite() && pol.best_val < best_val {
                best_val = pol.best_val;
                best_pos = pol.best_pos;
            }
        }
    }

    let ne = n_evals.min(budget);
    let ng = budget.saturating_sub(ne).min(n_grads);
    AmsaResult {
        best_pos,
        best_val,
        n_evals: ne,
        n_grads: ng,
        n_reseeds,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Sphere {
        bounds: Bounds<f64>,
    }
    impl Sphere {
        fn new(d: usize) -> Self {
            Self {
                bounds: Bounds::new(Array1::from_elem(d, -2.0), Array1::from_elem(d, 2.0), 1e-12),
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

    /// Ill-conditioned quadratic: whitening must keep descent alive.
    struct Ellipse {
        bounds: Bounds<f64>,
    }
    impl Objective<f64> for Ellipse {
        fn dim(&self) -> usize {
            2
        }
        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            x[0] * x[0] + 1e4 * x[1] * x[1]
        }
    }
    impl Gradient<f64> for Ellipse {
        fn dim(&self) -> usize {
            2
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            ndarray::array![2.0 * x[0], 2e4 * x[1]]
        }
    }

    #[test]
    fn amsa_refines_sphere_within_budget() {
        let obj = Sphere::new(5);
        let res = amsa_optimize::<_, Sphere>(&obj, Some(&obj), 1200, 7, None);
        assert!(res.n_evals + res.n_grads <= 1200);
        assert!(res.best_val < 1e-6, "got {}", res.best_val);
    }

    #[test]
    fn amsa_whitens_ill_conditioned_quadratic() {
        let obj = Ellipse {
            bounds: Bounds::new(Array1::from_elem(2, -3.0), Array1::from_elem(2, 3.0), 1e-12),
        };
        let res = amsa_optimize::<_, Ellipse>(&obj, Some(&obj), 1500, 11, None);
        assert!(res.best_val < 1e-8, "got {}", res.best_val);
    }

    #[test]
    fn amsa_runs_without_gradient() {
        let obj = Sphere::new(3);
        let res = amsa_optimize::<_, Sphere>(&obj, None, 600, 3, None);
        assert!(res.n_evals <= 600);
        assert!(res.best_val < 0.5, "got {}", res.best_val);
    }
}
