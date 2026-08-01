//! Budget-Feasible Window Temperature (BFWT / D11).
//!
//! Combines D6 descent ceiling `T_hi = 2 gap / d` with D7 escape floor
//! `T_lo = b_hat / ln(B + e)` and design interior `T_des = θ⋆ gap / d`:
//!
//! ```text
//! if T_lo < T_hi { T = clamp(T_des, T_lo, T_hi) } else { T = T_lo }
//! ```
//!
//! When `b_hat → 0`, recovers GPMD (`T = T_des`). Not a dual-annealing
//! global solver; local law under stated models only.
//! Derivation: `proofs/d11_budget_feasible_temp.py`.

use eindir_core::{Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::methods::local_polish::projected_gradient_polish;
use crate::movekernel::reflect_into_box;

/// Design θ⋆ = 1/2 (D6 residual-rate operating point).
pub const THETA_STAR: f64 = 0.5;
/// Euler e in T_lo = b / log(B + e).
pub const EULER_E: f64 = std::f64::consts::E;
const GAP_EPS: f64 = 1e-12;
const SIGMA_FLOOR_FRAC: f64 = 1e-4;
const POLISH_FRACTION: f64 = 0.25;
const ALPHA_TARGET: f64 = 0.32;

/// Operating mode returned with the temperature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BfwtMode {
    /// T = T_des inside the D6∩D7 window.
    Design,
    /// T raised to escape floor (still below descent ceiling).
    EscapeFloor,
    /// T capped at descent ceiling.
    DescentCap,
    /// Window empty: T forced to escape floor (T_lo ≥ T_hi).
    EscapeForced,
}

impl BfwtMode {
    /// Stable string tag for logs / Python bindings.
    pub fn as_str(self) -> &'static str {
        match self {
            BfwtMode::Design => "design",
            BfwtMode::EscapeFloor => "escape_floor",
            BfwtMode::DescentCap => "descent_cap",
            BfwtMode::EscapeForced => "escape_forced",
        }
    }
}

/// D6 ceiling: T_hi = 2 * gap / d.
#[inline]
pub fn t_hi(gap: f64, dim: usize) -> f64 {
    let d = (dim.max(1)) as f64;
    2.0 * gap.max(GAP_EPS) / d
}

/// Design interior: T_des = θ⋆ * gap / d.
#[inline]
pub fn t_des(gap: f64, dim: usize) -> f64 {
    let d = (dim.max(1)) as f64;
    THETA_STAR * gap.max(GAP_EPS) / d
}

/// D7 floor: T_lo = barrier / log(B + e).
#[inline]
pub fn t_lo(barrier: f64, budget_remaining: f64) -> f64 {
    let b = barrier.max(0.0);
    b / (budget_remaining.max(0.0) + EULER_E).ln()
}

/// Window nonempty ⇔ barrier * d < 2 * gap * log(B + e).
#[inline]
pub fn window_nonempty(gap: f64, dim: usize, barrier: f64, budget_remaining: f64) -> bool {
    let d = (dim.max(1)) as f64;
    let g = gap.max(GAP_EPS);
    barrier * d < 2.0 * g * (budget_remaining.max(0.0) + EULER_E).ln()
}

/// BFWT law: clamp design temperature into the D6∩D7 window.
#[inline]
pub fn budget_feasible_temp(
    f: f64,
    f_best: f64,
    dim: usize,
    budget_remaining: f64,
    barrier_hat: f64,
) -> (f64, BfwtMode) {
    let gap = (f - f_best).max(GAP_EPS);
    let hi = t_hi(gap, dim);
    let lo = t_lo(barrier_hat, budget_remaining);
    let des = t_des(gap, dim);
    if lo < hi {
        if des < lo {
            (lo, BfwtMode::EscapeFloor)
        } else if des > hi {
            (hi, BfwtMode::DescentCap)
        } else {
            (des, BfwtMode::Design)
        }
    } else {
        (lo, BfwtMode::EscapeForced)
    }
}

/// Result of one BFWT run.
#[derive(Clone, Debug)]
pub struct BfwtResult {
    /// Best feasible position found.
    pub best_pos: Array1<f64>,
    /// Objective at `best_pos`.
    pub best_val: f64,
    /// Objective evaluations charged.
    pub n_evals: usize,
    /// Gradient evaluations charged (polish).
    pub n_grads: usize,
    /// Metropolis accepts during the SA phase.
    pub n_accept: usize,
    /// Last BFWT mode applied before polish.
    pub last_mode: BfwtMode,
}

fn work(n_evals: usize, n_grads: usize) -> usize {
    n_evals + n_grads
}

fn gaussian<R: Rng>(rng: &mut R) -> f64 {
    let u1 = rng.random::<f64>().max(1e-12);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Run BFWT Metropolis under a work-unit budget.
///
/// `barrier_hat` is an external barrier-depth proxy (same units as f).
/// When zero, temperature reduces to GPMD gap-proportional law.
pub fn bfwt_optimize<O, G>(
    obj: &O,
    grad: Option<&G>,
    budget: usize,
    seed: u64,
    barrier_hat: f64,
    x0: Option<ArrayView1<f64>>,
) -> BfwtResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    let mut rng = StdRng::seed_from_u64(seed ^ 0xbff7_11d1_c011_a715);
    let bounds = obj.bounds().clone();
    let dim = bounds.dims.max(1);
    let budget = budget.max(4);
    let polish_budget = ((budget as f64) * POLISH_FRACTION).floor() as usize;
    let polish_budget = polish_budget.clamp(0, budget.saturating_sub(2));
    let sa_budget = budget.saturating_sub(polish_budget).max(2);

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
    let mut last_mode = BfwtMode::Design;

    // Running scale only (isotropic); size via RM toward α*.
    let mut log_scale = 0.0_f64;
    let mut rm_n = 0usize;
    let mut n_accept = 0usize;

    while work(n_evals, n_grads) < sa_budget {
        let rem = sa_budget.saturating_sub(work(n_evals, n_grads)) as f64;
        let (t, mode) = budget_feasible_temp(f, best_val, dim, rem, barrier_hat);
        last_mode = mode;
        let mut width_mean = 0.0;
        for i in 0..dim {
            width_mean += (bounds.high[i] - bounds.low[i]).abs();
        }
        width_mean = (width_mean / dim as f64).max(1e-12);
        let sigma0 = (0.2 * width_mean / (dim as f64).sqrt()).max(SIGMA_FLOOR_FRAC * width_mean);
        let sigma = (sigma0 * log_scale.exp()).max(1e-14);

        let mut y = x.clone();
        for i in 0..dim {
            y[i] = x[i] + sigma * gaussian(&mut rng);
        }
        y = reflect_into_box(y.view(), &bounds);

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
        rm_n += 1;
        let step = 1.0 / (10.0 + rm_n as f64);
        let a_ind = if accept { 1.0 } else { 0.0 };
        log_scale += step * (a_ind - ALPHA_TARGET);
        log_scale = log_scale.clamp(-8.0, 4.0);

        if accept {
            n_accept += 1;
            x = y;
            f = fy;
            if f < best_val {
                best_val = f;
                best_pos = x.clone();
            }
        }
    }

    if polish_budget >= 4
        && let Some(g) = grad
    {
        let remain = budget.saturating_sub(work(n_evals, n_grads));
        let polish_use = polish_budget.min(remain);
        if polish_use >= 4 {
            let pol = projected_gradient_polish(obj, g, best_pos.clone(), polish_use, 0.1, 1e-8);
            n_evals += pol.n_evals;
            n_grads += pol.n_grads;
            if pol.best_val < best_val && pol.best_val.is_finite() {
                best_val = pol.best_val;
                best_pos = pol.best_pos;
            }
        }
    }

    let ne = n_evals.min(budget);
    let ng = budget.saturating_sub(ne).min(n_grads);
    BfwtResult {
        best_pos,
        best_val,
        n_evals: ne,
        n_grads: ng,
        n_accept,
        last_mode,
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

    #[test]
    fn zero_barrier_recovers_gpmd_temp() {
        // f=4, f_best=0, d=4 → T_des = 0.5 * 4/4 = 0.5
        let (t, mode) = budget_feasible_temp(4.0, 0.0, 4, 1000.0, 0.0);
        assert_eq!(mode, BfwtMode::Design);
        assert!((t - 0.5).abs() < 1e-12);
    }

    #[test]
    fn design_is_quarter_ceiling() {
        // θ⋆=1/2 and θ_hi=2 ⇒ T_des / T_hi = 1/4
        let gap = 8.0;
        let d = 4;
        assert!((t_des(gap, d) - 0.25 * t_hi(gap, d)).abs() < 1e-12);
        assert!((t_des(gap, d) * (d as f64) / gap - THETA_STAR).abs() < 1e-12);
    }

    #[test]
    fn nonempty_clamps_into_window() {
        let gap = 10.0;
        let d = 5;
        let b = 2.0;
        let budget = 1e4_f64;
        assert!(window_nonempty(gap, d, b, budget));
        let (t, mode) = budget_feasible_temp(gap, 0.0, d, budget, b);
        let lo = t_lo(b, budget);
        let hi = t_hi(gap, d);
        assert!(lo <= t + 1e-12 && t <= hi + 1e-12);
        assert!(matches!(
            mode,
            BfwtMode::Design | BfwtMode::EscapeFloor | BfwtMode::DescentCap
        ));
    }

    #[test]
    fn empty_window_forces_escape_floor() {
        let (t, mode) = budget_feasible_temp(0.1, 0.0, 50, 10.0, 100.0);
        assert_eq!(mode, BfwtMode::EscapeForced);
        assert!((t - t_lo(100.0, 10.0)).abs() < 1e-12);
        assert!(t + 1e-12 >= t_hi(0.1, 50));
    }

    #[test]
    fn bfwt_improves_sphere_and_respects_budget() {
        let obj = Sphere::new(5);
        let res = bfwt_optimize::<_, Sphere>(&obj, Some(&obj), 800, 7, 0.0, None);
        assert!(res.best_val.is_finite());
        assert!(res.n_evals + res.n_grads <= 800);
        assert!(
            res.best_val < 1.0,
            "sphere should refine substantially, got {}",
            res.best_val
        );
        assert_eq!(res.last_mode, BfwtMode::Design); // b=0
    }

    #[test]
    fn escape_floor_raises_temp_above_design() {
        // Moderate barrier, large gap, mid budget → T_lo may exceed T_des
        let gap = 4.0;
        let d = 4;
        let des = t_des(gap, d);
        // Choose b so T_lo is between T_des and T_hi
        let budget = 100.0_f64;
        let lo_target = des * 1.5;
        let b = lo_target * (budget + EULER_E).ln();
        assert!(window_nonempty(gap, d, b, budget));
        let (t, mode) = budget_feasible_temp(gap, 0.0, d, budget, b);
        assert_eq!(mode, BfwtMode::EscapeFloor);
        assert!(t > des + 1e-12);
        assert!(t < t_hi(gap, d) + 1e-12);
    }
}
