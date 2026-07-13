//! Classical population-controlled diffusion search (DMC-inspired).
//!
//! Engineering inspiration from projector Monte Carlo / DMC walkers in codes
//! such as QMCPACK: a **population of walkers**, **diffusion proposals**,
//! **weight-based branch/kill**, and **population control** to a target size.
//!
//! This module is **not** quantum DMC. Energies are a classical objective
//! `f(x)`. There is no trial wavefunction, fixed-node constraint, or
//! electronic Hamiltonian. The claimed product is a budgeted global-search
//! mechanism with multi-walker bookkeeping, not a ground-state projector.
//!
//! Pure dynamics (propose, weight, resample) are unit-tested without I/O.

use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::movekernel::reflect_into_box;

/// Default target walker population when the caller does not set one.
pub const DEFAULT_TARGET_WALKERS: usize = 16;
/// Diffusion steps between population-control events.
pub const DEFAULT_STEPS_PER_CONTROL: usize = 4;
/// Initial inverse-temperature scale relative to energy gap heuristics.
pub const DEFAULT_BETA0: f64 = 1.0;

/// One walker: position and last evaluated energy.
#[derive(Clone, Debug)]
pub struct Walker {
    pub pos: Array1<f64>,
    pub energy: f64,
}

/// Population of walkers with a target size for control.
#[derive(Clone, Debug)]
pub struct Population {
    pub walkers: Vec<Walker>,
    pub target_n: usize,
}

impl Population {
    /// Build `n` walkers by reflecting random (or provided) starts into the box.
    pub fn new_random<R: Rng>(
        n: usize,
        bounds: &Bounds<f64>,
        energy_fn: impl Fn(ArrayView1<f64>) -> f64,
        rng: &mut R,
    ) -> Self {
        let n = n.max(1);
        let dim = bounds.dims;
        let mut walkers = Vec::with_capacity(n);
        for _ in 0..n {
            let mut x = Array1::zeros(dim);
            for i in 0..dim {
                let lo = bounds.low[i];
                let hi = bounds.high[i];
                x[i] = if hi > lo {
                    lo + rng.random::<f64>() * (hi - lo)
                } else {
                    lo
                };
            }
            let x = reflect_into_box(x.view(), bounds);
            let energy = energy_fn(x.view());
            walkers.push(Walker { pos: x, energy });
        }
        Self {
            walkers,
            target_n: n,
        }
    }

    /// Best finite energy in the population.
    pub fn best_energy(&self) -> f64 {
        self.walkers
            .iter()
            .map(|w| w.energy)
            .filter(|e| e.is_finite())
            .fold(f64::INFINITY, f64::min)
    }

    /// Position of the best finite walker (or box center if none).
    pub fn best_pos(&self, bounds: &Bounds<f64>) -> Array1<f64> {
        let mut best_e = f64::INFINITY;
        let mut best_x = None;
        for w in &self.walkers {
            if w.energy.is_finite() && w.energy < best_e {
                best_e = w.energy;
                best_x = Some(w.pos.clone());
            }
        }
        best_x.unwrap_or_else(|| (&bounds.low + &bounds.high) * 0.5)
    }

    pub fn len(&self) -> usize {
        self.walkers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.walkers.is_empty()
    }
}

/// Unnormalized DMC-style weight from energy relative to the population minimum.
///
/// `weight = exp(-beta * (E - E_min))`, floored away from zero for resampling.
pub fn walker_weight(energy: f64, e_min: f64, beta: f64) -> f64 {
    if !energy.is_finite() {
        return 1e-300;
    }
    let beta = beta.max(0.0);
    let de = (energy - e_min).max(0.0);
    (-beta * de).exp().max(1e-300)
}

/// Multinomial branch/kill: resample `target_n` walkers with replacement
/// proportional to weights (classical population control).
///
/// Walkers with higher weight are copied more often; low-weight walkers die.
/// This is the DMC engineering pattern of branching/killing to a fixed
/// population size, not independent multi-start.
pub fn population_control<R: Rng>(
    walkers: &[Walker],
    target_n: usize,
    beta: f64,
    rng: &mut R,
) -> Vec<Walker> {
    let target_n = target_n.max(1);
    if walkers.is_empty() {
        return Vec::new();
    }
    let e_min = walkers
        .iter()
        .map(|w| w.energy)
        .filter(|e| e.is_finite())
        .fold(f64::INFINITY, f64::min);
    let e_min = if e_min.is_finite() { e_min } else { 0.0 };
    let weights: Vec<f64> = walkers
        .iter()
        .map(|w| walker_weight(w.energy, e_min, beta))
        .collect();
    let total: f64 = weights.iter().sum();
    if !(total.is_finite() && total > 0.0) {
        // Degenerate: clone first walker.
        return vec![walkers[0].clone(); target_n];
    }
    let mut cdf = Vec::with_capacity(weights.len());
    let mut acc = 0.0;
    for w in &weights {
        acc += *w;
        cdf.push(acc / total);
    }
    let mut out = Vec::with_capacity(target_n);
    for _ in 0..target_n {
        let u = rng.random::<f64>();
        let idx = cdf
            .iter()
            .position(|&c| u <= c)
            .unwrap_or(walkers.len() - 1);
        out.push(walkers[idx].clone());
    }
    out
}

/// Isotropic diffusion proposal, reflected into the box.
///
/// `sigma` is per-coordinate Gaussian scale. Optional Langevin drift uses
/// a crude Euler–Maruyama step when `grad` is provided:  
/// `x <- x - 0.5 * dt * g + sigma * z` with `dt = sigma^2`.
pub fn diffusion_displace(
    x: ArrayView1<f64>,
    bounds: &Bounds<f64>,
    sigma: f64,
    grad: Option<ArrayView1<f64>>,
    rng: &mut impl Rng,
) -> Array1<f64> {
    let dim = x.len();
    let sigma = sigma.max(1e-12);
    let mut y = x.to_owned();
    if let Some(g) = grad {
        let dt = sigma * sigma;
        for i in 0..dim {
            let gi = if i < g.len() && g[i].is_finite() {
                g[i]
            } else {
                0.0
            };
            let z = {
                // Box-Muller
                let u1 = rng.random::<f64>().max(1e-12);
                let u2 = rng.random::<f64>();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            };
            y[i] = x[i] - 0.5 * dt * gi + sigma * z;
        }
    } else {
        for i in 0..dim {
            let z = {
                let u1 = rng.random::<f64>().max(1e-12);
                let u2 = rng.random::<f64>();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            };
            y[i] = x[i] + sigma * z;
        }
    }
    reflect_into_box(y.view(), bounds)
}

/// Mean box half-width for setting a default diffusion scale.
pub fn default_sigma(bounds: &Bounds<f64>) -> f64 {
    let dim = bounds.dims.max(1) as f64;
    let mut mean = 0.0;
    let mut n = 0.0;
    for i in 0..bounds.dims {
        let w = bounds.high[i] - bounds.low[i];
        if w.is_finite() && w > 0.0 {
            mean += w;
            n += 1.0;
        }
    }
    let mean = if n > 0.0 { mean / n } else { 1.0 };
    (0.15 * mean / dim.sqrt()).max(1e-6)
}

/// Result of a budgeted population-controlled diffusion run.
#[derive(Clone, Debug)]
pub struct DmcPopulationResult {
    pub best_pos: Array1<f64>,
    pub best_val: f64,
    pub n_evals: usize,
    pub n_grads: usize,
    pub final_population: usize,
    pub controls: usize,
}

/// Run population-controlled diffusion under a hard evaluation budget.
///
/// Each objective evaluation increments `n_evals`. Optional gradients for
/// Langevin drift increment `n_grads`. Population control (branch/kill to
/// `target_n`) runs every `steps_per_control` diffusion rounds.
pub fn run_dmc_population<O, G, R>(
    obj: &O,
    grad: Option<&G>,
    budget: usize,
    seed: u64,
    target_n: usize,
    steps_per_control: usize,
    beta0: f64,
    rng: &mut R,
) -> DmcPopulationResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
    R: Rng,
{
    let bounds = obj.bounds().clone();
    let dim = bounds.dims.max(1);
    let target_n = target_n.max(2).min(budget.max(2));
    let steps_per_control = steps_per_control.max(1);
    let mut n_evals = 0usize;
    let mut n_grads = 0usize;
    let mut controls = 0usize;

    let work = |ne: usize, ng: usize| ne + ng;
    let mut charge_obj = |x: ArrayView1<f64>, ne: &mut usize, ng: usize| -> Option<f64> {
        if work(*ne, ng) >= budget {
            return None;
        }
        *ne += 1;
        Some(obj.eval(x))
    };

    // Seed population (charges `target_n` evaluations).
    let mut pop = {
        let mut walkers = Vec::with_capacity(target_n);
        let mut rng_seed = StdRng::seed_from_u64(seed);
        for k in 0..target_n {
            if n_evals >= budget {
                break;
            }
            let mut x = Array1::zeros(dim);
            for i in 0..dim {
                let lo = bounds.low[i];
                let hi = bounds.high[i];
                x[i] = if hi > lo {
                    lo + rng_seed.random::<f64>() * (hi - lo)
                } else {
                    lo
                };
            }
            // Mild diversification from seed stream.
            let _ = k;
            let x = reflect_into_box(x.view(), &bounds);
            let e = match charge_obj(x.view(), &mut n_evals, n_grads) {
                Some(v) => v,
                None => break,
            };
            walkers.push(Walker { pos: x, energy: e });
        }
        if walkers.is_empty() {
            let mid = (&bounds.low + &bounds.high) * 0.5;
            return DmcPopulationResult {
                best_pos: mid,
                best_val: f64::INFINITY,
                n_evals,
                n_grads,
                final_population: 0,
                controls: 0,
            };
        }
        Population {
            target_n,
            walkers,
        }
    };

    let mut sigma = default_sigma(&bounds);
    let mut beta = beta0.max(1e-6);
    let mut step = 0usize;
    let mut best_val = pop.best_energy();
    let mut best_pos = pop.best_pos(&bounds);

    while work(n_evals, n_grads) < budget && !pop.walkers.is_empty() {
        // One diffusion round over the whole population.
        for w in &mut pop.walkers {
            if work(n_evals, n_grads) >= budget {
                break;
            }
            let g = if let Some(gr) = grad {
                if work(n_evals, n_grads) + 1 > budget {
                    None
                } else {
                    n_grads += 1;
                    Some(gr.grad(w.pos.view()))
                }
            } else {
                None
            };
            let y = diffusion_displace(
                w.pos.view(),
                &bounds,
                sigma,
                g.as_ref().map(|a| a.view()),
                rng,
            );
            let e = match charge_obj(y.view(), &mut n_evals, n_grads) {
                Some(v) => v,
                None => break,
            };
            // Metropolis filter at current beta (keeps diffusion from pure random walk).
            let accept = if e <= w.energy {
                true
            } else {
                let de = e - w.energy;
                rng.random::<f64>() < (-beta * de).exp()
            };
            if accept {
                w.pos = y;
                w.energy = e;
            }
            if e.is_finite() && e < best_val {
                best_val = e;
                best_pos = w.pos.clone();
            }
        }
        step += 1;
        // Cool beta gently; shrink sigma slowly (annealed diffusion).
        beta = beta0 * (1.0 + 0.15 * step as f64);
        sigma = (default_sigma(&bounds) / (1.0 + 0.05 * step as f64)).max(1e-6);

        if step % steps_per_control == 0 {
            pop.walkers = population_control(&pop.walkers, pop.target_n, beta, rng);
            controls += 1;
            // Re-evaluate after branch only if needed: cloned walkers keep energy.
            // Compact: drop non-finite.
            pop.walkers.retain(|w| w.energy.is_finite() || w.pos.iter().all(|x| x.is_finite()));
            if pop.walkers.is_empty() {
                break;
            }
        }
    }

    DmcPopulationResult {
        best_pos,
        best_val,
        n_evals,
        n_grads,
        final_population: pop.walkers.len(),
        controls,
    }
}

/// Convenience wrapper with a fresh RNG from `seed`.
pub fn dmc_population_optimize<O, G>(
    obj: &O,
    grad: Option<&G>,
    budget: usize,
    seed: u64,
) -> DmcPopulationResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    let mut rng = StdRng::seed_from_u64(seed ^ 0xd1c_00b0_u64);
    run_dmc_population(
        obj,
        grad,
        budget,
        seed,
        DEFAULT_TARGET_WALKERS,
        DEFAULT_STEPS_PER_CONTROL,
        DEFAULT_BETA0,
        &mut rng,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use eindir_core::{Bounds, Objective};
    use ndarray::{Array1, ArrayView1};

    struct Sphere {
        bounds: Bounds<f64>,
    }

    impl Sphere {
        fn new(dim: usize) -> Self {
            Self {
                bounds: Bounds::new(
                    Array1::from_elem(dim, -2.0),
                    Array1::from_elem(dim, 2.0),
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
    fn walker_weight_prefers_lower_energy() {
        let w_low = walker_weight(0.0, 0.0, 1.0);
        let w_high = walker_weight(2.0, 0.0, 1.0);
        assert!(w_low > w_high);
        assert!((w_low - 1.0).abs() < 1e-12);
    }

    #[test]
    fn population_control_preserves_target_size() {
        let mut rng = StdRng::seed_from_u64(7);
        let walkers: Vec<Walker> = (0..10)
            .map(|i| Walker {
                pos: Array1::from_elem(2, i as f64),
                energy: i as f64,
            })
            .collect();
        let out = population_control(&walkers, 16, 2.0, &mut rng);
        assert_eq!(out.len(), 16);
        // Low energy walker (i=0) should appear more often than high energy.
        let n0 = out.iter().filter(|w| w.energy == 0.0).count();
        let n9 = out.iter().filter(|w| w.energy == 9.0).count();
        assert!(n0 >= n9);
    }

    #[test]
    fn diffusion_stays_in_bounds() {
        let bounds = Bounds::new(
            Array1::from_vec(vec![-1.0, -1.0]),
            Array1::from_vec(vec![1.0, 1.0]),
            1e-12,
        );
        let mut rng = StdRng::seed_from_u64(1);
        let x = Array1::zeros(2);
        for _ in 0..50 {
            let y = diffusion_displace(x.view(), &bounds, 0.5, None, &mut rng);
            assert!(bounds.contains(y.view()), "y={y:?}");
        }
    }

    #[test]
    fn run_controls_population_and_improves_sphere() {
        let obj = Sphere::new(3);
        let mut rng = StdRng::seed_from_u64(11);
        let res = run_dmc_population::<_, Sphere, _>(
            &obj,
            None,
            400,
            11,
            12,
            3,
            1.0,
            &mut rng,
        );
        assert!(res.n_evals <= 400);
        assert!(res.final_population > 0);
        assert!(res.final_population <= 12 + 2); // control targets 12
        assert!(res.controls >= 1);
        assert!(res.best_val.is_finite());
        // Random init on [-2,2]^3 has mean energy ~ 3; diffusion+control should beat 1.0 often.
        assert!(
            res.best_val < 1.5,
            "expected sphere improvement, got {}",
            res.best_val
        );
        assert!(obj.bounds().contains(res.best_pos.view()));
    }

    /// Fixed-protocol head-to-head: population DMC vs pure multi-start sampling.
    ///
    /// Protocol: Rastrigin D=5 on [-5.12,5.12]^5, budget 800, seeds 0..4.
    /// Primary metric: mean best (lower better). DMC population must beat
    /// independent uniform multi-start with the same evaluation count.
    #[test]
    fn dmc_beats_uniform_multistart_on_rastrigin() {
        struct Rastrigin5 {
            bounds: Bounds<f64>,
        }
        impl Rastrigin5 {
            fn new() -> Self {
                Self {
                    bounds: Bounds::new(
                        Array1::from_elem(5, -5.12),
                        Array1::from_elem(5, 5.12),
                        1e-12,
                    ),
                }
            }
        }
        impl Objective<f64> for Rastrigin5 {
            fn dim(&self) -> usize {
                5
            }
            fn bounds(&self) -> &Bounds<f64> {
                &self.bounds
            }
            fn eval(&self, x: ArrayView1<f64>) -> f64 {
                let d = x.len() as f64;
                10.0 * d
                    + x.iter()
                        .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                        .sum::<f64>()
            }
        }
        let obj = Rastrigin5::new();
        let budget = 800usize;
        let seeds: [u64; 5] = [0, 1, 2, 3, 4];
        let mut dmc_bests = Vec::new();
        let mut uni_bests = Vec::new();
        for &seed in &seeds {
            let mut rng = StdRng::seed_from_u64(seed);
            let dmc = run_dmc_population::<_, Rastrigin5, _>(
                &obj, None, budget, seed, 16, 4, 1.0, &mut rng,
            );
            assert!(dmc.n_evals + dmc.n_grads <= budget);
            dmc_bests.push(dmc.best_val);

            // Uniform multi-start: same number of objective evaluations.
            let mut rng = StdRng::seed_from_u64(seed ^ 0x55aa);
            let mut best = f64::INFINITY;
            for _ in 0..budget {
                let mut x = Array1::zeros(5);
                for i in 0..5 {
                    x[i] = -5.12 + rng.random::<f64>() * (5.12 - -5.12);
                }
                let e = obj.eval(x.view());
                if e < best {
                    best = e;
                }
            }
            uni_bests.push(best);
        }
        let mean_dmc = dmc_bests.iter().sum::<f64>() / dmc_bests.len() as f64;
        let mean_uni = uni_bests.iter().sum::<f64>() / uni_bests.len() as f64;
        eprintln!(
            "rastrigin_d5 budget={budget} seeds={seeds:?} mean_dmc={mean_dmc:.4} mean_uniform={mean_uni:.4} dmc_bests={dmc_bests:?} uni_bests={uni_bests:?}"
        );
        assert!(
            mean_dmc < mean_uni,
            "DMC population mean best {mean_dmc} should beat uniform multi-start {mean_uni}"
        );
    }
}
