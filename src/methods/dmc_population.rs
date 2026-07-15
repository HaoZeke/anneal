//! Classical population-controlled diffusion search.
//!
//! Engineering pattern from the diffusion Monte Carlo literature (branching
//! random walks + population control; e.g. Reynolds–Ceperley–Alder–Lester
//! 1982; Foulkes et al. 2001) and sequential Monte Carlo residual resampling
//! (Liu). Tempering / multi-replica cooling is closer to population annealing
//! (Machta 2010) than to a quantum projector. Inter-walker DE/rand/1 proposals
//! follow the classical differential-evolution literature (Storn–Price).
//!
//! This module is **not** quantum DMC. Energies are a classical objective
//! `f(x)`. There is no trial wavefunction, fixed-node constraint, or
//! electronic Hamiltonian. The product is a budgeted multi-walker global
//! search with branch/kill bookkeeping.
//!
//! Loop structure: QMC-seeded population → mixture of isotropic diffusion,
//! adaptive DE (SHADE-style F/CR memory; current-to-pbest / rand / best),
//! and heavy-tailed long jumps → residual weight-based population control
//! with soft min/median reference → elite multi-start coordinate + projected
//! gradient polish.

use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::movekernel::reflect_into_box;

/// Default target walker population when the caller does not set one.
pub const DEFAULT_TARGET_WALKERS: usize = 24;
/// Diffusion steps between population-control events.
pub const DEFAULT_STEPS_PER_CONTROL: usize = 2;
/// Initial inverse-temperature scale relative to energy gap heuristics.
pub const DEFAULT_BETA0: f64 = 1.0;
/// Coordinate polish steps per elite refine burst.
const ELITE_COORD_POLISH: usize = 3;
/// SHADE-style success-history memory length.
const SHADE_H: usize = 5;
/// External archive cap as a multiple of population size (JADE/SHADE).
const ARCHIVE_MULT: usize = 2;

/// Sample a positive scale from a Cauchy location-scale truncated to (0, 1].
fn sample_cauchy_01<R: Rng>(loc: f64, scale: f64, rng: &mut R) -> f64 {
    for _ in 0..20 {
        let u = rng.random::<f64>().clamp(1e-12, 1.0 - 1e-12);
        let c = loc + scale * (std::f64::consts::PI * (u - 0.5)).tan();
        if c > 0.0 && c <= 1.0 {
            return c;
        }
    }
    loc.clamp(0.05, 1.0)
}

/// Weighted Lehmer mean for SHADE memory update.
fn lehmer_mean(vals: &[f64], weights: &[f64]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (v, w) in vals.iter().zip(weights.iter()) {
        num += w * v * v;
        den += w * v;
    }
    if den > 1e-18 {
        (num / den).clamp(0.05, 1.0)
    } else {
        0.5
    }
}

/// Recommend a walker count from the remaining evaluation budget.
pub fn recommend_target_n(budget: usize, dim: usize) -> usize {
    let dim = dim.max(1);
    // Leave most of the budget for proposals/polish, not init.
    let from_budget = ((budget as f64).sqrt() * 0.65).round() as usize;
    let from_dim = 6 + dim;
    let cap = (budget / 6).max(8);
    from_budget.max(from_dim).clamp(8, 36).min(cap)
}

/// One walker: position and last evaluated energy.
#[derive(Clone, Debug)]
pub struct Walker {
    /// Box-feasible coordinates.
    pub pos: Array1<f64>,
    /// Last evaluated objective value at `pos`.
    pub energy: f64,
}

/// Population of walkers with a target size for control.
#[derive(Clone, Debug)]
pub struct Population {
    /// Live walkers (length may transiently differ from `target_n`).
    pub walkers: Vec<Walker>,
    /// Population size restored after each control step.
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

    /// Current number of walkers.
    pub fn len(&self) -> usize {
        self.walkers.len()
    }

    /// Whether the population has no walkers.
    pub fn is_empty(&self) -> bool {
        self.walkers.is_empty()
    }
}

/// Unnormalized DMC-style weight from energy relative to a reference energy.
///
/// `weight = exp(-beta * (E - E_ref))`, floored away from zero for resampling.
/// Callers typically set `E_ref` to the population minimum or a soft quantile.
pub fn walker_weight(energy: f64, e_ref: f64, beta: f64) -> f64 {
    if !energy.is_finite() {
        return 1e-300;
    }
    let beta = beta.max(0.0);
    let de = (energy - e_ref).max(0.0);
    (-beta * de).exp().max(1e-300)
}

/// Soft reference energy for branching: blends min and median so mediocre
/// walkers are not instantly zeroed when the elite is far ahead.
fn branch_reference(energies: &[f64]) -> f64 {
    let mut finite: Vec<f64> = energies.iter().copied().filter(|e| e.is_finite()).collect();
    if finite.is_empty() {
        return 0.0;
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let e_min = finite[0];
    let e_med = finite[finite.len() / 2];
    0.65 * e_min + 0.35 * e_med
}

/// Residual + multinomial branch/kill (population control).
///
/// Residual resampling (standard sequential Monte Carlo / particle filtering)
/// first takes the integer part of each weight, then multinomial-fills the
/// remainder. This has lower variance than pure multinomial branching while
/// preserving the DMC engineering pattern of population control to a fixed
/// target size (Liu, *Monte Carlo Strategies in Scientific Computing*).
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
    let energies: Vec<f64> = walkers.iter().map(|w| w.energy).collect();
    let e_ref = branch_reference(&energies);
    let weights: Vec<f64> = walkers
        .iter()
        .map(|w| walker_weight(w.energy, e_ref, beta))
        .collect();
    let total: f64 = weights.iter().sum();
    if !(total.is_finite() && total > 0.0) {
        return vec![walkers[0].clone(); target_n];
    }
    // Expected copies under target_n.
    let expected: Vec<f64> = weights
        .iter()
        .map(|w| (*w / total) * target_n as f64)
        .collect();
    let mut out = Vec::with_capacity(target_n);
    let mut residual = Vec::with_capacity(walkers.len());
    let mut residual_w = Vec::with_capacity(walkers.len());
    for (i, &e) in expected.iter().enumerate() {
        let k = e.floor() as usize;
        for _ in 0..k {
            if out.len() < target_n {
                out.push(walkers[i].clone());
            }
        }
        let r = e - k as f64;
        if r > 0.0 {
            residual.push(i);
            residual_w.push(r);
        }
    }
    // Multinomial fill for residual mass.
    let need = target_n.saturating_sub(out.len());
    if need > 0 && !residual.is_empty() {
        let rtot: f64 = residual_w.iter().sum();
        if rtot > 0.0 && rtot.is_finite() {
            let mut cdf = Vec::with_capacity(residual.len());
            let mut acc = 0.0;
            for w in &residual_w {
                acc += *w;
                cdf.push(acc / rtot);
            }
            for _ in 0..need {
                let u = rng.random::<f64>();
                let j = cdf
                    .iter()
                    .position(|&c| u <= c)
                    .unwrap_or(residual.len() - 1);
                out.push(walkers[residual[j]].clone());
            }
        }
    }
    // Safety pad if residual under-filled.
    while out.len() < target_n {
        let i = rng.random_range(0..walkers.len());
        out.push(walkers[i].clone());
    }
    out.truncate(target_n);
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
    /// Best feasible point found under the budget.
    pub best_pos: Array1<f64>,
    /// Objective value at `best_pos`.
    pub best_val: f64,
    /// Objective evaluations charged.
    pub n_evals: usize,
    /// Gradient evaluations charged (Langevin drift).
    pub n_grads: usize,
    /// Walker count at exit.
    pub final_population: usize,
    /// Number of population-control (branch/kill) events.
    pub controls: usize,
}

/// Run population-controlled diffusion under a hard evaluation budget.
///
/// Each objective evaluation increments `n_evals`. Optional gradients for
/// Langevin drift increment `n_grads`. Population control (branch/kill to
/// `target_n`) runs every `steps_per_control` diffusion rounds.
///
/// When `seed_x` is `Some`, the first walker is placed at that point (clipped
/// into the box) so portfolio slices can continue from the incumbent.
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
    run_dmc_population_seeded(
        obj,
        grad,
        budget,
        seed,
        target_n,
        steps_per_control,
        beta0,
        None,
        rng,
    )
}

/// Same as [`run_dmc_population`] with an optional seed position for walker 0.
pub fn run_dmc_population_seeded<O, G, R>(
    obj: &O,
    grad: Option<&G>,
    budget: usize,
    seed: u64,
    target_n: usize,
    steps_per_control: usize,
    beta0: f64,
    seed_x: Option<ArrayView1<f64>>,
    rng: &mut R,
) -> DmcPopulationResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
    R: Rng,
{
    let bounds = obj.bounds().clone();
    let dim = bounds.dims.max(1);
    // Honour the caller's target; only clamp to a feasible range.
    let target_n = target_n.clamp(4, budget.max(4).min(64));
    let steps_per_control = steps_per_control.max(1);
    let mut n_evals = 0usize;
    let mut n_grads = 0usize;
    let mut controls = 0usize;

    let work = |ne: usize, ng: usize| ne + ng;
    let charge_obj = |x: ArrayView1<f64>, ne: &mut usize, ng: usize| -> Option<f64> {
        if work(*ne, ng) >= budget {
            return None;
        }
        *ne += 1;
        Some(obj.eval(x))
    };

    // Seed population: QMC covering set, with walker 0 optionally forced to seed_x.
    let mut pop = {
        let mut walkers = Vec::with_capacity(target_n);
        let qmc = eindir_core::shifted_low_discrepancy_points(&bounds, target_n, 1, seed);
        for k in 0..target_n {
            if n_evals >= budget {
                break;
            }
            let x = if k == 0 {
                if let Some(sx) = seed_x {
                    reflect_into_box(sx, &bounds)
                } else if k < qmc.nrows() {
                    reflect_into_box(qmc.row(k), &bounds)
                } else {
                    (&bounds.low + &bounds.high) * 0.5
                }
            } else if k < qmc.nrows() {
                reflect_into_box(qmc.row(k), &bounds)
            } else {
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
                reflect_into_box(x.view(), &bounds)
            };
            // A few walkers near the seed/incumbent for local exploitation.
            let x = if k > 0 && k <= 3 {
                if let Some(sx) = seed_x {
                    let mut y = sx.to_owned();
                    let jitter = default_sigma(&bounds) * 0.25;
                    for i in 0..dim {
                        let u1 = rng.random::<f64>().max(1e-12);
                        let u2 = rng.random::<f64>();
                        let z = (-2.0 * u1.ln()).sqrt()
                            * (2.0 * std::f64::consts::PI * u2).cos();
                        y[i] += jitter * z;
                    }
                    reflect_into_box(y.view(), &bounds)
                } else {
                    x
                }
            } else {
                x
            };
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

    let base_sigma = default_sigma(&bounds) * 2.0;
    let mut sigma = base_sigma;
    let mut beta = (beta0 * 0.15).max(1e-6);
    let mut step = 0usize;
    let mut best_val = pop.best_energy();
    let mut best_pos = pop.best_pos(&bounds);
    // Adaptive split: small budgets polish earlier so the elite is refined.
    let polish_frac = if budget < 1500 {
        0.48
    } else if budget < 4000 {
        0.52
    } else {
        0.58
    };
    let polish_start = (budget as f64 * polish_frac) as usize;
    let target_n0 = pop.target_n;

    // SHADE-style success-history memories for F and CR (Tanabe & Fukunaga).
    let mut mem_f = [0.5_f64; SHADE_H];
    let mut mem_cr = [0.9_f64; SHADE_H];
    let mut mem_pos = 0usize;
    // JADE/SHADE external archive of discarded parents (for difference vectors).
    let mut archive: Vec<Array1<f64>> = Vec::new();
    let archive_cap = (target_n0 * ARCHIVE_MULT).max(8);
    let mut accept_ema = 0.35_f64;

    while work(n_evals, n_grads) < budget && !pop.walkers.is_empty() {
        let polishing = work(n_evals, n_grads) >= polish_start;
        if polishing {
            // Multi-start elite polish: top-k walkers + global best.
            let mut elites: Vec<(Array1<f64>, f64)> = pop
                .walkers
                .iter()
                .map(|w| (w.pos.clone(), w.energy))
                .collect();
            elites.push((best_pos.clone(), best_val));
            elites.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            elites.dedup_by(|a, b| (a.1 - b.1).abs() < 1e-14);
            let k_elite = elites.len().min(3).max(1);
            let mut ei = 0usize;
            while work(n_evals, n_grads) < budget {
                let (mut x, mut fx) = elites[ei % k_elite].clone();
                ei += 1;
                let p = (work(n_evals, n_grads) as f64 / budget as f64).clamp(0.0, 1.0);
                let local_sigma = (base_sigma * (0.10 - 0.08 * p)).max(1e-6);
                // Coordinate opposition + axial polls.
                for c in 0..dim {
                    if work(n_evals, n_grads) >= budget {
                        break;
                    }
                    let mid = 0.5 * (bounds.low[c] + bounds.high[c]);
                    let mut trial = x.clone();
                    trial[c] =
                        (mid - 1.05 * (x[c] - mid)).clamp(bounds.low[c], bounds.high[c]);
                    let e = match charge_obj(trial.view(), &mut n_evals, n_grads) {
                        Some(v) => v,
                        None => break,
                    };
                    if e < fx {
                        x = trial;
                        fx = e;
                    }
                }
                for _ in 0..ELITE_COORD_POLISH.max(1) {
                    if work(n_evals, n_grads) >= budget {
                        break;
                    }
                    let c = rng.random_range(0..dim);
                    let span = (bounds.high[c] - bounds.low[c]).abs().max(1e-12);
                    let step_c = local_sigma * span.max(1.0).min(2.0);
                    for dir in [-1.0_f64, 1.0] {
                        if work(n_evals, n_grads) >= budget {
                            break;
                        }
                        let mut trial = x.clone();
                        trial[c] = (x[c] + dir * step_c).clamp(bounds.low[c], bounds.high[c]);
                        let e = match charge_obj(trial.view(), &mut n_evals, n_grads) {
                            Some(v) => v,
                            None => break,
                        };
                        if e < fx {
                            x = trial;
                            fx = e;
                        }
                    }
                }
                // Projected gradient / Langevin burst when grad is available.
                if let Some(gr) = grad {
                    for _ in 0..4 {
                        if work(n_evals, n_grads) + 1 >= budget {
                            break;
                        }
                        n_grads += 1;
                        let g = gr.grad(x.view());
                        let step_len = local_sigma * local_sigma;
                        let mut y = x.clone();
                        for i in 0..dim {
                            let gi = if i < g.len() && g[i].is_finite() {
                                g[i]
                            } else {
                                0.0
                            };
                            let u1 = rng.random::<f64>().max(1e-12);
                            let u2 = rng.random::<f64>();
                            let z = (-2.0 * u1.ln()).sqrt()
                                * (2.0 * std::f64::consts::PI * u2).cos();
                            y[i] = x[i] - step_len * gi + local_sigma * 0.35 * z;
                        }
                        let y = reflect_into_box(y.view(), &bounds);
                        let e = match charge_obj(y.view(), &mut n_evals, n_grads) {
                            Some(v) => v,
                            None => break,
                        };
                        if e <= fx {
                            x = y;
                            fx = e;
                        }
                    }
                } else {
                    for _ in 0..3 {
                        if work(n_evals, n_grads) >= budget {
                            break;
                        }
                        let y = diffusion_displace(x.view(), &bounds, local_sigma, None, rng);
                        let e = match charge_obj(y.view(), &mut n_evals, n_grads) {
                            Some(v) => v,
                            None => break,
                        };
                        if e <= fx {
                            x = y;
                            fx = e;
                        }
                    }
                }
                if fx < best_val {
                    best_val = fx;
                    best_pos = x.clone();
                }
                // Update this elite slot for the next pass.
                if ei <= k_elite {
                    elites[(ei - 1) % k_elite] = (x, fx);
                }
            }
            break;
        }

        // Population phase: adaptive DE + diffusion + long jumps.
        let n_walkers = pop.walkers.len();
        let snapshot: Vec<Walker> = pop.walkers.clone();
        // p-best pool: top max(2, 0.15 N) for current-to-pbest.
        let mut order: Vec<usize> = (0..n_walkers).collect();
        order.sort_by(|&a, &b| {
            snapshot[a]
                .energy
                .partial_cmp(&snapshot[b].energy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let p_best_n = ((0.15 * n_walkers as f64).ceil() as usize).clamp(2, n_walkers.max(2));
        let mut sf_success = Vec::new();
        let mut scr_success = Vec::new();
        let mut s_delta = Vec::new();
        let mut n_accept = 0usize;
        let mut n_trial = 0usize;

        for (wi, w) in pop.walkers.iter_mut().enumerate() {
            if work(n_evals, n_grads) >= budget {
                break;
            }
            n_trial += 1;
            let u = rng.random::<f64>();
            let long_jump = wi * 2 >= n_walkers && u < 0.10;
            let use_de = !long_jump && n_walkers >= 4 && u < 0.70;
            // Sample F, CR from success-history (SHADE).
            let r_idx = rng.random_range(0..SHADE_H);
            let f_i = sample_cauchy_01(mem_f[r_idx], 0.1, rng);
            let cr_i = {
                let n01 = {
                    let u1 = rng.random::<f64>().max(1e-12);
                    let u2 = rng.random::<f64>();
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                };
                (mem_cr[r_idx] + 0.1 * n01).clamp(0.0, 1.0)
            };

            let y = if long_jump {
                let mut z = Array1::zeros(dim);
                let use_anchor = !order.is_empty() && rng.random::<f64>() < 0.55;
                let aidx = if use_anchor {
                    order[rng.random_range(0..p_best_n.min(order.len()))]
                } else {
                    0
                };
                for i in 0..dim {
                    let lo = bounds.low[i];
                    let hi = bounds.high[i];
                    let half = 0.5 * (hi - lo).abs().max(1e-12);
                    let t = (rng.random::<f64>() - 0.5) * std::f64::consts::PI * 0.9;
                    if use_anchor {
                        z[i] = snapshot[aidx].pos[i] + half * 0.25 * t.tan();
                    } else {
                        let mid = 0.5 * (lo + hi);
                        z[i] = mid + half * 0.4 * t.tan();
                    }
                }
                reflect_into_box(z.view(), &bounds)
            } else if use_de {
                // Prefer current-to-pbest (SHADE/JADE), then best/1, then rand/1.
                let rmode = rng.random::<f64>();
                let mode = if rmode < 0.5 {
                    0
                } else if rmode < 0.8 {
                    1
                } else {
                    2
                };
                let mut trial = Array1::zeros(dim);
                let j_rand = rng.random_range(0..dim);
                let pick_not = |forbid: &[usize], rng: &mut R| -> usize {
                    // Bounded rejection; fall back if population is tiny.
                    for _ in 0..32 {
                        let j = rng.random_range(0..n_walkers);
                        if !forbid.contains(&j) {
                            return j;
                        }
                    }
                    (wi + 1) % n_walkers
                };
                let arch_or_pop = |rng: &mut R| -> Array1<f64> {
                    if !archive.is_empty() && rng.random::<f64>() < 0.5 {
                        archive[rng.random_range(0..archive.len())].clone()
                    } else {
                        snapshot[rng.random_range(0..n_walkers)].pos.clone()
                    }
                };
                if mode == 0 && n_walkers >= 3 {
                    // current-to-pbest/1: x + F(x_pbest - x) + F(x_r1 - x_r2)
                    let pbest = order[rng.random_range(0..p_best_n.min(order.len()))];
                    let r1 = pick_not(&[wi, pbest], rng);
                    let x_r2 = arch_or_pop(rng);
                    for i in 0..dim {
                        if i == j_rand || rng.random::<f64>() < cr_i {
                            trial[i] = w.pos[i]
                                + f_i * (snapshot[pbest].pos[i] - w.pos[i])
                                + f_i * (snapshot[r1].pos[i] - x_r2[i]);
                        } else {
                            trial[i] = w.pos[i];
                        }
                    }
                } else if mode == 1 && n_walkers >= 3 {
                    // best/1
                    let best_i = order[0];
                    let r1 = pick_not(&[wi, best_i], rng);
                    let x_r2 = arch_or_pop(rng);
                    for i in 0..dim {
                        if i == j_rand || rng.random::<f64>() < cr_i {
                            trial[i] = snapshot[best_i].pos[i]
                                + f_i * (snapshot[r1].pos[i] - x_r2[i]);
                        } else {
                            trial[i] = w.pos[i];
                        }
                    }
                } else {
                    // rand/1
                    let r1 = pick_not(&[wi], rng);
                    let r2 = pick_not(&[wi, r1], rng);
                    let x_r3 = arch_or_pop(rng);
                    for i in 0..dim {
                        if i == j_rand || rng.random::<f64>() < cr_i {
                            trial[i] = snapshot[r1].pos[i]
                                + f_i * (snapshot[r2].pos[i] - x_r3[i]);
                        } else {
                            trial[i] = w.pos[i];
                        }
                    }
                }
                reflect_into_box(trial.view(), &bounds)
            } else {
                diffusion_displace(w.pos.view(), &bounds, sigma, None, rng)
            };
            let e = match charge_obj(y.view(), &mut n_evals, n_grads) {
                Some(v) => v,
                None => break,
            };
            // DE is greedy (Storn–Price); diffusion/long-jump use Metropolis.
            let accept = if e <= w.energy {
                true
            } else if use_de {
                false
            } else if long_jump {
                rng.random::<f64>() < (-beta * (e - w.energy)).exp() * 0.3
            } else {
                rng.random::<f64>() < (-beta * (e - w.energy)).exp()
            };
            if e.is_finite() && e < best_val {
                best_val = e;
                best_pos = y.clone();
            }
            if accept {
                n_accept += 1;
                if use_de {
                    let delta = (w.energy - e).max(0.0);
                    sf_success.push(f_i);
                    scr_success.push(cr_i);
                    s_delta.push(delta.max(1e-18));
                }
                // Archive discarded parent (JADE).
                if archive.len() < archive_cap {
                    archive.push(w.pos.clone());
                } else if !archive.is_empty() && rng.random::<f64>() < 0.25 {
                    let j = rng.random_range(0..archive.len());
                    archive[j] = w.pos.clone();
                }
                w.pos = y;
                w.energy = e;
            }
        }
        // Update SHADE memories from successful F/CR.
        if !sf_success.is_empty() {
            mem_f[mem_pos] = lehmer_mean(&sf_success, &s_delta);
            mem_cr[mem_pos] = lehmer_mean(&scr_success, &s_delta);
            mem_pos = (mem_pos + 1) % SHADE_H;
        }
        if n_trial > 0 {
            let rate = n_accept as f64 / n_trial as f64;
            accept_ema = 0.8 * accept_ema + 0.2 * rate;
        }
        step += 1;

        let mut energies: Vec<f64> = pop
            .walkers
            .iter()
            .map(|w| w.energy)
            .filter(|e| e.is_finite())
            .collect();
        energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = if energies.is_empty() {
            best_val
        } else {
            energies[energies.len() / 2]
        };
        let gap = (med - best_val).abs().max(1e-3);
        let progress = (work(n_evals, n_grads) as f64 / budget as f64).clamp(0.0, 1.0);
        // Population-annealing style cool: beta grows; sigma tracks acceptance.
        beta = (beta0 * (0.4 + 8.0 * progress) / gap).max(1e-6);
        let sigma_scale = if accept_ema < 0.2 {
            1.15
        } else if accept_ema > 0.5 {
            0.9
        } else {
            1.0
        };
        sigma = (base_sigma * (1.0 - 0.85 * progress) * sigma_scale).max(1e-6);
        pop.target_n = ((target_n0 as f64) * (1.0 - 0.5 * progress))
            .round()
            .max(8.0) as usize;

        if step % steps_per_control == 0 {
            pop.walkers = population_control(&pop.walkers, pop.target_n, beta, rng);
            controls += 1;
            pop.walkers
                .retain(|w| w.energy.is_finite() && w.pos.iter().all(|x| x.is_finite()));
            if best_val.is_finite() && !pop.walkers.is_empty() {
                if let Some((wi, _)) = pop
                    .walkers
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.energy
                            .partial_cmp(&b.energy)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                {
                    pop.walkers[wi] = Walker {
                        pos: best_pos.clone(),
                        energy: best_val,
                    };
                }
            }
            // Inject diversity: re-seed worst 15% — half near best, half global QMC.
            if pop.walkers.len() >= 8 && work(n_evals, n_grads) + 2 < budget {
                let n_re = ((pop.walkers.len() as f64) * 0.15).ceil() as usize;
                let mut worst: Vec<usize> = (0..pop.walkers.len()).collect();
                worst.sort_by(|&a, &b| {
                    pop.walkers[b]
                        .energy
                        .partial_cmp(&pop.walkers[a].energy)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let fresh = eindir_core::shifted_low_discrepancy_points(
                    &bounds,
                    n_re,
                    step as u64 + 17,
                    seed ^ (step as u64).wrapping_mul(0x9e37),
                );
                for (k, &wi) in worst.iter().take(n_re).enumerate() {
                    if work(n_evals, n_grads) >= budget {
                        break;
                    }
                    let y = if k % 2 == 0 {
                        let mut y = best_pos.clone();
                        let jit = (sigma * 3.0).max(default_sigma(&bounds) * 0.5);
                        for i in 0..dim {
                            let u1 = rng.random::<f64>().max(1e-12);
                            let u2 = rng.random::<f64>();
                            let z = (-2.0 * u1.ln()).sqrt()
                                * (2.0 * std::f64::consts::PI * u2).cos();
                            y[i] += jit * z;
                        }
                        reflect_into_box(y.view(), &bounds)
                    } else if k < fresh.nrows() {
                        reflect_into_box(fresh.row(k), &bounds)
                    } else {
                        continue;
                    };
                    if let Some(e) = charge_obj(y.view(), &mut n_evals, n_grads) {
                        pop.walkers[wi] = Walker { pos: y, energy: e };
                        if e < best_val {
                            best_val = e;
                            best_pos = pop.walkers[wi].pos.clone();
                        }
                    }
                }
            }
            // Short elite refine after every control.
            if best_val.is_finite() {
                let mut x = best_pos.clone();
                let mut fx = best_val;
                let refine_sigma = (sigma * 0.18).max(1e-6);
                for _ in 0..8 {
                    if work(n_evals, n_grads) >= budget {
                        break;
                    }
                    let y = diffusion_displace(x.view(), &bounds, refine_sigma, None, rng);
                    let e = match charge_obj(y.view(), &mut n_evals, n_grads) {
                        Some(v) => v,
                        None => break,
                    };
                    if e <= fx {
                        x = y;
                        fx = e;
                    }
                }
                if fx < best_val {
                    best_val = fx;
                    best_pos = x.clone();
                }
                if let Some(w) = pop.walkers.first_mut() {
                    w.pos = best_pos.clone();
                    w.energy = best_val;
                }
            }
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
    let dim = obj.dim().max(1);
    let n = recommend_target_n(budget, dim);
    run_dmc_population(
        obj,
        grad,
        budget,
        seed,
        n,
        DEFAULT_STEPS_PER_CONTROL,
        DEFAULT_BETA0,
        &mut rng,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use eindir_core::{Bounds, Gradient, Objective};
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
        impl Gradient<f64> for Rastrigin5 {
            fn dim(&self) -> usize {
                5
            }
            fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
                x.mapv(|xi| 2.0 * xi + 20.0 * std::f64::consts::PI * (2.0 * std::f64::consts::PI * xi).sin())
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

    /// Head-to-head vs classical logarithmic Boltzmann SA (same obj budget).
    #[test]
    fn dmc_beats_classical_boltzmann_on_rastrigin() {
        use crate::runner::run_rs_variant;
        use crate::variant;

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
            fn dim(&self) -> usize { 5 }
            fn bounds(&self) -> &Bounds<f64> { &self.bounds }
            fn eval(&self, x: ArrayView1<f64>) -> f64 {
                let d = x.len() as f64;
                10.0 * d
                    + x.iter()
                        .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                        .sum::<f64>()
            }
        }
        impl Gradient<f64> for Rastrigin5 {
            fn dim(&self) -> usize { 5 }
            fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
                x.mapv(|xi| 2.0 * xi + 20.0 * std::f64::consts::PI * (2.0 * std::f64::consts::PI * xi).sin())
            }
        }
        // Fair protocol: both methods spend objective-only work units (no
        // gradient charge on the classical SA side, so DMC also omits grads).
        let budget = 1200usize;
        let seeds: [u64; 5] = [0, 1, 2, 3, 4];
        let mut dmc_bests = Vec::new();
        let mut sa_bests = Vec::new();
        for &seed in &seeds {
            let obj = Rastrigin5::new();
            let mut rng = StdRng::seed_from_u64(seed);
            let dmc = run_dmc_population::<_, Rastrigin5, _>(
                &obj, None, budget, seed, 16, 3, 1.0, &mut rng,
            );
            dmc_bests.push(dmc.best_val);

            // Classical SA: epochs * steps ≈ budget objective evals.
            let steps = 30usize;
            let epochs = (budget / steps).max(5);
            let obj2 = Rastrigin5::new();
            let variant = variant::boltzmann(obj2, 8.0, 0.5).expect("boltzmann");
            let hist = run_rs_variant(variant, epochs, steps, seed);
            sa_bests.push(hist.best.val);
        }
        let mean_dmc = dmc_bests.iter().sum::<f64>() / dmc_bests.len() as f64;
        let mean_sa = sa_bests.iter().sum::<f64>() / sa_bests.len() as f64;
        eprintln!(
            "vs_classical_boltzmann budget={budget} mean_dmc={mean_dmc:.4} mean_sa={mean_sa:.4} dmc={dmc_bests:?} sa={sa_bests:?}"
        );
        assert!(
            mean_dmc < mean_sa,
            "DMC mean {mean_dmc} should beat classical SA mean {mean_sa}"
        );
    }

}
