//! Classical population-controlled diffusion search (`dmc_pop`).
//!
//! Budgeted multi-walker global search for box-constrained continuous
//! objectives. Walkers propose via adaptive DE (SHADE F/CR memory;
//! current-to-pbest / best / rand), Tsallis/GSA long jumps, and isotropic
//! diffusion; residual resampling controls population size with **D8
//! entropy-calibrated inverse temperature** (`calibrate_beta` /
//! `population_control_ecit`); dual-style visit→L-BFGS polish (or
//! derivative-free local search) refines elites.
//!
//! Engineering pattern draws on diffusion Monte Carlo multi-walker
//! bookkeeping (Reynolds–Ceperley–Alder–Lester; Foulkes et al.) with
//! classical `f(x)` only — no trial wavefunction or electronic Hamiltonian.
//!
//! Public API: [`dmc_population_optimize`], [`run_dmc_population_seeded`]
//! (Python `anneal.dmc_population_optimize`, portfolio `DmcPop` arm).

use std::cell::RefCell;
use std::time::{Duration, Instant};

use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::accept::{AcceptRule, TsallisAccept};
use crate::methods::local_polish::projected_gradient_polish;
use crate::movekernel::{MoveKernel, TsallisVisit, reflect_into_box};

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
///
/// A success by a walker that sat at `+inf` improves by `+inf`. Weighed as
/// such it turns the mean into `inf / inf`, and a NaN scale factor makes every
/// later trial coordinate NaN, so it is weighed like the largest finite
/// improvement instead.
fn lehmer_mean(vals: &[f64], weights: &[f64]) -> f64 {
    let largest = weights
        .iter()
        .copied()
        .filter(|w| w.is_finite())
        .fold(0.0_f64, f64::max);
    let cap = if largest > 0.0 { largest } else { 1.0 };
    let mut num = 0.0;
    let mut den = 0.0;
    for (v, w) in vals.iter().zip(weights.iter()) {
        let w = if w.is_finite() { *w } else { cap };
        num += w * v * v;
        den += w * v;
    }
    let mean = num / den;
    if den > 1e-18 && mean.is_finite() {
        mean.clamp(0.05, 1.0)
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

// ---------------------------------------------------------------------------
// D8: Entropy-calibrated inverse temperature (ECIT) for residual control.
// ---------------------------------------------------------------------------

/// Pure D8.2 softmax probabilities \(p_i\propto e^{-\beta E_i}\) (normalized).
///
/// Non-finite energies get mass 0; if every energy is non-finite, returns a
/// uniform distribution over the length of `energies`. Shift by \(\min E\)
/// is for numerical stability only.
pub fn softmax_probs(energies: &[f64], beta: f64) -> Vec<f64> {
    let n = energies.len();
    if n == 0 {
        return Vec::new();
    }
    let mut e_min = f64::INFINITY;
    for &e in energies {
        if e.is_finite() && e < e_min {
            e_min = e;
        }
    }
    if !e_min.is_finite() {
        return vec![1.0 / n as f64; n];
    }
    let beta = beta.max(0.0);
    let mut weights = Vec::with_capacity(n);
    let mut z = 0.0;
    for &e in energies {
        let w = if e.is_finite() {
            (-beta * (e - e_min)).exp()
        } else {
            0.0
        };
        weights.push(w);
        z += w;
    }
    if !(z.is_finite() && z > 0.0) {
        return vec![1.0 / n as f64; n];
    }
    for w in &mut weights {
        *w /= z;
    }
    weights
}

/// Shannon entropy \(H(\beta)=-\sum p_i\log p_i\) of the exponential family
/// \(p_i\propto e^{-\beta E_i}\) (D8.3). Energies are shifted by their min
/// for numerical stability; the shift does not change \(H\).
pub fn softmax_entropy(energies: &[f64], beta: f64) -> f64 {
    let p = softmax_probs(energies, beta);
    if p.is_empty() {
        return 0.0;
    }
    let mut h = 0.0;
    for &pi in &p {
        if pi > 0.0 {
            h -= pi * pi.ln();
        }
    }
    h
}

/// Shannon entropy of an arbitrary positive mass vector (normalized in-place
/// for the sum). Used to audit residual weight laws against \(H_\star\).
pub fn mass_entropy(weights: &[f64]) -> f64 {
    let total: f64 = weights
        .iter()
        .copied()
        .filter(|w| w.is_finite() && *w > 0.0)
        .sum();
    if !(total.is_finite() && total > 0.0) {
        return 0.0;
    }
    let mut h = 0.0;
    for &w in weights {
        if w.is_finite() && w > 0.0 {
            let p = w / total;
            h -= p * p.ln();
        }
    }
    h
}

/// Progress-linked target entropy (D8.5):
/// \(H_\star(\rho)=(1-\rho)\log N+\rho\log(\max\{1,f_{\mathrm{elite}}\})\).
///
/// `elite_floor` is the elite count in entropy units (default 2).
pub fn target_entropy(n: usize, progress: f64, elite_floor: f64) -> f64 {
    let n = n.max(1);
    let rho = progress.clamp(0.0, 1.0);
    let floor_n = elite_floor.clamp(1.0, n as f64);
    (1.0 - rho) * (n as f64).ln() + rho * floor_n.ln()
}

/// Safe bisection upper bound \(\beta_{\max}=20\log N / R\) (D8.6).
pub fn beta_search_max(energies: &[f64]) -> f64 {
    let finite: Vec<f64> = energies.iter().copied().filter(|e| e.is_finite()).collect();
    if finite.len() < 2 {
        return 0.0;
    }
    let e_min = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let e_max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let r = e_max - e_min;
    if r <= 0.0 {
        return 0.0;
    }
    20.0 * (finite.len() as f64).ln() / r
}

/// Unique \(\beta^\star=H^{-1}(H_\star)\) by bisection (Corollary D8.4).
pub fn calibrate_beta(energies: &[f64], h_star: f64) -> f64 {
    let finite: Vec<f64> = energies.iter().copied().filter(|e| e.is_finite()).collect();
    let n = finite.len();
    if n < 2 {
        return 0.0;
    }
    let h0 = softmax_entropy(&finite, 0.0);
    let e_min = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let m = finite
        .iter()
        .filter(|&&e| (e - e_min).abs() < 1e-15 * (1.0 + e_min.abs()))
        .count()
        .max(1);
    let h_inf = (m as f64).ln();
    // Degenerate spectrum: H is flat → any beta is equivalent; use 0.
    if h0 <= h_inf + 1e-9 {
        return 0.0;
    }
    let lo_h = h_inf + 1e-12;
    let h_star = if h_star < lo_h {
        lo_h
    } else if h_star > h0 {
        h0
    } else {
        h_star
    };
    const TOL: f64 = 1e-8;
    if (h_star - h0).abs() <= TOL {
        return 0.0;
    }
    let mut lo = 0.0;
    let mut hi = beta_search_max(&finite).max(1e-12);
    for _ in 0..40 {
        if softmax_entropy(&finite, hi) <= h_star + TOL {
            break;
        }
        hi *= 2.0;
    }
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        let h_mid = softmax_entropy(&finite, mid);
        if h_mid > h_star {
            lo = mid;
        } else {
            hi = mid;
        }
        if (h_mid - h_star).abs() <= TOL || (hi - lo) <= TOL * (1.0 + mid) {
            return mid;
        }
    }
    0.5 * (lo + hi)
}

/// Expected residual offspring counts \(m_i = N_{\mathrm{tgt}}\,w_i/\sum w\)
/// from a positive weight vector (length must match the walker count).
pub fn residual_expected_counts(weights: &[f64], target_n: usize) -> Vec<f64> {
    let target_n = target_n.max(1) as f64;
    let total: f64 = weights
        .iter()
        .copied()
        .filter(|w| w.is_finite() && *w > 0.0)
        .sum();
    if !(total.is_finite() && total > 0.0) {
        return vec![0.0; weights.len()];
    }
    weights
        .iter()
        .map(|&w| {
            if w.is_finite() && w > 0.0 {
                (w / total) * target_n
            } else {
                0.0
            }
        })
        .collect()
}

/// Residual + multinomial branch/kill given explicit positive weights.
///
/// Residual resampling (Liu): place \(\lfloor m_i\rfloor\) copies, then
/// multinomial-fill fractional remainders. Weight law is caller-supplied so
/// D8 pure-softmax and legacy soft-ref weights share one residual core.
pub fn population_control_weighted<R: Rng>(
    walkers: &[Walker],
    target_n: usize,
    weights: &[f64],
    rng: &mut R,
) -> Vec<Walker> {
    let target_n = target_n.max(1);
    if walkers.is_empty() {
        return Vec::new();
    }
    debug_assert_eq!(walkers.len(), weights.len());
    let expected = residual_expected_counts(weights, target_n);
    if expected.iter().all(|&e| e <= 0.0) {
        return vec![walkers[0].clone(); target_n];
    }
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
    while out.len() < target_n {
        let i = rng.random_range(0..walkers.len());
        out.push(walkers[i].clone());
    }
    out.truncate(target_n);
    out
}

/// Residual population control with **legacy soft-reference** weights
/// \(w_i=\exp(-\beta(E_i-E_{\mathrm{ref}})_+)\) where \(E_{\mathrm{ref}}\) is
/// the min/median blend from [`branch_reference`]. Prefer
/// [`population_control_ecit`] for the D8 pure-softmax residual law.
pub fn population_control<R: Rng>(
    walkers: &[Walker],
    target_n: usize,
    beta: f64,
    rng: &mut R,
) -> Vec<Walker> {
    if walkers.is_empty() {
        return Vec::new();
    }
    let energies: Vec<f64> = walkers.iter().map(|w| w.energy).collect();
    let e_ref = branch_reference(&energies);
    let weights: Vec<f64> = walkers
        .iter()
        .map(|w| walker_weight(w.energy, e_ref, beta))
        .collect();
    population_control_weighted(walkers, target_n, &weights, rng)
}

/// D8 ECIT residual masses: pure softmax \(p_i^\star(\beta^\star)\) (D8.2) at the
/// entropy-calibrated \(\beta^\star=H^{-1}(H_\star(\rho))\).
///
/// Returns `(normalized_probs, beta_star, h_star)`. These are the **exact**
/// residual weight law used by [`population_control_ecit`] (not the soft-ref
/// legacy path in [`population_control`]).
pub fn ecit_residual_probs(
    energies: &[f64],
    progress: f64,
    elite_floor: f64,
) -> (Vec<f64>, f64, f64) {
    let n = energies.iter().filter(|e| e.is_finite()).count().max(1);
    let h_star = target_entropy(n, progress, elite_floor);
    let beta_star = calibrate_beta(energies, h_star);
    let probs = softmax_probs(energies, beta_star);
    (probs, beta_star, h_star)
}

/// D8 ECIT residual control: calibrate \(\beta^\star\) to \(H_\star(\rho)\), form
/// pure D8.2 softmax masses, residual-resample. Returns `(offspring, beta_star)`.
pub fn population_control_ecit<R: Rng>(
    walkers: &[Walker],
    target_n: usize,
    progress: f64,
    elite_floor: f64,
    rng: &mut R,
) -> (Vec<Walker>, f64) {
    let energies: Vec<f64> = walkers.iter().map(|w| w.energy).collect();
    let (probs, beta_star, _h_star) = ecit_residual_probs(&energies, progress, elite_floor);
    let out = population_control_weighted(walkers, target_n, &probs, rng);
    (out, beta_star)
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

/// Wall-clock phases for one diffusion population run.
///
/// The five fields sum to the search wall. Setup is construction before
/// the main loop. Selection is ranking, proposals, and population control.
/// Potential-energy calls are objective and gradient work. Synchronization
/// is join and idle time inside a parallel region. Allocation is the
/// roster snapshot and the trial matrix.
#[derive(Clone, Copy, Debug, Default)]
pub struct DmcPhaseSeconds {
    /// Construction before the main loop, including pool setup folded in later.
    pub setup_s: f64,
    /// Ranking, proposals, accepts, and population control.
    pub selection_s: f64,
    /// Objective and gradient work, as a share of the wall.
    pub pes_s: f64,
    /// Join and idle time inside parallel regions.
    pub sync_s: f64,
    /// Roster snapshots and the trial matrix.
    pub alloc_s: f64,
}

impl DmcPhaseSeconds {
    /// Sum of the five phase walls.
    pub fn total_s(self) -> f64 {
        self.setup_s + self.selection_s + self.pes_s + self.sync_s + self.alloc_s
    }
}

const PH_SETUP: usize = 0;
const PH_SELECTION: usize = 1;
const PH_PES: usize = 2;
const PH_SYNC: usize = 3;
const PH_ALLOC: usize = 4;

struct PhaseMeter {
    cursor: Instant,
    bucket: usize,
    ns: [u128; 5],
    polish_sync_ns: u128,
    wave_sync_ns: u128,
}

impl PhaseMeter {
    fn new() -> Self {
        Self {
            cursor: Instant::now(),
            bucket: PH_SETUP,
            ns: [0; 5],
            polish_sync_ns: 0,
            wave_sync_ns: 0,
        }
    }

    fn flush(&mut self) {
        let now = Instant::now();
        let dt = now.saturating_duration_since(self.cursor).as_nanos();
        self.ns[self.bucket] = self.ns[self.bucket].saturating_add(dt);
        self.cursor = now;
    }

    fn set_bucket(&mut self, bucket: usize) {
        self.flush();
        self.bucket = bucket;
    }

    fn pause(&mut self) {
        self.flush();
    }

    fn resume_pes(&mut self, ns: u128) {
        self.ns[PH_PES] = self.ns[PH_PES].saturating_add(ns);
        self.cursor = Instant::now();
    }

    fn resume_parallel(&mut self, wall: Duration, busy: Duration, threads: usize, wave: bool) {
        let threads = threads.max(1) as u128;
        let wall_ns = wall.as_nanos();
        let pes_ns = busy.as_nanos() / threads;
        let pes_ns = pes_ns.min(wall_ns);
        let sync_ns = wall_ns.saturating_sub(pes_ns);
        self.ns[PH_PES] = self.ns[PH_PES].saturating_add(pes_ns);
        self.ns[PH_SYNC] = self.ns[PH_SYNC].saturating_add(sync_ns);
        if wave {
            self.wave_sync_ns = self.wave_sync_ns.saturating_add(sync_ns);
        } else {
            self.polish_sync_ns = self.polish_sync_ns.saturating_add(sync_ns);
        }
        self.cursor = Instant::now();
    }

    fn snapshot(&mut self) -> DmcPhaseSeconds {
        self.flush();
        let sec = |ns: u128| ns as f64 / 1.0e9;
        DmcPhaseSeconds {
            setup_s: sec(self.ns[PH_SETUP]),
            selection_s: sec(self.ns[PH_SELECTION]),
            pes_s: sec(self.ns[PH_PES]),
            sync_s: sec(self.ns[PH_SYNC]),
            alloc_s: sec(self.ns[PH_ALLOC]),
        }
    }

    fn polish_sync_s(&self) -> f64 {
        self.polish_sync_ns as f64 / 1.0e9
    }

    fn wave_sync_s(&self) -> f64 {
        self.wave_sync_ns as f64 / 1.0e9
    }
}

/// Record a strictly better incumbent and the potential calls spent so far.
fn mark_best(best_val: &mut f64, candidate: f64, work_now: usize, first_hit: &mut usize) -> bool {
    if candidate.is_finite() && candidate < *best_val {
        *best_val = candidate;
        *first_hit = work_now;
        true
    } else {
        false
    }
}

/// p-best focus already used when a proposal picks parents.
///
/// The count is the ceiling of fifteen percent of the roster, and it is
/// at least two when the roster can hold two parents.
fn focus_slots(roster: usize) -> usize {
    let upper = roster.max(2);
    ((0.15 * roster as f64).ceil() as usize).clamp(2, upper)
}

/// Wave width is the p-best focus of the roster, limited by the elites on hand.
///
/// The tail budget is split across that focus, not across every elite.
/// A wider split would thin each L-BFGS. The pool only schedules those passes.
fn derived_wave_width(elite_count: usize, roster: usize) -> usize {
    focus_slots(roster).min(elite_count.max(1)).max(1)
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
    /// Proposal accepts. Gradient polish updates are not accepts.
    pub n_accepts: usize,
    /// Phase walls for this run. The fields sum to the search wall.
    pub phases: DmcPhaseSeconds,
    /// Idle time inside per-control gradient passes. This is already part of `phases.sync_s`.
    pub polish_sync_s: f64,
    /// Idle time inside endgame elite waves. This is already part of `phases.sync_s`.
    pub wave_sync_s: f64,
    /// Potential calls spent when the final best value first appeared.
    pub first_hit_work: usize,
    /// Elite-wave width used for this run. Zero when the wave did not run.
    pub wave_width: usize,
}

/// Run population-controlled diffusion under a hard evaluation budget.
///
/// Each objective evaluation increments `n_evals`. Optional gradients for
/// Langevin drift increment `n_grads`. Population control (branch/kill to
/// `target_n`) runs every `steps_per_control` diffusion rounds.
///
/// When `seed_x` is `Some`, the first walker is placed at that point (clipped
/// into the box) so portfolio slices can continue from the incumbent.
#[allow(clippy::too_many_arguments)]
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
    O: Objective<f64> + Sync,
    G: Gradient<f64> + Sync,
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
#[allow(clippy::too_many_arguments)]
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
    O: Objective<f64> + Sync,
    G: Gradient<f64> + Sync,
    R: Rng,
{
    run_dmc_population_seeded_with_proposals(
        obj,
        grad,
        budget,
        seed,
        target_n,
        steps_per_control,
        beta0,
        seed_x,
        rng,
        None,
    )
}

struct ElitePass {
    pos: Array1<f64>,
    energy: f64,
    n_evals: usize,
    n_grads: usize,
}

/// One gradient-polish iteration with a private budget and a private generator.
///
/// The caller snapshots the elite and the generator before the pool runs a
/// wave, so the iteration does not read another slot's result.
fn gradient_elite_pass<O, G>(
    obj: &O,
    gr: &G,
    mut x: Array1<f64>,
    mut fx: f64,
    cap: usize,
    dim: usize,
    e_span: f64,
    bounds: &Bounds<f64>,
    visit: &TsallisVisit,
    rng: &mut StdRng,
) -> ElitePass
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    let mut n_evals = 0usize;
    let mut n_grads = 0usize;
    let spent = |ne: usize, ng: usize| ne + ng;
    if cap < 3 {
        return ElitePass {
            pos: x,
            energy: fx,
            n_evals,
            n_grads,
        };
    }
    let remain = cap;
    let max_fe = (remain / 3).max(4).min((2 * dim + 24).max(16));
    let step0 = (default_sigma(bounds) * 0.5).max(1e-4);
    let pol = projected_gradient_polish(obj, gr, x.clone(), max_fe, step0, 1e-10);
    let room = cap.saturating_sub(spent(n_evals, n_grads));
    let charged = (pol.n_evals + pol.n_grads).min(room);
    let ce = pol.n_evals.min(charged);
    n_evals += ce;
    n_grads += (charged - ce).min(pol.n_grads);
    if pol.best_val.is_finite() && pol.best_val <= fx {
        x = pol.best_pos;
        fx = pol.best_val;
    }
    let hops = if remain > 4 * dim { 3 } else { 2 };
    for hop in 0..hops {
        if spent(n_evals, n_grads) + 8 >= cap {
            break;
        }
        let t_hop = (e_span * (0.8 / (1.0 + hop as f64))).max(0.5);
        let raw = visit.propose(x.view(), t_hop, rng);
        let y = reflect_into_box(raw.view(), bounds);
        if spent(n_evals, n_grads) >= cap {
            break;
        }
        n_evals += 1;
        let ey = obj.eval(y.view());
        if !ey.is_finite() {
            continue;
        }
        let room2 = cap.saturating_sub(spent(n_evals, n_grads));
        if room2 < 6 {
            if ey < fx {
                x = y;
                fx = ey;
            }
            break;
        }
        let max_fe2 = (room2 / 2).max(4).min((2 * dim + 20).max(12));
        let pol2 = projected_gradient_polish(obj, gr, y.clone(), max_fe2, step0, 1e-10);
        let room3 = cap.saturating_sub(spent(n_evals, n_grads));
        let charged2 = (pol2.n_evals + pol2.n_grads).min(room3);
        let ce2 = pol2.n_evals.min(charged2);
        n_evals += ce2;
        n_grads += (charged2 - ce2).min(pol2.n_grads);
        if pol2.best_val.is_finite() && pol2.best_val < fx {
            x = pol2.best_pos;
            fx = pol2.best_val;
        } else if ey < fx {
            x = y;
            fx = ey;
        }
    }
    for hop in 0..3 {
        if spent(n_evals, n_grads) + 10 >= cap {
            break;
        }
        let t_hop = (e_span * (0.5 / (1.0 + hop as f64))).max(0.3);
        let raw = visit.propose(x.view(), t_hop, rng);
        let y = reflect_into_box(raw.view(), bounds);
        if spent(n_evals, n_grads) >= cap {
            break;
        }
        n_evals += 1;
        let _ey = obj.eval(y.view());
        let room2 = cap.saturating_sub(spent(n_evals, n_grads));
        if room2 < 8 {
            break;
        }
        let max_fe2 = (room2 / 2).max(6).min((3 * dim + 24).max(16));
        let pol2 = projected_gradient_polish(obj, gr, y, max_fe2, step0, 1e-10);
        let room3 = cap.saturating_sub(spent(n_evals, n_grads));
        let charged2 = (pol2.n_evals + pol2.n_grads).min(room3);
        let ce2 = pol2.n_evals.min(charged2);
        n_evals += ce2;
        n_grads += (charged2 - ce2).min(pol2.n_grads);
        if pol2.best_val.is_finite() && pol2.best_val < fx {
            x = pol2.best_pos;
            fx = pol2.best_val;
        }
    }
    ElitePass {
        pos: x,
        energy: fx,
        n_evals,
        n_grads,
    }
}

/// Gradient-polish waves on the current rayon pool.
///
/// Each wave snapshots every slot it will polish and draws that slot's
/// generator on the calling thread. The pool runs the passes, and the
/// merge writes them in slot order. Width is the p-best focus of the roster.
/// One worker and many workers share that list and the same per-pass cap.
#[allow(clippy::too_many_arguments)]
fn gradient_polish_waves<O, G, R>(
    obj: &O,
    gr: &G,
    bounds: &Bounds<f64>,
    visit: &TsallisVisit,
    rng: &mut R,
    elites: &mut Vec<(Array1<f64>, f64)>,
    best_pos: &mut Array1<f64>,
    best_val: &mut f64,
    n_evals: &mut usize,
    n_grads: &mut usize,
    budget: usize,
    dim: usize,
    e_span: f64,
    roster: usize,
    first_hit: &mut usize,
    wave_width: &mut usize,
    meter: &RefCell<PhaseMeter>,
) where
    O: Objective<f64> + Sync,
    G: Gradient<f64> + Sync,
    R: Rng,
{
    let k_elite = derived_wave_width(elites.len(), roster);
    *wave_width = k_elite;
    let mut ei = 0usize;
    while n_evals.saturating_add(*n_grads) < budget && !crate::caller_stop::raised() {
        let remain = budget.saturating_sub(n_evals.saturating_add(*n_grads));
        if remain < 3 {
            break;
        }
        let slots = k_elite.min((remain / 3).max(1));
        let share = (remain / slots).max(3);
        let mut caps = vec![share; slots];
        caps[slots - 1] = remain - share * (slots - 1);
        struct WaveJob {
            x: Array1<f64>,
            fx: f64,
            cap: usize,
            seed: u64,
            write_at: usize,
        }
        let mut jobs = Vec::with_capacity(slots);
        for s in 0..slots {
            let read_at = (ei + s) % k_elite;
            let (x, fx) = elites[read_at].clone();
            jobs.push(WaveJob {
                x,
                fx,
                cap: caps[s],
                seed: rng.random::<u64>(),
                write_at: (ei + s + 1) % k_elite,
            });
        }
        let spent_before = n_evals.saturating_add(*n_grads);
        let threads = rayon::current_num_threads().max(1);
        meter.borrow_mut().pause();
        let wave_started = Instant::now();
        let passes: Vec<(ElitePass, Duration)> = jobs
            .par_iter()
            .map(|job| {
                let mut child = StdRng::seed_from_u64(job.seed);
                let task_started = Instant::now();
                let pass = gradient_elite_pass(
                    obj,
                    gr,
                    job.x.clone(),
                    job.fx,
                    job.cap,
                    dim,
                    e_span,
                    bounds,
                    visit,
                    &mut child,
                );
                (pass, task_started.elapsed())
            })
            .collect();
        let wave_wall = wave_started.elapsed();
        let wave_busy = passes.iter().map(|(_, busy)| *busy).sum();
        meter
            .borrow_mut()
            .resume_parallel(wave_wall, wave_busy, threads, true);
        for (job, (pass, _)) in jobs.iter().zip(passes.iter()) {
            let room = budget.saturating_sub(n_evals.saturating_add(*n_grads));
            let charged = (pass.n_evals + pass.n_grads).min(room);
            let ce = pass.n_evals.min(charged);
            *n_evals += ce;
            *n_grads += (charged - ce).min(pass.n_grads);
            if mark_best(
                best_val,
                pass.energy,
                n_evals.saturating_add(*n_grads),
                first_hit,
            ) {
                *best_pos = pass.pos.clone();
            }
            elites[job.write_at] = (pass.pos.clone(), pass.energy);
            ei += 1;
            if k_elite >= 2 && ei.is_multiple_of(4) && n_evals.saturating_add(*n_grads) + 1 < budget
            {
                let a = elites[0].0.clone();
                let b = elites[1 + (ei % (k_elite - 1))].0.clone();
                let mut trial = Array1::zeros(dim);
                for i in 0..dim {
                    trial[i] = if rng.random::<f64>() < 0.5 {
                        a[i]
                    } else {
                        b[i]
                    };
                }
                let trial = reflect_into_box(trial.view(), bounds);
                if n_evals.saturating_add(*n_grads) < budget {
                    *n_evals += 1;
                    meter.borrow_mut().pause();
                    let eval_started = Instant::now();
                    let energy = obj.eval(trial.view());
                    let eval_ns = eval_started.elapsed().as_nanos();
                    meter.borrow_mut().resume_pes(eval_ns);
                    if energy < elites[k_elite - 1].1 {
                        elites[k_elite - 1] = (trial.clone(), energy);
                        elites.sort_by(|p, q| {
                            p.1.partial_cmp(&q.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                    if mark_best(
                        best_val,
                        energy,
                        n_evals.saturating_add(*n_grads),
                        first_hit,
                    ) {
                        *best_pos = trial;
                    }
                }
            }
        }
        if n_evals.saturating_add(*n_grads) == spent_before {
            break;
        }
    }
}

/// Global proposal correction precedes evaluation and walker-state updates.
/// Local refinement evaluates the raw objective without this correction.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn run_dmc_population_seeded_with_proposals<O, G, R>(
    obj: &O,
    grad: Option<&G>,
    budget: usize,
    seed: u64,
    target_n: usize,
    steps_per_control: usize,
    beta0: f64,
    seed_x: Option<ArrayView1<f64>>,
    rng: &mut R,
    prepare: Option<&dyn Fn(ArrayView1<f64>, &mut Array1<f64>, Option<usize>) -> bool>,
) -> DmcPopulationResult
where
    O: Objective<f64> + Sync,
    G: Gradient<f64> + Sync,
    R: Rng,
{
    let bounds = obj.bounds().clone();
    let dim = bounds.dims.max(1);
    // Honour the caller's target; only clamp to a feasible range.
    let target_n = target_n.clamp(4, budget.clamp(4, 64));
    let steps_per_control = steps_per_control.max(1);
    let mut n_evals = 0usize;
    let mut n_grads = 0usize;
    let mut controls = 0usize;
    let mut n_accepts = 0usize;

    let work = |ne: usize, ng: usize| ne + ng;
    let meter = RefCell::new(PhaseMeter::new());
    let charge_obj = |x: ArrayView1<f64>, ne: &mut usize, ng: usize| -> Option<f64> {
        if work(*ne, ng) >= budget {
            return None;
        }
        *ne += 1;
        meter.borrow_mut().pause();
        let eval_started = Instant::now();
        let value = obj.eval(x);
        let eval_ns = eval_started.elapsed().as_nanos();
        meter.borrow_mut().resume_pes(eval_ns);
        Some(value)
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
            let mut x = if k > 0 && k <= 3 {
                if let Some(sx) = seed_x {
                    let mut y = sx.to_owned();
                    let jitter = default_sigma(&bounds) * 0.25;
                    for i in 0..dim {
                        let u1 = rng.random::<f64>().max(1e-12);
                        let u2 = rng.random::<f64>();
                        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                        y[i] += jitter * z;
                    }
                    reflect_into_box(y.view(), &bounds)
                } else {
                    x
                }
            } else {
                x
            };
            // The supplied incumbent is an occupied state, not a new launch.
            if let Some(prepare) = prepare
                && (k > 0 || seed_x.is_none())
            {
                let anchor = seed_x.unwrap_or_else(|| x.view()).to_owned();
                prepare(anchor.view(), &mut x, None);
            }
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
                n_accepts: 0,
                phases: meter.borrow_mut().snapshot(),
                polish_sync_s: meter.borrow().polish_sync_s(),
                wave_sync_s: meter.borrow().wave_sync_s(),
                first_hit_work: 0,
                wave_width: 0,
            };
        }
        Population { target_n, walkers }
    };

    let base_sigma = default_sigma(&bounds) * 2.0;
    let mut sigma = base_sigma;
    let mut beta = (beta0 * 0.15).max(1e-6);
    let mut step = 0usize;
    let mut best_val = pop.best_energy();
    let mut best_pos = pop.best_pos(&bounds);
    let mut first_hit_work = work(n_evals, n_grads);
    let mut wave_width = 0usize;
    // Population phase long enough for DE to place walkers in good basins;
    // dual-style visit→L-BFGS endgame then spends the tail aggressively.
    let polish_frac = if budget < 1500 {
        0.50
    } else if budget < 4000 {
        0.55
    } else {
        0.60
    };
    let polish_start = (budget as f64 * polish_frac) as usize;
    let target_n0 = pop.target_n;
    // GSA visiting (same family as SciPy dual_annealing) for long jumps.
    let visit = TsallisVisit::new(2.62);
    let accept_rule = TsallisAccept::new(2.7);
    let e_span = {
        let es: Vec<f64> = pop
            .walkers
            .iter()
            .map(|w| w.energy)
            .filter(|e| e.is_finite())
            .collect();
        let lo = es.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = es.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (hi - lo).abs().max(1.0)
    };
    let mut visit_temp = (e_span * 2.0).max(1.0);

    // SHADE-style success-history memories for F and CR (Tanabe & Fukunaga).
    let mut mem_f = [0.5_f64; SHADE_H];
    let mut mem_cr = [0.9_f64; SHADE_H];
    let mut mem_pos = 0usize;
    // JADE/SHADE external archive of discarded parents (for difference vectors).
    let mut archive: Vec<Array1<f64>> = Vec::new();
    let archive_cap = (target_n0 * ARCHIVE_MULT).max(8);
    let mut accept_ema = 0.35_f64;

    while work(n_evals, n_grads) < budget
        && !pop.walkers.is_empty()
        && !crate::caller_stop::raised()
    {
        meter.borrow_mut().set_bucket(PH_SELECTION);
        let polishing = work(n_evals, n_grads) >= polish_start;
        if polishing {
            // Multi-start elite polish: L-BFGS when grad is available (dual
            // annealing style), else continuous coordinate/pattern DF search.
            // With a gradient and no proposal correction, each wave polishes
            // disjoint elite slots on the current rayon pool.
            if let Some(gr) = grad
                && prepare.is_none()
            {
                meter.borrow_mut().set_bucket(PH_ALLOC);
                let mut elites: Vec<(Array1<f64>, f64)> = pop
                    .walkers
                    .iter()
                    .map(|w| (w.pos.clone(), w.energy))
                    .collect();
                elites.push((best_pos.clone(), best_val));
                elites.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                elites.dedup_by(|a, b| (a.1 - b.1).abs() < 1e-14);
                meter.borrow_mut().set_bucket(PH_SELECTION);
                gradient_polish_waves(
                    obj,
                    gr,
                    &bounds,
                    &visit,
                    rng,
                    &mut elites,
                    &mut best_pos,
                    &mut best_val,
                    &mut n_evals,
                    &mut n_grads,
                    budget,
                    dim,
                    e_span,
                    target_n0,
                    &mut first_hit_work,
                    &mut wave_width,
                    &meter,
                );
                break;
            }
            let mut elites: Vec<(Array1<f64>, f64)> = pop
                .walkers
                .iter()
                .map(|w| (w.pos.clone(), w.energy))
                .collect();
            elites.push((best_pos.clone(), best_val));
            elites.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            elites.dedup_by(|a, b| (a.1 - b.1).abs() < 1e-14);
            let k_elite = derived_wave_width(elites.len(), target_n0);
            let mut ei = 0usize;
            // Dual annealing endgame: visit (Tsallis) → local (L-BFGS) cycles.
            while work(n_evals, n_grads) < budget && !crate::caller_stop::raised() {
                let (mut x, mut fx) = elites[ei % k_elite].clone();
                ei += 1;
                let remain = budget.saturating_sub(work(n_evals, n_grads));
                if remain < 3 {
                    break;
                }
                if let Some(gr) = grad {
                    // 1) Local L-BFGS polish from current elite.
                    let max_fe = (remain / 3).max(4).min((2 * dim + 24).max(16));
                    let step0 = (default_sigma(&bounds) * 0.5).max(1e-4);
                    let pol = projected_gradient_polish(obj, gr, x.clone(), max_fe, step0, 1e-10);
                    let room = budget.saturating_sub(work(n_evals, n_grads));
                    let charge = (pol.n_evals + pol.n_grads).min(room);
                    let ce = pol.n_evals.min(charge);
                    n_evals += ce;
                    n_grads += (charge - ce).min(pol.n_grads);
                    if pol.best_val.is_finite() && pol.best_val <= fx {
                        x = pol.best_pos;
                        fx = pol.best_val;
                    }
                    if mark_best(
                        &mut best_val,
                        fx,
                        work(n_evals, n_grads),
                        &mut first_hit_work,
                    ) {
                        best_pos = x.clone();
                    }
                    // 2) Dual-style basin hop: Tsallis visit then re-polish.
                    // Continuous visits explore basins independently of coordinate origin.
                    let hops = if remain > 4 * dim { 3 } else { 2 };
                    for hop in 0..hops {
                        if work(n_evals, n_grads) + 8 >= budget {
                            break;
                        }
                        let t_hop = (e_span * (0.8 / (1.0 + hop as f64))).max(0.5);
                        let raw = visit.propose(x.view(), t_hop, rng);
                        let mut y = reflect_into_box(raw.view(), &bounds);
                        if let Some(prepare) = prepare {
                            prepare(x.view(), &mut y, None);
                        }
                        let ey = match charge_obj(y.view(), &mut n_evals, n_grads) {
                            Some(v) => v,
                            None => break,
                        };
                        if !ey.is_finite() {
                            continue;
                        }
                        let room2 = budget.saturating_sub(work(n_evals, n_grads));
                        if room2 < 6 {
                            if ey < fx {
                                x = y;
                                fx = ey;
                            }
                            break;
                        }
                        let max_fe2 = (room2 / 2).max(4).min((2 * dim + 20).max(12));
                        let pol2 =
                            projected_gradient_polish(obj, gr, y.clone(), max_fe2, step0, 1e-10);
                        let room3 = budget.saturating_sub(work(n_evals, n_grads));
                        let charge2 = (pol2.n_evals + pol2.n_grads).min(room3);
                        let ce2 = pol2.n_evals.min(charge2);
                        n_evals += ce2;
                        n_grads += (charge2 - ce2).min(pol2.n_grads);
                        if pol2.best_val.is_finite() && pol2.best_val < fx {
                            x = pol2.best_pos;
                            fx = pol2.best_val;
                        } else if ey < fx {
                            x = y;
                            fx = ey;
                        }
                        if mark_best(
                            &mut best_val,
                            fx,
                            work(n_evals, n_grads),
                            &mut first_hit_work,
                        ) {
                            best_pos = x.clone();
                        }
                    }
                    // Additional continuous visits use the same local refinement
                    // regardless of proximity to any absolute coordinate lattice.
                    for hop in 0..3 {
                        if work(n_evals, n_grads) + 10 >= budget {
                            break;
                        }
                        let t_hop = (e_span * (0.5 / (1.0 + hop as f64))).max(0.3);
                        let raw = visit.propose(x.view(), t_hop, rng);
                        let mut y = reflect_into_box(raw.view(), &bounds);
                        if let Some(prepare) = prepare {
                            prepare(x.view(), &mut y, None);
                        }
                        if charge_obj(y.view(), &mut n_evals, n_grads).is_none() {
                            break;
                        }
                        let room2 = budget.saturating_sub(work(n_evals, n_grads));
                        if room2 < 8 {
                            break;
                        }
                        let max_fe2 = (room2 / 2).max(6).min((3 * dim + 24).max(16));
                        let pol2 = projected_gradient_polish(obj, gr, y, max_fe2, step0, 1e-10);
                        let room3 = budget.saturating_sub(work(n_evals, n_grads));
                        let charge2 = (pol2.n_evals + pol2.n_grads).min(room3);
                        let ce2 = pol2.n_evals.min(charge2);
                        n_evals += ce2;
                        n_grads += (charge2 - ce2).min(pol2.n_grads);
                        if pol2.best_val.is_finite() && pol2.best_val < fx {
                            x = pol2.best_pos;
                            fx = pol2.best_val;
                        }
                    }
                    if mark_best(
                        &mut best_val,
                        fx,
                        work(n_evals, n_grads),
                        &mut first_hit_work,
                    ) {
                        best_pos = x.clone();
                    }
                } else {
                    let p = (work(n_evals, n_grads) as f64 / budget as f64).clamp(0.0, 1.0);
                    let local_sigma = (base_sigma * (0.12 - 0.09 * p)).max(1e-6);
                    // Opposition + multi-scale axial.
                    for c in 0..dim {
                        if work(n_evals, n_grads) >= budget {
                            break;
                        }
                        let mid = 0.5 * (bounds.low[c] + bounds.high[c]);
                        let mut trial = x.clone();
                        trial[c] = (mid - 1.05 * (x[c] - mid)).clamp(bounds.low[c], bounds.high[c]);
                        if let Some(prepare) = prepare {
                            prepare(x.view(), &mut trial, Some(c));
                        }
                        let e = match charge_obj(trial.view(), &mut n_evals, n_grads) {
                            Some(v) => v,
                            None => break,
                        };
                        if e < fx {
                            x = trial;
                            fx = e;
                        }
                    }
                    for scale in [1.0_f64, 0.35, 0.12, 2.0] {
                        for _ in 0..ELITE_COORD_POLISH.max(1) {
                            if work(n_evals, n_grads) >= budget {
                                break;
                            }
                            let c = rng.random_range(0..dim);
                            let span = (bounds.high[c] - bounds.low[c]).abs().max(1e-12);
                            let step_c =
                                (local_sigma * scale * span.clamp(1.0, 2.0)).min(0.2 * span);
                            for dir in [-1.0_f64, 1.0] {
                                if work(n_evals, n_grads) >= budget {
                                    break;
                                }
                                let mut trial = x.clone();
                                trial[c] =
                                    (x[c] + dir * step_c).clamp(bounds.low[c], bounds.high[c]);
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
                    }
                    // Short isotropic refine.
                    for _ in 0..4 {
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
                if mark_best(
                    &mut best_val,
                    fx,
                    work(n_evals, n_grads),
                    &mut first_hit_work,
                ) {
                    best_pos = x.clone();
                }
                elites[ei % k_elite] = (x, fx);
                // Occasional elite crossover (basin recombination).
                if k_elite >= 2 && ei.is_multiple_of(4) && work(n_evals, n_grads) + 1 < budget {
                    let a = &elites[0].0;
                    let b = &elites[1 + (ei % (k_elite - 1))].0;
                    let mut trial = Array1::zeros(dim);
                    for i in 0..dim {
                        trial[i] = if rng.random::<f64>() < 0.5 {
                            a[i]
                        } else {
                            b[i]
                        };
                    }
                    let mut trial = reflect_into_box(trial.view(), &bounds);
                    if let Some(prepare) = prepare {
                        prepare(a.view(), &mut trial, None);
                    }
                    if let Some(e) = charge_obj(trial.view(), &mut n_evals, n_grads) {
                        if e < elites[k_elite - 1].1 {
                            elites[k_elite - 1] = (trial.clone(), e);
                            elites.sort_by(|p, q| {
                                p.1.partial_cmp(&q.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                        }
                        if mark_best(
                            &mut best_val,
                            e,
                            work(n_evals, n_grads),
                            &mut first_hit_work,
                        ) {
                            best_pos = trial;
                        }
                    }
                }
            }
            break;
        }

        // Population phase: propose (serial RNG) → batch-eval (rayon) → accept.
        meter.borrow_mut().set_bucket(PH_ALLOC);
        let n_walkers = pop.walkers.len();
        let snapshot: Vec<Walker> = pop.walkers.clone();
        meter.borrow_mut().set_bucket(PH_SELECTION);
        let mut order: Vec<usize> = (0..n_walkers).collect();
        order.sort_by(|&a, &b| {
            snapshot[a]
                .energy
                .partial_cmp(&snapshot[b].energy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let p_best_n = focus_slots(n_walkers);
        let mut sf_success = Vec::new();
        let mut scr_success = Vec::new();
        let mut s_delta = Vec::new();
        let mut n_accept = 0usize;
        let mut n_trial = 0usize;

        let remain = budget.saturating_sub(work(n_evals, n_grads));
        let n_prop = n_walkers.min(remain);
        if n_prop == 0 {
            break;
        }

        // Meta for accept after parallel eval.
        struct PropMeta {
            wi: usize,
            use_de: bool,
            long_jump: bool,
            f_i: f64,
            cr_i: f64,
            u_accept: f64,
        }
        meter.borrow_mut().set_bucket(PH_ALLOC);
        let mut metas = Vec::with_capacity(n_prop);
        let mut trial_mat = Array2::<f64>::zeros((n_prop, dim));
        meter.borrow_mut().set_bucket(PH_SELECTION);

        for (pi, wi) in (0..n_prop).enumerate() {
            let w = &snapshot[wi];
            n_trial += 1;
            let u = rng.random::<f64>();
            let long_jump = wi * 2 >= n_walkers && u < 0.12;
            let use_de = !long_jump && n_walkers >= 4 && u < 0.72;
            let r_idx = rng.random_range(0..SHADE_H);
            let f_i = sample_cauchy_01(mem_f[r_idx], 0.1, rng);
            let cr_i = {
                let u1 = rng.random::<f64>().max(1e-12);
                let u2 = rng.random::<f64>();
                let n01 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                (mem_cr[r_idx] + 0.1 * n01).clamp(0.0, 1.0)
            };
            let u_accept = rng.random::<f64>();

            let mut y = if long_jump {
                if rng.random::<f64>() < 0.65 {
                    let raw = visit.propose(w.pos.view(), visit_temp.max(1e-12), rng);
                    reflect_into_box(raw.view(), &bounds)
                } else {
                    let mut z = Array1::zeros(dim);
                    let aidx = if !order.is_empty() {
                        order[rng.random_range(0..p_best_n.min(order.len()))]
                    } else {
                        0
                    };
                    for i in 0..dim {
                        let lo = bounds.low[i];
                        let hi = bounds.high[i];
                        let half = 0.5 * (hi - lo).abs().max(1e-12);
                        let t = (rng.random::<f64>() - 0.5) * std::f64::consts::PI * 0.9;
                        z[i] = snapshot[aidx].pos[i] + half * 0.25 * t.tan();
                    }
                    reflect_into_box(z.view(), &bounds)
                }
            } else if use_de {
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
                    let best_i = order[0];
                    let r1 = pick_not(&[wi, best_i], rng);
                    let x_r2 = arch_or_pop(rng);
                    for i in 0..dim {
                        if i == j_rand || rng.random::<f64>() < cr_i {
                            trial[i] =
                                snapshot[best_i].pos[i] + f_i * (snapshot[r1].pos[i] - x_r2[i]);
                        } else {
                            trial[i] = w.pos[i];
                        }
                    }
                } else {
                    let r1 = pick_not(&[wi], rng);
                    let r2 = pick_not(&[wi, r1], rng);
                    let x_r3 = arch_or_pop(rng);
                    for i in 0..dim {
                        if i == j_rand || rng.random::<f64>() < cr_i {
                            trial[i] = snapshot[r1].pos[i] + f_i * (snapshot[r2].pos[i] - x_r3[i]);
                        } else {
                            trial[i] = w.pos[i];
                        }
                    }
                }
                reflect_into_box(trial.view(), &bounds)
            } else {
                diffusion_displace(w.pos.view(), &bounds, sigma, None, rng)
            };
            if let Some(prepare) = prepare {
                prepare(w.pos.view(), &mut y, None);
            }
            trial_mat.row_mut(pi).assign(&y);
            metas.push(PropMeta {
                wi,
                use_de,
                long_jump,
                f_i,
                cr_i,
                u_accept,
            });
        }

        // Multi-walker proposals evaluated as one batch.
        // - Python/CUTEst: CallableObjective::eval_batch → Counter.eval_batch
        // - Native Sync: eindir default is serial; use Rayon via eval_batch
        //   override only where provided (see eval_batch_parallel for explicit
        //   native fan-out at other call sites).
        meter.borrow_mut().pause();
        let batch_started = Instant::now();
        let energies = obj.eval_batch(trial_mat.view());
        let batch_ns = batch_started.elapsed().as_nanos();
        meter.borrow_mut().resume_pes(batch_ns);
        n_evals += energies.len();

        for (pi, meta) in metas.into_iter().enumerate() {
            let e = energies[pi];
            let y = trial_mat.row(pi).to_owned();
            let w = &mut pop.walkers[meta.wi];
            let de = e - w.energy;
            let accept = if e <= w.energy {
                true
            } else if meta.use_de {
                false
            } else if meta.long_jump {
                meta.u_accept
                    < accept_rule
                        .accept_prob(de, visit_temp.max(1e-12))
                        .clamp(0.0, 1.0)
            } else {
                meta.u_accept < (-beta * de).exp()
            };
            if mark_best(
                &mut best_val,
                e,
                work(n_evals, n_grads),
                &mut first_hit_work,
            ) {
                best_pos = y.clone();
            }
            if accept {
                n_accept += 1;
                if meta.use_de {
                    let delta = (w.energy - e).max(0.0);
                    sf_success.push(meta.f_i);
                    scr_success.push(meta.cr_i);
                    s_delta.push(delta.max(1e-18));
                }
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
        n_accepts += n_accept;
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
        // D8 ECIT: residual-control beta is calibrated to H_*(progress).
        // Tsallis visit temperature and diffusion scale still cool with progress.
        let _gap_beta = (beta0 * (0.4 + 8.0 * progress) / gap).max(1e-6);
        visit_temp = ((e_span * 2.0) * (1.0 - progress).powi(2)).max(1e-6);
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

        if step.is_multiple_of(steps_per_control) {
            // Algorithm D8: entropy-calibrated residual population control.
            let (next, beta_star) =
                population_control_ecit(&pop.walkers, pop.target_n, progress, 2.0, rng);
            pop.walkers = next;
            beta = beta_star.max(1e-6);
            controls += 1;
            pop.walkers
                .retain(|w| w.energy.is_finite() && w.pos.iter().all(|x| x.is_finite()));
            if best_val.is_finite()
                && !pop.walkers.is_empty()
                && let Some((wi, _)) = pop.walkers.iter().enumerate().max_by(|(_, a), (_, b)| {
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
            // Dual-style interleaved L-BFGS on elites when gradient is available.
            if let Some(gr) = grad
                && work(n_evals, n_grads) + 12 < polish_start.min(budget)
            {
                let remain = polish_start.saturating_sub(work(n_evals, n_grads));
                let per = ((2 * dim + 16).max(20))
                    .min(remain / 4)
                    .max(10)
                    .min(remain.saturating_sub(4));
                let mut order_w: Vec<usize> = (0..pop.walkers.len()).collect();
                order_w.sort_by(|&a, &b| {
                    pop.walkers[a]
                        .energy
                        .partial_cmp(&pop.walkers[b].energy)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                // One long polish is serial inside the timed wall. Split the same
                // per-control budget across the live walkers. The width is the
                // roster, not the thread count, so one worker and many workers
                // polish the same starts for the same step cap.
                let width = order_w.len().clamp(1, 32);
                let each = (per / width).max(4);
                let step0 = (default_sigma(&bounds) * 0.5).max(1e-4);
                meter.borrow_mut().set_bucket(PH_ALLOC);
                let mut plan = Vec::new();
                let mut shadow = work(n_evals, n_grads);
                for &wi in order_w.iter().take(width) {
                    if shadow + 8 >= polish_start.min(budget) {
                        break;
                    }
                    plan.push((
                        wi,
                        pop.walkers[wi].pos.clone(),
                        pop.walkers[wi].energy,
                        each,
                    ));
                    shadow = shadow.saturating_add(each.saturating_mul(2));
                }
                let threads = rayon::current_num_threads().max(1);
                meter.borrow_mut().pause();
                let polish_started = Instant::now();
                let polished: Vec<_> = plan
                    .par_iter()
                    .map(|(wi, x0, f0, max_fe)| {
                        let task_started = Instant::now();
                        let pol =
                            projected_gradient_polish(obj, gr, x0.clone(), *max_fe, step0, 1e-10);
                        (*wi, *f0, pol, task_started.elapsed())
                    })
                    .collect();
                let polish_wall = polish_started.elapsed();
                let polish_busy = polished.iter().map(|(_, _, _, busy)| *busy).sum();
                meter
                    .borrow_mut()
                    .resume_parallel(polish_wall, polish_busy, threads, false);
                meter.borrow_mut().set_bucket(PH_SELECTION);
                for (wi, f0, pol, _) in polished {
                    let room = budget.saturating_sub(work(n_evals, n_grads));
                    let charge = (pol.n_evals + pol.n_grads).min(room);
                    let ce = pol.n_evals.min(charge);
                    n_evals += ce;
                    n_grads += (charge - ce).min(pol.n_grads);
                    if pol.best_val.is_finite() && pol.best_val <= f0 {
                        pop.walkers[wi] = Walker {
                            pos: pol.best_pos.clone(),
                            energy: pol.best_val,
                        };
                        if mark_best(
                            &mut best_val,
                            pol.best_val,
                            work(n_evals, n_grads),
                            &mut first_hit_work,
                        ) {
                            best_pos = pol.best_pos;
                        }
                    }
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
                    let mut y = if k.is_multiple_of(2) {
                        let mut y = best_pos.clone();
                        let jit = (sigma * 3.0).max(default_sigma(&bounds) * 0.5);
                        for i in 0..dim {
                            let u1 = rng.random::<f64>().max(1e-12);
                            let u2 = rng.random::<f64>();
                            let z =
                                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                            y[i] += jit * z;
                        }
                        reflect_into_box(y.view(), &bounds)
                    } else if k < fresh.nrows() {
                        reflect_into_box(fresh.row(k), &bounds)
                    } else {
                        continue;
                    };
                    if let Some(prepare) = prepare {
                        prepare(pop.walkers[wi].pos.view(), &mut y, None);
                    }
                    if let Some(e) = charge_obj(y.view(), &mut n_evals, n_grads) {
                        pop.walkers[wi] = Walker { pos: y, energy: e };
                        if mark_best(
                            &mut best_val,
                            e,
                            work(n_evals, n_grads),
                            &mut first_hit_work,
                        ) {
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
                for _ in 0..6 {
                    if work(n_evals, n_grads) >= budget {
                        break;
                    }
                    let c = rng.random_range(0..dim);
                    let span = (bounds.high[c] - bounds.low[c]).abs().max(1e-12);
                    let h = (refine_sigma * span.clamp(1.0, 2.0)).min(0.15 * span);
                    for dir in [-1.0_f64, 1.0] {
                        if work(n_evals, n_grads) >= budget {
                            break;
                        }
                        let mut trial = x.clone();
                        trial[c] = (x[c] + dir * h).clamp(bounds.low[c], bounds.high[c]);
                        if let Some(e) = charge_obj(trial.view(), &mut n_evals, n_grads) {
                            if e < fx {
                                x = trial;
                                fx = e;
                            }
                        } else {
                            break;
                        }
                    }
                }
                for _ in 0..4 {
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
                if mark_best(
                    &mut best_val,
                    fx,
                    work(n_evals, n_grads),
                    &mut first_hit_work,
                ) {
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

    let (phases, polish_sync_s, wave_sync_s) = {
        let mut guard = meter.borrow_mut();
        let phases = guard.snapshot();
        let polish_sync_s = guard.polish_sync_s();
        let wave_sync_s = guard.wave_sync_s();
        (phases, polish_sync_s, wave_sync_s)
    };
    DmcPopulationResult {
        best_pos,
        best_val,
        n_evals,
        n_grads,
        final_population: pop.walkers.len(),
        controls,
        n_accepts,
        phases,
        polish_sync_s,
        wave_sync_s,
        first_hit_work,
        wave_width,
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
    O: Objective<f64> + Sync,
    G: Gradient<f64> + Sync,
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

    /// D8.3–D8.5: H(0)=log N, H decreases in beta, calibrate hits H_star.
    #[test]
    fn d8_entropy_calibrated_beta_matches_target() {
        let energies = [0.0, 0.5, 1.0, 2.0, 5.0];
        let n = energies.len();
        let h0 = softmax_entropy(&energies, 0.0);
        assert!((h0 - (n as f64).ln()).abs() < 1e-12, "H(0)={h0}");
        let h1 = softmax_entropy(&energies, 1.0);
        let h5 = softmax_entropy(&energies, 5.0);
        assert!(h1 <= h0 + 1e-12, "H not monotone: {h1} > {h0}");
        assert!(h5 <= h1 + 1e-12, "H not monotone: {h5} > {h1}");
        for progress in [0.0, 0.35, 0.7, 1.0] {
            let h_star = target_entropy(n, progress, 2.0);
            let beta = calibrate_beta(&energies, h_star);
            let h = softmax_entropy(&energies, beta);
            assert!(
                (h - h_star).abs() < 1e-5 * (1.0 + h_star.abs()),
                "progress={progress}: H(beta*)={h} vs H*={h_star} beta={beta}"
            );
        }
    }

    #[test]
    fn softmax_does_not_assign_mass_to_underflowed_states() {
        let p = softmax_probs(&[0.0, 1.0e6], 1.0);
        assert_eq!(p[0], 1.0);
        assert_eq!(p[1], 0.0);
        assert_eq!(softmax_entropy(&[0.0, 1.0e6], 1.0), 0.0);
    }

    /// D8 residual control preserves target size and returns finite beta*.
    #[test]
    fn d8_population_control_ecit_preserves_size() {
        let mut rng = StdRng::seed_from_u64(7);
        let walkers: Vec<Walker> = (0..12)
            .map(|i| Walker {
                pos: Array1::from_elem(2, i as f64),
                energy: (i as f64) * 0.4,
            })
            .collect();
        let (out, beta_star) = population_control_ecit(&walkers, 16, 0.4, 2.0, &mut rng);
        assert_eq!(out.len(), 16);
        assert!(beta_star.is_finite() && beta_star >= 0.0);
        assert!(out.iter().all(|w| w.energy.is_finite()));
    }

    /// Residual masses used by ECIT are pure D8.2 softmax and realize H*.
    ///
    /// Regression guard: legacy soft-ref walker_weight (min/median blend) must
    /// NOT be the residual law for population_control_ecit — that path was
    /// calibrating H* on pure softmax then resampling with a different law.
    #[test]
    fn d8_ecit_residual_weights_realize_h_star() {
        let energies = [0.0_f64, 0.5, 1.0, 2.0, 5.0];
        let n = energies.len();
        for progress in [0.0, 0.35, 0.7, 1.0] {
            let (probs, beta_star, h_star) = ecit_residual_probs(&energies, progress, 2.0);
            // Pure softmax entropy matches the target.
            let h_probs = mass_entropy(&probs);
            assert!(
                (h_probs - h_star).abs() < 1e-5 * (1.0 + h_star.abs()),
                "rho={progress}: mass_entropy(probs)={h_probs} H*={h_star} beta={beta_star}"
            );
            // probs == softmax_probs(energies, beta_star) elementwise (D8.2).
            let p_ref = softmax_probs(&energies, beta_star);
            assert_eq!(probs.len(), p_ref.len());
            for (a, b) in probs.iter().zip(p_ref.iter()) {
                assert!((a - b).abs() < 1e-12, "prob mismatch {a} vs {b}");
            }
            // Expected residual counts proportional to pure softmax, not soft-ref.
            let expected = residual_expected_counts(&probs, 16);
            assert!((expected.iter().sum::<f64>() - 16.0).abs() < 1e-9);
            let e_ref = branch_reference(&energies);
            let soft_ref: Vec<f64> = energies
                .iter()
                .map(|&e| walker_weight(e, e_ref, beta_star))
                .collect();
            let h_soft = mass_entropy(&soft_ref);
            // Soft-ref entropy must differ from H* when progress forces concentration
            // (except near rho=0 where both are near log N).
            if progress >= 0.7 {
                assert!(
                    (h_soft - h_star).abs() > 0.05,
                    "soft-ref accidentally matches H* (test would be weak): h_soft={h_soft} H*={h_star}"
                );
            }
            // Shannon entropy of the residual *counts* equals H* (same as probs).
            let h_counts = mass_entropy(&expected);
            assert!(
                (h_counts - h_star).abs() < 1e-5 * (1.0 + h_star.abs()),
                "rho={progress}: residual count entropy {h_counts} != H* {h_star}"
            );
            let _ = n;
        }
    }

    /// population_control_ecit uses the same residual mass law as ecit_residual_probs.
    #[test]
    fn d8_population_control_ecit_uses_pure_softmax_masses() {
        let mut rng = StdRng::seed_from_u64(11);
        let energies = [0.0, 0.4, 0.8, 1.5, 3.0, 6.0];
        let walkers: Vec<Walker> = energies
            .iter()
            .enumerate()
            .map(|(i, &e)| Walker {
                pos: Array1::from_elem(2, i as f64),
                energy: e,
            })
            .collect();
        let progress = 1.0;
        let (probs, beta_star, h_star) = ecit_residual_probs(&energies, progress, 2.0);
        let (out, beta_out) = population_control_ecit(&walkers, 18, progress, 2.0, &mut rng);
        assert_eq!(out.len(), 18);
        assert!((beta_out - beta_star).abs() < 1e-15);
        // Empirical offspring frequencies should be closer to pure softmax
        // expected counts than to soft-ref expected counts (Monte Carlo, but
        // expected counts themselves are deterministic and pure-softmax).
        let expected_soft = residual_expected_counts(&probs, 18);
        let e_ref = branch_reference(&energies);
        let soft_w: Vec<f64> = energies
            .iter()
            .map(|&e| walker_weight(e, e_ref, beta_star))
            .collect();
        let expected_legacy = residual_expected_counts(&soft_w, 18);
        // Deterministic: pure-softmax and soft-ref expected vectors differ.
        let l1: f64 = expected_soft
            .iter()
            .zip(expected_legacy.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            l1 > 0.5,
            "pure softmax and soft-ref residual counts nearly equal (L1={l1}); H*={h_star}"
        );
        // Soft-ref residual entropy is NOT H* at full concentration.
        let h_legacy = mass_entropy(&expected_legacy);
        assert!(
            (h_legacy - h_star).abs() > 0.05,
            "legacy residual entropy {h_legacy} accidentally equals H* {h_star}"
        );
        let h_pure = mass_entropy(&expected_soft);
        assert!((h_pure - h_star).abs() < 1e-5);
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
    fn lehmer_mean_stays_finite_when_a_success_left_infinity() {
        let mean = lehmer_mean(&[0.4, 0.8, 0.6], &[f64::INFINITY, 2.0, 1.0]);
        assert!(mean.is_finite() && (0.05..=1.0).contains(&mean), "{mean}");
        let mean = lehmer_mean(&[0.4, 0.8], &[f64::INFINITY, f64::INFINITY]);
        assert!(mean.is_finite() && (0.05..=1.0).contains(&mean), "{mean}");
    }

    #[test]
    fn run_controls_population_and_improves_sphere() {
        let obj = Sphere::new(3);
        let mut rng = StdRng::seed_from_u64(11);
        let res = run_dmc_population::<_, Sphere, _>(&obj, None, 400, 11, 12, 3, 1.0, &mut rng);
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
                x.mapv(|xi| {
                    2.0 * xi + 20.0 * std::f64::consts::PI * (2.0 * std::f64::consts::PI * xi).sin()
                })
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
                x.mapv(|xi| {
                    2.0 * xi + 20.0 * std::f64::consts::PI * (2.0 * std::f64::consts::PI * xi).sin()
                })
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

    /// Budget accounting: work units never exceed the requested budget.
    #[test]
    fn batch_eval_matches_across_thread_counts() {
        use rayon::prelude::*;

        struct BatchSphere {
            bounds: Bounds<f64>,
        }

        impl BatchSphere {
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

        impl Objective<f64> for BatchSphere {
            fn dim(&self) -> usize {
                self.bounds.dims
            }
            fn bounds(&self) -> &Bounds<f64> {
                &self.bounds
            }
            fn eval(&self, x: ArrayView1<f64>) -> f64 {
                x.iter().map(|v| v * v).sum()
            }
            fn eval_batch(&self, x: ndarray::ArrayView2<f64>) -> Array1<f64> {
                let mut out = vec![0.0; x.nrows()];
                out.par_iter_mut().enumerate().for_each(|(i, slot)| {
                    *slot = self.eval(x.row(i));
                });
                Array1::from(out)
            }
        }

        impl Gradient<f64> for BatchSphere {
            fn dim(&self) -> usize {
                self.bounds.dims
            }
            fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
                x.mapv(|v| 2.0 * v)
            }
        }

        let once = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            pool.install(|| {
                let obj = BatchSphere::new(4);
                let mut rng = StdRng::seed_from_u64(11);
                run_dmc_population::<_, BatchSphere, _>(&obj, None, 80, 11, 8, 2, 1.0, &mut rng)
            })
        };
        let one = once(1);
        let three = once(3);
        assert_eq!(one.best_val.to_bits(), three.best_val.to_bits());
        assert_eq!(one.n_evals, three.n_evals);
        assert_eq!(one.final_population, three.final_population);
    }

    #[test]
    fn dmc_population_optimize_respects_budget() {
        let obj = Sphere::new(4);
        let res = dmc_population_optimize::<_, Sphere>(&obj, Some(&obj), 500, 42);
        assert!(res.best_val.is_finite());
        assert!(
            res.n_evals + res.n_grads <= 500,
            "work {} exceeds budget 500",
            res.n_evals + res.n_grads
        );
        assert!(
            res.best_val < 2.0,
            "sphere should refine, got {}",
            res.best_val
        );
        assert!(obj.bounds().contains(res.best_pos.view()));
    }

    /// Gradient-wave width follows the pool. The search is no longer one trajectory.
    #[test]
    fn parallel_gradient_wave_width_follows_the_pool() {
        use crate::potentials::PairPotential;

        let once = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            pool.install(|| {
                let obj = PairPotential::lennard_jones(16);
                let mut rng = StdRng::seed_from_u64(7);
                run_dmc_population(&obj, Some(&obj), 600, 7, 8, 2, 1.0, &mut rng)
            })
        };
        let serial = once(1);
        let parallel = once(4);
        assert_eq!(serial.best_val.to_bits(), parallel.best_val.to_bits());
        assert_eq!(serial.wave_width, parallel.wave_width);
        assert!(serial.wave_width >= 1);
        assert_eq!(serial.first_hit_work, parallel.first_hit_work);
        assert!(serial.first_hit_work <= 600);
        assert_eq!(serial.n_evals, parallel.n_evals);
        assert_eq!(serial.n_grads, parallel.n_grads);
        assert!(serial.n_grads > 0, "gradient polish did not run");
        assert!(serial.n_accepts > 0, "no proposal accepts");
    }
}
