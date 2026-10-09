//! Resumable box-constrained CMA-ES with BIPOP restart planning.
//!
//! The engine is Hansen's (mu/mu_w, lambda)-CMA-ES: log-rank recombination
//! weights, cumulative step-size adaptation, and the rank-one plus rank-mu
//! covariance update with the defaults of Hansen's tutorial (arXiv:1604.00772,
//! Table 1). It is an ask/tell state machine that hands out one candidate at a
//! time, so a caller can stop after any evaluation and resume later on an
//! identical trajectory: the sampling stream belongs to the engine and the
//! update depends only on the values told back.
//!
//! Box handling is Lamarckian repair. A sample outside the box is mirror-
//! reflected into it, the reflected point is what gets evaluated, and the
//! reflected step enters the update. The step's Mahalanobis length is clipped
//! at `sqrt(n) + 2n/(n+2)` (Hansen, arXiv:1110.4181, injected solutions), so a
//! repair cannot inflate the evolution paths. Every candidate lies in the box,
//! and the mean stays in it because it is a convex combination of in-box
//! points.
//!
//! An elitist run ([`CmaEs::with_elite`]) carries the best point it knows into
//! every generation as one pre-evaluated candidate, entered the same way as a
//! repaired one, so the elite costs no evaluation and is never lost.
//!
//! The eigendecomposition is refreshed lazily, once per
//! `lambda / ((c1 + c_mu) n 10)` evaluations, through the crate's Jacobi
//! solver. A separable run ([`CmaEs::separable`], sep-CMA-ES of Ros and
//! Hansen 2008) keeps only the diagonal of the covariance, with learning
//! rates raised by `(n + 2) / 3`: linear time and memory per candidate, for
//! dimensions or budgets at which the full matrix cannot be learned (see
//! [`covariance_learning_evaluations`]).
//!
//! Restarts are planned by [`Bipop`]: large-population runs double
//! lambda (IPOP) and small-population runs draw a reduced population and a
//! step size up to two decades smaller, each regime taking its turn when it
//! has spent less of the budget (Hansen 2009, BIPOP-CMA-ES).

use std::collections::VecDeque;

use eindir_core::Bounds;
use ndarray::{Array1, Array2, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use crate::movekernel::reflect_into_box;

/// Relative fitness range that ends a run (TolFun and TolHistFun).
const TOL_FUN_REL: f64 = 1e-12;
/// Step size, relative to the run's initial step, that ends a run (TolX).
const TOL_X_REL: f64 = 1e-11;
/// Covariance condition number that ends a run.
const MAX_CONDITION: f64 = 1e14;
/// Jacobi sweep cap; the solver stops earlier once off-diagonals vanish.
const EIGEN_SWEEPS: usize = 64;
/// A run whose sampling width exceeds this many box widths has diverged.
const MAX_WIDTH_RATIO: f64 = 10.0;

/// Why a CMA-ES run stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmaStop {
    /// Recent best values and the latest generation span a negligible range.
    TolFun,
    /// At least ten recent best values span less than the configured
    /// absolute tolerance (pycma's `tolfunhist`; see
    /// [`CmaEs::with_tol_fun_hist`]).
    TolFunHist,
    /// The sampling width shrank below the resolution of the run.
    TolX,
    /// A tenth of a principal axis no longer moves the mean.
    NoEffectAxis,
    /// A fifth of a coordinate deviation no longer moves the mean.
    NoEffectCoord,
    /// The covariance matrix became numerically singular.
    ConditionCov,
    /// The run's best value did not improve over the history window.
    Stagnation,
    /// The step size became non-finite or far wider than the box.
    Diverged,
}

/// Default population size `4 + floor(3 ln n)`.
pub fn default_lambda(dim: usize) -> usize {
    4 + (3.0 * (dim.max(1) as f64).ln()).floor() as usize
}

/// Log-rank recombination weights of a population of `lambda` (at least 2)
/// and their variance-effective selection mass.
fn recombination(lambda: usize) -> (Vec<f64>, f64) {
    let mu = lambda.max(2) / 2;
    let raw: Vec<f64> = (1..=mu)
        .map(|i| ((lambda.max(2) as f64 + 1.0) / 2.0).ln() - (i as f64).ln())
        .collect();
    let total: f64 = raw.iter().sum();
    let weights: Vec<f64> = raw.iter().map(|w| w / total).collect();
    let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
    (weights, mu_eff)
}

/// Rank-one and rank-mu learning rates of the full covariance.
fn learning_rates(n: usize, mu_eff: f64) -> (f64, f64) {
    let nf = n as f64;
    let c_1 = 2.0 / ((nf + 1.3).powi(2) + mu_eff);
    let c_mu = (1.0 - c_1).min(2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((nf + 2.0).powi(2) + mu_eff));
    (c_1, c_mu)
}

/// Evaluations a full-covariance run of population `lambda` needs to learn
/// its covariance, `lambda / (c1 + c_mu)`: the matrix forgets its start at
/// rate `c1 + c_mu` per generation. A run given less than this adapts only a
/// fraction of the matrix it pays to decompose.
pub fn covariance_learning_evaluations(dim: usize, lambda: usize) -> f64 {
    let lambda = lambda.max(2);
    let (_, mu_eff) = recombination(lambda);
    let (c_1, c_mu) = learning_rates(dim.max(1), mu_eff);
    lambda as f64 / (c_1 + c_mu)
}

/// One CMA-ES run over a box.
pub struct CmaEs {
    n: usize,
    lambda: usize,
    weights: Vec<f64>,
    mu_eff: f64,
    c_sigma: f64,
    d_sigma: f64,
    c_c: f64,
    c_1: f64,
    c_mu: f64,
    chi_n: f64,
    bounds: Bounds<f64>,
    max_width: f64,
    mean: Array1<f64>,
    sigma: f64,
    sigma0: f64,
    /// Full covariance and its eigenbasis; empty in a separable run.
    cov: Array2<f64>,
    basis: Array2<f64>,
    /// Diagonal covariance of a separable run; empty otherwise.
    diag: Array1<f64>,
    separable: bool,
    scales: Array1<f64>,
    p_sigma: Array1<f64>,
    p_c: Array1<f64>,
    generation: usize,
    evals: usize,
    eigen_evals: usize,
    candidates: Vec<Array1<f64>>,
    steps: Vec<Array1<f64>>,
    values: Vec<f64>,
    next: usize,
    rng: StdRng,
    best_x: Array1<f64>,
    best_f: f64,
    history: VecDeque<f64>,
    history_len: usize,
    stall: usize,
    stop: Option<CmaStop>,
    elitist: bool,
    tol_fun_hist: f64,
}

impl CmaEs {
    /// Starts a run at `mean` (clipped into the box) with step size `sigma`
    /// and population `lambda` (at least 2).
    pub fn new(
        mean: ArrayView1<f64>,
        sigma: f64,
        lambda: usize,
        bounds: &Bounds<f64>,
        seed: u64,
    ) -> Self {
        let n = bounds.dims;
        assert!(n > 0, "CMA-ES needs a positive dimension");
        assert_eq!(mean.len(), n, "mean length must match the box");
        assert!(
            sigma.is_finite() && sigma > 0.0,
            "sigma must be positive and finite"
        );
        let lambda = lambda.max(2);
        let (weights, mu_eff) = recombination(lambda);
        let nf = n as f64;
        let c_sigma = (mu_eff + 2.0) / (nf + mu_eff + 5.0);
        let d_sigma = 1.0 + 2.0 * (((mu_eff - 1.0) / (nf + 1.0)).sqrt() - 1.0).max(0.0) + c_sigma;
        let c_c = (4.0 + mu_eff / nf) / (nf + 4.0 + 2.0 * mu_eff / nf);
        let (c_1, c_mu) = learning_rates(n, mu_eff);
        let chi_n = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));
        let max_width = (0..n)
            .map(|i| bounds.high[i] - bounds.low[i])
            .fold(0.0_f64, f64::max);
        let mean = bounds.clip(mean);
        Self {
            n,
            lambda,
            weights,
            mu_eff,
            c_sigma,
            d_sigma,
            c_c,
            c_1,
            c_mu,
            chi_n,
            bounds: bounds.clone(),
            max_width,
            best_x: mean.clone(),
            mean,
            sigma,
            sigma0: sigma,
            cov: Array2::eye(n),
            basis: Array2::eye(n),
            diag: Array1::zeros(0),
            separable: false,
            scales: Array1::ones(n),
            p_sigma: Array1::zeros(n),
            p_c: Array1::zeros(n),
            generation: 0,
            evals: 0,
            eigen_evals: 0,
            candidates: Vec::with_capacity(lambda),
            steps: Vec::with_capacity(lambda),
            values: Vec::with_capacity(lambda),
            next: 0,
            rng: StdRng::seed_from_u64(seed),
            best_f: f64::INFINITY,
            history: VecDeque::new(),
            history_len: 10 + (30.0 * nf / lambda as f64).ceil() as usize,
            stall: 0,
            stop: None,
            elitist: false,
            tol_fun_hist: 0.0,
        }
    }

    /// Restricts the covariance to its diagonal (sep-CMA-ES, Ros and Hansen
    /// 2008). The rank-one and rank-mu rates grow by `(n + 2) / 3`, the rate
    /// at which `n` free parameters can be learned in place of `n^2 / 2`, and
    /// no eigendecomposition or `n x n` matrix is kept, so a candidate costs
    /// linear time and memory. Call before the first [`CmaEs::ask`].
    pub fn separable(mut self) -> Self {
        assert_eq!(self.evals, 0, "a run turns separable before sampling");
        let boost = (self.n as f64 + 2.0) / 3.0;
        self.c_1 = (self.c_1 * boost).min(1.0);
        self.c_mu = (self.c_mu * boost).min(1.0 - self.c_1);
        self.cov = Array2::zeros((0, 0));
        self.basis = Array2::zeros((0, 0));
        self.diag = Array1::ones(self.n);
        self.separable = true;
        self
    }

    /// Ends the run once more than nine generation bests are on record and
    /// the recent ones, up to the `10 + 30n/lambda` generations of the
    /// TolHistFun window, span less than `tol` ([`CmaStop::TolFunHist`],
    /// pycma's `tolfunhist`). A run that starts converged, or converges,
    /// stops within ten generations of its last improvement larger than
    /// `tol`. Zero (the default) disables it.
    pub fn with_tol_fun_hist(mut self, tol: f64) -> Self {
        assert!(tol >= 0.0 && tol.is_finite(), "tol_fun_hist must be finite");
        self.tol_fun_hist = tol;
        self
    }

    /// Whether the run keeps only the diagonal of its covariance.
    pub fn is_separable(&self) -> bool {
        self.separable
    }

    fn variance(&self, i: usize) -> f64 {
        if self.separable {
            self.diag[i]
        } else {
            self.cov[[i, i]]
        }
    }

    /// Makes the run elitist, starting from `x` with the known value
    /// `value`. Every generation then carries the best point the run knows
    /// as one pre-evaluated candidate in place of a sample, entered like an
    /// injected solution, so selection never loses it and the mean cannot
    /// drift off a valley narrower than the step size.
    pub fn with_elite(mut self, x: ArrayView1<f64>, value: f64) -> Self {
        assert_eq!(x.len(), self.n, "elite length must match the box");
        if value.is_finite() {
            self.best_x = self.bounds.clip(x);
            self.best_f = value;
        }
        self.elitist = true;
        self
    }

    /// Population size of this run.
    pub fn lambda(&self) -> usize {
        self.lambda
    }

    /// Current step size.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Current distribution mean.
    pub fn mean(&self) -> ArrayView1<'_, f64> {
        self.mean.view()
    }

    /// Completed generations.
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Values told back so far in this run.
    pub fn evaluations(&self) -> usize {
        self.evals
    }

    /// Best candidate and value told back in this run.
    pub fn best(&self) -> (ArrayView1<'_, f64>, f64) {
        (self.best_x.view(), self.best_f)
    }

    /// Why the run stopped, once a generation boundary detects it.
    pub fn stop_reason(&self) -> Option<CmaStop> {
        self.stop
    }

    /// True between generations (no candidate of the current one handed out).
    pub fn at_generation_boundary(&self) -> bool {
        self.next == 0 && self.candidates.is_empty()
    }

    /// Next candidate to evaluate; always inside the box.
    pub fn ask(&mut self) -> Array1<f64> {
        if self.candidates.is_empty() {
            self.sample_generation();
        }
        self.candidates[self.next].clone()
    }

    /// Value of the candidate last returned by [`CmaEs::ask`]. Non-finite
    /// values rank last.
    pub fn tell(&mut self, value: f64) {
        assert!(
            self.next < self.candidates.len(),
            "tell without a pending ask"
        );
        self.values
            .push(if value.is_nan() { f64::INFINITY } else { value });
        self.evals += 1;
        self.next += 1;
        if self.next == self.lambda {
            self.update();
        }
    }

    fn mahalanobis_norm(&self, step: &Array1<f64>) -> f64 {
        let rotated = if self.separable {
            step.clone()
        } else {
            self.basis.t().dot(step)
        };
        rotated
            .iter()
            .zip(self.scales.iter())
            .map(|(r, s)| (r / s).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn clip_step(&self, mut step: Array1<f64>) -> Array1<f64> {
        let n = self.n as f64;
        let clip_len = n.sqrt() + 2.0 * n / (n + 2.0);
        let length = self.mahalanobis_norm(&step);
        if length > clip_len {
            step *= clip_len / length;
        }
        step
    }

    fn sample_generation(&mut self) {
        let n = self.n;
        self.candidates.clear();
        self.steps.clear();
        self.values.clear();
        self.next = 0;
        if self.elitist && self.best_f.is_finite() {
            let step = self.clip_step((&self.best_x - &self.mean) / self.sigma);
            self.candidates.push(self.best_x.clone());
            self.steps.push(step);
            self.values.push(self.best_f);
            self.next = 1;
        }
        while self.candidates.len() < self.lambda {
            let z: Array1<f64> =
                Array1::from_iter((0..n).map(|_| StandardNormal.sample(&mut self.rng)));
            let y = if self.separable {
                &self.scales * &z
            } else {
                self.basis.dot(&(&self.scales * &z))
            };
            let x = &self.mean + &(&y * self.sigma);
            let repaired = reflect_into_box(x.view(), &self.bounds);
            let step = if repaired == x {
                y
            } else {
                self.clip_step((&repaired - &self.mean) / self.sigma)
            };
            self.candidates.push(repaired);
            self.steps.push(step);
        }
    }

    fn update(&mut self) {
        let n = self.n;
        let mut order: Vec<usize> = (0..self.lambda).collect();
        order.sort_by(|&a, &b| self.values[a].total_cmp(&self.values[b]).then(a.cmp(&b)));
        let generation_best = self.values[order[0]];
        if generation_best < self.best_f {
            self.best_f = generation_best;
            self.best_x = self.candidates[order[0]].clone();
            self.stall = 0;
        } else {
            self.stall += 1;
        }

        let mut y_w = Array1::<f64>::zeros(n);
        for (rank, &idx) in order.iter().take(self.weights.len()).enumerate() {
            y_w.scaled_add(self.weights[rank], &self.steps[idx]);
        }
        self.mean.scaled_add(self.sigma, &y_w);
        self.mean = self.bounds.clip(self.mean.view());

        let whitened = if self.separable {
            &y_w / &self.scales
        } else {
            let rotated = self.basis.t().dot(&y_w);
            let scaled =
                Array1::from_iter(rotated.iter().zip(self.scales.iter()).map(|(r, s)| r / s));
            self.basis.dot(&scaled)
        };
        let ps_gain = (self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff).sqrt();
        self.p_sigma *= 1.0 - self.c_sigma;
        self.p_sigma.scaled_add(ps_gain, &whitened);
        let ps_norm = self.p_sigma.dot(&self.p_sigma).sqrt();
        let decay = 1.0 - (1.0 - self.c_sigma).powi(2 * (self.generation as i32 + 1));
        let h_sigma =
            ps_norm / decay.max(1e-300).sqrt() < (1.4 + 2.0 / (n as f64 + 1.0)) * self.chi_n;
        let pc_gain = (self.c_c * (2.0 - self.c_c) * self.mu_eff).sqrt();
        self.p_c *= 1.0 - self.c_c;
        if h_sigma {
            self.p_c.scaled_add(pc_gain, &y_w);
        }
        let delta_h = if h_sigma {
            0.0
        } else {
            self.c_c * (2.0 - self.c_c)
        };

        let keep = 1.0 - self.c_1 - self.c_mu + self.c_1 * delta_h;
        if self.separable {
            for i in 0..n {
                let mut add = self.c_1 * self.p_c[i] * self.p_c[i];
                for (rank, &idx) in order.iter().take(self.weights.len()).enumerate() {
                    let step = &self.steps[idx];
                    add += self.c_mu * self.weights[rank] * step[i] * step[i];
                }
                self.diag[i] = keep * self.diag[i] + add;
            }
        } else {
            self.cov *= keep;
            for i in 0..n {
                for j in 0..=i {
                    let mut add = self.c_1 * self.p_c[i] * self.p_c[j];
                    for (rank, &idx) in order.iter().take(self.weights.len()).enumerate() {
                        let step = &self.steps[idx];
                        add += self.c_mu * self.weights[rank] * step[i] * step[j];
                    }
                    self.cov[[i, j]] += add;
                    if i != j {
                        self.cov[[j, i]] = self.cov[[i, j]];
                    }
                }
            }
        }

        let exponent = (self.c_sigma / self.d_sigma) * (ps_norm / self.chi_n - 1.0);
        self.sigma *= exponent.min(1.0).exp();
        // Flat fitness: equal ranks carry no selection signal, so widen the
        // search instead of letting the paths shrink it.
        let kth = order[(0.1 + self.lambda as f64 / 4.0).ceil() as usize % self.lambda];
        if self.values[order[0]] == self.values[kth] {
            self.sigma *= (0.2 + self.c_sigma / self.d_sigma).exp();
        }

        self.generation += 1;
        if self.separable {
            self.rescale_diagonal();
        } else {
            let gap = self.lambda as f64 / ((self.c_1 + self.c_mu) * n as f64 * 10.0);
            if (self.evals - self.eigen_evals) as f64 > gap {
                self.decompose();
            }
        }
        self.history.push_back(generation_best);
        while self.history.len() > self.history_len {
            self.history.pop_front();
        }
        self.stop = self.check_stop();
        self.candidates.clear();
        self.steps.clear();
        self.values.clear();
        self.next = 0;
    }

    fn decompose(&mut self) {
        self.eigen_evals = self.evals;
        let sym = (&self.cov + &self.cov.t()) * 0.5;
        let (values, vectors) = crate::spectral::symmetric_eigen(sym.view(), EIGEN_SWEEPS);
        let top = values.iter().copied().fold(0.0_f64, f64::max);
        if !(top.is_finite() && top > 0.0) {
            self.stop = Some(CmaStop::ConditionCov);
            return;
        }
        let floor = top * 1e-20;
        self.scales = values.mapv(|v| v.max(floor).sqrt());
        self.basis = vectors;
        self.cov = sym;
    }

    fn rescale_diagonal(&mut self) {
        let top = self.diag.iter().copied().fold(0.0_f64, f64::max);
        if !(top.is_finite() && top > 0.0) {
            self.stop = Some(CmaStop::ConditionCov);
            return;
        }
        let floor = top * 1e-20;
        self.scales = self.diag.mapv(|v| v.max(floor).sqrt());
    }

    fn check_stop(&self) -> Option<CmaStop> {
        if self.stop.is_some() {
            return self.stop;
        }
        let max_scale = self.scales.iter().copied().fold(0.0_f64, f64::max);
        let min_scale = self.scales.iter().copied().fold(f64::INFINITY, f64::min);
        if !self.sigma.is_finite() || self.sigma * max_scale > MAX_WIDTH_RATIO * self.max_width {
            return Some(CmaStop::Diverged);
        }
        let finite = |v: &&f64| v.is_finite();
        let hist_max = self
            .history
            .iter()
            .filter(finite)
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let hist_min = self
            .history
            .iter()
            .filter(finite)
            .copied()
            .fold(f64::INFINITY, f64::min);
        if self.history.len() >= self.history_len {
            let scale = self.best_f.abs().max(hist_max.abs());
            if hist_max.is_finite()
                && hist_min.is_finite()
                && hist_max - hist_min <= TOL_FUN_REL * scale
            {
                return Some(CmaStop::TolFun);
            }
            if self.stall >= self.history_len {
                return Some(CmaStop::Stagnation);
            }
        }
        if self.tol_fun_hist > 0.0
            && self.history.len() > 9
            && hist_max.is_finite()
            && hist_min.is_finite()
            && hist_max - hist_min < self.tol_fun_hist
        {
            return Some(CmaStop::TolFunHist);
        }
        let width = (0..self.n)
            .map(|i| self.variance(i).sqrt().max(self.p_c[i].abs()))
            .fold(0.0_f64, f64::max);
        if self.sigma * width < TOL_X_REL * self.sigma0 {
            return Some(CmaStop::TolX);
        }
        let axis = self.generation % self.n;
        let shift = 0.1 * self.sigma * self.scales[axis];
        let no_effect_axis = if self.separable {
            self.mean[axis] + shift == self.mean[axis]
        } else {
            (0..self.n).all(|i| self.mean[i] + shift * self.basis[[i, axis]] == self.mean[i])
        };
        if no_effect_axis {
            return Some(CmaStop::NoEffectAxis);
        }
        if (0..self.n)
            .any(|i| self.mean[i] + 0.2 * self.sigma * self.variance(i).sqrt() == self.mean[i])
        {
            return Some(CmaStop::NoEffectCoord);
        }
        if (max_scale / min_scale).powi(2) > MAX_CONDITION {
            return Some(CmaStop::ConditionCov);
        }
        None
    }
}

/// Restart regime of a planned run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmaRegime {
    /// Large population with the global step size (IPOP doubling).
    Large,
    /// Reduced population with a step size up to two decades smaller.
    Small,
}

/// Population size, step size, and regime of the next run.
#[derive(Clone, Copy, Debug)]
pub struct CmaRunPlan {
    /// Population size.
    pub lambda: usize,
    /// Initial step size.
    pub sigma: f64,
    /// Regime whose budget the run is charged to.
    pub regime: CmaRegime,
}

/// BIPOP restart planner: alternates IPOP large-population runs and
/// small-population local runs by spent budget.
#[derive(Clone, Debug)]
pub struct Bipop {
    default_lambda: usize,
    max_lambda: usize,
    large_sigma: f64,
    /// Large-regime runs so far, the first run included when it was one.
    large_runs: u32,
    last_large_lambda: usize,
    spent_large: usize,
    spent_small: usize,
}

impl Bipop {
    /// Planner for the default population `default_lambda`, capped at
    /// `max_lambda`, with `large_sigma` as the global step size. `first` is
    /// the regime of the run started before the planner is consulted, to
    /// which [`Bipop::record`] charges it: a large first run is the
    /// default-population IPOP run, so the first planned large run doubles
    /// it, while after a local (small) first run the first planned large run
    /// uses the default population.
    pub fn new(
        default_lambda: usize,
        max_lambda: usize,
        large_sigma: f64,
        first: CmaRegime,
    ) -> Self {
        let default_lambda = default_lambda.max(2);
        Self {
            default_lambda,
            max_lambda: max_lambda.max(default_lambda),
            large_sigma,
            large_runs: u32::from(first == CmaRegime::Large),
            last_large_lambda: default_lambda,
            spent_large: 0,
            spent_small: 0,
        }
    }

    /// Charges a finished run's evaluations to its regime.
    pub fn record(&mut self, regime: CmaRegime, evals: usize) {
        match regime {
            CmaRegime::Large => self.spent_large += evals,
            CmaRegime::Small => self.spent_small += evals,
        }
    }

    /// Plans the next restart.
    pub fn next_run<R: Rng + ?Sized>(&mut self, rng: &mut R) -> CmaRunPlan {
        if self.spent_large <= self.spent_small {
            let lambda = self
                .default_lambda
                .saturating_mul(1usize << self.large_runs.min(20))
                .min(self.max_lambda);
            self.large_runs += 1;
            self.last_large_lambda = lambda;
            CmaRunPlan {
                lambda,
                sigma: self.large_sigma,
                regime: CmaRegime::Large,
            }
        } else {
            let u: f64 = rng.random();
            let ratio = 0.5 * self.last_large_lambda as f64 / self.default_lambda as f64;
            let lambda =
                ((self.default_lambda as f64) * ratio.max(1.0).powf(u * u)).floor() as usize;
            let sigma = self.large_sigma * 10f64.powf(-2.0 * rng.random::<f64>());
            CmaRunPlan {
                lambda: lambda.clamp(self.default_lambda, self.max_lambda),
                sigma,
                regime: CmaRegime::Small,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_box(n: usize, half: f64) -> Bounds<f64> {
        Bounds::new(Array1::from_elem(n, -half), Array1::from_elem(n, half), 0.0)
    }

    fn drive<F: Fn(ArrayView1<f64>) -> f64>(es: &mut CmaEs, f: F, evals: usize) -> Vec<f64> {
        let mut seen = Vec::with_capacity(evals);
        for _ in 0..evals {
            let x = es.ask();
            let v = f(x.view());
            seen.push(v);
            es.tell(v);
        }
        seen
    }

    #[test]
    fn default_lambda_matches_hansen() {
        assert_eq!(default_lambda(2), 6);
        assert_eq!(default_lambda(10), 10);
        assert_eq!(default_lambda(39), 14);
    }

    #[test]
    fn converges_on_rotated_ellipsoid() {
        let n = 8;
        let bounds = unit_box(n, 5.0);
        let f = |x: ArrayView1<f64>| {
            (0..n)
                .map(|i| {
                    let s: f64 = (0..=i).map(|j| x[j] - 1.0).sum();
                    10f64.powf(3.0 * i as f64 / (n - 1) as f64) * s * s
                })
                .sum::<f64>()
        };
        let mut es = CmaEs::new(Array1::zeros(n).view(), 1.0, default_lambda(n), &bounds, 3);
        drive(&mut es, f, 6000);
        assert!(es.best().1 < 1e-8, "best {}", es.best().1);
    }

    #[test]
    fn candidates_stay_in_box_and_mean_follows() {
        let n = 5;
        let bounds = unit_box(n, 1.0);
        // Minimum outside the box pushes the population onto a face.
        let f = |x: ArrayView1<f64>| x.iter().map(|v| (v - 3.0).powi(2)).sum::<f64>();
        let mut es = CmaEs::new(Array1::zeros(n).view(), 2.0, 12, &bounds, 11);
        for _ in 0..3000 {
            let x = es.ask();
            assert!(bounds.contains(x.view()), "candidate left the box: {x}");
            es.tell(f(x.view()));
            assert!(bounds.contains(es.mean()), "mean left the box");
        }
        assert!(es.best().0.iter().all(|v| (v - 1.0).abs() < 1e-3));
    }

    #[test]
    fn same_seed_replays_and_pauses_are_invisible() {
        let n = 6;
        let bounds = unit_box(n, 3.0);
        let f = |x: ArrayView1<f64>| x.iter().map(|v| v * v + (3.0 * v).sin()).sum::<f64>();
        let mut a = CmaEs::new(Array1::from_elem(n, 1.0).view(), 0.5, 9, &bounds, 42);
        let mut b = CmaEs::new(Array1::from_elem(n, 1.0).view(), 0.5, 9, &bounds, 42);
        let full = drive(&mut a, f, 400);
        let mut split = Vec::new();
        for chunk in [1usize, 7, 50, 3, 139, 200] {
            split.extend(drive(&mut b, f, chunk));
        }
        assert_eq!(full, split);
        assert_eq!(a.best().1, b.best().1);
    }

    #[test]
    fn flat_objective_stops_instead_of_spinning() {
        let n = 3;
        let bounds = unit_box(n, 1.0);
        let mut es = CmaEs::new(Array1::zeros(n).view(), 0.3, 6, &bounds, 5);
        let mut evals = 0;
        while es.stop_reason().is_none() && evals < 20_000 {
            let _ = es.ask();
            es.tell(1.0);
            evals += 1;
        }
        assert!(es.stop_reason().is_some());
    }

    #[test]
    fn separable_run_learns_an_axis_aligned_ellipsoid_in_linear_memory() {
        let n = 40;
        let bounds = unit_box(n, 5.0);
        let f = |x: ArrayView1<f64>| {
            (0..n)
                .map(|i| 10f64.powf(4.0 * i as f64 / (n - 1) as f64) * (x[i] - 1.0).powi(2))
                .sum::<f64>()
        };
        let mut es =
            CmaEs::new(Array1::zeros(n).view(), 1.0, default_lambda(n), &bounds, 7).separable();
        assert!(es.is_separable());
        drive(&mut es, f, 20_000);
        assert_eq!(es.cov.len() + es.basis.len(), 0, "no n x n storage");
        assert!(es.best().1 < 1e-6, "best {}", es.best().1);
    }

    #[test]
    fn learning_horizon_grows_with_dimension() {
        let h10 = covariance_learning_evaluations(10, default_lambda(10));
        let h39 = covariance_learning_evaluations(39, default_lambda(39));
        let h100 = covariance_learning_evaluations(100, default_lambda(100));
        assert!((250.0..400.0).contains(&h10), "{h10}");
        assert!((3000.0..4500.0).contains(&h39), "{h39}");
        assert!((15_000.0..30_000.0).contains(&h100), "{h100}");
    }

    #[test]
    fn history_tolerance_ends_a_converged_run_early() {
        // A local run near the minimum of a shifted sphere: the default
        // stops wait for the history to flatten to 1e-12 of |f|, the
        // history tolerance only for gains below 1e-4 of it.
        let n = 10;
        let bounds = unit_box(n, 1.0);
        let f = |x: ArrayView1<f64>| 5.0 + x.iter().map(|v| (v - 0.3).powi(2)).sum::<f64>();
        let start = Array1::from_elem(n, 0.31);
        let run = |tol: f64| {
            let mut es = CmaEs::new(start.view(), 0.01, default_lambda(n), &bounds, 3)
                .with_tol_fun_hist(tol);
            let mut evals = 0;
            while es.stop_reason().is_none() && evals < 50_000 {
                let x = es.ask();
                es.tell(f(x.view()));
                evals += 1;
            }
            (evals, es.stop_reason(), es.best().1)
        };
        let (plain_evals, plain_stop, _) = run(0.0);
        let (hist_evals, hist_stop, hist_best) = run(5e-4);
        assert_ne!(plain_stop, Some(CmaStop::TolFunHist));
        assert_eq!(hist_stop, Some(CmaStop::TolFunHist));
        assert!(
            2 * hist_evals < plain_evals,
            "history tolerance {hist_evals} against default {plain_evals}"
        );
        assert!(hist_best - 5.0 < 1e-3, "best {hist_best}");
    }

    #[test]
    fn bipop_after_a_local_first_run_plans_a_default_large_run() {
        let mut planner = Bipop::new(10, 640, 0.3, CmaRegime::Small);
        let mut rng = StdRng::seed_from_u64(2);
        planner.record(CmaRegime::Small, 500);
        let large = planner.next_run(&mut rng);
        assert_eq!(large.regime, CmaRegime::Large);
        assert_eq!(large.lambda, 10);
        assert_eq!(large.sigma, 0.3);
        planner.record(CmaRegime::Large, 900);
        assert_eq!(planner.next_run(&mut rng).regime, CmaRegime::Small);
        planner.record(CmaRegime::Small, 400);
        let large = planner.next_run(&mut rng);
        assert_eq!(large.regime, CmaRegime::Large);
        assert_eq!(large.lambda, 20);
    }

    #[test]
    fn bipop_alternates_regimes_by_budget() {
        let mut planner = Bipop::new(10, 640, 2.0, CmaRegime::Large);
        let mut rng = StdRng::seed_from_u64(1);
        planner.record(CmaRegime::Large, 500);
        let small = planner.next_run(&mut rng);
        assert_eq!(small.regime, CmaRegime::Small);
        assert!(small.sigma <= 2.0 && small.sigma >= 0.02 - 1e-12);
        planner.record(CmaRegime::Small, 600);
        let large = planner.next_run(&mut rng);
        assert_eq!(large.regime, CmaRegime::Large);
        assert_eq!(large.lambda, 20);
        planner.record(CmaRegime::Large, 2000);
        let small = planner.next_run(&mut rng);
        assert_eq!(small.regime, CmaRegime::Small);
        assert!(small.lambda >= 10 && small.lambda <= 20);
    }
}
