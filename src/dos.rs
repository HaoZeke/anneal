//! A posterior over the density of minima, and acceptance by entropy.
//!
//! *Measured negative as an acceptance rule.* Flat-histogram acceptance built
//! on this estimator was refuted against paired controls (numbers and
//! mechanism in `docs/derivations/`), and no acceptance rule here is part of
//! the recommended configuration. The joint multi-sweep estimator itself
//! stands, with its tests, as a density-of-states tool.
//!
//! # Why the acceptance rule is the trap
//!
//! Basin hopping samples minima from `pi(m) ∝ exp(-E~_m / T)` on the quenched
//! surface. Marginalised onto energy that is
//!
//! ```text
//! p(E~) ∝ g(E~) exp(-E~ / T)
//! ```
//!
//! where `g` counts minima per unit quenched energy. On a multi-funnel
//! landscape `g` is not flat and not slowly varying: the icosahedral funnel of
//! a 38-point Lennard-Jones cluster holds exponentially more minima than the
//! face-centred-cubic funnel that contains the global minimum. At a temperature
//! loose enough to keep the chain moving, `g` dominates and the chain sits
//! where the states are. At a temperature tight enough for energy to beat
//! multiplicity, the chain stops moving. The obstruction is a property of the
//! target distribution and no proposal fixes it, which is why better moves --
//! Hamiltonian dynamics, tempering, surrogate-accelerated acceptance, escape
//! along soft modes -- all measure at chance on this landscape.
//!
//! # Sampling flat in energy instead
//!
//! Give each state weight `1 / g(E~)` rather than `exp(-E~ / T)`. The marginal
//! over energy becomes `g(E~) / g(E~)`, a constant, so the chain spends as much
//! time at each quenched energy as at any other, and the rare deep energies get
//! the same share as the abundant shallow ones. With `S = ln g` the acceptance
//! for a symmetric proposal is
//!
//! ```text
//! a = min(1, exp(-[S(E~_new) - S(E~_old)]))
//! ```
//!
//! Metropolis with entropy where the energy was. This is the multicanonical
//! construction of Berg and Neuhaus (doi:10.1016/0370-2693(91)90256-P) and the
//! flat-histogram target of Wang and Landau
//! (doi:10.1103/PhysRevLett.86.2050), applied to the density of *minima*
//! rather than of configurations, which is the object Bogdan, Wales and Calvo
//! estimate for cluster thermodynamics (doi:10.1063/1.2148958).
//!
//! It is also the coarse-graining the landscape actually admits. Under this
//! target two minima at the same quenched energy are the same state, so the
//! state space collapses from an exponential number of minima onto a single
//! energy axis, and no structural descriptor is needed to say which states are
//! equivalent.
//!
//! # What is Bayesian about it
//!
//! `S` is unknown and has to be learned from the run. Wang and Landau learn it
//! by a histogram with a multiplicative schedule `f -> sqrt(f)`, whose error
//! saturates rather than converging (Belardinelli and Pereyra,
//! doi:10.1103/PhysRevE.75.046701). Here `S` carries a posterior instead:
//!
//! - **Likelihood.** Under a frozen weight `w = exp(-S^)` the chain's
//!   stationary distribution over bins is `p_k ∝ exp(S_k - S^_k)`, so a visit
//!   histogram `n_k` observes the residual directly: `n_k ~ Poisson(exp(S_k -
//!   S^_k + c))`.
//! - **Prior.** A second-difference Gaussian random walk on `S`, the intrinsic
//!   smoothing prior of Poisson density estimation. It is proper on second
//!   differences, improper on the two degrees of freedom it deliberately does
//!   not constrain -- level and slope -- which is what lets the posterior
//!   extrapolate `S` linearly past the deepest energy yet seen, with variance
//!   that grows as it goes.
//! - **Exploration.** The sampling weight is a draw from the posterior rather
//!   than its mean, so a bin the run has little evidence about is over-weighted
//!   in proportion to how little is known, and gets visited. Thompson sampling
//!   on the weight function.
//!
//! The unvisited bin is where this separates from Wang-Landau in kind rather
//! than in degree: a histogram method has nothing to say about a bin with no
//! counts, and the prior gives it a value with an honest error bar.
//!
//! # Validity
//!
//! Changing the weight function invalidates detailed balance, so the weight is
//! frozen for a fixed sweep and refreshed between sweeps. Each sweep is then an
//! exact Markov chain for its own target and the sequence is the standard
//! stochastic-approximation scheme, not an adaptive chain whose invariance has
//! to be argued.

use ndarray::{Array1, Array2};
use rand::Rng;

/// Default number of energy bins.
pub const BINS: usize = 96;

/// A posterior over `S = ln g` on a binned energy axis.
#[derive(Debug, Clone)]
pub struct DensityOfStates {
    /// Lower edge of the binned window.
    lo: f64,
    /// Bin width.
    width: f64,
    /// Posterior mean of `S` per bin.
    mean: Array1<f64>,
    /// Posterior standard deviation of `S` per bin.
    sd: Array1<f64>,
    /// Visit counts accumulated since the last refresh.
    counts: Array1<f64>,
    /// Bins that have ever been visited, so extrapolation knows where the
    /// evidence stops.
    seen: Vec<bool>,
    /// Smoothness scale of the second-difference prior, chosen by marginal
    /// likelihood at each refresh.
    tau: f64,
    /// Every sweep so far: the weight curve that was in force and the counts
    /// collected under it.
    ///
    /// `S` is fitted against all of them jointly rather than against the most
    /// recent one. Fitting sweep by sweep and adding the residual to a running
    /// mean is Wang-Landau's recursion, and it inherits Wang-Landau's defect:
    /// a sweep short enough to be affordable estimates each bin from a handful
    /// of visits, so every update injects noise of order one nat and the curve
    /// random-walks by more than the structure it is estimating. Measured on 38
    /// points, that arm solved 3 seeds in 24 against 15 for the plain rule.
    /// Fitting jointly has no gain schedule to get wrong and the posterior
    /// narrows with total counts.
    history: Vec<(Array1<f64>, Array1<f64>)>,
    /// The curve the current sweep is being collected under.
    active: Array1<f64>,
    /// Refreshes performed.
    pub refreshes: usize,
    /// Samples recorded since the last refresh.
    pending: usize,
    /// Samples that fell outside the window and were clamped into an end bin.
    pub clamped: usize,
}

impl DensityOfStates {
    /// A posterior over `[lo, hi]` split into `bins` bins, flat and uncertain
    /// before any evidence arrives.
    ///
    /// The initial state is deliberately `S = 0` everywhere with a wide spread:
    /// flat `S` makes the first sweep plain random-walk acceptance, which is
    /// the least committed thing to do before the run has seen an energy.
    pub fn new(lo: f64, hi: f64, bins: usize) -> Self {
        let bins = bins.max(4);
        let width = ((hi - lo) / bins as f64).max(1e-9);
        Self {
            lo,
            width,
            mean: Array1::zeros(bins),
            sd: Array1::from_elem(bins, 1.0),
            counts: Array1::zeros(bins),
            seen: vec![false; bins],
            tau: 1.0,
            history: Vec::new(),
            active: Array1::zeros(bins),
            refreshes: 0,
            pending: 0,
            clamped: 0,
        }
    }

    /// Number of bins.
    pub fn bins(&self) -> usize {
        self.mean.len()
    }

    /// Centre energy of bin `k`.
    pub fn centre(&self, k: usize) -> f64 {
        self.lo + (k as f64 + 0.5) * self.width
    }

    /// The bin an energy falls in, clamped to the window.
    fn bin(&self, e: f64) -> usize {
        let raw = ((e - self.lo) / self.width).floor();
        if raw < 0.0 {
            0
        } else if raw >= self.mean.len() as f64 {
            self.mean.len() - 1
        } else {
            raw as usize
        }
    }

    /// Whether an energy sits inside the binned window.
    pub fn inside(&self, e: f64) -> bool {
        e >= self.lo && e < self.lo + self.width * self.mean.len() as f64
    }

    /// Records a visit at quenched energy `e`.
    pub fn observe(&mut self, e: f64) {
        if !e.is_finite() {
            return;
        }
        if !self.inside(e) {
            self.clamped += 1;
        }
        let k = self.bin(e);
        self.counts[k] += 1.0;
        self.seen[k] = true;
        self.pending += 1;
    }

    /// Samples pending since the last refresh.
    pub fn pending(&self) -> usize {
        self.pending
    }

    /// Posterior mean and standard deviation of `S` at an energy.
    ///
    /// Inside the window this reads the bin. Below the window it extrapolates
    /// linearly from the slope at the low end, with the standard deviation
    /// growing linearly in the distance extrapolated, which is the behaviour
    /// the second-difference prior implies: level and slope are unconstrained
    /// by it, so a step past the evidence keeps the slope and loses confidence
    /// at a constant rate.
    pub fn entropy(&self, e: f64) -> (f64, f64) {
        let n = self.mean.len();
        let hi = self.lo + self.width * n as f64;
        if e >= self.lo && e < hi {
            let k = self.bin(e);
            return (self.mean[k], self.sd[k]);
        }
        if e < self.lo {
            let slope = (self.mean[1] - self.mean[0]) / self.width;
            let d = self.lo + 0.5 * self.width - e;
            (
                self.mean[0] - slope * d,
                self.sd[0] + self.sd[0].max(0.5) * d / self.width,
            )
        } else {
            // Flat above the evidence, with the spread still widening: nothing
            // observed supports a claim about how many states sit up there, and
            // the honest posterior says so rather than charging for the rise.
            let d = e - (hi - 0.5 * self.width);
            (
                self.mean[n - 1],
                self.sd[n - 1] + self.sd[n - 1].max(0.5) * d / self.width,
            )
        }
    }

    /// Posterior mean and standard deviation of the statistical temperature at
    /// an energy.
    ///
    /// The statistical temperature is the reciprocal slope of the entropy,
    /// `T_S = (dS/dE~)^-1`, and it is the temperature at which a chain standing
    /// at that energy is critically mobile: the Metropolis ratio for a typical
    /// move is order one, neither frozen nor free. Kim, Straub and Keyes use it
    /// to drive dynamics directly (doi:10.1103/PhysRevLett.97.050601).
    ///
    /// Its behaviour is what a search wants and is the opposite of a cooling
    /// curve. At a funnel floor few new minima appear per unit energy, so the
    /// slope is small, the temperature is high, and the chain climbs out. In
    /// the high-energy sea minima are dense, the slope is large, the
    /// temperature is low, and the chain descends. It is read off the run's own
    /// entropy rather than scheduled against the clock, so a chain that has
    /// stopped finding new minima at its current depth heats itself.
    ///
    /// The slope is taken by central difference on the fitted mean, which is
    /// smooth by construction under the second-difference prior, and its spread
    /// follows from the two bins it is taken across.
    pub fn temperature(&self, e: f64) -> (f64, f64) {
        let n = self.mean.len();
        let k = self.bin(e).clamp(1, n.saturating_sub(2).max(1));
        let slope = (self.mean[k + 1] - self.mean[k - 1]) / (2.0 * self.width);
        let spread = (self.sd[k + 1] + self.sd[k - 1]) / (2.0 * self.width);
        if slope <= 1e-9 {
            return (f64::INFINITY, f64::INFINITY);
        }
        // The reciprocal's spread by the delta method, which is all that is
        // wanted here: a scale for how much the temperature could plausibly
        // differ, not a calibrated interval.
        (1.0 / slope, spread / (slope * slope))
    }

    /// A weight function drawn from the posterior, frozen for one sweep.
    ///
    /// Thompson sampling on `S`: one draw per bin, correlated through the same
    /// standard normal along the axis so the drawn curve is smooth rather than
    /// a bin-independent rattle, since an unsmooth weight would make the
    /// acceptance ratio between neighbouring bins meaningless.
    pub fn draw<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Weight {
        let n = self.mean.len();
        let mut s = Array1::zeros(n);
        // A single smooth perturbation: a random level and slope, which are the
        // two directions the prior leaves free, plus a damped local term.
        let z0: f64 = normal(rng);
        let z1: f64 = normal(rng);
        let mut local = 0.0;
        for k in 0..n {
            let t = (k as f64 - 0.5 * n as f64) / n as f64;
            local = 0.8 * local + 0.6 * normal(rng);
            s[k] = self.mean[k] + self.sd[k] * (0.5 * z0 + z1 * t + 0.5 * local);
        }
        self.active = s.clone();
        Weight {
            lo: self.lo,
            width: self.width,
            s,
            top: self.top_seen(),
        }
    }

    /// The posterior mean as a weight function, for runs that want no
    /// exploration term.
    pub fn mean_weight(&mut self) -> Weight {
        self.active = self.mean.clone();
        Weight {
            lo: self.lo,
            width: self.width,
            s: self.mean.clone(),
            top: self.top_seen(),
        }
    }

    /// Highest bin index the run has ever visited.
    pub fn top_seen(&self) -> usize {
        self.seen
            .iter()
            .rposition(|v| *v)
            .unwrap_or(self.mean.len() - 1)
    }

    /// Folds the pending histogram into the posterior and clears it.
    ///
    /// Returns false when there is nothing to learn from.
    pub fn refresh(&mut self) -> bool {
        if self.pending == 0 {
            return false;
        }
        let n = self.mean.len();
        // The sweep joins the record with the curve it ran under, and every
        // sweep is refitted together. Nothing is discarded and no gain
        // schedule decides how much of this sweep to believe.
        self.history
            .push((self.active.clone(), self.counts.clone()));
        if self.history.len() > 128 {
            self.history.remove(0);
        }
        let mut best = (
            f64::NEG_INFINITY,
            self.tau,
            self.mean.clone(),
            self.sd.clone(),
        );
        // The smoothness scale is chosen by the Laplace-approximated marginal
        // likelihood rather than fixed, so the amount of smoothing is set by
        // how much the histogram actually supports.
        for step in 0..7 {
            let tau = 0.05 * 4.0_f64.powi(step as i32 - 3);
            if let Some((ev, mean, sd)) = self.fit(tau) {
                if ev > best.0 {
                    best = (ev, tau, mean, sd);
                }
            }
        }
        if best.0 == f64::NEG_INFINITY {
            self.counts.fill(0.0);
            self.pending = 0;
            return false;
        }
        self.tau = best.1;
        self.mean = best.2;
        self.sd = best.3;
        // Level is not identified by the likelihood, so pin it at the lowest
        // bin. Only differences of `S` enter the acceptance.
        let anchor = self.mean[0];
        for k in 0..n {
            self.mean[k] -= anchor;
        }
        self.counts.fill(0.0);
        self.pending = 0;
        self.refreshes += 1;
        true
    }

    /// Penalised fit of `S` against every recorded sweep at one smoothness
    /// scale.
    ///
    /// Sweep `j` ran under curve `S^_j` and produced counts `n_.j` totalling
    /// `N_j`. Conditional on `N_j` the counts are multinomial with cell
    /// probabilities `p_kj = exp(S_k - S^_kj) / sum_m exp(S_m - S^_mj)`, which
    /// profiles the per-sweep normalising constant out rather than fitting it,
    /// so the only unknown is the curve itself. The log posterior is
    ///
    /// ```text
    /// sum_j [ sum_k n_kj (S_k - S^_kj) - N_j ln sum_k exp(S_k - S^_kj) ]
    ///     - (1/2) S' P S
    /// ```
    ///
    /// with `P` the second-difference precision. Newton on a dense system; the
    /// axis is short and the sweep count is capped, so the cost is negligible
    /// beside a single relaxation.
    fn fit(&self, tau: f64) -> Option<(f64, Array1<f64>, Array1<f64>)> {
        let n = self.mean.len();
        if self.history.is_empty() {
            return None;
        }
        let totals: Vec<f64> = self.history.iter().map(|(_, c)| c.sum()).collect();
        if totals.iter().sum::<f64>() <= 0.0 {
            return None;
        }
        let lambda = 1.0 / (tau * tau);
        let mut p = Array2::<f64>::zeros((n, n));
        for k in 1..n - 1 {
            let idx = [k - 1, k, k + 1];
            let w = [1.0_f64, -2.0, 1.0];
            for a in 0..3 {
                for b in 0..3 {
                    p[[idx[a], idx[b]]] += lambda * w[a] * w[b];
                }
            }
        }
        // The level is not identified by a multinomial likelihood and the prior
        // leaves it free by design, so a whisper of ridge keeps the system
        // solvable without informing the slope.
        for k in 0..n {
            p[[k, k]] += 1e-8;
        }
        let mut s = self.mean.clone();
        let mut prob = vec![Array1::<f64>::zeros(n); self.history.len()];
        for _ in 0..40 {
            let mut grad = p.dot(&s);
            let mut h = p.clone();
            for (j, (used, counts)) in self.history.iter().enumerate() {
                if totals[j] <= 0.0 {
                    continue;
                }
                let mut max = f64::NEG_INFINITY;
                for k in 0..n {
                    let v = s[k] - used[k];
                    if v > max {
                        max = v;
                    }
                }
                let mut acc = 0.0;
                for k in 0..n {
                    let v = (s[k] - used[k] - max).exp();
                    prob[j][k] = v;
                    acc += v;
                }
                if !(acc > 0.0) {
                    return None;
                }
                for k in 0..n {
                    prob[j][k] /= acc;
                }
                for k in 0..n {
                    grad[k] += totals[j] * prob[j][k] - counts[k];
                }
                // Multinomial information: diagonal minus the outer product,
                // which is what makes the level direction flat and is why the
                // prior has to supply it.
                for a in 0..n {
                    let pa = prob[j][a];
                    if pa < 1e-14 {
                        continue;
                    }
                    h[[a, a]] += totals[j] * pa;
                    for b in 0..n {
                        h[[a, b]] -= totals[j] * pa * prob[j][b];
                    }
                }
            }
            let l = cholesky(&h)?;
            let step = cholesky_solve(&l, &grad);
            let mut moved = 0.0_f64;
            for k in 0..n {
                let d = step[k].clamp(-2.0, 2.0);
                s[k] -= d;
                moved = moved.max(d.abs());
            }
            if moved < 1e-9 {
                break;
            }
        }
        // Laplace evidence, up to terms constant in tau.
        let mut ll = 0.0;
        let mut h = p.clone();
        for (j, (used, counts)) in self.history.iter().enumerate() {
            if totals[j] <= 0.0 {
                continue;
            }
            let mut max = f64::NEG_INFINITY;
            for k in 0..n {
                let v = s[k] - used[k];
                if v > max {
                    max = v;
                }
            }
            let mut acc = 0.0;
            for k in 0..n {
                let v = (s[k] - used[k] - max).exp();
                prob[j][k] = v;
                acc += v;
            }
            for k in 0..n {
                prob[j][k] /= acc;
                ll += counts[k] * (s[k] - used[k]);
            }
            ll -= totals[j] * (acc.ln() + max);
            for a in 0..n {
                let pa = prob[j][a];
                if pa < 1e-14 {
                    continue;
                }
                h[[a, a]] += totals[j] * pa;
                for b in 0..n {
                    h[[a, b]] -= totals[j] * pa * prob[j][b];
                }
            }
        }
        let pen = 0.5 * s.dot(&p.dot(&s));
        let lh = cholesky(&h)?;
        let lp = cholesky(&p)?;
        let mut logdet_h = 0.0;
        let mut logdet_p = 0.0;
        for k in 0..n {
            logdet_h += lh[[k, k]].ln();
            logdet_p += lp[[k, k]].ln();
        }
        let evidence = ll - pen + logdet_p - logdet_h;
        if !evidence.is_finite() {
            return None;
        }
        let mut sd = Array1::<f64>::zeros(n);
        for k in 0..n {
            let mut e = Array1::<f64>::zeros(n);
            e[k] = 1.0;
            let col = cholesky_solve(&lh, &e);
            sd[k] = col[k].max(1e-12).sqrt();
        }
        Some((evidence, s, sd))
    }
}

/// A frozen weight function: the entropy curve one sweep accepts against.
#[derive(Debug, Clone)]
pub struct Weight {
    lo: f64,
    width: f64,
    s: Array1<f64>,
    /// Highest bin the run has ever visited. Above it the curve is held flat,
    /// because the smoothing prior would otherwise carry the observed trend
    /// through the empty bins and charge the chain for a rise no evidence
    /// supports.
    top: usize,
}

impl Weight {
    /// The entropy this weight assigns to an energy, extrapolated linearly
    /// outside the binned window.
    pub fn at(&self, e: f64) -> f64 {
        let n = self.s.len();
        let hi = self.lo + self.width * n as f64;
        if e < self.lo {
            let slope = (self.s[1] - self.s[0]) / self.width;
            return self.s[0] - slope * (self.lo + 0.5 * self.width - e);
        }
        // Above the highest bin the run has reached there is no evidence about
        // how many states there are, so the curve is held at the last value it
        // has any. Carrying the trend up instead is what charges the chain for
        // the rise it needs to leave a funnel.
        if e >= self.lo + self.width * (self.top + 1) as f64 {
            return self.s[self.top];
        }
        if e >= hi {
            // Held flat above the evidence rather than extrapolated along the
            // trend. Continuing the trend asserts that the unobserved high
            // energies hold ever more states, which charges the chain for
            // exactly the rise it has to make to leave a funnel and rebuilds
            // the trap this target exists to dissolve. Measured on the
            // two-funnel test, extrapolating upward put the crossing state
            // beyond reach and the rule lost to Metropolis 3 to 7.
            return self.s[n - 1];
        }
        let k = (((e - self.lo) / self.width).floor() as usize).min(n - 1);
        self.s[k]
    }

    /// Acceptance probability for a move between two quenched energies.
    ///
    /// Metropolis against `1 / g`: a move to a rarer energy is always taken,
    /// and a move to a more abundant one is taken with the ratio of
    /// multiplicities. Both directions in energy are treated alike, which is
    /// what makes the sampled histogram flat rather than funnel-weighted.
    pub fn accept_prob(&self, e_old: f64, e_new: f64) -> f64 {
        let d = self.at(e_new) - self.at(e_old);
        if d <= 0.0 { 1.0 } else { (-d).exp() }
    }
}

/// A flat target below a cut and a Boltzmann one above it.
///
/// Flat sampling across a whole energy range is the wrong target for a search.
/// It buys barrier crossing by giving every energy an equal share of the run,
/// and most of the range holds nothing worth the share: measured on 38 points,
/// flat over the visited range solved 3 seeds in 24 where the plain Metropolis
/// rule solved 15, because the budget went to the high-energy sea.
///
/// Restricting the flat region fixes that, and the restriction cannot be a
/// wall. Crossing between funnels *requires* rising in quenched energy, so
/// forbidding the rise rebuilds the trap the flat target was adopted to
/// dissolve. What works is flat below and suppressed above:
///
/// ```text
/// -ln pi(E~) = S(E~) + max(0, E~ - c) / w
/// ```
///
/// Below `c` the multiplicity of a funnel does not decide anything, so a
/// barrier inside the reachable region is invisible. Above `c` the cost grows
/// linearly, so the chain can still climb `w`-worth to cross and does not walk
/// off to the top. The cut descends with the chain, which is the annealing:
/// the schedule is read off the run's own visited energies rather than
/// supplied as a cooling curve.
#[derive(Debug, Clone)]
pub struct CutWeight {
    /// The entropy curve.
    pub weight: Weight,
    /// Energy above which the target stops being flat.
    pub cut: f64,
    /// Width of the suppression above the cut, in energy.
    pub width: f64,
}

impl CutWeight {
    /// The negative log target at an energy.
    pub fn cost(&self, e: f64) -> f64 {
        self.weight.at(e) + (e - self.cut).max(0.0) / self.width.max(1e-9)
    }

    /// Acceptance probability for a move between two quenched energies, with an
    /// additive term for any external bias the caller carries.
    pub fn accept_prob(&self, e_old: f64, e_new: f64, bias_delta: f64) -> f64 {
        let d = self.cost(e_new) - self.cost(e_old) + bias_delta;
        if d <= 0.0 { 1.0 } else { (-d).exp() }
    }
}

/// The cut and width implied by a sweep's visited energies.
///
/// The cut is a quantile of what the chain saw and the width is a spread of the
/// same sample, so both descend as the run descends and neither is a tuned
/// constant. A run that has stopped improving keeps the cut where it is, which
/// is the right behaviour: the schedule follows progress rather than the clock.
pub fn cut_from(seen: &[f64], quantile: f64) -> Option<(f64, f64)> {
    if seen.len() < 8 {
        return None;
    }
    let mut v: Vec<f64> = seen.iter().cloned().filter(|x| x.is_finite()).collect();
    if v.len() < 8 {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pick = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    let cut = pick(quantile.clamp(0.01, 0.99));
    // A quarter of the interdecile range, floored so a stalled sweep whose
    // energies have collapsed onto one value still allows a climb.
    let width = ((pick(0.9) - pick(0.1)) / 4.0).max(0.05);
    Some((cut, width))
}

/// A standard normal draw by Box-Muller, since the crate carries no normal
/// sampler at this layer.
fn normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-12);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Lower Cholesky factor of a symmetric positive definite matrix.
fn cholesky(a: &Array2<f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    Some(l)
}

/// Solves `L L^T x = b` for `x`.
fn cholesky_solve(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = l.nrows();
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[[i, k]] * y[k];
        }
        y[i] = sum / l[[i, i]];
    }
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in i + 1..n {
            sum -= l[[k, i]] * x[k];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Counts drawn from a known entropy curve have to recover it, or the
    /// posterior is not measuring what it claims to.
    #[test]
    fn the_posterior_recovers_an_entropy_it_is_shown() {
        let mut d = DensityOfStates::new(0.0, 10.0, 40);
        let mut rng = StdRng::seed_from_u64(1);
        // S(E) = 0.9 E, so bin k holds exp(0.9 * centre) states. Sampling under
        // a flat weight visits bin k in proportion to exp(S_k).
        let truth: Vec<f64> = (0..40).map(|k| 0.9 * d.centre(k)).collect();
        let max = truth.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        for _ in 0..40000 {
            let k = rng.random_range(0..40);
            if rng.random::<f64>() < (truth[k] - max).exp() {
                d.observe(d.centre(k));
            }
        }
        assert!(d.refresh(), "refresh found nothing to fit");
        // Only differences matter, so compare slopes over the range where the
        // histogram has counts.
        let got = d.entropy(d.centre(30)).0 - d.entropy(d.centre(10)).0;
        let want = truth[30] - truth[10];
        assert!(
            (got - want).abs() < 0.15 * want.abs(),
            "recovered {got}, truth {want}"
        );
    }

    /// A bin with no counts still has to come back with a finite value and a
    /// wider spread than a bin that has evidence. That is the property a
    /// histogram method does not have.
    #[test]
    fn an_unvisited_bin_gets_a_value_and_a_wider_error_bar() {
        let mut d = DensityOfStates::new(0.0, 10.0, 40);
        for k in 10..30 {
            for _ in 0..200 {
                d.observe(d.centre(k));
            }
        }
        assert!(d.refresh());
        let (m_seen, sd_seen) = d.entropy(d.centre(20));
        let (m_unseen, sd_unseen) = d.entropy(d.centre(2));
        assert!(m_seen.is_finite() && m_unseen.is_finite());
        assert!(
            sd_unseen > sd_seen,
            "unvisited sd {sd_unseen} not wider than visited {sd_seen}"
        );
    }

    /// Past the deepest energy seen, the mean has to keep the slope and the
    /// spread has to grow. That is what makes the posterior say something about
    /// how many minima remain below.
    #[test]
    fn the_posterior_extrapolates_below_the_window() {
        let mut d = DensityOfStates::new(0.0, 10.0, 40);
        let mut rng = StdRng::seed_from_u64(7);
        let truth: Vec<f64> = (0..40).map(|k| 0.8 * d.centre(k)).collect();
        let max = truth.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        for _ in 0..40000 {
            let k = rng.random_range(0..40);
            if rng.random::<f64>() < (truth[k] - max).exp() {
                d.observe(d.centre(k));
            }
        }
        assert!(d.refresh());
        let (m0, sd0) = d.entropy(d.centre(0));
        let (m_out, sd_out) = d.entropy(-2.0);
        assert!(m_out < m0, "extrapolated {m_out} did not fall below {m0}");
        assert!(sd_out > sd0, "spread {sd_out} did not grow past {sd0}");
    }

    /// A frozen weight has to be a pure function of energy, or the sweep it
    /// governs is not a Markov chain for any target.
    #[test]
    fn a_frozen_weight_is_deterministic() {
        let mut d = DensityOfStates::new(0.0, 10.0, 40);
        for k in 0..40 {
            for _ in 0..(10 + k) {
                d.observe(d.centre(k));
            }
        }
        d.refresh();
        let mut rng = StdRng::seed_from_u64(3);
        let w = d.draw(&mut rng);
        for _ in 0..10 {
            assert_eq!(w.at(4.2), w.at(4.2));
            assert_eq!(w.accept_prob(2.0, 7.0), w.accept_prob(2.0, 7.0));
        }
    }

    /// Downhill in entropy is always taken and uphill decays with the ratio of
    /// multiplicities, which is the rule the module exists to apply.
    #[test]
    fn acceptance_is_metropolis_in_entropy() {
        let mut d = DensityOfStates::new(0.0, 10.0, 40);
        let mut w = d.mean_weight();
        for k in 0..40 {
            w.s[k] = 0.5 * (k as f64);
        }
        assert_eq!(w.accept_prob(9.0, 1.0), 1.0);
        let up = w.accept_prob(1.0, 9.0);
        assert!(up > 0.0 && up < 1e-6, "uphill acceptance {up}");
    }

    /// The double funnel, reduced to the part that matters and nothing else.
    ///
    /// Two funnels joined by a single high-energy bridge, which is the only
    /// state a proposal can use to cross. The wide funnel holds four thousand
    /// states around the same moderate energy; the narrow one holds a dozen and
    /// bottoms out below anything in the wide one. Proposals are symmetric,
    /// mostly stay in the funnel the chain is in, and both rules are handed the
    /// same ones from the same generator.
    ///
    /// The crossing rate is what separates the rules. Metropolis has to pay
    /// `exp(-dE / T)` for the rise onto the bridge. Accepting against `1 / g`
    /// pays nothing for it, because the bridge is a *rarer* energy than the
    /// wide funnel's floor and moving toward rare is the direction this rule
    /// takes for free. The barrier is thermodynamically invisible to it.
    fn two_funnels<R: Rng>(rng: &mut R, flat: bool, steps: usize, temp: f64) -> bool {
        const W: usize = 4000;
        const N: usize = 12;
        let bridge = W;
        let target = W + N;
        let energy = |i: usize| -> f64 {
            if i < W {
                0.9 + 0.5 * (i as f64 / W as f64)
            } else if i == bridge {
                3.5
            } else {
                0.9 - 0.075 * (i - bridge) as f64
            }
        };
        // The wide funnel is a sea: a proposal lands anywhere in it. The narrow
        // funnel has to be walked down one step at a time, which is what makes
        // a temperature high enough to cross the bridge too high to hold the
        // descent. Only the narrow funnel's top rung sees the bridge.
        let propose = |rng: &mut R, cur: usize| -> usize {
            if cur == bridge {
                // Leaving the bridge lands in a funnel in proportion to how
                // many states it holds, which is what a random perturbation
                // from a crossing structure does. This is the entropic trap
                // itself: the way back into the wide funnel is four thousand
                // times likelier than the way into the narrow one, so a rule
                // that does not discount multiplicity is pulled back every
                // time it gets out.
                if rng.random_range(0..W + 1) < W {
                    rng.random_range(0..W)
                } else {
                    bridge + 1
                }
            } else if cur < W {
                if rng.random::<f64>() < 0.001 {
                    bridge
                } else {
                    rng.random_range(0..W)
                }
            } else if cur == bridge + 1 && rng.random::<f64>() < 0.5 {
                bridge
            } else if cur == target || rng.random::<bool>() {
                cur - 1
            } else {
                cur + 1
            }
        };
        let mut cur = W - 1;
        if !flat {
            for _ in 0..steps {
                let prop = propose(rng, cur);
                let de = energy(prop) - energy(cur);
                if de <= 0.0 || rng.random::<f64>() < (-de / temp).exp() {
                    cur = prop;
                }
                if cur == target {
                    return true;
                }
            }
            return false;
        }
        let mut d = DensityOfStates::new(-0.2, 2.4, 48);
        let mut used = 0usize;
        while used < steps {
            let w = d.draw(rng);
            for _ in 0..500 {
                used += 1;
                let prop = propose(rng, cur);
                if rng.random::<f64>() < w.accept_prob(energy(cur), energy(prop)) {
                    cur = prop;
                }
                d.observe(energy(cur));
                if cur == target {
                    return true;
                }
            }
            d.refresh();
        }
        false
    }

    #[test]
    fn flat_sampling_crosses_the_funnel_barrier_and_metropolis_does_not() {
        let steps = 300000;
        // Metropolis is given the best of a temperature sweep rather than one
        // guess, so the comparison is against the rule at its own optimum.
        let mut best_metro = 0usize;
        let mut best_temp = 0.0;
        for t in [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0] {
            let mut rng = StdRng::seed_from_u64(11);
            let hits = (0..8)
                .filter(|_| two_funnels(&mut rng, false, steps, t))
                .count();
            if hits > best_metro {
                best_metro = hits;
                best_temp = t;
            }
        }
        let mut rng = StdRng::seed_from_u64(101);
        let hits_dos = (0..8)
            .filter(|_| two_funnels(&mut rng, true, steps, 0.0))
            .count();
        eprintln!("flat {hits_dos}/8, metropolis {best_metro}/8 at T={best_temp}");
        assert!(
            hits_dos > best_metro,
            "flat {hits_dos}/8, metropolis {best_metro}/8 at its best temperature {best_temp}"
        );
    }
}

/// A well-tempered bias deposited in quenched energy.
///
/// The per-basin bias this crate carries fills the basin the chain stands in.
/// Measured at 38 points, that is not the shape of the trap: of 17 failing
/// seeds, 12 end at exactly -173.252378 and 4 at -173.134317, the floor of the
/// icosahedral funnel. A funnel holds exponentially many basins, so filling
/// them one at a time cannot fill it, and coarsening the basin metric does not
/// help because a single length in coordinate space cannot tell a funnel's
/// variants from a different funnel: at a radius of 0.7 the run registers 365
/// basins and solves 55 of 72, at 2.0 it registers 7 and solves 24.
///
/// Energy separates what the length cannot. The funnel floor is an energy the
/// chain returns to, so depositing there fills the funnel rather than one of
/// its basins, and the coordinate costs nothing because every hop computes it.
///
/// Deposits are well tempered (Barducci, Bussi and Parrinello,
/// doi:10.1103/PhysRevLett.100.020603): height falls as `exp(-V/((gamma-1) T))`
/// where the bias already stands, so the sum converges instead of growing
/// without bound, and the sampled distribution is the well-tempered ensemble of
/// Bonomi and Parrinello (doi:10.1103/PhysRevLett.104.190601) rather than a
/// flat one. This is the part the flat-histogram acceptance got wrong: it
/// forced every energy to an equal share, where tempering only broadens what
/// the chain already samples.
#[derive(Debug, Clone)]
pub struct EnergyBias {
    lo: f64,
    width: f64,
    v: Array1<f64>,
    /// Initial deposit height.
    pub w0: f64,
    /// Tempering factor. One is no bias; large is untempered metadynamics.
    pub gamma: f64,
    /// Width of a deposit, in bins.
    pub sigma_bins: f64,
    /// Deposits made.
    pub deposits: usize,
}

impl EnergyBias {
    /// The number of deposits that fill a well one standard deviation deep.
    ///
    /// The one free number in this construction, and it is dimensionless: every
    /// other scale is taken from the run's own quenched-energy distribution, so
    /// nothing here carries units of a particular system's energy and nothing
    /// is set per system.
    pub const FILL_DEPOSITS: f64 = 100.0;

    /// A bias whose scales come from a sample of quenched energies.
    ///
    /// The tempering factor is set by `(gamma - 1) T = sigma`, so the bias
    /// broadens the energy distribution by about its own width, and the deposit
    /// height by `w0 = sigma / FILL_DEPOSITS`, so a well one standard deviation
    /// deep fills in a fixed number of deposits whatever the system's energy
    /// scale. The window spans the sample padded by a standard deviation on
    /// each side.
    ///
    /// Returns `None` when the sample is too small or degenerate to set a
    /// scale, which is the honest answer rather than a default.
    pub fn from_sample(seen: &[f64], temp: f64, bins: usize) -> Option<Self> {
        let v: Vec<f64> = seen.iter().cloned().filter(|x| x.is_finite()).collect();
        if v.len() < 32 {
            return None;
        }
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / v.len() as f64;
        let sigma = var.sqrt();
        if !(sigma > 0.0) || !sigma.is_finite() {
            return None;
        }
        let lo = v.iter().cloned().fold(f64::INFINITY, f64::min) - sigma;
        let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + sigma;
        let gamma = 1.0 + sigma / temp.max(1e-12);
        Some(Self::new(lo, hi, bins, sigma / Self::FILL_DEPOSITS, gamma))
    }

    /// A bias over `[lo, hi]`, empty until something is deposited.
    pub fn new(lo: f64, hi: f64, bins: usize, w0: f64, gamma: f64) -> Self {
        let bins = bins.max(8);
        Self {
            lo,
            width: ((hi - lo) / bins as f64).max(1e-9),
            v: Array1::zeros(bins),
            w0,
            gamma: gamma.max(1.0 + 1e-9),
            sigma_bins: 1.5,
            deposits: 0,
        }
    }

    /// The bias at an energy, held flat outside the binned window so a chain
    /// that leaves the range is neither pushed back nor pulled out.
    pub fn at(&self, e: f64) -> f64 {
        let n = self.v.len();
        let k = ((e - self.lo) / self.width).floor();
        if k < 0.0 {
            return self.v[0];
        }
        if k >= n as f64 {
            return self.v[n - 1];
        }
        self.v[k as usize]
    }

    /// Deposits at an energy, at the well-tempered height for the bias already
    /// standing there.
    pub fn deposit(&mut self, e: f64, temp: f64) {
        if !e.is_finite() {
            return;
        }
        let n = self.v.len();
        let centre = (e - self.lo) / self.width;
        let h = self.w0 * (-self.at(e) / ((self.gamma - 1.0) * temp.max(1e-12))).exp();
        if !h.is_finite() || h <= 0.0 {
            return;
        }
        let reach = (3.0 * self.sigma_bins).ceil() as isize;
        let c = centre.floor() as isize;
        for d in -reach..=reach {
            let k = c + d;
            if k < 0 || k >= n as isize {
                continue;
            }
            let dx = (k as f64 + 0.5 - centre) / self.sigma_bins;
            self.v[k as usize] += h * (-0.5 * dx * dx).exp();
        }
        self.deposits += 1;
    }

    /// The bias difference a move carries, in units of temperature, ready to be
    /// added to a Metropolis exponent.
    pub fn delta(&self, e_old: f64, e_new: f64, temp: f64) -> f64 {
        (self.at(e_new) - self.at(e_old)) / temp.max(1e-12)
    }

    /// Largest bias standing anywhere, for reporting how filled the range is.
    pub fn peak(&self) -> f64 {
        self.v.iter().cloned().fold(0.0_f64, f64::max)
    }
}

#[cfg(test)]
mod energy_bias_tests {
    use super::*;

    /// A deposit has to raise the energy it was made at, or the bias does
    /// nothing.
    #[test]
    fn a_deposit_raises_where_it_lands() {
        let mut b = EnergyBias::new(-180.0, -140.0, 64, 0.1, 5.0);
        let before = b.at(-173.25);
        b.deposit(-173.25, 0.8);
        assert!(b.at(-173.25) > before);
    }

    /// Well tempering has to make the bias grow as `(gamma - 1) T ln t` at an
    /// energy that keeps being deposited into, which is the result that makes
    /// the sum converge in the sense that matters: the *rate* falls to zero, so
    /// a funnel fills and then stops deepening, where an untempered deposit
    /// would keep pushing forever.
    ///
    /// Checked against the law rather than against a tolerance on the value,
    /// since the value genuinely does keep rising.
    #[test]
    fn the_bias_grows_by_the_well_tempered_law() {
        let gamma = 5.0;
        let temp = 0.8;
        let mut b = EnergyBias::new(-180.0, -140.0, 64, 0.1, gamma);
        for _ in 0..20000 {
            b.deposit(-173.25, temp);
        }
        let v1 = b.at(-173.25);
        for _ in 0..40000 {
            b.deposit(-173.25, temp);
        }
        let v2 = b.at(-173.25);
        let want = (gamma - 1.0) * temp * 3.0_f64.ln();
        assert!(
            ((v2 - v1) - want).abs() < 0.1 * want,
            "tripling the deposits raised the bias by {}, law says {want}",
            v2 - v1
        );
    }

    /// The trap the bias exists to break: a chain pinned at one energy has to
    /// find that energy uphill of a neighbour once enough has been deposited.
    #[test]
    fn filling_a_floor_makes_leaving_it_downhill() {
        let mut b = EnergyBias::new(-180.0, -140.0, 64, 0.1, 5.0);
        let floor = -173.25;
        let out = -171.0;
        for _ in 0..5000 {
            b.deposit(floor, 0.8);
        }
        assert!(
            b.delta(floor, out, 0.8) < 0.0,
            "leaving the filled floor still costs {}",
            b.delta(floor, out, 0.8)
        );
    }

    /// Outside the window the bias is flat, so a chain that escapes the range
    /// is neither pushed back nor pulled out.
    #[test]
    fn the_bias_is_flat_outside_the_window() {
        let mut b = EnergyBias::new(-180.0, -140.0, 64, 0.1, 5.0);
        b.deposit(-179.0, 0.8);
        assert_eq!(b.at(-200.0), b.at(-179.9));
        assert_eq!(b.at(-100.0), b.at(-140.1));
    }
}
