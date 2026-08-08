//! A posterior over the density of minima, and acceptance by entropy.
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
            (self.mean[0] - slope * d, self.sd[0] + self.sd[0].max(0.5) * d / self.width)
        } else {
            let slope = (self.mean[n - 1] - self.mean[n - 2]) / self.width;
            let d = e - (hi - 0.5 * self.width);
            (
                self.mean[n - 1] + slope * d,
                self.sd[n - 1] + self.sd[n - 1].max(0.5) * d / self.width,
            )
        }
    }

    /// A weight function drawn from the posterior, frozen for one sweep.
    ///
    /// Thompson sampling on `S`: one draw per bin, correlated through the same
    /// standard normal along the axis so the drawn curve is smooth rather than
    /// a bin-independent rattle, since an unsmooth weight would make the
    /// acceptance ratio between neighbouring bins meaningless.
    pub fn draw<R: Rng + ?Sized>(&self, rng: &mut R) -> Weight {
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
        Weight {
            lo: self.lo,
            width: self.width,
            s,
        }
    }

    /// The posterior mean as a weight function, for runs that want no
    /// exploration term.
    pub fn mean_weight(&self) -> Weight {
        Weight {
            lo: self.lo,
            width: self.width,
            s: self.mean.clone(),
        }
    }

    /// Folds the pending histogram into the posterior and clears it.
    ///
    /// Returns false when there is nothing to learn from.
    pub fn refresh(&mut self) -> bool {
        if self.pending == 0 {
            return false;
        }
        let n = self.mean.len();
        let mut best = (f64::NEG_INFINITY, self.tau, self.mean.clone(), self.sd.clone());
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

    /// Penalised Poisson fit at one smoothness scale.
    ///
    /// Model: `n_k ~ Poisson(exp(d_k + c))` where `d = S_new - S_old` is the
    /// residual the histogram measures, under a second-difference Gaussian
    /// prior on `S_new`. Newton iteration on a dense symmetric positive
    /// definite system; the axis is short enough that banded structure is not
    /// worth the code.
    fn fit(&self, tau: f64) -> Option<(f64, Array1<f64>, Array1<f64>)> {
        let n = self.mean.len();
        let total: f64 = self.counts.sum();
        if total <= 0.0 {
            return None;
        }
        // Second-difference precision, scaled by the smoothness.
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
        // A whisper of ridge so the two unconstrained directions (level and
        // slope) stay solvable without being informed. Small enough that it
        // does not shrink the extrapolated slope.
        for k in 0..n {
            p[[k, k]] += 1e-8;
        }
        let mut s = self.mean.clone();
        let mut mu = Array1::<f64>::zeros(n);
        for _ in 0..64 {
            // The offset that matches total counts, which is what the
            // normalising constant of the multinomial contributes.
            let mut acc = 0.0;
            for k in 0..n {
                acc += (s[k] - self.mean[k]).min(30.0).exp();
            }
            if !(acc > 0.0) {
                return None;
            }
            let c = (total / acc).ln();
            let mut grad = Array1::<f64>::zeros(n);
            for k in 0..n {
                mu[k] = (s[k] - self.mean[k] + c).min(30.0).exp();
                grad[k] = mu[k] - self.counts[k];
            }
            let ps = p.dot(&s);
            for k in 0..n {
                grad[k] += ps[k];
            }
            let mut h = p.clone();
            for k in 0..n {
                h[[k, k]] += mu[k];
            }
            let l = cholesky(&h)?;
            let step = cholesky_solve(&l, &grad);
            let mut moved = 0.0_f64;
            for k in 0..n {
                let d = step[k].clamp(-4.0, 4.0);
                s[k] -= d;
                moved = moved.max(d.abs());
            }
            if moved < 1e-8 {
                break;
            }
        }
        // Laplace evidence up to terms constant in tau: log-likelihood at the
        // mode, minus the penalty, plus half the log determinant of the prior
        // over that of the posterior.
        let mut acc = 0.0;
        for k in 0..n {
            acc += (s[k] - self.mean[k]).min(30.0).exp();
        }
        let c = (total / acc).ln();
        let mut ll = 0.0;
        for k in 0..n {
            let eta = s[k] - self.mean[k] + c;
            ll += self.counts[k] * eta - eta.min(30.0).exp();
        }
        let ps = p.dot(&s);
        let pen = 0.5 * s.dot(&ps);
        let mut h = p.clone();
        for k in 0..n {
            h[[k, k]] += (s[k] - self.mean[k] + c).min(30.0).exp();
        }
        let lh = cholesky(&h)?;
        let lp = cholesky(&p)?;
        let mut logdet_h = 0.0;
        let mut logdet_p = 0.0;
        for k in 0..n {
            logdet_h += lh[[k, k]].ln();
            logdet_p += lp[[k, k]].ln();
        }
        let evidence = ll - pen + logdet_p - logdet_h;
        // Marginal standard deviations from the inverse of the posterior
        // precision.
        let mut sd = Array1::<f64>::zeros(n);
        for k in 0..n {
            let mut e = Array1::<f64>::zeros(n);
            e[k] = 1.0;
            let col = cholesky_solve(&lh, &e);
            sd[k] = col[k].max(1e-12).sqrt();
        }
        if !evidence.is_finite() {
            return None;
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
        if e >= hi {
            let slope = (self.s[n - 1] - self.s[n - 2]) / self.width;
            return self.s[n - 1] + slope * (e - (hi - 0.5 * self.width));
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
        if d <= 0.0 {
            1.0
        } else {
            (-d).exp()
        }
    }
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
    use rand::rngs::StdRng;
    use rand::SeedableRng;

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
        let d = DensityOfStates::new(0.0, 10.0, 40);
        let mut w = d.mean_weight();
        for k in 0..40 {
            w.s[k] = 0.5 * (k as f64);
        }
        assert_eq!(w.accept_prob(9.0, 1.0), 1.0);
        let up = w.accept_prob(1.0, 9.0);
        assert!(up > 0.0 && up < 1e-6, "uphill acceptance {up}");
    }

    /// The mechanism, on a landscape with the pathology and nothing else.
    ///
    /// Two funnels. The wide one holds thousands of states at moderate energy;
    /// the narrow one holds a handful and contains the lowest state. Moves are
    /// symmetric and both rules see the same ones. Metropolis at any
    /// temperature that keeps the chain moving sits in the wide funnel because
    /// that is where the states are, and the flat-histogram rule does not,
    /// because it is not weighting by how many there are.
    #[test]
    fn flat_sampling_finds_the_rare_deep_state_and_metropolis_does_not() {
        // State i in 0..W is wide-funnel, energy 1.0 + small spread.
        // State W+j for j in 0..N is narrow-funnel, energy falling to 0.
        const W: usize = 4000;
        const N: usize = 12;
        let energy = |i: usize| -> f64 {
            if i < W {
                1.0 + (i % 7) as f64 * 0.05
            } else {
                let j = i - W;
                0.9 - 0.08 * j as f64
            }
        };
        let target = W + N - 1;
        let steps = 60000;

        // Metropolis on energy at the temperature that keeps it moving.
        let mut rng = StdRng::seed_from_u64(11);
        let mut hits_metro = 0usize;
        for _ in 0..8 {
            let mut cur = 0usize;
            let mut found = false;
            for _ in 0..steps {
                let prop = rng.random_range(0..W + N);
                let de = energy(prop) - energy(cur);
                if de <= 0.0 || rng.random::<f64>() < (-de / 0.25).exp() {
                    cur = prop;
                }
                if cur == target {
                    found = true;
                    break;
                }
            }
            if found {
                hits_metro += 1;
            }
        }

        // The same chain accepting against a learned entropy.
        let mut hits_dos = 0usize;
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(101 + seed);
            let mut d = DensityOfStates::new(-0.3, 1.5, 48);
            let mut cur = 0usize;
            let mut found = false;
            let mut used = 0usize;
            'outer: while used < steps {
                let w = d.draw(&mut rng);
                for _ in 0..1500 {
                    used += 1;
                    let prop = rng.random_range(0..W + N);
                    if rng.random::<f64>() < w.accept_prob(energy(cur), energy(prop)) {
                        cur = prop;
                    }
                    d.observe(energy(cur));
                    if cur == target {
                        found = true;
                        break 'outer;
                    }
                }
                d.refresh();
            }
            if found {
                hits_dos += 1;
            }
        }

        assert!(
            hits_dos > hits_metro,
            "flat sampling {hits_dos}/8, metropolis {hits_metro}/8: the mechanism did not separate"
        );
    }
}
