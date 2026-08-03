//! Deciding which trials are worth relaxing, under a posterior.
//!
//! The expensive step in a cluster search is the local relaxation, at roughly
//! thirty charged evaluations against one for a proposal, and most trials are
//! not worth it. The cheap version of that decision is a threshold on the
//! energy after a few relaxation steps, and it is the one mechanism in this
//! crate that is measurably worth having: at 75 points it takes the success
//! rate from 2 seeds in 8 to 13 in 24. Its threshold is a hand-set margin.
//!
//! This replaces the margin with inference. A partial relaxation supplies cheap
//! features, the full relaxation supplies the answer, and a Bayesian linear
//! model of the second given the first says how plausible it is that finishing
//! the relaxation would improve on the incumbent. Spending the thirty
//! evaluations is then a decision under a posterior rather than a comparison
//! against a constant, which is what probabilistic numerics asks of a numerical
//! procedure: treat the quantity you have not computed as unknown, not as
//! absent.
//!
//! # Why the exploration floor is not optional
//!
//! The model is trained on the trials it chose to relax, so a rule that relaxes
//! only where it already predicts improvement never sees a counterexample and
//! its own confidence is self-confirming. A fixed fraction of trials is relaxed
//! regardless of the posterior, which keeps the training set from being
//! censored by the decision rule it trains. This is the same reason a bandit
//! keeps a floor on every arm, and [`crate::allocate`] does it there.
//!
//! # The approximation, stated
//!
//! The conjugate Normal-Inverse-Gamma posterior gives a Student-t predictive.
//! The tail probability here uses the Gaussian with the same variance, which is
//! exact in the limit of many observations and understates the tails below it.
//! That is why nothing is decided by the model until [`Screen::warmup`]
//! observations have arrived; before then every trial is relaxed.

use ndarray::{Array1, Array2, ArrayView1};

/// Features of a partially relaxed trial, and whether it deserves a full one.
///
/// The design vector is the caller's. What the model needs is that it be cheap
/// relative to a relaxation and computed the same way at fit time and at
/// decision time.
#[derive(Debug, Clone)]
pub struct Screen {
    /// Observations required before the posterior is used at all.
    pub warmup: usize,
    /// Fraction of trials relaxed regardless of what the model says.
    pub exploration: f64,
    /// Posterior probability of improvement above which a trial is relaxed.
    ///
    /// Set from the cost asymmetry rather than by taste: relaxing costs a fixed
    /// number of evaluations, and failing to relax a trial that would have
    /// improved costs the search that improvement. A low threshold spends more
    /// and misses less.
    pub threshold: f64,
    /// Precision matrix of the coefficient posterior, `V0^-1 + X'X`.
    precision: Array2<f64>,
    /// `V0^-1 m0 + X'y`.
    rhs: Array1<f64>,
    /// Inverse-gamma shape, `a0 + n/2`.
    a: f64,
    /// Inverse-gamma rate, updated from the residuals.
    b: f64,
    /// `y'y`, kept so the rate can be formed without storing the data.
    yy: f64,
    /// Observations folded in.
    n: usize,
    /// Decisions made, and how many said relax.
    pub decided: usize,
    /// Of those, how many were relaxed.
    pub relaxed: usize,
    /// Relaxations forced by the exploration floor.
    pub explored: usize,
}

impl Screen {
    /// A screen over `d` features, with a weak prior.
    ///
    /// The prior precision is a small multiple of the identity, which is a
    /// ridge: it keeps the first few updates from producing an ill-conditioned
    /// posterior without expressing an opinion about the coefficients.
    pub fn new(d: usize, warmup: usize, exploration: f64, threshold: f64) -> Self {
        assert!(d > 0, "a design needs at least one feature");
        assert!(
            (0.0..=1.0).contains(&exploration),
            "the exploration floor is a fraction, got {exploration}"
        );
        assert!(
            (0.0..1.0).contains(&threshold),
            "the decision threshold is a probability, got {threshold}"
        );
        Self {
            warmup,
            exploration,
            threshold,
            precision: Array2::eye(d) * 1e-3,
            rhs: Array1::zeros(d),
            a: 1e-3,
            b: 1e-3,
            yy: 0.0,
            n: 0,
            decided: 0,
            relaxed: 0,
            explored: 0,
        }
    }

    /// Observations folded in so far.
    pub fn observations(&self) -> usize {
        self.n
    }

    /// Records a trial whose full relaxation is known.
    pub fn observe(&mut self, x: ArrayView1<f64>, y: f64) {
        if !y.is_finite() || x.iter().any(|v| !v.is_finite()) {
            return;
        }
        let d = self.rhs.len();
        if x.len() != d {
            return;
        }
        for i in 0..d {
            for j in 0..d {
                self.precision[[i, j]] += x[i] * x[j];
            }
            self.rhs[i] += x[i] * y;
        }
        self.yy += y * y;
        self.n += 1;
        self.a += 0.5;
    }

    /// Posterior mean of the coefficients, or `None` before anything is known.
    pub fn coefficients(&self) -> Option<Array1<f64>> {
        solve(self.precision.view(), self.rhs.view())
    }

    /// Predictive mean and variance at `x`.
    ///
    /// The variance is the Student-t scale, which carries both the noise and
    /// the uncertainty in the coefficients; the second term is what makes a
    /// trial unlike anything seen before come out uncertain rather than
    /// confidently predicted.
    pub fn predict(&self, x: ArrayView1<f64>) -> Option<(f64, f64)> {
        let m = self.coefficients()?;
        if x.len() != m.len() {
            return None;
        }
        let mean: f64 = x.iter().zip(m.iter()).map(|(a, b)| a * b).sum();
        // b_n = b0 + (y'y - m' rhs) / 2, the residual form that avoids keeping
        // the data.
        let mrhs: f64 = m.iter().zip(self.rhs.iter()).map(|(a, c)| a * c).sum();
        let bn = (self.b + 0.5 * (self.yy - mrhs)).max(1e-12);
        let s2 = bn / self.a.max(1e-12);
        let vx = quad_form(self.precision.view(), x)?;
        Some((mean, (s2 * (1.0 + vx)).max(1e-12)))
    }

    /// Whether to pay for the full relaxation of a trial with features `x`,
    /// given the incumbent `best`.
    ///
    /// `u` is a uniform draw supplied by the caller, so the decision is
    /// reproducible under the caller's seed rather than reaching for its own
    /// randomness.
    pub fn decide(&mut self, x: ArrayView1<f64>, best: f64, u: f64) -> bool {
        self.decided += 1;
        if u < self.exploration {
            self.explored += 1;
            self.relaxed += 1;
            return true;
        }
        if self.n < self.warmup {
            self.relaxed += 1;
            return true;
        }
        let p = match self.predict(x) {
            Some((mean, var)) => normal_cdf((best - mean) / var.sqrt().max(1e-12)),
            None => {
                self.relaxed += 1;
                return true;
            }
        };
        let go = p >= self.threshold;
        if go {
            self.relaxed += 1;
        }
        go
    }

    /// Posterior probability that a full relaxation would land below `best`.
    pub fn probability_of_improvement(&self, x: ArrayView1<f64>, best: f64) -> Option<f64> {
        let (mean, var) = self.predict(x)?;
        Some(normal_cdf((best - mean) / var.sqrt().max(1e-12)))
    }
}

/// Solves `a z = b` for small symmetric positive-definite `a`, by Cholesky.
fn solve(a: ndarray::ArrayView2<f64>, b: ArrayView1<f64>) -> Option<Array1<f64>> {
    let n = b.len();
    if a.nrows() != n || a.ncols() != n {
        return None;
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    // Forward then back substitution.
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[[i, k]] * y[k];
        }
        y[i] = s / l[[i, i]];
    }
    let mut z = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[[k, i]] * z[k];
        }
        z[i] = s / l[[i, i]];
    }
    Some(z)
}

/// `x' A^-1 x` for symmetric positive-definite `A`.
fn quad_form(a: ndarray::ArrayView2<f64>, x: ArrayView1<f64>) -> Option<f64> {
    let z = solve(a, x)?;
    Some(x.iter().zip(z.iter()).map(|(p, q)| p * q).sum())
}

/// Standard normal CDF, by the error function's rational approximation.
fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Abramowitz and Stegun 7.1.26, absolute error below 1.5e-7.
fn erf(x: f64) -> f64 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    s * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Features of a trial: an intercept and two cheap measurements.
    fn feat(a: f64, b: f64) -> Array1<f64> {
        array![1.0, a, b]
    }

    #[test]
    fn it_recovers_a_known_linear_relationship() {
        let mut s = Screen::new(3, 10, 0.0, 0.5);
        // y = -2 + 3 a - 1.5 b, exactly.
        for i in 0..200 {
            let a = ((i * 37) % 100) as f64 / 50.0 - 1.0;
            let b = ((i * 53) % 100) as f64 / 50.0 - 1.0;
            s.observe(feat(a, b).view(), -2.0 + 3.0 * a - 1.5 * b);
        }
        let c = s.coefficients().unwrap();
        assert!((c[0] + 2.0).abs() < 1e-3, "intercept {}", c[0]);
        assert!((c[1] - 3.0).abs() < 1e-3, "slope a {}", c[1]);
        assert!((c[2] + 1.5).abs() < 1e-3, "slope b {}", c[2]);
    }

    /// The point of carrying a posterior rather than a fit: a trial unlike
    /// anything observed has to come out uncertain, not confidently predicted.
    #[test]
    fn a_trial_unlike_the_data_is_predicted_with_more_variance() {
        let mut s = Screen::new(3, 10, 0.0, 0.5);
        for i in 0..100 {
            let a = ((i % 10) as f64) / 100.0;
            s.observe(feat(a, 0.0).view(), a);
        }
        let (_, near) = s.predict(feat(0.05, 0.0).view()).unwrap();
        let (_, far) = s.predict(feat(50.0, 30.0).view()).unwrap();
        assert!(
            far > near * 10.0,
            "variance {far} away from the data against {near} inside it"
        );
    }

    #[test]
    fn variance_falls_as_observations_arrive() {
        let mut s = Screen::new(3, 5, 0.0, 0.5);
        for i in 0..8 {
            let a = (i as f64) / 8.0;
            s.observe(feat(a, 0.0).view(), 2.0 * a + 0.1);
        }
        let (_, early) = s.predict(feat(0.5, 0.0).view()).unwrap();
        for i in 0..500 {
            let a = ((i % 8) as f64) / 8.0;
            s.observe(feat(a, 0.0).view(), 2.0 * a + 0.1);
        }
        let (_, late) = s.predict(feat(0.5, 0.0).view()).unwrap();
        assert!(late < early, "variance rose from {early} to {late}");
    }

    /// Everything is relaxed until the model has seen enough to be worth
    /// consulting, which is what makes the Gaussian tail approximation safe.
    #[test]
    fn nothing_is_refused_before_warmup() {
        let mut s = Screen::new(3, 50, 0.0, 0.99);
        for i in 0..49 {
            let a = i as f64 / 49.0;
            assert!(s.decide(feat(a, 0.0).view(), -1000.0, 0.9));
            s.observe(feat(a, 0.0).view(), 500.0);
        }
        assert_eq!(s.relaxed, 49);
    }

    /// The decision has to follow the posterior in both directions: a trial the
    /// model expects to land well below the incumbent is relaxed, one it
    /// expects to land well above is not.
    #[test]
    fn the_decision_follows_the_posterior() {
        let mut s = Screen::new(3, 20, 0.0, 0.5);
        // Screened energy predicts final energy one for one.
        for i in 0..400 {
            let e = -400.0 + ((i * 17) % 100) as f64 / 10.0;
            s.observe(feat(e, 0.0).view(), e - 1.0);
        }
        // Incumbent at -396. A trial screening at -399 relaxes to about -400.
        assert!(
            s.decide(feat(-399.0, 0.0).view(), -396.0, 0.9),
            "a trial heading well below the incumbent was refused"
        );
        // One screening at -390 relaxes to about -391, above the incumbent.
        assert!(
            !s.decide(feat(-390.0, 0.0).view(), -396.0, 0.9),
            "a trial heading well above the incumbent was relaxed"
        );
    }

    /// A rule that relaxes only where it predicts improvement never observes a
    /// counterexample, so the floor is what keeps the training set honest.
    #[test]
    fn the_exploration_floor_relaxes_regardless_of_the_posterior() {
        let mut s = Screen::new(3, 1, 0.2, 0.999);
        for i in 0..400 {
            let e = -300.0 + (i % 50) as f64;
            s.observe(feat(e, 0.0).view(), e + 100.0);
        }
        let mut forced = 0;
        for i in 0..1000 {
            // Deterministic sweep of the unit interval in place of a sampler.
            let u = (i as f64) / 1000.0;
            // A trial the model is sure is hopeless.
            if s.decide(feat(0.0, 0.0).view(), -1e6, u) {
                forced += 1;
            }
        }
        assert_eq!(forced, s.explored);
        assert!(
            (180..=220).contains(&forced),
            "{forced} forced relaxations, not about a fifth of 1000"
        );
    }

    #[test]
    fn non_finite_observations_are_ignored() {
        let mut s = Screen::new(3, 1, 0.0, 0.5);
        s.observe(feat(f64::NAN, 0.0).view(), 1.0);
        s.observe(feat(1.0, 0.0).view(), f64::INFINITY);
        assert_eq!(s.observations(), 0);
    }
}
