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
//! # The predictive tail
//!
//! The conjugate Normal-Inverse-Gamma posterior gives a Student-t predictive
//! with `ν = 2a` degrees of freedom and scale `√(s² (1 + x'Vx))`. The
//! probability of landing below the incumbent is that distribution's cdf.
//! A normal cdf of the same scale understates the tails while `ν` is small.
//! Nothing is decided by the model until [`Screen::warmup`] observations have
//! arrived; before then every trial is relaxed.

use ndarray::{Array1, Array2, ArrayView1};

/// Bayes threshold from the hop's own step counts.
///
/// Finishing the quench costs `Q = R - S` extra evaluations. A discarded
/// winner costs one full hop `R` of opportunity. Net value of quenching is
/// `p R - (1-p) Q`, which is positive exactly when
/// `p > Q / (Q + R) = (R - S) / (2 R - S)`.
///
/// For the measured hop (`S = 25`, `R = 200`) this is `7/15`, not a knob.
pub fn cost_asymmetric_threshold(screen_steps: usize, relax_steps: usize) -> f64 {
    assert!(
        screen_steps < relax_steps && relax_steps > 0,
        "screen_steps must be strictly less than relax_steps (got {screen_steps}, {relax_steps})"
    );
    let s = screen_steps as f64;
    let r = relax_steps as f64;
    (r - s) / (2.0 * r - s)
}

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

    /// Predictive mean and Student-t scale squared at `x`.
    ///
    /// The scale is `s² (1 + x'Vx)`. It carries both the noise and the
    /// uncertainty in the coefficients; the second term is what makes a
    /// trial unlike anything seen before come out uncertain rather than
    /// confidently predicted. The degrees of freedom are `2a`, and the
    /// tail in [`Screen::probability_of_improvement`] is the Student-t cdf
    /// of that scale, not a normal cdf.
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
        let p = match self.probability_of_improvement(x, best) {
            Some(probability) => probability,
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
    ///
    /// Student-t with `ν = 2a` and the scale from [`Screen::predict`].
    pub fn probability_of_improvement(&self, x: ArrayView1<f64>, best: f64) -> Option<f64> {
        let (mean, scale2) = self.predict(x)?;
        let z = (best - mean) / scale2.sqrt().max(1e-12);
        Some(student_t_cdf(z, (2.0 * self.a).max(1e-12)))
    }
}

/// Remaining drop `D = E_sc - E_full` of a purchased quench.
///
/// Features are an intercept and the screened energy. The predicted landing
/// is `E_sc - E[D | E_sc]`. A trial whose landing sits on a known floor is
/// refused by the caller; this type only predicts the drop.
#[derive(Debug, Clone)]
pub struct DropModel {
    inner: Screen,
}

impl DropModel {
    /// Weak prior, no exploration floor: the caller decides whether to buy.
    pub fn new() -> Self {
        Self {
            inner: Screen::new(2, 2, 0.0, 0.5),
        }
    }

    /// Fold in one purchased pair.
    pub fn observe(&mut self, e_sc: f64, e_full: f64) {
        if !e_sc.is_finite() || !e_full.is_finite() {
            return;
        }
        let d = e_sc - e_full;
        self.inner.observe(Array1::from(vec![1.0, e_sc]).view(), d);
    }

    /// Predictive remaining drop at `e_sc`.
    pub fn predict_drop(&self, e_sc: f64) -> Option<(f64, f64)> {
        self.inner.predict(Array1::from(vec![1.0, e_sc]).view())
    }

    /// Predicted full-quench energy.
    pub fn predicted_full(&self, e_sc: f64) -> Option<f64> {
        let (d, _) = self.predict_drop(e_sc)?;
        Some(e_sc - d)
    }

    /// Observations folded in.
    pub fn observations(&self) -> usize {
        self.inner.observations()
    }

    /// Purchased pairs required before a predicted landing may refuse a quench.
    ///
    /// This is the screen warmup the model was built with, not a search-level
    /// knob: refusing before it has seen that many pairs starves the driver.
    pub fn warmup(&self) -> usize {
        self.inner.warmup
    }

    /// Whether enough purchased pairs have arrived to trust a refuse.
    pub fn calibrated(&self) -> bool {
        self.observations() >= self.warmup()
    }
}

impl Default for DropModel {
    fn default() -> Self {
        Self::new()
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

/// Student-t cdf at `t` with `nu` degrees of freedom.
///
/// `P(T ≤ t) = 1 - ½ I_x(ν/2, ½)` for `t ≥ 0`, and the complement for
/// `t < 0`, with `x = ν / (ν + t²)`. `I_x` is the regularized incomplete
/// beta. At `ν = 1` this is the Cauchy cdf `½ + arctan(t)/π`.
fn student_t_cdf(t: f64, nu: f64) -> f64 {
    if !nu.is_finite() || nu <= 0.0 {
        return f64::NAN;
    }
    if t.is_nan() {
        return f64::NAN;
    }
    if t.is_infinite() {
        return if t.is_sign_positive() { 1.0 } else { 0.0 };
    }
    let x = nu / (nu + t * t);
    let tail = 0.5 * regularized_incomplete_beta(0.5 * nu, 0.5, x);
    if t >= 0.0 { 1.0 - tail } else { tail }
}

/// Regularized incomplete beta `I_x(a, b)`, by Lentz's continued fraction.
fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_prefix = a * x.ln() + b * (1.0 - x).ln() - ln_beta(a, b);
    let prefix = ln_prefix.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        prefix * beta_continued_fraction(a, b, x) / a
    } else {
        1.0 - prefix * beta_continued_fraction(b, a, 1.0 - x) / b
    }
}

/// Continued fraction for the incomplete beta, Numerical Recipes `betacf`.
fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 200;
    const EPS: f64 = 3.0e-14;
    const FPMIN: f64 = 1.0e-30;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let m = m as f64;
        let m2 = 2.0 * m;
        let mut aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// `ln B(a, b)` from the Lanczos approximation of `ln Γ`.
fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Lanczos `ln Γ(z)` for `z > 0`, reflection below `1/2`.
fn ln_gamma(z: f64) -> f64 {
    if z < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * z).sin()).ln() - ln_gamma(1.0 - z);
    }
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let z = z - 1.0;
    let mut acc = COEFFICIENTS[0];
    for (i, coefficient) in COEFFICIENTS.iter().enumerate().skip(1) {
        acc += coefficient / (z + i as f64);
    }
    let t = z + 7.5;
    (2.0 * std::f64::consts::PI).sqrt().ln() + (z + 0.5) * t.ln() - t + acc.ln()
}

/// Standard normal CDF, by the error function's rational approximation.
///
/// Kept for the Student-t reference test. The screen tail does not call it.
#[cfg(test)]
fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Abramowitz and Stegun 7.1.26, absolute error below 1.5e-7.
#[cfg(test)]
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
    fn student_t_matches_the_cauchy_and_a_published_quantile() {
        assert!((student_t_cdf(0.0, 5.0) - 0.5).abs() < 1e-12);
        assert!((student_t_cdf(0.0, 1.0) - 0.5).abs() < 1e-12);
        // Cauchy: 1/2 + arctan(1)/pi = 3/4.
        let cauchy = 0.5 + 1.0_f64.atan() / std::f64::consts::PI;
        assert!((student_t_cdf(1.0, 1.0) - cauchy).abs() < 1e-12);
        assert!((student_t_cdf(-1.0, 1.0) - (1.0 - cauchy)).abs() < 1e-12);
        // t_5 0.95 quantile, to 9 digits.
        assert!((student_t_cdf(2.015_048_373, 5.0) - 0.95).abs() < 1e-8);
        assert!((student_t_cdf(-2.015_048_373, 5.0) - 0.05).abs() < 1e-8);
        let z = 1.644_853_626_951_472_2;
        assert!((student_t_cdf(z, 1.0e6) - normal_cdf(z)).abs() < 1e-5);
    }

    #[test]
    fn improvement_probability_is_the_student_t_of_the_nig_scale() {
        let mut screen = Screen::new(2, 1, 0.0, 0.5);
        for i in 0..12 {
            let a = (i as f64) - 6.0;
            screen.observe(array![1.0, a].view(), a);
        }
        let x = array![1.0, 0.25];
        let (mean, scale2) = screen.predict(x.view()).unwrap();
        let best = mean;
        let z = (best - mean) / scale2.sqrt();
        let reference = student_t_cdf(z, 2.0 * screen.a);
        let got = screen.probability_of_improvement(x.view(), best).unwrap();
        assert!((got - reference).abs() < 1e-12, "{got} against {reference}");
        assert!(
            (got - 0.5).abs() < 1e-12,
            "a mean landing is a half tail, got {got}"
        );
        let z_off = -2.0;
        let off = screen
            .probability_of_improvement(x.view(), mean + z_off * scale2.sqrt())
            .unwrap();
        assert!((off - student_t_cdf(z_off, 2.0 * screen.a)).abs() < 1e-12);
        assert!(
            (off - normal_cdf(z_off)).abs() > 1e-3,
            "the Student-t tail must not collapse to the normal at nu {}",
            2.0 * screen.a
        );
    }

    #[test]
    fn cost_threshold_is_seven_fifteenths_on_the_measured_hop() {
        let t = cost_asymmetric_threshold(25, 200);
        assert!((t - 7.0 / 15.0).abs() < 1e-12, "{t}");
        assert!(t > 0.0 && t < 1.0);
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
    /// consulting. The Student-t tail is not used before that.
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

    #[test]
    fn drop_model_predicts_a_constant_descent() {
        let mut d = DropModel::new();
        for e_sc in [0.0, 1.0, 2.0, 3.0, 4.0] {
            d.observe(e_sc, e_sc - 0.5);
        }
        let hat = d.predicted_full(10.0).unwrap();
        assert!(
            (hat - 9.5).abs() < 0.2,
            "predicted landing {hat}, expected about 9.5"
        );
    }

    #[test]
    fn drop_model_is_uncalibrated_until_its_own_warmup() {
        let mut d = DropModel::new();
        let need = d.warmup();
        assert!(need > 0);
        assert!(!d.calibrated());
        for i in 0..need {
            d.observe(i as f64, i as f64 - 0.5);
            if i + 1 < need {
                assert!(!d.calibrated());
            }
        }
        assert!(d.calibrated());
    }
}
