//! Stopping a screening quench on a decision rather than an iteration count.
//!
//! Measured on 38 points, 89 to 92 per cent of the charged budget goes into the
//! screening pass: 12 383 hops spent 357 000 evaluations screening and 30 000
//! relaxing, because every proposal pays a fixed 25 descent steps whether or
//! not the answer was settled after three. The screen exists to decide one
//! question, whether finishing this relaxation can beat the incumbent, and a
//! fixed length answers it at whatever precision that length happens to buy.
//!
//! The descent supplies the information to stop earlier. A quench into a basin
//! that is locally quadratic decreases its energy by decrements that shrink
//! geometrically, so a short prefix of the trajectory already says roughly
//! where it is going. Fitting that decay and extrapolating gives a predicted
//! limit with an uncertainty attached, and the uncertainty is what makes it a
//! decision: stop when the predicted limit is above the incumbent by more than
//! the prediction error, or below it by more, and keep going only while the
//! two are within reach of each other.
//!
//! # The model
//!
//! Write the decrements as `d_k = E_k - E_{k+1}`. Under a geometric decay
//! `d_k = d_0 r^k` with `0 < r < 1`, the energy remaining below `E_k` is the
//! tail `sum_{j>=k} d_j = d_k r / (1 - r)`, so the limit is
//!
//! ```text
//! E_inf = E_k - d_k * r / (1 - r)
//! ```
//!
//! and the ratio is estimated by least squares on `log d_k` against `k`. The
//! residual scatter of that fit propagates to the tail, which is where the
//! uncertainty comes from. Fitting in log space rather than on the decrements
//! themselves is what keeps a single large early decrement from setting the
//! rate: the first steps of a quench from a perturbed structure are not in the
//! quadratic region at all, and on a linear fit they dominate every later
//! point.
//!
//! # Measured, and off
//!
//! The model does not hold on real quenches. Scoring each extrapolation
//! against the value the full pass reaches, without acting on it, gives a mean
//! absolute error of 12442 at the step where the rule would have stopped, 993
//! at a warmup of eight, 19.1 at twelve and 3.6 at sixteen. Minima near the
//! bottom of the 38 point landscape are separated by well under one unit, so a
//! usable energy costs about twenty of the twenty-five steps.
//!
//! Acting on it is worse than the error alone suggests: with the extrapolated
//! energy driving acceptance, eight seeds solved nothing where the fixed screen
//! solved eight, because a chain handed energies off by four orders of
//! magnitude rejects everything and stops exploring.
//!
//! What this settles is not the rule but the premise behind it. The screening
//! pass looked like overhead because it takes 89 to 93 per cent of the charged
//! budget against 8 per cent for the relaxations it guards. It is not
//! overhead. It is the quench, and the relaxation that follows is the polish on
//! the 2 per cent of trials that survive. There is no factor of two there.
//!
//! # Why this is not a gradient tolerance
//!
//! A tolerance asks whether the point has stopped moving. This asks whether the
//! answer has stopped mattering, which is a different and much weaker question,
//! and weaker questions are cheaper. A proposal 40 units above the incumbent
//! can be abandoned after two steps while its gradient is still enormous, and a
//! proposal within 0.01 has to be finished no matter how flat it looks.

/// What a partial descent says about where it is going.
#[derive(Debug, Clone, Copy)]
pub struct Prediction {
    /// The extrapolated limit of the descent.
    pub limit: f64,
    /// One standard deviation of the extrapolation.
    ///
    /// Infinite before the fit has enough points, which is what stops a caller
    /// from acting on two observations.
    pub sigma: f64,
    /// The fitted decay ratio.
    pub ratio: f64,
}

/// What to do with a screening quench that has run `k` steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The limit is decisively above the incumbent; the trial cannot improve.
    Hopeless,
    /// The limit is decisively below the incumbent; finish the relaxation.
    Promising,
    /// The two are still within reach of each other.
    Undecided,
}

/// Accumulates a descent's energies and reports when the question is settled.
///
/// One instance per screening quench. It holds the trajectory, not the
/// structure, so it costs nothing beyond the energies the descent produced
/// anyway.
#[derive(Debug, Clone)]
pub struct QuenchPredictor {
    energies: Vec<f64>,
    /// Steps taken before any verdict is allowed.
    ///
    /// Three decrements are the fewest that give a rate and a residual, and a
    /// rate with no residual is a point estimate a caller would act on as
    /// though it were certain.
    pub warmup: usize,
    /// How many standard deviations of separation a verdict needs.
    pub confidence: f64,
    /// Extra separation required regardless of the fitted uncertainty.
    ///
    /// The geometric model is an approximation, and near the incumbent an
    /// approximation that is slightly wrong is expensive: abandoning a trial
    /// that would have improved costs the whole reason the search is running.
    /// The margin buys asymmetry against that.
    pub margin: f64,
}

impl Default for QuenchPredictor {
    fn default() -> Self {
        Self {
            energies: Vec::new(),
            warmup: 4,
            confidence: 2.0,
            margin: 1e-3,
        }
    }
}

impl QuenchPredictor {
    /// A predictor with the default warmup and confidence.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records the energy after a descent step.
    pub fn observe(&mut self, e: f64) {
        if e.is_finite() {
            self.energies.push(e);
        }
    }

    /// Steps observed.
    pub fn len(&self) -> usize {
        self.energies.len()
    }

    /// Whether nothing has been observed.
    pub fn is_empty(&self) -> bool {
        self.energies.is_empty()
    }

    /// The current energy, if any.
    pub fn last(&self) -> Option<f64> {
        self.energies.last().copied()
    }

    /// The extrapolated limit of the descent so far.
    ///
    /// `None` before there are enough decrements to fit, or when the descent is
    /// not decreasing, which happens when a line search has stalled and means
    /// the geometric model does not apply.
    pub fn predict(&self) -> Option<Prediction> {
        let n = self.energies.len();
        if n < self.warmup {
            return None;
        }
        // Decrements, in log space. A non-positive decrement is not evidence
        // about the rate of a decreasing sequence, so it is dropped rather than
        // clamped: clamping invents a decrement that was never observed.
        let mut ks = Vec::new();
        let mut ls = Vec::new();
        for k in 0..(n - 1) {
            let d = self.energies[k] - self.energies[k + 1];
            if d > 0.0 {
                ks.push(k as f64);
                ls.push(d.ln());
            }
        }
        if ks.len() < 3 {
            return None;
        }
        let m = ks.len() as f64;
        let kbar = ks.iter().sum::<f64>() / m;
        let lbar = ls.iter().sum::<f64>() / m;
        let mut sxx = 0.0;
        let mut sxy = 0.0;
        for (k, l) in ks.iter().zip(ls.iter()) {
            sxx += (k - kbar) * (k - kbar);
            sxy += (k - kbar) * (l - lbar);
        }
        if sxx <= 0.0 {
            return None;
        }
        let slope = sxy / sxx;
        // A slope at or above zero says the decrements are not shrinking, so
        // the tail does not converge and there is nothing to extrapolate to.
        if !(slope < 0.0) {
            return None;
        }
        let ratio = slope.exp();
        if !(ratio < 1.0) {
            return None;
        }
        let intercept = lbar - slope * kbar;

        // Residual scatter of the log-linear fit, which is the only measure of
        // how well the geometric model describes this particular quench.
        let mut ss = 0.0;
        for (k, l) in ks.iter().zip(ls.iter()) {
            let r = l - (intercept + slope * k);
            ss += r * r;
        }
        let dof = (m - 2.0).max(1.0);
        let s_log = (ss / dof).sqrt();

        // The tail below the current energy, from the fitted decrement at the
        // last observed step rather than the observed one, so a single noisy
        // step does not set the whole extrapolation.
        let last_k = (n - 2) as f64;
        let d_fit = (intercept + slope * last_k).exp();
        let tail = d_fit * ratio / (1.0 - ratio);
        let current = self.energies[n - 1];
        let limit = current - tail;

        // A multiplicative error on the fitted decrement carries through the
        // tail unchanged, since the tail is proportional to it. The rate's own
        // uncertainty enters through `1 / (1 - r)`, which is the term that
        // blows up as the descent approaches a flat direction, and that is
        // correct behaviour: near `r = 1` the limit genuinely is not
        // determined by a short prefix.
        let se_slope = s_log / sxx.sqrt();
        let d_rel = s_log;
        let r_rel = (se_slope * ratio / (1.0 - ratio)).abs();
        let sigma = tail.abs() * (d_rel * d_rel + r_rel * r_rel).sqrt();

        Some(Prediction {
            limit,
            sigma,
            ratio,
        })
    }

    /// The energy to hand a caller that stopped this descent early.
    ///
    /// The extrapolated limit, floored so that it cannot sit at or below
    /// `best`. The floor is the invariant, not a correction: a structure whose
    /// descent was cut short is not a minimum, and an energy that beats the
    /// incumbent is recorded as the run's answer. Arguing that the verdict
    /// already guarantees it is not enough. It was argued, and a run came back
    /// reporting a structure with a gradient of 7.1e2 where a relaxed one comes
    /// back at 1e-6, because a trial can leave the screen by a second route:
    /// the return screen takes the screened energy and structure directly,
    /// without the full relaxation that the promising verdict assumed would
    /// follow.
    ///
    /// `fallback` is the value at the point where the descent stopped, used
    /// when there is no usable extrapolation.
    pub fn stopped_energy(&self, best: f64, fallback: f64) -> f64 {
        let raw = match self.predict() {
            Some(p) if p.limit.is_finite() && p.limit < fallback => p.limit,
            _ => fallback,
        };
        let floor = best + self.margin;
        if raw > floor { raw } else { floor }
    }

    /// Whether the descent can still reach `best`.
    ///
    /// Asymmetric on purpose. Calling a trial hopeless discards it, and a
    /// discarded improvement is unrecoverable, so that direction pays the
    /// margin as well as the confidence interval. Calling one promising only
    /// spends the evaluations the fixed-length screen would have spent anyway.
    pub fn verdict(&self, best: f64) -> Verdict {
        let Some(p) = self.predict() else {
            return Verdict::Undecided;
        };
        if !p.sigma.is_finite() {
            return Verdict::Undecided;
        }
        if p.limit - self.confidence * p.sigma > best + self.margin {
            return Verdict::Hopeless;
        }
        if p.limit + self.confidence * p.sigma < best - self.margin {
            return Verdict::Promising;
        }
        Verdict::Undecided
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A textbook geometric descent: the extrapolation has to find the limit it
    /// was built from, which is the whole claim of the model.
    #[test]
    fn it_extrapolates_a_geometric_descent() {
        let limit = -100.0;
        let mut q = QuenchPredictor::new();
        let mut gap = 10.0;
        for _ in 0..8 {
            q.observe(limit + gap);
            gap *= 0.5;
        }
        let p = q.predict().expect("no prediction");
        assert!(
            (p.limit - limit).abs() < 0.05,
            "predicted {} against {limit}",
            p.limit
        );
        assert!((p.ratio - 0.5).abs() < 0.02, "ratio {}", p.ratio);
    }

    /// The verdict that saves the budget: a descent heading well above the
    /// incumbent is abandoned rather than finished.
    #[test]
    fn a_descent_heading_high_is_called_hopeless() {
        let mut q = QuenchPredictor::new();
        let mut gap = 5.0;
        for _ in 0..6 {
            q.observe(-40.0 + gap);
            gap *= 0.6;
        }
        assert_eq!(q.verdict(-100.0), Verdict::Hopeless);
    }

    /// And the verdict that spends it: a descent heading below the incumbent
    /// goes straight on to the full relaxation.
    #[test]
    fn a_descent_heading_low_is_called_promising() {
        let mut q = QuenchPredictor::new();
        let mut gap = 5.0;
        for _ in 0..6 {
            q.observe(-140.0 + gap);
            gap *= 0.6;
        }
        assert_eq!(q.verdict(-100.0), Verdict::Promising);
    }

    /// A descent whose limit sits on the incumbent must not be decided: this is
    /// the case where the screen has to pay, and a rule that resolves it is
    /// discarding the improvements the search exists to find.
    #[test]
    fn a_descent_landing_on_the_incumbent_stays_undecided() {
        let mut q = QuenchPredictor::new();
        let mut gap = 2.0;
        for _ in 0..5 {
            q.observe(-100.0 + gap);
            gap *= 0.7;
        }
        assert_eq!(q.verdict(-100.0), Verdict::Undecided);
    }

    /// Two observations are not a fit. Acting on them would be acting on a
    /// point estimate with no error, which is the failure this module is
    /// supposed to avoid.
    #[test]
    fn it_refuses_to_decide_before_the_warmup() {
        let mut q = QuenchPredictor::new();
        q.observe(-10.0);
        q.observe(-20.0);
        assert!(q.predict().is_none());
        assert_eq!(q.verdict(-100.0), Verdict::Undecided);
    }

    /// A stalled line search produces a flat or rising sequence. The geometric
    /// model does not describe it and the predictor has to say so rather than
    /// report an extrapolation of nothing.
    #[test]
    fn a_stalled_descent_yields_no_prediction() {
        let mut q = QuenchPredictor::new();
        for _ in 0..8 {
            q.observe(-50.0);
        }
        assert!(q.predict().is_none());
        assert_eq!(q.verdict(-100.0), Verdict::Undecided);
    }

    /// Decrements that shrink slowly leave the limit genuinely undetermined,
    /// and the reported uncertainty has to grow to say so. A rule with a fixed
    /// error would call these decided and be wrong in exactly the region where
    /// being wrong costs an improvement.
    #[test]
    fn a_slower_decay_carries_a_larger_uncertainty() {
        let build = |r: f64| {
            let mut q = QuenchPredictor::new();
            let mut gap = 10.0;
            for _ in 0..7 {
                q.observe(-100.0 + gap);
                gap *= r;
            }
            // A little scatter, or the fit is exact and every sigma is zero.
            q.energies[3] += 0.01;
            q.predict().expect("no prediction").sigma
        };
        let fast = build(0.3);
        let slow = build(0.9);
        assert!(
            slow > fast,
            "slow decay reported {slow}, fast reported {fast}"
        );
    }

    /// The invariant, stated as a test rather than as an argument: whatever a
    /// stopped descent extrapolates to, the caller is never handed a value
    /// that would be recorded as an improvement over a real minimum.
    #[test]
    fn a_stopped_descent_never_reports_below_the_incumbent() {
        let mut q = QuenchPredictor::new();
        // A descent plunging well past the incumbent, which is exactly the
        // case that produced a reported structure that was not a minimum.
        let mut gap = 5.0;
        for _ in 0..7 {
            q.observe(-500.0 + gap);
            gap *= 0.5;
        }
        let p = q.predict().expect("no prediction");
        assert!(
            p.limit < -400.0,
            "limit {} is not below the incumbent",
            p.limit
        );
        let e = q.stopped_energy(-100.0, -495.0);
        assert!(e > -100.0, "stopped energy {e} beats the incumbent -100");
    }

    /// And it still passes the extrapolation through when that is above the
    /// incumbent, or the floor would flatten every screened energy onto one
    /// value and the chain would stop distinguishing proposals.
    #[test]
    fn a_stopped_descent_above_the_incumbent_keeps_its_estimate() {
        let mut q = QuenchPredictor::new();
        let mut gap = 5.0;
        for _ in 0..7 {
            q.observe(-40.0 + gap);
            gap *= 0.5;
        }
        let e = q.stopped_energy(-100.0, -39.0);
        assert!(
            (e - q.predict().unwrap().limit).abs() < 1e-12,
            "estimate {e} was replaced"
        );
    }

    /// The asymmetry, stated as a test: a prediction sitting just above the
    /// incumbent inside the margin is not called hopeless.
    #[test]
    fn the_margin_protects_a_marginal_trial() {
        let mut q = QuenchPredictor::new();
        q.margin = 0.5;
        let mut gap = 1.0;
        for _ in 0..7 {
            q.observe(-99.9 + gap);
            gap *= 0.4;
        }
        assert_ne!(q.verdict(-100.0), Verdict::Hopeless);
    }
}
