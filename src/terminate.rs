//! Stopping a relaxation when its answer is no longer in doubt.
//!
//! A local relaxation runs to an iteration cap, and most of those iterations
//! are spent near convergence buying very little energy. For the great majority
//! of trials the question being asked is not "what is the exact minimum" but
//! "does this beat the incumbent", and that question is settled long before the
//! relaxation is.
//!
//! Treating the unconverged limit as an unknown with a distribution, rather
//! than as something you must compute exactly, is what probabilistic numerics
//! asks of an iterative solver. The limit of a quasi-Newton descent is
//! estimated from the decrements it has produced so far, and the relaxation
//! stops as soon as the estimate says the answer to the caller's question is
//! decided.
//!
//! # The model
//!
//! Near a minimum a quasi-Newton method converges linearly on the energy: the
//! decrements `d_k = e_{k-1} - e_k` fall by roughly a constant factor. If they
//! do, the energy still to be gained is a geometric tail,
//!
//! ```text
//! e_inf = e_k - d_k r / (1 - r),   r = the ratio the decrements are falling by
//! ```
//!
//! and the spread of the observed ratios says how much to trust it. The ratio
//! is estimated in log space, where a multiplicative process is additive and a
//! Gaussian on it is not obviously wrong.
//!
//! # Measured, and it does not pay here
//!
//! Wired to the screening pass at 75 points and three million evaluations it
//! scored 0 seeds in 8, against 13 in 24 for the same driver without it. It
//! fired on 107520 of 108893 hops and saved 857049 relaxation iterations, so
//! the saving was real and the search was worse for it.
//!
//! Stopping nearly every screen means the chain acts on limits extrapolated
//! from a handful of decrements, and the screen's job is to decide which
//! trials deserve a full relaxation. Buying hops by degrading that decision is
//! a bad trade at this ratio: hops cost about thirty evaluations and a missed
//! crossing costs the run.
//!
//! Kept, tested, and off. It is a correct estimator of a limit; the fault is in
//! spending its output on a decision this sensitive. A caller with a cheaper
//! objective or a coarser question may find it pays.
//!
//! # What it will not do
//!
//! It never stops before [`Terminator::min_iters`], because two decrements
//! estimate a ratio and no decrements estimate nothing, and it never reports a
//! limit when the decrements are not falling (`r >= 1`), because then the tail
//! is not geometric and the extrapolation would be an invention rather than an
//! estimate.

/// Watches a decreasing sequence and says when its limit is decided.
#[derive(Debug, Clone)]
pub struct Terminator {
    /// Iterations before any early stop is considered.
    pub min_iters: usize,
    /// Standard deviations of headroom required to call the question settled.
    ///
    /// The estimate is an extrapolation, so stopping on the mean alone stops
    /// early and wrong. Two standard deviations of margin is the difference
    /// between saving iterations and losing minima.
    pub confidence: f64,
    last: Option<f64>,
    last_decrement: Option<f64>,
    /// Running mean and sum of squares of the log decrement ratio.
    log_mean: f64,
    log_m2: f64,
    ratios: usize,
    /// Values seen.
    pub steps: usize,
}

impl Default for Terminator {
    fn default() -> Self {
        Self::new(4, 2.0)
    }
}

impl Terminator {
    /// A terminator that waits `min_iters` and requires `confidence` sigma.
    pub fn new(min_iters: usize, confidence: f64) -> Self {
        assert!(min_iters >= 3, "a ratio needs at least three values");
        assert!(confidence >= 0.0, "confidence is a number of sigma");
        Self {
            min_iters,
            confidence,
            last: None,
            last_decrement: None,
            log_mean: 0.0,
            log_m2: 0.0,
            ratios: 0,
            steps: 0,
        }
    }

    /// Forgets the sequence, for reuse on the next relaxation.
    pub fn reset(&mut self) {
        self.last = None;
        self.last_decrement = None;
        self.log_mean = 0.0;
        self.log_m2 = 0.0;
        self.ratios = 0;
        self.steps = 0;
    }

    /// Records the energy after another iteration.
    pub fn observe(&mut self, e: f64) {
        if !e.is_finite() {
            return;
        }
        self.steps += 1;
        if let Some(prev) = self.last {
            let d = prev - e;
            if d > 0.0 {
                if let Some(pd) = self.last_decrement {
                    if pd > 0.0 {
                        // Welford on the log ratio: a multiplicative process is
                        // additive there, and the variance means something.
                        let lr = (d / pd).ln();
                        if lr.is_finite() {
                            self.ratios += 1;
                            let delta = lr - self.log_mean;
                            self.log_mean += delta / self.ratios as f64;
                            self.log_m2 += delta * (lr - self.log_mean);
                        }
                    }
                }
                self.last_decrement = Some(d);
            }
        }
        self.last = Some(e);
    }

    /// Estimated limit and its standard deviation, if the tail is geometric.
    ///
    /// `None` when there is too little to go on, or when the decrements are not
    /// falling, in which case the sequence is not in its linear regime and
    /// extrapolating it would be invention.
    pub fn limit(&self) -> Option<(f64, f64)> {
        if self.ratios < 2 {
            return None;
        }
        let e = self.last?;
        let d = self.last_decrement?;
        let var = self.log_m2 / (self.ratios as f64 - 1.0);
        let sd = var.max(0.0).sqrt();
        let r = self.log_mean.exp();
        if !(r < 1.0) || !r.is_finite() {
            return None;
        }
        let tail = d * r / (1.0 - r);
        // Propagate the ratio's spread through the tail. d/dr of r/(1-r) is
        // 1/(1-r)^2, and the log-normal spread of r is r * sd.
        let dtail = d * (r * sd) / ((1.0 - r) * (1.0 - r));
        Some((e - tail, dtail.abs()))
    }

    /// Whether the sequence can be stopped because it will not reach `target`.
    ///
    /// True only when the estimated limit sits `confidence` standard deviations
    /// *above* the target: the caller wanted to know whether this relaxation
    /// beats an incumbent, and it will not.
    pub fn settled_above(&self, target: f64) -> bool {
        if self.steps < self.min_iters {
            return false;
        }
        match self.limit() {
            Some((mean, sd)) => mean - self.confidence * sd > target,
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A geometric tail, which is what a quasi-Newton method produces near a
    /// minimum, and the limit is known exactly.
    fn geometric(limit: f64, first: f64, r: f64, n: usize) -> Vec<f64> {
        let mut v = Vec::with_capacity(n);
        let mut gap = first;
        for _ in 0..n {
            v.push(limit + gap);
            gap *= r;
        }
        v
    }

    #[test]
    fn it_estimates_the_limit_of_a_geometric_tail() {
        let mut t = Terminator::new(3, 2.0);
        for e in geometric(-173.928, 4.0, 0.5, 12) {
            t.observe(e);
        }
        let (mean, sd) = t.limit().unwrap();
        assert!(
            (mean + 173.928).abs() < 0.01,
            "limit {mean} against -173.928, sd {sd}"
        );
    }

    /// The property that makes it safe: a sequence still far from its limit and
    /// heading below the target must not be stopped.
    #[test]
    fn a_sequence_still_heading_below_the_target_is_not_stopped() {
        let mut t = Terminator::new(3, 2.0);
        // Limit -400, target -396: this one wins and must run.
        for e in geometric(-400.0, 20.0, 0.6, 8) {
            t.observe(e);
            assert!(
                !t.settled_above(-396.0),
                "stopped a relaxation that reaches -400 against a target of -396"
            );
        }
    }

    #[test]
    fn a_sequence_that_will_not_reach_the_target_is_stopped() {
        let mut t = Terminator::new(3, 2.0);
        let mut stopped = None;
        for (i, e) in geometric(-390.0, 5.0, 0.5, 20).into_iter().enumerate() {
            t.observe(e);
            if stopped.is_none() && t.settled_above(-396.0) {
                stopped = Some(i);
            }
        }
        let at = stopped.expect("never stopped a relaxation that cannot reach the target");
        assert!(at < 12, "stopped only at iteration {at} of 20");
    }

    #[test]
    fn nothing_is_stopped_before_the_minimum_iterations() {
        let mut t = Terminator::new(8, 0.0);
        for (i, e) in geometric(-100.0, 1.0, 0.1, 7).into_iter().enumerate() {
            t.observe(e);
            assert!(!t.settled_above(-1e9), "stopped at iteration {i}");
        }
    }

    /// Decrements that are not falling mean the sequence is not in its linear
    /// regime, and extrapolating it would be invention rather than estimation.
    #[test]
    fn a_sequence_that_is_not_converging_yields_no_limit() {
        let mut t = Terminator::new(3, 2.0);
        // Decrements growing: 1, 2, 4, 8.
        let mut e = 0.0;
        let mut d = 1.0;
        for _ in 0..6 {
            t.observe(e);
            e -= d;
            d *= 2.0;
        }
        assert!(t.limit().is_none(), "extrapolated a diverging tail");
        assert!(!t.settled_above(1e9));
    }

    #[test]
    fn a_noisier_tail_gives_a_wider_estimate() {
        let mut clean = Terminator::new(3, 2.0);
        for e in geometric(-50.0, 2.0, 0.5, 10) {
            clean.observe(e);
        }
        let mut noisy = Terminator::new(3, 2.0);
        let mut gap = 2.0;
        let mut e = -50.0 + gap;
        for i in 0..10 {
            noisy.observe(e);
            // Ratio alternating either side of 0.5.
            let r = if i % 2 == 0 { 0.3 } else { 0.75 };
            gap *= r;
            e = -50.0 + gap;
        }
        let (_, sd_clean) = clean.limit().unwrap();
        let (_, sd_noisy) = noisy.limit().unwrap();
        assert!(
            sd_noisy > sd_clean,
            "noisy {sd_noisy} should exceed clean {sd_clean}"
        );
    }

    #[test]
    fn reset_forgets_the_previous_relaxation() {
        let mut t = Terminator::new(3, 2.0);
        for e in geometric(-10.0, 1.0, 0.5, 8) {
            t.observe(e);
        }
        assert!(t.limit().is_some());
        t.reset();
        assert!(t.limit().is_none());
        assert_eq!(t.steps, 0);
    }
}
