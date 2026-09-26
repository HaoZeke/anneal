//! Setting the "same basin" threshold from the search's own steps.
//!
//! Every basin-keyed mechanism in this crate compares a distance against a
//! threshold, and until now that threshold was a number. Numbers do not
//! transfer. A radius calibrated at 38 points is wrong at 75, one calibrated in
//! a sorted-distance spectrum is wrong in a shape metric, and a campaign that
//! sweeps it has tuned a constant rather than found a method.
//!
//! The structure-prediction literature settled this by deriving the threshold
//! from the descriptor's own statistics rather than choosing it. Oganov and
//! Valle's fingerprint distance is used with a cutoff read off the distribution
//! of distances the search actually produces, which is what makes fingerprint
//! niching work across systems in USPEX rather than per system.
//!
//! The definition here is the one the geometry supplies. Two structures are the
//! same basin when one accepted hop can carry the chain between them, so the
//! threshold is a high quantile of the distance a single accepted hop covers.
//! Nothing about that is specific to a size or a potential: the search reports
//! its own step length and the threshold follows.
//!
//! Measured at 75 points, one accepted hop covers 0.4766 in shape distance and
//! independent minima sit at 0.9212, so the two distributions are separated and
//! a quantile of the first lands between them. That separation is the reason
//! the definition works, and a system where it fails is one where the
//! descriptor cannot tell the two apart, which the caller wants to know.

/// A threshold tracking a quantile of the steps it is shown.
///
/// Robbins-Monro on the quantile: each sample moves the estimate up by `step *
/// q` when it lands above and down by `step * (1 - q)` when it lands below, so
/// the fixed point is the level with `q` of the mass beneath it. The step
/// decays as `1 / sqrt(n)`, which converges without needing a schedule.
#[derive(Debug, Clone)]
pub struct StepCalibrator {
    /// Quantile of the step-length distribution the threshold aims at.
    pub quantile: f64,
    /// Steps required before the estimate replaces the prior.
    pub warmup: u64,
    estimate: f64,
    prior: f64,
    scale: f64,
    samples: u64,
}

impl StepCalibrator {
    /// Tracker aiming at `quantile`, holding `prior` until `warmup` steps.
    ///
    /// The prior matters. Until the search has taken enough accepted hops to
    /// say anything about its own step length, a threshold read off three
    /// samples is worse than the value the caller came with.
    pub fn new(quantile: f64, warmup: u64, prior: f64) -> Self {
        assert!(
            (0.0..1.0).contains(&quantile),
            "quantile must lie in [0, 1), got {quantile}"
        );
        assert!(
            prior > 0.0 && prior.is_finite(),
            "the prior is a distance and must be positive, got {prior}"
        );
        Self {
            quantile,
            warmup,
            estimate: prior,
            prior,
            scale: prior,
            samples: 0,
        }
    }

    /// Records the distance one accepted hop covered.
    ///
    /// Non-finite and non-positive steps are ignored rather than folded in: a
    /// hop that did not move, or a shape match that failed and returned
    /// infinity, says nothing about how far a hop reaches.
    pub fn observe(&mut self, step: f64) {
        if !step.is_finite() || step <= 0.0 {
            return;
        }
        self.samples += 1;
        // A running mean, used only to scale the update so the tracker does not
        // depend on the units of the caller's metric.
        self.scale += (step - self.scale) / self.samples as f64;
        let rate = self.scale / (self.samples as f64).sqrt().max(1.0);
        if step > self.estimate {
            self.estimate += rate * self.quantile;
        } else {
            self.estimate -= rate * (1.0 - self.quantile);
        }
        // A threshold cannot go to zero: there it recognises nothing as the
        // same basin, every hop opens a new one, and the bias never
        // accumulates. That failure is on record, at 4423 basins and 2.6
        // revisits each where the working radius gave 250 basins at 25.
        if self.estimate < 1e-9 {
            self.estimate = 1e-9;
        }
    }

    /// Current threshold: the prior until warmed up, the estimate after.
    pub fn threshold(&self) -> f64 {
        if self.samples < self.warmup {
            self.prior
        } else {
            self.estimate
        }
    }

    /// Whether enough steps have been seen for the estimate to be used.
    pub fn warm(&self) -> bool {
        self.samples >= self.warmup
    }

    /// Steps recorded.
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Mean step length seen, which is what the threshold is a quantile of.
    pub fn mean_step(&self) -> f64 {
        self.scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Steps drawn from a fixed distribution, deterministically.
    fn steps(n: usize, lo: f64, hi: f64) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let u = ((i * 7919 + 104_729) % 1000) as f64 / 1000.0;
                lo + u * (hi - lo)
            })
            .collect()
    }

    #[test]
    fn it_finds_the_quantile_of_the_steps_it_is_shown() {
        // Uniform on [0.2, 1.2]; the 0.9 quantile is 1.1.
        let mut c = StepCalibrator::new(0.9, 50, 0.5);
        for s in steps(20_000, 0.2, 1.2) {
            c.observe(s);
        }
        let t = c.threshold();
        assert!(
            (t - 1.1).abs() < 0.06,
            "threshold {t} should be near the 0.9 quantile of 1.1"
        );
    }

    /// The property that makes this transfer: the same code on a distribution
    /// scaled by ten returns a threshold scaled by ten, with no constant
    /// anywhere that knows about the units.
    #[test]
    fn the_threshold_scales_with_the_steps() {
        let mut a = StepCalibrator::new(0.9, 50, 0.5);
        for s in steps(20_000, 0.2, 1.2) {
            a.observe(s);
        }
        let mut b = StepCalibrator::new(0.9, 50, 5.0);
        for s in steps(20_000, 2.0, 12.0) {
            b.observe(s);
        }
        let ratio = b.threshold() / a.threshold();
        assert!(
            (ratio - 10.0).abs() < 0.7,
            "a tenfold larger metric gave a ratio of {ratio}"
        );
    }

    #[test]
    fn it_holds_the_prior_until_it_has_seen_enough() {
        let mut c = StepCalibrator::new(0.9, 100, 0.7);
        for s in steps(99, 5.0, 6.0) {
            c.observe(s);
        }
        assert!(!c.warm());
        assert_eq!(c.threshold(), 0.7, "the prior was abandoned early");
        c.observe(5.5);
        assert!(c.warm());
        assert_ne!(c.threshold(), 0.7);
    }

    #[test]
    fn hops_that_did_not_move_are_not_evidence_about_how_far_a_hop_reaches() {
        let mut c = StepCalibrator::new(0.9, 1, 0.5);
        c.observe(f64::INFINITY);
        c.observe(f64::NAN);
        c.observe(0.0);
        c.observe(-1.0);
        assert_eq!(c.samples(), 0);
        assert_eq!(c.threshold(), 0.5);
    }

    /// The failure this exists to prevent, kept as a test rather than a
    /// comment: a threshold driven to zero recognises nothing as the same
    /// basin and the bias never accumulates.
    #[test]
    fn the_threshold_cannot_be_driven_to_zero() {
        let mut c = StepCalibrator::new(0.5, 1, 1.0);
        for _ in 0..100_000 {
            c.observe(1e-12);
        }
        assert!(c.threshold() > 0.0, "the threshold collapsed");
    }
}
