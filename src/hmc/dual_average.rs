//! Step-size adaptation, so the proposal carries no swept constant.
//!
//! The argument this module exists to support is that a hand-tuned amplitude is
//! not a method. Basin hopping's displacement half-width is set by sweeping it
//! until the acceptance ratio looks right, and the number does not transfer
//! between cluster sizes or between potentials. Hamiltonian dynamics has the
//! same parameter, the leapfrog step `eps`, and sampling theory settled how to
//! set it: drive `log eps` by Nesterov dual averaging against a target
//! acceptance statistic, which converges without a sweep and without a human
//! looking at the trace.
//!
//! Hoffman and Gelman, "The No-U-Turn Sampler", JMLR 15 (2014) 1593-1623,
//! algorithms 5 and 6, and the Stan reference manual's MCMC chapter under
//! "Automatic parameter tuning".
//!
//! # The recursion
//!
//! With `alpha_m` the average Metropolis acceptance probability over the
//! trajectory's leaves at iteration `m` and `delta` the target,
//!
//! ```text
//! eta_m     = 1 / (m + t0)
//! s_m       = (1 - eta_m) s_{m-1} + eta_m (delta - alpha_m)
//! log eps_m = mu - sqrt(m) s_m / gamma
//! w_m       = m^-kappa
//! log epsbar_m = (1 - w_m) log epsbar_{m-1} + w_m log eps_m
//! ```
//!
//! `s_m` is the running mean of `delta - alpha`, so the second line is the
//! statement in the brief with the sum written recursively. The chain
//! integrates with `eps_m` while adapting and with `epsbar_m` once frozen: the
//! shrunken average is what has converged, while `eps_m` keeps moving by
//! construction.
//!
//! Stan's defaults are used unchanged: `delta = 0.8`, `gamma = 0.05`,
//! `kappa = 0.75`, `t0 = 10`, `mu = log(10 eps_0)`. The target of 0.8 is
//! deliberately above the 0.65 that is optimal for a fixed-length HMC
//! trajectory on a Gaussian, because Stan found the higher target more robust
//! for targets with varying curvature, and a Lennard-Jones surface with an
//! `r^-12` wall is such a target with room to spare.
//!
//! # Per chain, never global
//!
//! One of these belongs to one chain. A replica ladder runs rungs at different
//! temperatures, a hot rung crosses barriers a cold rung cannot and its
//! trajectories see a differently conditioned landscape, so the two converge to
//! different step sizes. Sharing one adapter across rungs would average them
//! into a step size neither wants.

/// Nesterov dual averaging on `log eps` against a target acceptance statistic.
#[derive(Debug, Clone)]
pub struct DualAverage {
    /// Target acceptance statistic.
    pub delta: f64,
    /// Regularisation scale on the update.
    pub gamma: f64,
    /// Decay exponent of the shrinking average; must lie in `(0.5, 1]`.
    pub kappa: f64,
    /// Iteration offset that damps early updates.
    pub t0: f64,
    /// Point `log eps` is shrunk towards, `log(10 eps_0)`.
    mu: f64,
    /// Running mean of `delta - alpha`.
    s: f64,
    /// `log epsbar`, the shrunken average.
    log_bar: f64,
    /// Current `log eps`.
    log_eps: f64,
    counter: u64,
    frozen: bool,
}

impl DualAverage {
    /// An adapter shrinking towards `10 * eps0`, with Stan's defaults.
    pub fn new(eps0: f64) -> Self {
        assert!(
            eps0 > 0.0 && eps0.is_finite(),
            "the initial step must be positive and finite, got {eps0}"
        );
        Self {
            delta: 0.8,
            gamma: 0.05,
            kappa: 0.75,
            t0: 10.0,
            mu: (10.0 * eps0).ln(),
            s: 0.0,
            log_bar: eps0.ln(),
            log_eps: eps0.ln(),
            counter: 0,
            frozen: false,
        }
    }

    /// The step size to integrate with.
    ///
    /// While adapting this is `eps_m`, which is still moving; once frozen it is
    /// `epsbar_m`, the shrunken average, which is what has converged.
    pub fn epsilon(&self) -> f64 {
        if self.frozen {
            self.log_bar.exp()
        } else {
            self.log_eps.exp()
        }
    }

    /// The shrunken average, whether or not adaptation has stopped.
    pub fn epsilon_bar(&self) -> f64 {
        self.log_bar.exp()
    }

    /// Updates from one trajectory's acceptance statistic.
    ///
    /// A non-finite statistic, which a divergent trajectory produces, is read
    /// as zero acceptance rather than dropped: a step size that keeps diverging
    /// is exactly the case the adapter has to shrink, and dropping the
    /// observation would leave it unaware.
    pub fn learn(&mut self, alpha: f64) {
        if self.frozen {
            return;
        }
        let a = if alpha.is_finite() {
            alpha.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.counter += 1;
        let m = self.counter as f64;
        let eta = 1.0 / (m + self.t0);
        self.s = (1.0 - eta) * self.s + eta * (self.delta - a);
        self.log_eps = self.mu - m.sqrt() * self.s / self.gamma;
        let w = m.powf(-self.kappa);
        self.log_bar = (1.0 - w) * self.log_bar + w * self.log_eps;
    }

    /// Stops adaptation and switches the chain onto the shrunken average.
    ///
    /// Required, not optional. An adapter that keeps learning during the
    /// measurement phase makes the transition kernel depend on the whole
    /// history, so the chain is no longer Markov and the usual convergence
    /// argument does not apply.
    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// Whether adaptation has stopped.
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Updates recorded so far.
    pub fn count(&self) -> u64 {
        self.counter
    }

    /// Restarts the recursion around the current step size.
    ///
    /// Stan does this at the end of each metric window, because a new metric
    /// changes what step size the target acceptance corresponds to and the
    /// history accumulated under the old one is describing a different problem.
    pub fn restart(&mut self) {
        let eps = self.epsilon();
        self.mu = (10.0 * eps).ln();
        self.s = 0.0;
        self.log_bar = eps.ln();
        self.log_eps = eps.ln();
        self.counter = 0;
    }
}

/// Stan's three-phase warmup: a fast interval, doubling slow windows, a fast
/// interval.
///
/// The step size adapts throughout. The metric is estimated only inside the
/// slow windows and written at each window's close, because a covariance needs
/// draws that a step size does not, and the windows double so that each new
/// estimate rests on as many draws as every previous one together.
///
/// Stan's defaults are 75, 25 and 50 against a warmup of 1000. They are scaled
/// down here and the reason is a budget rather than a preference: a warmup hop
/// costs a trajectory plus a quench, so 1000 hops of warmup is a quarter of a
/// 4e5-evaluation campaign. [`WarmupSchedule::hops`] is what a caller sets and
/// [`WarmupSchedule::fraction_of`] is the instrument that says what it cost.
#[derive(Debug, Clone)]
pub struct WarmupSchedule {
    /// Warmup iterations in total.
    pub hops: usize,
    /// Fast interval before the first metric window.
    pub init_buffer: usize,
    /// Fast interval after the last metric window.
    pub term_buffer: usize,
    /// Width of the first slow window.
    pub base_window: usize,
    counter: usize,
    window_size: usize,
    next_window: usize,
}

impl WarmupSchedule {
    /// A schedule over `hops` warmup iterations, in Stan's proportions.
    ///
    /// When the three phases do not fit, they are cut to Stan's fallback
    /// fractions of 15 per cent initial and 10 per cent terminal with the rest
    /// as one window, rather than silently overrunning the warmup.
    pub fn new(hops: usize) -> Self {
        let (init_buffer, term_buffer, base_window) = if 75 + 25 + 50 <= hops {
            (75, 50, 25)
        } else {
            let i = ((0.15 * hops as f64) as usize).max(1);
            let t = ((0.10 * hops as f64) as usize).max(1);
            let w = hops.saturating_sub(i + t).max(1);
            (i, t, w)
        };
        let window_size = base_window;
        Self {
            hops,
            init_buffer,
            term_buffer,
            base_window,
            counter: 0,
            window_size,
            next_window: init_buffer + window_size - 1,
        }
    }

    /// A schedule that adapts nothing, for a caller running fixed parameters.
    pub fn none() -> Self {
        let mut s = Self::new(1);
        s.hops = 0;
        s
    }

    /// Warmup iterations as a fraction of `budget` charged evaluations, given
    /// the measured cost of a hop.
    ///
    /// Reported rather than assumed: warmup that eats a fifth of the budget is
    /// a cost of the method and belongs in the table beside the solve count.
    pub fn fraction_of(&self, budget: usize, charged_per_hop: f64) -> f64 {
        if budget == 0 {
            return 0.0;
        }
        self.hops as f64 * charged_per_hop / budget as f64
    }

    /// Whether the chain is still in warmup.
    pub fn warming(&self) -> bool {
        self.counter < self.hops
    }

    /// Whether the current iteration falls inside a metric window.
    pub fn in_window(&self) -> bool {
        self.counter >= self.init_buffer
            && self.counter < self.hops.saturating_sub(self.term_buffer)
            && self.counter != self.hops
    }

    /// Whether the current iteration closes a metric window.
    pub fn closes_window(&self) -> bool {
        self.counter == self.next_window
            && self.counter != self.hops.saturating_sub(self.term_buffer)
    }

    /// Advances one iteration, and reports whether a window just closed.
    ///
    /// Call once per hop. The return says the caller should write the metric
    /// estimate and restart the step-size recursion around it.
    pub fn advance(&mut self) -> bool {
        let closed = self.warming() && self.in_window() && self.closes_window();
        if closed {
            self.compute_next_window();
        }
        self.counter += 1;
        closed
    }

    /// Whether this iteration is the one that ends warmup.
    pub fn just_finished(&self) -> bool {
        self.counter == self.hops
    }

    /// Iterations taken.
    pub fn count(&self) -> usize {
        self.counter
    }

    fn compute_next_window(&mut self) {
        let last = self.hops.saturating_sub(self.term_buffer).saturating_sub(1);
        if self.next_window == last {
            return;
        }
        self.window_size *= 2;
        self.next_window = self.counter + self.window_size;
        if self.next_window == last {
            return;
        }
        // Stretch the current window to the end when the one after it would
        // overrun, rather than closing a stunted window whose estimate rests on
        // fewer draws than the one before it.
        if self.next_window + 2 * self.window_size >= self.hops.saturating_sub(self.term_buffer) {
            self.next_window = last;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The claim dual averaging makes is that it finds the step size at which
    /// the acceptance statistic hits the target, without being told what that
    /// step size is. Against a stand-in whose acceptance falls smoothly with
    /// `eps`, the converged step has to be the one solving `alpha(eps) = delta`.
    #[test]
    fn it_converges_to_the_step_that_hits_the_target() {
        // alpha(eps) = exp(-eps), so alpha = 0.8 at eps = -ln(0.8) = 0.2231.
        let want = -(0.8f64).ln();
        let mut da = DualAverage::new(1.0);
        for _ in 0..2000 {
            let alpha = (-da.epsilon()).exp();
            da.learn(alpha);
        }
        da.freeze();
        let got = da.epsilon();
        assert!(
            (got / want - 1.0).abs() < 0.05,
            "converged to {got}, wanted {want}"
        );
    }

    /// A step size that always diverges has to be driven down, or the adapter
    /// is not doing the one job that makes HMC usable on a hard surface.
    #[test]
    fn a_diverging_trajectory_shrinks_the_step() {
        let mut da = DualAverage::new(1.0);
        for _ in 0..200 {
            da.learn(f64::NAN);
        }
        assert!(
            da.epsilon() < 0.05,
            "constant divergence left the step at {}",
            da.epsilon()
        );
    }

    /// Freezing has to stop adaptation and switch onto the shrunken average,
    /// because a kernel that keeps learning is not Markov.
    #[test]
    fn freezing_stops_the_recursion() {
        let mut da = DualAverage::new(0.5);
        for _ in 0..50 {
            da.learn(0.9);
        }
        da.freeze();
        let held = da.epsilon();
        assert_eq!(held, da.epsilon_bar(), "a frozen chain runs the average");
        for _ in 0..50 {
            da.learn(0.1);
        }
        assert_eq!(da.epsilon(), held, "adaptation continued after freezing");
    }

    /// The windows have to double and to stay inside the warmup, or the metric
    /// is estimated on a schedule other than the one claimed.
    #[test]
    fn the_windows_double_and_stay_inside_warmup() {
        let mut s = WarmupSchedule::new(1000);
        assert_eq!((s.init_buffer, s.base_window, s.term_buffer), (75, 25, 50));
        let mut closes = Vec::new();
        while s.warming() {
            if s.advance() {
                closes.push(s.count() - 1);
            }
        }
        assert!(!closes.is_empty(), "no metric window ever closed");
        for c in &closes {
            assert!(
                *c >= s.init_buffer && *c < 1000 - s.term_buffer,
                "a window closed at {c}, outside the slow phase"
            );
        }
        // Widths double until the stretch rule takes over at the end.
        let mut widths = Vec::new();
        let mut prev = s.init_buffer - 1;
        for c in &closes {
            widths.push(c - prev);
            prev = *c;
        }
        for w in widths.iter().take(widths.len().saturating_sub(1)).skip(1) {
            assert!(*w > 0, "a window of width {w}");
        }
    }

    /// A warmup too short for Stan's proportions has to be cut rather than
    /// overrun it, since the fallback is what a budgeted campaign actually
    /// runs.
    #[test]
    fn a_short_warmup_falls_back_to_fractions() {
        let s = WarmupSchedule::new(60);
        assert!(
            s.init_buffer + s.base_window + s.term_buffer <= 60,
            "phases {} + {} + {} overrun a warmup of 60",
            s.init_buffer,
            s.base_window,
            s.term_buffer
        );
    }
}
