//! Allocating proposals among move kernels, and the budget-window temperature.
//!
//! The cluster driver ran a fixed Metropolis temperature and drew its move
//! kernel uniformly, while the theory this crate implements derives both. This
//! module supplies the two, so the driver runs the law rather than a setting.

use std::f64::consts::E;

use rand::Rng;

/// Budget-window temperature: the descent boundary and the escape floor.
///
/// The law clamps a design point into the window where descent and escape are
/// both feasible:
///
/// ```text
/// T = clamp(theta * g / d,   b / ln(B + e),   2 g / d)
/// ```
///
/// The ceiling `2g/d` is the sphere-model descent boundary, above which the
/// expected one-step progress is negative. The floor `b / ln(B + e)` is the
/// birth-death escape requirement: a barrier of depth `b` is not crossed within
/// `B` remaining proposals at a temperature below it.
///
/// When the floor exceeds the ceiling the window is empty. The law then holds
/// the floor and records the step as escape-forced, which is the regime a
/// funnelled landscape sits in and is worth counting rather than hiding: on a
/// multi-funnel cluster it is not an edge case but the normal condition.
///
/// On a hopping chain the state is a quenched minimum, so the gap is measured
/// against the incumbent and the barrier is estimated from the uphill steps the
/// chain actually failed to take.
#[derive(Debug, Clone)]
pub struct BudgetWindowTemperature {
    /// Problem dimension, `d` in the law.
    pub dim: f64,
    /// Design point as a fraction of the descent boundary; must be under two.
    pub theta: f64,
    /// Decay of the barrier estimate, in `(0, 1)`.
    pub decay: f64,
    /// Temperature never returned below this, so the chain never freezes hard.
    pub floor_temp: f64,
    barrier: f64,
    /// Steps where the escape floor exceeded the descent ceiling.
    pub escape_forced: usize,
    /// Times the law was evaluated.
    pub calls: usize,
}

impl BudgetWindowTemperature {
    /// The law in `dim` dimensions at design point `theta`.
    pub fn new(dim: usize, theta: f64) -> Self {
        assert!(dim > 0, "dimension must be positive");
        assert!(
            theta > 0.0 && theta < 2.0,
            "theta must lie strictly inside the descent boundary, got {theta}"
        );
        Self {
            dim: dim as f64,
            theta,
            decay: 0.98,
            floor_temp: 1e-6,
            barrier: 0.0,
            escape_forced: 0,
            calls: 0,
        }
    }

    /// Records an uphill step the chain declined, which estimates the barrier.
    ///
    /// Only rejections carry the information. An accepted uphill step was
    /// already affordable and says nothing about what the chain cannot cross.
    pub fn observe_rejection(&mut self, uphill: f64) {
        if uphill > 0.0 && uphill.is_finite() {
            self.barrier = self.decay * self.barrier + (1.0 - self.decay) * uphill;
        }
    }

    /// Current estimate of the barrier the chain is failing to cross.
    pub fn barrier(&self) -> f64 {
        self.barrier
    }

    /// Temperature for a chain `gap` above its incumbent with `remaining` left.
    pub fn temperature(&mut self, gap: f64, remaining: usize) -> f64 {
        self.calls += 1;
        let g = gap.max(1e-12);
        let hi = 2.0 * g / self.dim;
        let lo = self.barrier / ((remaining.max(1) as f64) + E).ln();
        let design = self.theta * g / self.dim;
        if lo < hi {
            design.clamp(lo, hi)
        } else {
            self.escape_forced += 1;
            lo.max(self.floor_temp)
        }
    }
}

/// Discounted Beta-Bernoulli allocation with a decaying uniform floor.
///
/// Each arm carries a Beta posterior over its success probability and is chosen
/// by Thompson sampling. Evidence is discounted, so the allocator tracks a
/// landscape whose best move changes as the search moves through it rather than
/// converging on whatever worked at the start. A uniform floor decaying as
/// `1/sqrt(t)` keeps every arm reachable, so no mechanism is starved
/// permanently on early evidence.
pub struct FlooredThompson {
    alpha: Vec<f64>,
    beta: Vec<f64>,
    pulls: Vec<usize>,
    /// Discount applied to all evidence at each update, in `(0, 1]`.
    pub discount: f64,
    /// Scale of the exploration floor.
    pub floor_scale: f64,
    t: usize,
}

impl FlooredThompson {
    /// Allocator over `n_arms`.
    pub fn new(n_arms: usize) -> Self {
        assert!(n_arms > 0, "an allocator needs at least one arm");
        Self {
            alpha: vec![1.0; n_arms],
            beta: vec![1.0; n_arms],
            pulls: vec![0; n_arms],
            discount: 0.995,
            floor_scale: 1.0,
            t: 0,
        }
    }

    /// Arms available.
    pub fn len(&self) -> usize {
        self.alpha.len()
    }

    /// True when there are no arms.
    pub fn is_empty(&self) -> bool {
        self.alpha.is_empty()
    }

    /// Times each arm was chosen.
    pub fn pulls(&self) -> &[usize] {
        &self.pulls
    }

    /// Posterior mean success rate of each arm.
    pub fn rates(&self) -> Vec<f64> {
        (0..self.len())
            .map(|k| self.alpha[k] / (self.alpha[k] + self.beta[k]))
            .collect()
    }

    /// Chooses an arm.
    pub fn select<R: Rng + ?Sized>(&mut self, rng: &mut R) -> usize {
        self.t += 1;
        let floor = (self.floor_scale / (self.t as f64).sqrt()).min(0.5);
        if rng.random::<f64>() < floor {
            return rng.random_range(0..self.len());
        }
        let mut best = 0;
        let mut best_draw = f64::NEG_INFINITY;
        for k in 0..self.len() {
            let draw = sample_beta(self.alpha[k], self.beta[k], rng);
            if draw > best_draw {
                best_draw = draw;
                best = k;
            }
        }
        best
    }

    /// Records whether the chosen arm succeeded.
    ///
    /// Every arm is discounted, not only the one pulled, so evidence ages with
    /// time rather than with how often an arm happens to be selected. Otherwise
    /// a starved arm keeps stale evidence indefinitely and never recovers.
    pub fn update(&mut self, arm: usize, success: bool) {
        for k in 0..self.len() {
            self.alpha[k] = 1.0 + (self.alpha[k] - 1.0) * self.discount;
            self.beta[k] = 1.0 + (self.beta[k] - 1.0) * self.discount;
        }
        if success {
            self.alpha[arm] += 1.0;
        } else {
            self.beta[arm] += 1.0;
        }
        self.pulls[arm] += 1;
    }
}

/// One auditable EXP3-IX arm draw.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exp3IxSelection {
    arm: usize,
    probability: f64,
    learning_rate: f64,
    implicit_exploration: f64,
    round: u64,
}

impl Exp3IxSelection {
    /// Selected arm.
    pub fn arm(self) -> usize {
        self.arm
    }

    /// Probability assigned to the selected arm.
    pub fn probability(self) -> f64 {
        self.probability
    }

    /// Exponential-weights learning rate `eta_t`.
    pub fn learning_rate(self) -> f64 {
        self.learning_rate
    }

    /// Implicit-exploration bias `gamma_t = eta_t / 2`.
    pub fn implicit_exploration(self) -> f64 {
        self.implicit_exploration
    }

    /// One-indexed bandit round.
    pub fn round(self) -> u64 {
        self.round
    }
}

/// Anytime EXP3-IX allocation for overlapping, non-stationary arms.
///
/// The learner follows Algorithm 1 and the horizon-free rates in Theorem 1 of
/// Neu, *Explore no more: Improved high-probability regret bounds for
/// non-stochastic bandits* (NeurIPS 2015):
/// `eta_t = sqrt(log(K) / (K t))` and `gamma_t = eta_t / 2`. Observed losses
/// must lie in `[0, 1]`. Unlike [`FlooredThompson`], this rule assumes neither a
/// stationary Bernoulli likelihood nor a hand-set discount or exploration
/// floor.
#[derive(Debug, Clone)]
pub struct Exp3Ix {
    estimated_losses: Vec<f64>,
    pulls: Vec<usize>,
    rewards: Vec<f64>,
    rounds: u64,
    pending: Option<Exp3IxSelection>,
}

impl Exp3Ix {
    /// Construct an equal-weight learner over `arms` actions.
    pub fn new(arms: usize) -> Self {
        assert!(arms > 0, "an EXP3-IX learner needs at least one arm");
        Self {
            estimated_losses: vec![0.0; arms],
            pulls: vec![0; arms],
            rewards: vec![0.0; arms],
            rounds: 0,
            pending: None,
        }
    }

    fn next_rates(&self) -> (f64, f64) {
        let arms = self.estimated_losses.len() as f64;
        let round = self.rounds.saturating_add(1) as f64;
        let eta = (arms.ln() / (arms * round)).sqrt();
        (eta, eta / 2.0)
    }

    /// Sampling distribution for the next round.
    pub fn next_probabilities(&self) -> Vec<f64> {
        let (eta, _) = self.next_rates();
        let logits = self
            .estimated_losses
            .iter()
            .map(|loss| -eta * loss)
            .collect::<Vec<_>>();
        let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let minimum_log_weight = f64::MIN_POSITIVE.ln();
        let weights = logits
            .iter()
            .map(|logit| (*logit - maximum).max(minimum_log_weight).exp())
            .collect::<Vec<_>>();
        let total = weights.iter().sum::<f64>();
        weights.into_iter().map(|weight| weight / total).collect()
    }

    /// Draw one arm from the current exponential-weights distribution.
    ///
    /// Every draw must be followed by exactly one [`Exp3Ix::update`].
    pub fn select<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Exp3IxSelection {
        assert!(self.pending.is_none(), "EXP3-IX selection needs its loss");
        let probabilities = self.next_probabilities();
        let draw = rng.random::<f64>();
        let mut cumulative = 0.0;
        let mut arm = probabilities.len() - 1;
        for (index, probability) in probabilities.iter().copied().enumerate() {
            cumulative += probability;
            if draw < cumulative {
                arm = index;
                break;
            }
        }
        let (learning_rate, implicit_exploration) = self.next_rates();
        let selection = Exp3IxSelection {
            arm,
            probability: probabilities[arm],
            learning_rate,
            implicit_exploration,
            round: self.rounds.saturating_add(1),
        };
        self.pending = Some(selection);
        selection
    }

    /// Apply the selected arm's observed loss in `[0, 1]`.
    pub fn update(&mut self, loss: f64) {
        assert!(loss.is_finite() && (0.0..=1.0).contains(&loss));
        let selection = self
            .pending
            .take()
            .expect("EXP3-IX update needs a selected arm");
        let estimate = loss / (selection.probability + selection.implicit_exploration);
        self.estimated_losses[selection.arm] += estimate;
        self.pulls[selection.arm] = self.pulls[selection.arm]
            .checked_add(1)
            .expect("EXP3-IX pull count must fit usize");
        self.rewards[selection.arm] += 1.0 - loss;
        self.rounds = self.rounds.saturating_add(1);
    }

    /// Completed selections per arm.
    pub fn pulls(&self) -> &[usize] {
        &self.pulls
    }

    /// Empirical mean reward per arm without a prior or discount.
    pub fn success_rates(&self) -> Vec<f64> {
        self.rewards
            .iter()
            .zip(&self.pulls)
            .map(|(reward, pulls)| {
                if *pulls == 0 {
                    0.0
                } else {
                    reward / *pulls as f64
                }
            })
            .collect()
    }
}

/// Exact diagnostic counters for stationary-object discovery mechanisms.
///
/// Selection belongs to the coordinator's exact-species discovery rule. This
/// type only reports completed attempts, distinct-object yield, and charged PES
/// work without a prior, discount, exploration floor, or sampling policy.
#[derive(Debug, Clone)]
pub struct DiscoveryAccounting {
    pulls: Vec<usize>,
    discoveries: Vec<u64>,
    charged_calls: Vec<u64>,
}

impl DiscoveryAccounting {
    /// Create zeroed counters for `n_arms` discovery mechanisms.
    pub fn new(n_arms: usize) -> Self {
        assert!(n_arms > 0, "discovery accounting needs at least one arm");
        Self {
            pulls: vec![0; n_arms],
            discoveries: vec![0; n_arms],
            charged_calls: vec![0; n_arms],
        }
    }

    /// Record one completed attempt and its distinct-object yield.
    pub fn observe(&mut self, arm: usize, discoveries: u64, charged_calls: u64) {
        assert!(arm < self.pulls.len(), "discovery arm is out of range");
        self.pulls[arm] = self.pulls[arm]
            .checked_add(1)
            .expect("discovery attempt count must fit usize");
        self.discoveries[arm] = self.discoveries[arm]
            .checked_add(discoveries)
            .expect("discovery count must fit u64");
        self.charged_calls[arm] = self.charged_calls[arm]
            .checked_add(charged_calls)
            .expect("charged discovery work must fit u64");
    }

    /// Empirical distinct discoveries per charged PES evaluation for each arm.
    pub fn rates(&self) -> Vec<f64> {
        self.discoveries
            .iter()
            .zip(&self.charged_calls)
            .map(|(discoveries, charged)| {
                if *charged == 0 {
                    0.0
                } else {
                    *discoveries as f64 / *charged as f64
                }
            })
            .collect()
    }

    /// Measured exposures assigned to each arm.
    pub fn pulls(&self) -> &[usize] {
        &self.pulls
    }

    /// Distinct stationary-object discoveries attributed to each arm.
    pub fn discoveries(&self) -> &[u64] {
        &self.discoveries
    }

    /// Charged PES evaluations attributed to each arm.
    pub fn charged_calls(&self) -> &[u64] {
        &self.charged_calls
    }
}

/// A Beta draw from two Gamma draws, since the crate carries no Beta sampler.
/// A Beta draw, public for posteriors held outside this module.
pub fn beta_draw<R: Rng + ?Sized>(a: f64, b: f64, rng: &mut R) -> f64 {
    sample_beta(a, b, rng)
}

fn sample_beta<R: Rng + ?Sized>(a: f64, b: f64, rng: &mut R) -> f64 {
    let x = sample_gamma(a, rng);
    let y = sample_gamma(b, rng);
    if x + y <= 0.0 { 0.5 } else { x / (x + y) }
}

/// Marsaglia and Tsang's Gamma sampler, with the shape boost below one.
fn sample_gamma<R: Rng + ?Sized>(shape: f64, rng: &mut R) -> f64 {
    if shape < 1.0 {
        // Boosting: if G ~ Gamma(a+1) and U ~ Uniform, then G * U^(1/a) is
        // Gamma(a). The direct method is invalid below one.
        let u: f64 = rng.random::<f64>().max(1e-300);
        return sample_gamma(shape + 1.0, rng) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        // Standard normal by Box-Muller, so no distribution crate is needed.
        let u1: f64 = rng.random::<f64>().max(1e-300);
        let u2: f64 = rng.random::<f64>();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let v = 1.0 + c * z;
        if v <= 0.0 {
            continue;
        }
        let v3 = v * v * v;
        let u: f64 = rng.random::<f64>().max(1e-300);
        if u.ln() < 0.5 * z * z + d - d * v3 + d * v3.ln() {
            return d * v3;
        }
    }
}

#[cfg(test)]
mod discovery_accounting_tests {
    use super::DiscoveryAccounting;

    #[test]
    fn discovery_accounting_retains_exact_exposures() {
        let mut accounting = DiscoveryAccounting::new(3);
        accounting.observe(0, 0, 50);
        accounting.observe(1, 2, 75);
        accounting.observe(1, 1, 25);

        assert_eq!(accounting.pulls(), &[1, 2, 0]);
        assert_eq!(accounting.discoveries(), &[0, 3, 0]);
        assert_eq!(accounting.charged_calls(), &[50, 100, 0]);
    }

    #[test]
    fn discovery_rates_have_no_prior_or_discount() {
        let mut accounting = DiscoveryAccounting::new(2);
        accounting.observe(0, 1, 1_000);
        accounting.observe(1, 1, 100);

        assert_eq!(accounting.rates(), vec![0.001, 0.01]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn temperature_respects_the_descent_boundary() {
        let mut law = BudgetWindowTemperature::new(30, 0.5);
        // No barrier seen, so the floor is zero and the design point applies.
        let t = law.temperature(3.0, 1000);
        assert!(t <= 2.0 * 3.0 / 30.0 + 1e-12, "above the ceiling: {t}");
        assert!(
            (t - 0.5 * 3.0 / 30.0).abs() < 1e-12,
            "not the design point: {t}"
        );
    }

    #[test]
    fn a_deep_barrier_forces_the_escape_floor() {
        let mut law = BudgetWindowTemperature::new(30, 0.5);
        for _ in 0..500 {
            law.observe_rejection(5.0);
        }
        // A small gap makes the ceiling tiny while the barrier stays large, so
        // the window is empty. This is the multi-funnel regime.
        let t = law.temperature(0.01, 1000);
        assert!(law.escape_forced > 0, "an empty window was not recorded");
        assert!(t > 2.0 * 0.01 / 30.0, "the floor did not override: {t}");
    }

    #[test]
    fn accepted_uphill_steps_do_not_raise_the_barrier() {
        let mut law = BudgetWindowTemperature::new(10, 0.5);
        law.observe_rejection(-1.0);
        law.observe_rejection(0.0);
        law.observe_rejection(f64::NAN);
        assert_eq!(law.barrier(), 0.0);
    }

    #[test]
    fn the_floor_shrinks_so_exploration_decays() {
        let mut law = BudgetWindowTemperature::new(30, 0.5);
        for _ in 0..200 {
            law.observe_rejection(1.0);
        }
        let early = law.temperature(0.001, 10);
        let late = law.temperature(0.001, 1_000_000);
        assert!(
            late < early,
            "more remaining budget should need less heat: {early} then {late}"
        );
    }

    #[test]
    fn gamma_sampler_has_the_right_mean() {
        let mut rng = StdRng::seed_from_u64(1);
        for shape in [0.5_f64, 1.0, 3.0, 12.0] {
            let n = 20000;
            let mean: f64 = (0..n).map(|_| sample_gamma(shape, &mut rng)).sum::<f64>() / n as f64;
            assert!(
                (mean - shape).abs() < 0.1 * shape.max(1.0),
                "Gamma({shape}) mean {mean}"
            );
        }
    }

    #[test]
    fn beta_sampler_has_the_right_mean() {
        let mut rng = StdRng::seed_from_u64(2);
        let n = 20000;
        let mean: f64 = (0..n).map(|_| sample_beta(2.0, 6.0, &mut rng)).sum::<f64>() / n as f64;
        assert!((mean - 0.25).abs() < 0.02, "Beta(2,6) mean {mean}");
    }

    /// The property the allocator exists for: it must find the better arm.
    #[test]
    fn allocation_concentrates_on_the_arm_that_works() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut alloc = FlooredThompson::new(4);
        let truth = [0.05, 0.05, 0.6, 0.05];
        for _ in 0..3000 {
            let arm = alloc.select(&mut rng);
            let success = rng.random::<f64>() < truth[arm];
            alloc.update(arm, success);
        }
        let pulls = alloc.pulls();
        assert!(
            pulls[2] > pulls[0] + pulls[1] + pulls[3],
            "the good arm was not preferred: {pulls:?}"
        );
    }

    #[test]
    fn every_arm_keeps_being_reachable() {
        let mut rng = StdRng::seed_from_u64(4);
        let mut alloc = FlooredThompson::new(3);
        for _ in 0..4000 {
            let arm = alloc.select(&mut rng);
            // Arm 0 never succeeds, so without a floor it would be abandoned.
            alloc.update(arm, arm != 0 && rng.random::<f64>() < 0.5);
        }
        assert!(
            alloc.pulls()[0] > 20,
            "a losing arm was starved: {:?}",
            alloc.pulls()
        );
    }

    #[test]
    fn discounting_lets_a_reversal_be_learned() {
        let mut rng = StdRng::seed_from_u64(5);
        let mut alloc = FlooredThompson::new(2);
        // Arm 0 is good, then stops working and arm 1 takes over. Without
        // discounting the early evidence would keep arm 0 in front.
        for _ in 0..1500 {
            let arm = alloc.select(&mut rng);
            alloc.update(arm, arm == 0 && rng.random::<f64>() < 0.8);
        }
        let before = alloc.pulls().to_vec();
        for _ in 0..3000 {
            let arm = alloc.select(&mut rng);
            alloc.update(arm, arm == 1 && rng.random::<f64>() < 0.8);
        }
        let after: Vec<usize> = alloc
            .pulls()
            .iter()
            .zip(before.iter())
            .map(|(a, b)| a - b)
            .collect();
        assert!(
            after[1] > after[0],
            "the reversal was not tracked: {after:?}"
        );
    }

    #[test]
    fn exp3_ix_uses_the_anytime_learning_and_implicit_exploration_rates() {
        let mut rng = StdRng::seed_from_u64(17);
        let mut alloc = Exp3Ix::new(2);

        assert_eq!(alloc.next_probabilities(), vec![0.5, 0.5]);
        let selection = alloc.select(&mut rng);
        let expected_eta = (2.0_f64.ln() / 2.0).sqrt();
        assert!((selection.learning_rate() - expected_eta).abs() < 1e-15);
        assert!((selection.implicit_exploration() - expected_eta / 2.0).abs() < 1e-15);

        let penalized = selection.arm();
        alloc.update(1.0);
        let probabilities = alloc.next_probabilities();
        assert!(probabilities[penalized] < 0.5, "{probabilities:?}");
        assert!(probabilities[1 - penalized] > 0.5, "{probabilities:?}");
    }

    #[test]
    fn exp3_ix_tracks_the_best_fixed_kernel_under_adversarial_feedback() {
        let mut rng = StdRng::seed_from_u64(23);
        let mut alloc = Exp3Ix::new(3);
        for _ in 0..2_000 {
            let selection = alloc.select(&mut rng);
            alloc.update(if selection.arm() == 1 { 0.0 } else { 1.0 });
        }

        assert!(alloc.pulls()[1] > alloc.pulls()[0] + alloc.pulls()[2]);
        assert_eq!(alloc.success_rates()[1], 1.0);
        assert_eq!(alloc.success_rates()[0], 0.0);
        assert_eq!(alloc.success_rates()[2], 0.0);
    }
}

/// Mergeable sufficient statistics for finite depth rewards.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct RewardMoments {
    /// Number of independently credited observations.
    pub count: u64,
    /// Arithmetic mean reward.
    pub mean: f64,
    /// Sum of squared deviations from the mean.
    pub m2: f64,
}

impl RewardMoments {
    /// Reject invalid moments and counts that lose integer precision in a posterior.
    pub fn validate(self) -> Result<(), &'static str> {
        if self.count > (1_u64 << 53)
            || !self.mean.is_finite()
            || !self.m2.is_finite()
            || self.m2 < 0.0
        {
            return Err("invalid reward moments");
        }
        if (self.count == 0 && (self.mean != 0.0 || self.m2 != 0.0))
            || (self.count == 1 && self.m2 != 0.0)
        {
            return Err("inconsistent reward moments");
        }
        Ok(())
    }

    /// Add one observation without changing the record on invalid input.
    pub fn observe(&mut self, reward: f64) -> Result<(), &'static str> {
        *self = self.merge(Self {
            count: 1,
            mean: reward,
            m2: 0.0,
        })?;
        Ok(())
    }

    /// Combine disjoint observations using the parallel variance identity.
    pub fn merge(self, other: Self) -> Result<Self, &'static str> {
        self.validate()?;
        other.validate()?;
        if self.count == 0 {
            return Ok(other);
        }
        if other.count == 0 {
            return Ok(self);
        }
        let count = self
            .count
            .checked_add(other.count)
            .ok_or("reward count overflow")?;
        let delta = other.mean - self.mean;
        let weight = other.count as f64 / count as f64;
        let combined = Self {
            count,
            mean: self.mean + weight * delta,
            m2: self.m2 + other.m2 + delta * delta * self.count as f64 * weight,
        };
        combined.validate()?;
        Ok(combined)
    }
}

/// Thompson sampling over arms by the depth they reach, not by whether they are
/// accepted.
///
/// The Beta-Bernoulli allocator above is rewarded with `improved || accept`.
/// Beating the run's best is rare -- at 98 points a run registers about five
/// such events in ten thousand hops -- so the reward is carried almost entirely
/// by acceptance, and acceptance does not separate a move that produces deep
/// structures from one that is merely plausible. Measured, that is how a twin
/// arm survives on a system it does not suit: it is accepted readily, never
/// switched off, and takes draws from the arm the system needs.
///
/// Depth is dense and informative at once. Every hop yields a number, and its
/// magnitude says how close the arm brought the chain to the best it knows.
///
/// Each arm carries a Normal-Gamma posterior over that reward with unknown mean
/// and unknown precision, so no scale has to be supplied and a system whose
/// energies run in hundreds is handled the same as one running in tens. A draw
/// is taken from the posterior predictive, which is Student-t, and the best draw
/// wins.
#[derive(Debug, Clone)]
pub struct DepthAllocator {
    /// Prior and posterior mean per arm.
    mu: Vec<f64>,
    /// Pseudo-observations behind the mean.
    kappa: Vec<f64>,
    /// Shape of the precision posterior.
    alpha: Vec<f64>,
    /// Rate of the precision posterior.
    beta: Vec<f64>,
    /// Draws per arm, for reporting.
    pub draws: Vec<usize>,
}

impl DepthAllocator {
    /// Reconstruct the same Normal-Gamma posterior as sequential observations.
    pub fn from_moments(moments: &[RewardMoments]) -> Result<Self, &'static str> {
        if moments.is_empty() {
            return Err("an allocator needs at least one arm");
        }
        let mut allocator = Self::new(moments.len());
        for (arm, moment) in moments.iter().copied().enumerate() {
            moment.validate()?;
            let n = moment.count as f64;
            let kappa = 1e-6 + n;
            allocator.mu[arm] = n / kappa * moment.mean;
            allocator.kappa[arm] = kappa;
            allocator.alpha[arm] = 1.0 + 0.5 * n;
            allocator.beta[arm] =
                1.0 + 0.5 * moment.m2 + 0.5 * (1e-6 * n / kappa) * moment.mean * moment.mean;
            allocator.draws[arm] =
                usize::try_from(moment.count).map_err(|_| "reward count overflow")?;
            if !allocator.beta[arm].is_finite() {
                return Err("reward posterior overflow");
            }
        }
        Ok(allocator)
    }

    /// An allocator over `n_arms`, uninformative until fed.
    pub fn new(n_arms: usize) -> Self {
        Self {
            mu: vec![0.0; n_arms],
            kappa: vec![1e-6; n_arms],
            alpha: vec![1.0; n_arms],
            beta: vec![1.0; n_arms],
            draws: vec![0; n_arms],
        }
    }

    /// Number of arms.
    pub fn arms(&self) -> usize {
        self.mu.len()
    }

    /// Records the depth an arm reached.
    ///
    /// The conjugate Normal-Gamma update for one observation.
    pub fn update(&mut self, arm: usize, reward: f64) {
        if arm >= self.mu.len() || !reward.is_finite() {
            return;
        }
        let k = self.kappa[arm];
        let m = self.mu[arm];
        self.kappa[arm] = k + 1.0;
        self.mu[arm] = (k * m + reward) / (k + 1.0);
        self.alpha[arm] += 0.5;
        self.beta[arm] += 0.5 * k / (k + 1.0) * (reward - m) * (reward - m);
        self.draws[arm] += 1;
    }

    /// Draws an arm by Thompson sampling on the posterior predictive.
    pub fn select<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let mut best = 0usize;
        let mut best_v = f64::NEG_INFINITY;
        for k in 0..self.mu.len() {
            // Predictive is Student-t with 2 alpha degrees of freedom, centred
            // at mu, scaled by beta (kappa + 1) / (alpha kappa). An arm with no
            // evidence has a very small kappa and so an enormous spread, which
            // is what makes it get tried.
            let scale =
                (self.beta[k] * (self.kappa[k] + 1.0) / (self.alpha[k] * self.kappa[k])).sqrt();
            let v = self.mu[k] + scale * student_t(2.0 * self.alpha[k], rng);
            if v > best_v {
                best_v = v;
                best = k;
            }
        }
        best
    }

    /// Posterior mean reward per arm, so a run can report what it learned.
    pub fn means(&self) -> Vec<f64> {
        self.mu.clone()
    }
}

/// A Student-t draw with `nu` degrees of freedom, as a normal scaled by a
/// chi-square, which needs no special functions.
fn student_t<R: Rng + ?Sized>(nu: f64, rng: &mut R) -> f64 {
    let z = {
        let u1: f64 = rng.random::<f64>().max(1e-12);
        let u2: f64 = rng.random::<f64>();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    // Chi-square with nu degrees of freedom is Gamma(nu/2, 2).
    let g = sample_gamma(nu * 0.5, rng) * 2.0;
    if g <= 0.0 {
        return z;
    }
    z / (g / nu).sqrt()
}

#[cfg(test)]
mod depth_allocator_tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn sufficient_statistics_reconstruct_every_posterior_parameter() {
        let values = [[-4.0, 2.0, 1.0, -3.0, 8.0], [0.5, 0.25, 0.75, -1.0, 2.0]];
        let mut sequential = DepthAllocator::new(2);
        let mut moments = [RewardMoments::default(); 2];
        for (arm, values) in values.iter().enumerate() {
            for &value in values {
                sequential.update(arm, value);
                moments[arm].observe(value).unwrap();
            }
        }
        let restored = DepthAllocator::from_moments(&moments).unwrap();
        for (actual, expected) in [&restored.mu, &restored.kappa, &restored.alpha, &restored.beta].into_iter().zip([&sequential.mu, &sequential.kappa, &sequential.alpha, &sequential.beta]) {
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-12);
            }
        }
        assert_eq!(restored.draws, sequential.draws);
        let mut left = StdRng::seed_from_u64(79);
        let mut right = left.clone();
        for _ in 0..256 { assert_eq!(restored.select(&mut left), sequential.select(&mut right)); }
    }

    /// The allocator has to find the arm that reaches deeper, which is the
    /// whole point of rewarding depth instead of acceptance.
    #[test]
    fn the_deeper_arm_is_found() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut a = DepthAllocator::new(3);
        for _ in 0..600 {
            let k = a.select(&mut rng);
            // Arm 1 reaches deeper on average; the others are shallower.
            let r = match k {
                1 => -1.0,
                _ => -3.0,
            } + 0.3 * {
                let u1: f64 = rng.random::<f64>().max(1e-12);
                let u2: f64 = rng.random::<f64>();
                (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
            };
            a.update(k, r);
        }
        assert!(
            a.draws[1] > a.draws[0] && a.draws[1] > a.draws[2],
            "draws {:?}, means {:?}",
            a.draws,
            a.means()
        );
    }

    /// An arm that is frequently rewarded but shallow must lose to a rarely
    /// rewarded deep one, which is exactly the case the Bernoulli allocator
    /// gets wrong.
    #[test]
    fn frequent_and_shallow_loses_to_deep() {
        let mut rng = StdRng::seed_from_u64(5);
        let mut a = DepthAllocator::new(2);
        for _ in 0..800 {
            let k = a.select(&mut rng);
            // Arm 0 always lands a little way above the best; arm 1 usually
            // lands at the same place but occasionally much deeper, so its mean
            // depth is better.
            let r = if k == 0 {
                -2.0
            } else if rng.random::<f64>() < 0.1 {
                -0.1
            } else {
                -2.1
            };
            a.update(k, r);
        }
        assert!(
            a.draws[1] > a.draws[0],
            "draws {:?}, means {:?}",
            a.draws,
            a.means()
        );
    }

    /// A scale change must not change which arm wins, since one system's
    /// energies run in tens and another's in hundreds.
    #[test]
    fn the_choice_is_invariant_to_the_energy_scale() {
        let run = |scale: f64, seed: u64| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut a = DepthAllocator::new(3);
            for _ in 0..500 {
                let k = a.select(&mut rng);
                let r = scale
                    * match k {
                        2 => -1.0,
                        _ => -4.0,
                    };
                a.update(k, r);
            }
            a.draws
        };
        let small = run(1.0, 9);
        let large = run(100.0, 9);
        assert_eq!(
            small
                .iter()
                .enumerate()
                .max_by_key(|(_, v)| **v)
                .map(|(i, _)| i),
            large
                .iter()
                .enumerate()
                .max_by_key(|(_, v)| **v)
                .map(|(i, _)| i),
            "scale changed the winner: {small:?} against {large:?}"
        );
    }
}
