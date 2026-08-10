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
