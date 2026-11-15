// Copyright (c) anneal developers
//! Discounted bandit over move kernels and stall actions.
//!
//! One decision per hop. The reward of a decision is the drop in the
//! ledger's best energy divided by the charged calls between it and the
//! next decision, so an arm that buys a record cheaply outranks one that
//! buys the same record with a long relaxation, and a stall action is
//! priced by the calls it spends like any proposal. Records are a moving
//! target: the arm that improves a descent is not the arm that leaves a
//! funnel, so every statistic is discounted per decision (Garivier and
//! Moulines, ALT 2011). The reward has the potential's energy unit; both
//! rules divide it by the pooled spread, so no reward scale is set. The
//! record reward counts a new lowest basin as one, per charged call, and
//! has no unit at all.

use rand::Rng;

/// Weight kept by every past decision at each new one.
///
/// An effective memory of two hundred decisions, a few times the arm
/// count of any library.
pub const DISCOUNT: f64 = 0.995;

/// UCB exploration weight `xi` of discounted UCB.
const UCB_XI: f64 = 0.6;

/// Share of the preset allocator's kernel draws a floored bandit keeps.
///
/// Half the draws follow the preset's allocator and half are the bandit's
/// to move, so a kernel the bandit has written off is still drawn at half
/// the rate the preset would draw it.
pub const FLOOR_SHARE: f64 = 0.5;

/// What an arm does when played.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BanditArm {
    /// Propose from this kernel index.
    Kernel(usize),
    /// Relocate the worst-bound point.
    Angular,
    /// Quench a fresh cluster and adopt it.
    Reseed,
    /// Leave the plateau the way the preset leaves a call plateau.
    Leave,
}

/// Selection rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// Gaussian Thompson sampling on discounted sums.
    Thompson,
    /// Discounted UCB.
    Ucb,
}

/// What a settled decision is credited with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reward {
    /// The drop in the best energy per charged call.
    Gain,
    /// One per charged call when the best dropped at all.
    Record,
}

/// Discounted per-arm sums with a pending decision awaiting its reward.
#[derive(Debug, Clone)]
pub struct DiscountedBandit {
    arms: Vec<BanditArm>,
    rule: Rule,
    reward: Reward,
    count: Vec<f64>,
    sum: Vec<f64>,
    square: Vec<f64>,
    plays: Vec<usize>,
    pending: Option<(usize, usize, f64)>,
}

impl DiscountedBandit {
    /// A bandit over `arms`, none played.
    pub fn new(arms: Vec<BanditArm>, rule: Rule) -> Self {
        let k = arms.len();
        assert!(k > 0, "a bandit needs an arm");
        Self {
            arms,
            rule,
            reward: Reward::Gain,
            count: vec![0.0; k],
            sum: vec![0.0; k],
            square: vec![0.0; k],
            plays: vec![0; k],
            pending: None,
        }
    }

    /// The same bandit crediting `reward`.
    pub fn with_reward(mut self, reward: Reward) -> Self {
        self.reward = reward;
        self
    }

    /// Forget every arm's statistics. The decision awaiting its reward is
    /// still settled, and the play counts are kept.
    pub fn reset(&mut self) {
        self.count.fill(0.0);
        self.sum.fill(0.0);
        self.square.fill(0.0);
    }

    /// The arm set, in index order.
    pub fn arms(&self) -> &[BanditArm] {
        &self.arms
    }

    /// Decisions taken per arm.
    pub fn plays(&self) -> &[usize] {
        &self.plays
    }

    /// Credit one observed reward to `arm` after discounting every sum.
    pub fn update(&mut self, arm: usize, reward: f64) {
        let reward = if reward.is_finite() { reward } else { 0.0 };
        for index in 0..self.arms.len() {
            self.count[index] *= DISCOUNT;
            self.sum[index] *= DISCOUNT;
            self.square[index] *= DISCOUNT;
        }
        self.count[arm] += 1.0;
        self.sum[arm] += reward;
        self.square[arm] += reward * reward;
    }

    /// Settle the decision taken last, given the ledger's spend and best now.
    pub fn settle(&mut self, spent: usize, best: f64) {
        if let Some((arm, spent_then, best_then)) = self.pending.take() {
            let calls = spent.saturating_sub(spent_then).max(1) as f64;
            let gain = match self.reward {
                Reward::Gain if best_then.is_finite() && best.is_finite() => {
                    (best_then - best).max(0.0)
                }
                Reward::Gain => 0.0,
                Reward::Record if best < best_then => 1.0,
                Reward::Record => 0.0,
            };
            self.update(arm, gain / calls);
        }
    }

    /// Choose an arm among those `eligible` allows and hold it pending.
    pub fn choose<R: Rng + ?Sized>(
        &mut self,
        spent: usize,
        best: f64,
        eligible: impl Fn(BanditArm) -> bool,
        rng: &mut R,
    ) -> BanditArm {
        let index = self.select(&eligible, rng);
        self.plays[index] += 1;
        self.pending = Some((index, spent, best));
        self.arms[index]
    }

    /// Credit the decision awaiting its reward to `arm` instead, when `arm`
    /// is one of the bandit's.
    pub fn retarget(&mut self, arm: BanditArm) {
        if let Some(index) = self.arms.iter().position(|&a| a == arm)
            && let Some(pending) = self.pending.as_mut()
        {
            self.plays[pending.0] -= 1;
            self.plays[index] += 1;
            pending.0 = index;
        }
    }

    /// Index of the arm the rule picks, without holding it.
    pub fn select<R: Rng + ?Sized>(
        &self,
        eligible: &impl Fn(BanditArm) -> bool,
        rng: &mut R,
    ) -> usize {
        let open: Vec<usize> = (0..self.arms.len())
            .filter(|&index| eligible(self.arms[index]))
            .collect();
        let open = if open.is_empty() {
            (0..self.arms.len()).collect()
        } else {
            open
        };
        let spread = self.pooled_spread();
        let total: f64 = self.count.iter().sum();
        let mut best_index = open[0];
        let mut best_score = f64::NEG_INFINITY;
        let mut ties = 0usize;
        for &index in &open {
            let n = self.count[index];
            let score = match self.rule {
                Rule::Thompson => {
                    let mean = self.sum[index] / (n + 1.0);
                    mean + spread * standard_normal(rng) / (n + 1.0).sqrt()
                }
                Rule::Ucb => {
                    if n < 1e-9 {
                        f64::INFINITY
                    } else {
                        let mean = self.sum[index] / n;
                        let width = (UCB_XI * total.max(1.0).ln() / n).sqrt();
                        mean + 2.0 * spread * width
                    }
                }
            };
            if score > best_score {
                best_score = score;
                best_index = index;
                ties = 1;
            } else if score == best_score {
                ties += 1;
                if rng.random_range(0..ties) == 0 {
                    best_index = index;
                }
            }
        }
        best_index
    }

    fn pooled_spread(&self) -> f64 {
        let n: f64 = self.count.iter().sum();
        if n < 1e-9 {
            return 0.0;
        }
        let mean = self.sum.iter().sum::<f64>() / n;
        let second = self.square.iter().sum::<f64>() / n;
        (second - mean * mean).max(0.0).sqrt()
    }
}

/// Kernel draws owed to the preset's allocator under a floored bandit.
#[derive(Debug, Clone)]
pub struct KernelFloor {
    owed: Vec<f64>,
    drawn: Vec<f64>,
}

impl KernelFloor {
    /// A floor over `kernels` kernels, nothing owed.
    pub fn new(kernels: usize) -> Self {
        Self {
            owed: vec![0.0; kernels],
            drawn: vec![0.0; kernels],
        }
    }

    /// The preset's allocator would have drawn `kernel` on this hop.
    pub fn owe(&mut self, kernel: usize) {
        if let Some(owed) = self.owed.get_mut(kernel) {
            *owed += FLOOR_SHARE;
        }
    }

    /// The kernel most behind its floor, when one is a whole draw behind.
    pub fn due(&self) -> Option<usize> {
        let mut due = None;
        let mut most = 1.0 - 1e-12;
        for (kernel, (&owed, &drawn)) in self.owed.iter().zip(&self.drawn).enumerate() {
            if owed - drawn >= most {
                most = owed - drawn;
                due = Some(kernel);
            }
        }
        due
    }

    /// `kernel` was drawn on this hop.
    pub fn drew(&mut self, kernel: usize) {
        if let Some(drawn) = self.drawn.get_mut(kernel) {
            *drawn += 1.0;
        }
    }
}

fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2: f64 = rng.random::<f64>();
    super::portable_sqrt(-2.0 * super::portable_ln(u1))
        * super::portable_cos(std::f64::consts::TAU * u2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn arms() -> Vec<BanditArm> {
        vec![
            BanditArm::Kernel(0),
            BanditArm::Kernel(1),
            BanditArm::Reseed,
        ]
    }

    #[test]
    fn untried_arms_tie_and_are_drawn_evenly() {
        for rule in [Rule::Thompson, Rule::Ucb] {
            let bandit = DiscountedBandit::new(arms(), rule);
            let mut rng = StdRng::seed_from_u64(7);
            let mut seen = [0usize; 3];
            for _ in 0..3000 {
                seen[bandit.select(&|_| true, &mut rng)] += 1;
            }
            assert!(seen.iter().all(|&c| c > 800), "{rule:?} {seen:?}");
        }
    }

    #[test]
    fn the_arm_that_buys_records_cheaply_takes_the_draw() {
        for rule in [Rule::Thompson, Rule::Ucb] {
            let mut bandit = DiscountedBandit::new(arms(), rule);
            let mut rng = StdRng::seed_from_u64(11);
            let mut spent = 0usize;
            let mut best = 0.0;
            let mut picks = [0usize; 3];
            for _ in 0..2000 {
                bandit.settle(spent, best);
                let arm = bandit.choose(spent, best, |_| true, &mut rng);
                spent += 100;
                if arm == BanditArm::Kernel(1) && rng.random::<f64>() < 0.3 {
                    best -= 0.01;
                }
                let index = bandit.arms().iter().position(|a| *a == arm).unwrap();
                picks[index] += 1;
            }
            assert!(picks[1] > picks[0] + picks[2], "{rule:?} {picks:?}");
        }
    }

    #[test]
    fn discounting_forgets_an_arm_that_stopped_paying() {
        let mut bandit = DiscountedBandit::new(arms(), Rule::Thompson);
        for _ in 0..50 {
            bandit.update(0, 1.0);
        }
        for _ in 0..3000 {
            bandit.update(1, 0.0);
        }
        assert!(bandit.count[0] < 1e-3);
    }

    #[test]
    fn the_record_reward_counts_a_record_whatever_its_size() {
        let mut bandit = DiscountedBandit::new(arms(), Rule::Thompson).with_reward(Reward::Record);
        let mut rng = StdRng::seed_from_u64(5);
        bandit.choose(0, 0.0, |a| a == BanditArm::Kernel(0), &mut rng);
        bandit.settle(10, -1e-9);
        bandit.choose(10, -1e-9, |a| a == BanditArm::Kernel(1), &mut rng);
        bandit.settle(20, -100.0);
        bandit.choose(20, -100.0, |a| a == BanditArm::Reseed, &mut rng);
        bandit.settle(30, -100.0);
        assert!((bandit.sum[0] / bandit.count[0] - 0.1).abs() < 1e-12);
        assert!((bandit.sum[1] / bandit.count[1] - 0.1).abs() < 1e-12);
        assert_eq!(bandit.sum[2], 0.0);
    }

    #[test]
    fn a_reset_forgets_the_sums_and_keeps_the_pending_decision() {
        let mut bandit = DiscountedBandit::new(arms(), Rule::Ucb);
        let mut rng = StdRng::seed_from_u64(5);
        bandit.update(0, 1.0);
        bandit.choose(0, 0.0, |a| a == BanditArm::Reseed, &mut rng);
        bandit.reset();
        assert!(bandit.count.iter().all(|&c| c == 0.0));
        bandit.settle(10, -1.0);
        assert!(bandit.count[2] > 0.0);
        assert_eq!(bandit.plays()[2], 1);
    }

    #[test]
    fn the_floor_keeps_half_of_each_kernels_draws() {
        let mut floor = KernelFloor::new(3);
        let mut drawn = [0usize; 3];
        for hop in 0..3000 {
            let preset = hop % 3;
            floor.owe(preset);
            let kernel = floor.due().unwrap_or(0);
            floor.drew(kernel);
            drawn[kernel] += 1;
        }
        assert!(drawn[1] >= 499 && drawn[2] >= 499, "{drawn:?}");
        assert!(drawn[0] >= 1000, "{drawn:?}");
    }

    #[test]
    fn an_ineligible_arm_is_never_chosen() {
        let mut bandit = DiscountedBandit::new(arms(), Rule::Ucb);
        let mut rng = StdRng::seed_from_u64(3);
        for _ in 0..200 {
            bandit.settle(0, 0.0);
            let arm = bandit.choose(0, 0.0, |a| a != BanditArm::Reseed, &mut rng);
            assert_ne!(arm, BanditArm::Reseed);
        }
    }
}
