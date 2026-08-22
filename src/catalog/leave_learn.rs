//! Leave cover and action credit: FunnelModel EI plus Thompson on what paid.
//!
//! Occupancy already fits [`crate::funnel_bo::FunnelModel`] on packing
//! histograms and uses it to retire. That model never chose the next
//! Leave: cover indices were `replica + leave * wave`, and a
//! one-community book walked rather than proposed. Expected
//! improvement is how a search reaches a funnel it has no evidence
//! about (Jones, Schonlau & Welch, *J. Global Optim.* 1998). Thompson
//! on the covers and actions that later dropped the energy is how the
//! same replica stops repeating a hole that quenched back to ico.
//!
//! Arms of the cover bandit are the SoftSaddle covering points plus one
//! fivefold residual. Arms of the action bandit are local walk, explore,
//! and Leave.

use ndarray::Array1;
use rand::Rng;

use crate::catalog::archive::Curiosity;
use crate::funnel_bo::FunnelModel;
use crate::hypersphere;

/// Covering points plus the fivefold residual as the last arm.
pub fn cover_arm_count() -> usize {
    hypersphere::default_cover_size().saturating_add(1)
}

/// Arm index of the fivefold residual, after the covering points.
pub fn fivefold_arm() -> usize {
    hypersphere::default_cover_size()
}

/// Action bandit: keep walking the live minimum.
pub const ACTION_LOCAL: usize = 0;
/// Action bandit: descriptor-space explore.
pub const ACTION_EXPLORE: usize = 1;
/// Action bandit: Leave the occupied packing.
pub const ACTION_LEAVE: usize = 2;

/// Proposals scored by EI before one of them is quenched.
pub const LEAVE_EI_PROBES: usize = 8;

/// Shared learner for one replica (thread-local: one hop chain).
pub struct LeaveLearner {
    /// Lowest energy per packing histogram.
    pub funnel: FunnelModel,
    covers: Curiosity,
    actions: Curiosity,
    /// Best energy this replica has quenched to.
    pub best: f64,
}

impl LeaveLearner {
    /// Uniform priors, empty funnel.
    pub fn new() -> Self {
        let covers = cover_arm_count();
        Self {
            funnel: FunnelModel::new(0.15, 20.0, 1e-2),
            covers: Curiosity::new(covers.max(1)),
            actions: Curiosity::new(3),
            best: f64::INFINITY,
        }
    }

    /// Record a quenched landing.
    ///
    /// The funnel sees the packing histogram and the energy. The cover
    /// arm is rewarded only when the landing beats this replica's best:
    /// leaving ico for a \(-380\) defect is coverage, not improvement,
    /// and must not crowd out the next shot.
    pub fn observe(&mut self, histogram: &[f64], energy: f64, cover: Option<usize>) {
        if energy.is_finite()
            && !histogram.is_empty()
            && histogram.iter().all(|value| value.is_finite())
        {
            self.funnel
                .observe(Array1::from(histogram.to_vec()).view(), energy);
        }
        let improved = energy.is_finite() && energy < self.best - 1e-6;
        if improved {
            self.best = energy;
        }
        if let Some(cover) = cover {
            self.covers.ensure(cover + 1);
            if improved {
                self.covers.reward(cover);
            } else {
                self.covers.penalise(cover);
            }
        }
    }

    /// Credit an action after the slice that used it.
    pub fn credit_action(&mut self, action: usize, improved: bool) {
        self.actions.ensure(action + 1);
        if improved {
            self.actions.reward(action);
        } else {
            self.actions.penalise(action);
        }
    }

    /// Thompson draw over `n` cover arms (fivefold is `n-1` when
    /// `n == cover_arm_count()`).
    pub fn pick_cover<R: Rng + ?Sized>(&mut self, n: usize, rng: &mut R) -> usize {
        let n = n.max(1);
        self.covers.ensure(n);
        let allowed: Vec<usize> = (0..n).collect();
        self.covers.select(&allowed, rng).unwrap_or(0)
    }

    /// Thompson draw over the allowed policy actions.
    pub fn pick_action<R: Rng + ?Sized>(&mut self, allowed: &[usize], rng: &mut R) -> usize {
        if allowed.is_empty() {
            return ACTION_LOCAL;
        }
        self.actions
            .ensure(allowed.iter().copied().max().unwrap_or(0) + 1);
        self.actions.select(allowed, rng).unwrap_or(allowed[0])
    }

    /// Cover whose packing histogram the funnel rates highest under MES.
    ///
    /// `None` when the funnel has fewer than two observations, so the
    /// first Leaves are Thompson rather than a prior that has never
    /// seen a packing. Jones EI stays on retire; this ranks holes by
    /// \(I(E^\star; y)\).
    pub fn pick_cover_ei(&mut self, candidates: &[(usize, Vec<f64>)]) -> Option<usize> {
        if candidates.is_empty() || self.funnel.len() < 2 {
            return None;
        }
        let extras: Vec<Array1<f64>> = candidates
            .iter()
            .filter(|(_, histogram)| !histogram.is_empty())
            .map(|(_, histogram)| Array1::from(histogram.clone()))
            .collect();
        if extras.is_empty() {
            return None;
        }
        let views: Vec<ndarray::ArrayView1<f64>> = extras.iter().map(|x| x.view()).collect();
        let minima = self.funnel.sample_minima(&views, 16);
        let mut best: Option<(f64, usize)> = None;
        for (cover, histogram) in candidates {
            if histogram.is_empty() {
                continue;
            }
            let mes = self
                .funnel
                .max_value_entropy(Array1::from(histogram.clone()).view(), &minima);
            if mes.is_finite() && best.as_ref().is_none_or(|(held, _)| mes > *held) {
                best = Some((mes, *cover));
            }
        }
        best.map(|(_, cover)| cover)
    }
}

impl Default for LeaveLearner {
    fn default() -> Self {
        Self::new()
    }
}

std::thread_local! {
    static LEARNER: std::cell::RefCell<LeaveLearner> = std::cell::RefCell::new(LeaveLearner::new());
}

/// Observe a quenched Leave on this replica's learner.
pub fn observe_leave(histogram: &[f64], energy: f64, cover: Option<usize>) {
    LEARNER.with(|slot| slot.borrow_mut().observe(histogram, energy, cover));
}

/// Credit the policy action this slice took.
pub fn credit_action(action: usize, improved: bool) {
    LEARNER.with(|slot| slot.borrow_mut().credit_action(action, improved));
}

/// Thompson cover for a Leave that has no EI candidates yet.
pub fn pick_leave_cover<R: Rng + ?Sized>(n: usize, rng: &mut R) -> usize {
    LEARNER.with(|slot| slot.borrow_mut().pick_cover(n, rng))
}

/// EI cover if the funnel can score, otherwise Thompson.
pub fn pick_leave_cover_ei<R: Rng + ?Sized>(
    candidates: &[(usize, Vec<f64>)],
    n: usize,
    rng: &mut R,
) -> usize {
    LEARNER.with(|slot| {
        let mut held = slot.borrow_mut();
        held.pick_cover_ei(candidates)
            .unwrap_or_else(|| held.pick_cover(n, rng))
    })
}

/// Thompson action among `allowed`.
pub fn pick_leave_action<R: Rng + ?Sized>(allowed: &[usize], rng: &mut R) -> usize {
    LEARNER.with(|slot| slot.borrow_mut().pick_action(allowed, rng))
}

/// Whether this replica's funnel still expects improvement.
pub fn leave_ei_open() -> bool {
    LEARNER.with(|slot| {
        let mut held = slot.borrow_mut();
        !crate::catalog::occupancy_ei_exhausted(
            held.funnel.max_expected_improvement_at_data(),
            held.funnel.len(),
            held.funnel.noise,
        )
    })
}

/// Snapshot of the best energy the learner has been shown.
pub fn leave_best() -> f64 {
    LEARNER.with(|slot| slot.borrow().best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn ei_prefers_an_unvisited_histogram() {
        let mut learner = LeaveLearner::new();
        let ico = vec![1.0, 0.0, 0.0];
        let junk = vec![0.0, 1.0, 0.0];
        let unseen = vec![0.0, 0.0, 1.0];
        learner.observe(&ico, -396.28, Some(0));
        learner.observe(&junk, -380.0, Some(1));
        let picked = learner
            .pick_cover_ei(&[(0, ico), (1, junk), (2, unseen)])
            .expect("funnel with two observations scores");
        assert_eq!(picked, 2, "unvisited morphology has the MES score");
    }

    #[test]
    fn only_an_improvement_rewards_the_cover() {
        let mut learner = LeaveLearner::new();
        learner.observe(&[1.0], -396.28, Some(0));
        learner.observe(&[0.0, 1.0], -380.0, Some(1));
        assert!(
            learner.covers.score(0) > learner.covers.score(1),
            "a worse landing must not outvote the well it left"
        );
    }

    #[test]
    fn thompson_cover_stays_inside_the_arm_count() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut learner = LeaveLearner::new();
        for _ in 0..32 {
            let cover = learner.pick_cover(12, &mut rng);
            assert!(cover < 12, "cover {cover}");
        }
    }
}
