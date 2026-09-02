//! Successive-halving population control for a live coordinator roster.
//!
//! Rungs are \(r_0 \eta^i\). Replicas that have charged at least the current
//! rung are ranked by best energy; the bottom \(1 - 1/\eta\) are retired and
//! the same number of spawns is requested. A manual target emits the retire
//! or spawn decisions that reach that live count.

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

/// Invalid successive-halving parameters.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum ScalingError {
    /// The first rung must be a positive charged-work count.
    #[error("successive-halving r0 must be positive, got {0}")]
    InvalidR0(u64),
    /// The reduction factor must be finite and strictly greater than one.
    #[error("successive-halving eta must be finite and greater than 1, got {0}")]
    InvalidEta(f64),
    /// The population target must be at least one replica.
    #[error("successive-halving population target must be positive")]
    InvalidTarget,
}

/// One decision emitted by [`SuccessiveHalving`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleDecision {
    /// Remove this replica from the live roster.
    Retire(u32),
    /// Request this many new workers.
    Spawn(u32),
}

/// Resource-rung successive halving over live replicas.
#[derive(Clone, Debug)]
pub struct SuccessiveHalving {
    r0: u64,
    eta: f64,
    population_target: u32,
    charged: BTreeMap<u32, u64>,
    energy: BTreeMap<u32, f64>,
    live: BTreeSet<u32>,
    retired: BTreeSet<u32>,
    evaluated: BTreeSet<u32>,
}

impl SuccessiveHalving {
    /// Construct a policy with first rung `r0`, reduction `eta`, and a
    /// cohort size that must cross a rung before it is scored.
    pub fn new(r0: u64, eta: f64, population_target: u32) -> Result<Self, ScalingError> {
        if r0 == 0 {
            return Err(ScalingError::InvalidR0(r0));
        }
        if !eta.is_finite() || eta <= 1.0 {
            return Err(ScalingError::InvalidEta(eta));
        }
        if population_target == 0 {
            return Err(ScalingError::InvalidTarget);
        }
        Ok(Self {
            r0,
            eta,
            population_target,
            charged: BTreeMap::new(),
            energy: BTreeMap::new(),
            live: BTreeSet::new(),
            retired: BTreeSet::new(),
            evaluated: BTreeSet::new(),
        })
    }

    /// Charged-work threshold of rung `index`, \(r_0 \eta^{\mathrm{index}}\).
    pub fn rung(&self, index: u32) -> Option<u64> {
        let value = (self.r0 as f64) * self.eta.powi(index as i32);
        if !value.is_finite() || value < 1.0 || value > u64::MAX as f64 {
            None
        } else {
            Some(value.round() as u64)
        }
    }

    /// Record one replica's charged work and best energy, emitting retire
    /// and spawn decisions for every newly completed rung.
    pub fn observe(
        &mut self,
        replica: u32,
        charged_work: u64,
        best_energy: f64,
    ) -> Vec<ScaleDecision> {
        if self.retired.contains(&replica) {
            return Vec::new();
        }
        self.live.insert(replica);
        self.charged.insert(replica, charged_work);
        if best_energy.is_finite() {
            let stored = self.energy.entry(replica).or_insert(f64::INFINITY);
            if best_energy < *stored {
                *stored = best_energy;
            }
        }
        let mut decisions = Vec::new();
        for index in 0..64 {
            let Some(threshold) = self.rung(index) else {
                break;
            };
            if self.evaluated.contains(&index) {
                continue;
            }
            let crossed = self.crossed(threshold);
            let needed = (self.population_target as usize).max(2);
            if crossed.len() < needed {
                if charged_work < threshold {
                    break;
                }
                continue;
            }
            self.evaluated.insert(index);
            decisions.extend(self.retire_bottom(&crossed));
        }
        decisions
    }

    /// Emit retires or a spawn so the live set reaches `live`.
    pub fn set_target(&mut self, live: u32) -> Vec<ScaleDecision> {
        let current = u32::try_from(self.live.len()).unwrap_or(u32::MAX);
        if live < current {
            let mut ranked = self.rank(self.live.iter().copied());
            let drop = (current - live) as usize;
            let start = ranked.len().saturating_sub(drop);
            let victims = ranked.split_off(start);
            self.retire_ids(&victims)
        } else if live > current {
            vec![ScaleDecision::Spawn(live - current)]
        } else {
            Vec::new()
        }
    }

    fn crossed(&self, threshold: u64) -> Vec<u32> {
        self.live
            .iter()
            .copied()
            .filter(|replica| self.charged.get(replica).copied().unwrap_or(0) >= threshold)
            .collect()
    }

    fn rank(&self, replicas: impl IntoIterator<Item = u32>) -> Vec<u32> {
        let mut ranked = replicas.into_iter().collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            let left_energy = self.energy.get(left).copied().unwrap_or(f64::INFINITY);
            let right_energy = self.energy.get(right).copied().unwrap_or(f64::INFINITY);
            left_energy
                .partial_cmp(&right_energy)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.cmp(right))
        });
        ranked
    }

    fn retire_bottom(&mut self, crossed: &[u32]) -> Vec<ScaleDecision> {
        let n = crossed.len();
        let retire_count = ((n as f64) * (1.0 - 1.0 / self.eta)).floor() as usize;
        if retire_count == 0 {
            return Vec::new();
        }
        let ranked = self.rank(crossed.iter().copied());
        let start = ranked.len().saturating_sub(retire_count);
        let mut decisions = self.retire_ids(&ranked[start..]);
        let spawned = u32::try_from(decisions.len()).unwrap_or(0);
        if spawned > 0 {
            decisions.push(ScaleDecision::Spawn(spawned));
        }
        decisions
    }

    fn retire_ids(&mut self, ids: &[u32]) -> Vec<ScaleDecision> {
        let mut decisions = Vec::new();
        for &replica in ids {
            if self.retired.insert(replica) {
                self.live.remove(&replica);
                decisions.push(ScaleDecision::Retire(replica));
            }
        }
        decisions
    }
}

#[cfg(test)]
mod tests {
    use super::{ScaleDecision, ScalingError, SuccessiveHalving};

    #[test]
    fn rungs_are_r0_times_eta_to_the_i() {
        let policy = SuccessiveHalving::new(10, 2.0, 3).unwrap();
        assert_eq!(policy.rung(0), Some(10));
        assert_eq!(policy.rung(1), Some(20));
        assert_eq!(policy.rung(2), Some(40));
    }

    #[test]
    fn first_rung_retires_the_worst_of_three_and_requests_one_spawn() {
        let mut policy = SuccessiveHalving::new(10, 2.0, 3).unwrap();
        assert!(policy.observe(0, 10, -3.0).is_empty());
        assert!(policy.observe(1, 10, -2.0).is_empty());
        assert_eq!(
            policy.observe(2, 10, -1.0),
            vec![ScaleDecision::Retire(2), ScaleDecision::Spawn(1)]
        );
    }

    #[test]
    fn set_target_emits_retires_or_spawns_to_reach_live() {
        let mut policy = SuccessiveHalving::new(10, 2.0, 3).unwrap();
        policy.observe(0, 1, -3.0);
        policy.observe(1, 1, -2.0);
        policy.observe(2, 1, -1.0);
        assert_eq!(policy.set_target(2), vec![ScaleDecision::Retire(2)]);
        assert_eq!(policy.set_target(4), vec![ScaleDecision::Spawn(2)]);
    }

    #[test]
    fn work_below_the_first_rung_emits_nothing() {
        let mut policy = SuccessiveHalving::new(10, 2.0, 3).unwrap();
        assert!(policy.observe(0, 9, -1.0).is_empty());
        assert!(policy.observe(1, 9, -2.0).is_empty());
        assert!(policy.observe(2, 9, -3.0).is_empty());
    }

    #[test]
    fn constructor_rejects_invalid_parameters() {
        assert_eq!(
            SuccessiveHalving::new(0, 2.0, 3).unwrap_err(),
            ScalingError::InvalidR0(0)
        );
        assert_eq!(
            SuccessiveHalving::new(10, 1.0, 3).unwrap_err(),
            ScalingError::InvalidEta(1.0)
        );
        assert_eq!(
            SuccessiveHalving::new(10, 2.0, 0).unwrap_err(),
            ScalingError::InvalidTarget
        );
    }
}
