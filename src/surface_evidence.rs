//! Transferable relaxation-surface rewards keyed by the measured source.
//!
//! Moments pool only inside one source descriptor region and one declared
//! proposal and quench condition. Descriptor schema, quench schema, the
//! entering incumbent gap, and charged block work delimit that transfer.
//! A perturbed relaxation input is not a source. A checkpoint whose block
//! interval differs from the held block is not a source. Peer replies keep
//! the key they were credited under.

use std::collections::BTreeMap;

use crate::allocate::{DepthAllocator, RewardMoments};

/// Observations below this count stay on the uninformative local prior.
pub const MIN_TRANSFER_OBSERVATIONS: u64 = 8;

/// Measured source environment and the declared proposal and quench conditions.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceTransferKey {
    /// Descriptor schema name for the occupied validated minimum.
    pub descriptor_schema: String,
    /// Descriptor schema version.
    pub descriptor_version: u32,
    /// Attraction region of the occupied validated source.
    pub region: u64,
    /// Declared proposal condition.
    pub proposal: String,
    /// Declared quench schema.
    pub quench_schema: String,
    /// Declared hops per held block. A different interval is a different key.
    pub block: usize,
}

/// One producer's cumulative arm rewards for a single source key.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceEvidenceMessage {
    /// Producer that credited the block. Replies do not rewrite this.
    pub producer: u32,
    /// Source captured when the held block opened.
    pub key: SourceTransferKey,
    /// Cumulative moments per arm, plain surface first.
    pub arms: Vec<RewardMoments>,
    /// Entering incumbent gap carried with the block.
    pub incumbent_gap: f64,
    /// Charged objective work of the block. Zero is unresolved.
    pub charged_work: u64,
}

impl SurfaceEvidenceMessage {
    /// Schema, gap, and charged work must all be present before transfer.
    pub fn validate(&self, n_arms: usize) -> Result<(), &'static str> {
        if self.key.descriptor_schema.is_empty()
            || self.key.descriptor_version == 0
            || self.key.proposal.is_empty()
            || self.key.quench_schema.is_empty()
            || self.key.block == 0
            || self.arms.len() != n_arms
            || self.charged_work == 0
            || !self.incumbent_gap.is_finite()
        {
            return Err("surface evidence is unresolved or undeclared");
        }
        for arm in &self.arms {
            arm.validate()?;
        }
        Ok(())
    }
}

/// Cumulative snapshots per source key and producer.
#[derive(Debug, Clone)]
pub struct SurfaceEvidenceBook {
    n_arms: usize,
    reports: BTreeMap<SourceTransferKey, BTreeMap<u32, Vec<RewardMoments>>>,
}

impl SurfaceEvidenceBook {
    /// Empty book for `n_arms` surfaces, plain arm first.
    pub fn new(n_arms: usize) -> Self {
        Self {
            n_arms,
            reports: BTreeMap::new(),
        }
    }

    /// Number of arms.
    pub fn arms(&self) -> usize {
        self.n_arms
    }

    /// The occupied validated source, never the perturbed relaxation input.
    ///
    /// A checkpoint is that source only when its interval is the declared block.
    pub fn attribute_source<'a>(
        occupied: &'a SourceTransferKey,
        _perturbed: &SourceTransferKey,
        checkpoint: Option<(usize, &SourceTransferKey)>,
    ) -> Result<&'a SourceTransferKey, &'static str> {
        if occupied.descriptor_schema.is_empty()
            || occupied.descriptor_version == 0
            || occupied.proposal.is_empty()
            || occupied.quench_schema.is_empty()
            || occupied.block == 0
        {
            return Err("occupied source is unresolved");
        }
        if let Some((interval, _)) = checkpoint {
            if interval != occupied.block {
                return Ok(occupied);
            }
        }
        Ok(occupied)
    }

    /// Record one finite rewarded block under the source key it names.
    pub fn observe(
        &mut self,
        producer: u32,
        key: &SourceTransferKey,
        arm: usize,
        reward: f64,
        incumbent_gap: f64,
        charged_work: u64,
    ) -> Result<(), &'static str> {
        if arm >= self.n_arms || !reward.is_finite() {
            return Err("unresolved surface reward");
        }
        let mut arms = self
            .reports
            .get(key)
            .and_then(|producers| producers.get(&producer))
            .cloned()
            .unwrap_or_else(|| vec![RewardMoments::default(); self.n_arms]);
        arms[arm].observe(reward)?;
        self.exchange(SurfaceEvidenceMessage {
            producer,
            key: key.clone(),
            arms,
            incumbent_gap,
            charged_work,
        })?;
        Ok(())
    }

    /// Replace this producer's snapshot and return other producers on the same key.
    ///
    /// The reply keeps `message.key`. The caller's current region is not read.
    pub fn exchange(
        &mut self,
        message: SurfaceEvidenceMessage,
    ) -> Result<SurfaceEvidenceMessage, &'static str> {
        message.validate(self.n_arms)?;
        let mut peers = vec![RewardMoments::default(); self.n_arms];
        if let Some(producers) = self.reports.get(&message.key) {
            for (&producer, arms) in producers {
                if producer == message.producer {
                    for (old, new) in arms.iter().zip(&message.arms) {
                        if new.count < old.count || (new.count == old.count && new != old) {
                            return Err("surface evidence regressed or changed on replay");
                        }
                    }
                } else {
                    for (peer, arm) in peers.iter_mut().zip(arms) {
                        *peer = peer.merge(*arm)?;
                    }
                }
            }
        }
        self.reports
            .entry(message.key.clone())
            .or_default()
            .insert(message.producer, message.arms);
        Ok(SurfaceEvidenceMessage {
            producer: message.producer,
            key: message.key,
            arms: peers,
            incumbent_gap: message.incumbent_gap,
            charged_work: message.charged_work,
        })
    }

    /// Merged moments for one source. Missing keys are uninformative.
    pub fn moments(&self, key: &SourceTransferKey) -> Vec<RewardMoments> {
        let mut merged = vec![RewardMoments::default(); self.n_arms];
        let Some(producers) = self.reports.get(key) else {
            return merged;
        };
        for arms in producers.values() {
            for (slot, arm) in merged.iter_mut().zip(arms) {
                if let Ok(next) = slot.merge(*arm) {
                    *slot = next;
                }
            }
        }
        merged
    }

    /// Arithmetic means for one source, or `None` while that context is sparse.
    pub fn means(&self, key: &SourceTransferKey) -> Option<Vec<f64>> {
        let moments = self.moments(key);
        let total = moments.iter().map(|arm| arm.count).sum::<u64>();
        if total < MIN_TRANSFER_OBSERVATIONS {
            return None;
        }
        Some(moments.iter().map(|arm| arm.mean).collect())
    }

    /// Posterior used for transfer. Sparse and unresolved keys stay uninformative.
    pub fn decision_allocator(&self, key: &SourceTransferKey) -> Option<DepthAllocator> {
        let moments = self.moments(key);
        if moments.iter().map(|arm| arm.count).sum::<u64>() < MIN_TRANSFER_OBSERVATIONS {
            return None;
        }
        DepthAllocator::from_moments(&moments).ok()
    }

    /// Means pooled across regions that share proposal, quench, and block.
    ///
    /// This is the unconditioned aggregate the source key exists to avoid.
    pub fn pooled_means(
        &self,
        descriptor_schema: &str,
        descriptor_version: u32,
        proposal: &str,
        quench_schema: &str,
        block: usize,
    ) -> Vec<f64> {
        let mut merged = vec![RewardMoments::default(); self.n_arms];
        for (key, producers) in &self.reports {
            if key.descriptor_schema != descriptor_schema
                || key.descriptor_version != descriptor_version
                || key.proposal != proposal
                || key.quench_schema != quench_schema
                || key.block != block
            {
                continue;
            }
            for arms in producers.values() {
                for (slot, arm) in merged.iter_mut().zip(arms) {
                    if let Ok(next) = slot.merge(*arm) {
                        *slot = next;
                    }
                }
            }
        }
        merged.iter().map(|arm| arm.mean).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(region: u64) -> SourceTransferKey {
        SourceTransferKey {
            descriptor_schema: "universal".into(),
            descriptor_version: 1,
            region,
            proposal: "hop".into(),
            quench_schema: "lbfgs".into(),
            block: 100,
        }
    }

    fn constant(count: u64, mean: f64) -> RewardMoments {
        RewardMoments {
            count,
            mean,
            m2: 0.0,
        }
    }

    #[test]
    fn source_transfer_key_prefers_plain_where_pooled_moments_favor_the_transform() {
        let mut book = SurfaceEvidenceBook::new(2);
        let region_a = key(1);
        let region_b = key(2);
        let perturbed = key(9);
        let occupied =
            SurfaceEvidenceBook::attribute_source(&region_a, &perturbed, Some((40, &region_b)))
                .unwrap();
        assert_eq!(occupied, &region_a);

        book.exchange(SurfaceEvidenceMessage {
            producer: 1,
            key: region_a.clone(),
            arms: vec![constant(80, 0.0), constant(80, 1.0)],
            incumbent_gap: 1.0,
            charged_work: 80,
        })
        .unwrap();
        book.exchange(SurfaceEvidenceMessage {
            producer: 2,
            key: region_b.clone(),
            arms: vec![constant(20, 1.0), constant(20, -1.0)],
            incumbent_gap: -1.0,
            charged_work: 20,
        })
        .unwrap();

        let pooled = book.pooled_means("universal", 1, "hop", "lbfgs", 100);
        assert!((pooled[0] - 0.2).abs() < 1e-12, "{pooled:?}");
        assert!((pooled[1] - 0.6).abs() < 1e-12, "{pooled:?}");
        assert!(pooled[1] > pooled[0]);

        let in_b = book.means(&region_b).unwrap();
        assert!((in_b[0] - 1.0).abs() < 1e-12, "{in_b:?}");
        assert!((in_b[1] - (-1.0)).abs() < 1e-12, "{in_b:?}");
        assert!(in_b[0] > in_b[1]);

        let reply = book
            .exchange(SurfaceEvidenceMessage {
                producer: 3,
                key: region_a.clone(),
                arms: vec![constant(80, 0.0), constant(80, 1.0)],
                incumbent_gap: 1.0,
                charged_work: 80,
            })
            .unwrap();
        assert_eq!(reply.key.region, region_a.region);
        assert_eq!(book.means(&region_b).unwrap(), in_b);

        let unresolved = SurfaceEvidenceMessage {
            producer: 4,
            key: region_b.clone(),
            arms: vec![constant(4, 5.0), constant(4, 5.0)],
            incumbent_gap: f64::NAN,
            charged_work: 0,
        };
        assert!(book.exchange(unresolved).is_err());
        assert_eq!(book.means(&region_b).unwrap(), in_b);
        assert!(book.means(&key(3)).is_none());
    }
}
