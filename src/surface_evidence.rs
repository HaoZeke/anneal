//! Replay-safe sharing of relaxation-surface rewards between independent chains.
//!
//! Reports contain a producer's cumulative observations, never imported evidence.
//! A reply excludes the requesting producer so local observations can continue
//! during communication without being lost or counted twice.

use std::collections::BTreeMap;

use crate::allocate::{DepthAllocator, RewardMoments};

/// Cumulative arm rewards under an exact surface and reward configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceReport {
    /// Versioned reward definition, ordered transforms, and block length.
    pub schema: String,
    /// Independent reward moments for each arm, with the plain surface first.
    pub arms: Vec<RewardMoments>,
}

impl SurfaceReport {
    /// Validate a bounded report before using it to influence search.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.schema.is_empty()
            || self.schema.len() > 8192
            || self.arms.is_empty()
            || self.arms.len() > 64
        {
            return Err("invalid surface evidence schema");
        }
        DepthAllocator::from_moments(&self.arms).map(|_| ())
    }
}

/// Cumulative snapshots per experiment and producer, scoped to one system.
#[derive(Debug, Default, Clone)]
pub struct SurfaceEvidenceBook {
    reports: BTreeMap<String, BTreeMap<u32, Vec<RewardMoments>>>,
}

impl SurfaceEvidenceBook {
    /// Replace the producer's snapshot and return only other producers' evidence.
    ///
    /// Identical reports are idempotent. Regressing counters, changed moments at
    /// an unchanged counter, and incompatible arm counts cannot mutate the book.
    pub fn exchange(
        &mut self,
        producer: u32,
        report: SurfaceReport,
    ) -> Result<SurfaceReport, &'static str> {
        report.validate()?;
        if !self.reports.contains_key(&report.schema) && self.reports.len() >= 128 {
            return Err("surface experiment capacity exhausted");
        }
        let mut peers = vec![RewardMoments::default(); report.arms.len()];
        if let Some(experiment) = self.reports.get(&report.schema) {
            for (&replica, arms) in experiment {
                if arms.len() != report.arms.len() {
                    return Err("surface arm count mismatch");
                }
                if replica == producer {
                    for (old, new) in arms.iter().zip(&report.arms) {
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
        let reply = SurfaceReport {
            schema: report.schema.clone(),
            arms: peers,
        };
        reply.validate()?;
        let aggregate = reply
            .arms
            .iter()
            .zip(&report.arms)
            .map(|(peer, own)| peer.merge(*own))
            .collect::<Result<Vec<_>, _>>()?;
        DepthAllocator::from_moments(&aggregate)?;
        self.reports
            .entry(report.schema)
            .or_default()
            .insert(producer, report.arms);
        Ok(reply)
    }
}
