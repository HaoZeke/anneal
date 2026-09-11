//! Per-chain exploration scales learned from raw uphill energy differences.

use crate::bias::AdaptiveHeight;

pub(super) struct Temperatures {
    gaps: Vec<Option<AdaptiveHeight>>,
}

impl Temperatures {
    pub(super) fn new(replicas: usize) -> Self {
        Self {
            gaps: (0..replicas).map(|_| None).collect(),
        }
    }

    /// One scale feeds deposit tempering, acceptance and Langevin excitation.
    /// The objective's arbitrary additive constant supplies no scale evidence.
    pub(super) fn at(&self, replica: usize, generation: usize) -> f64 {
        let scale = self.gaps[replica]
            .as_ref()
            .map_or(1.0, AdaptiveHeight::gap_estimate)
            .max(1e-12);
        scale * 5.0 * std::f64::consts::LN_2 / (generation as f64 + 1.0).ln().max(1e-12)
    }

    /// Every finite uphill trial supplies evidence, including rejected trials.
    /// Learning occurs after the boundary's acceptance and coverage deposit.
    pub(super) fn observe(&mut self, replica: usize, occupied: f64, trial: f64) {
        let gap = trial - occupied;
        if !gap.is_finite() || gap <= 0.0 {
            return;
        }
        let estimate = self.gaps[replica].get_or_insert_with(|| AdaptiveHeight::new(0.5, 1.0, gap));
        estimate.observe(gap);
    }
}
