//! Borrowed and frozen forms of the same sampled-position correction.

use ndarray::{Array1, ArrayView1};
use rand::Rng;

use crate::bias::{BasinMetric, EuclideanMetric, Fingerprint};

use super::super::repulsion::{PeerSamples, Separation};
use super::NormalizedCoordinates;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RepulsionStats {
    pub(crate) sample_peer_checks: usize,
    pub(crate) sample_anchor_overlaps: usize,
    pub(crate) sample_anchor_only_overlaps: usize,
    pub(crate) sample_overlaps: usize,
    pub(crate) repelled_proposals: usize,
    pub(crate) constrained_repulsions: usize,
}

pub(super) struct Field<'a> {
    pub(super) coordinates: &'a NormalizedCoordinates,
    pub(super) peers: &'a PeerSamples,
    pub(super) radius: f64,
    pub(super) weight: f64,
    pub(super) enabled: bool,
}

/// Owned geometry contains no exchange handle or synchronization primitive.
pub(crate) struct RepulsionSnapshot {
    coordinates: NormalizedCoordinates,
    peers: PeerSamples,
    radius: f64,
    weight: f64,
    enabled: bool,
    stats: RepulsionStats,
}

impl RepulsionSnapshot {
    pub(crate) fn repel<R: Rng + ?Sized>(
        &mut self,
        anchor: ArrayView1<f64>,
        proposal: &mut Array1<f64>,
        rng: &mut R,
    ) {
        Field {
            coordinates: &self.coordinates,
            peers: &self.peers,
            radius: self.radius,
            weight: self.weight,
            enabled: self.enabled,
        }
        .repel(anchor, proposal, rng, &mut self.stats);
    }

    pub(crate) fn take_stats(&mut self) -> RepulsionStats {
        std::mem::take(&mut self.stats)
    }
}

impl Field<'_> {
    pub(super) fn snapshot(&self) -> RepulsionSnapshot {
        RepulsionSnapshot {
            coordinates: self.coordinates.clone(),
            peers: self.peers.clone(),
            radius: self.radius,
            weight: self.weight,
            enabled: self.enabled,
            stats: RepulsionStats::default(),
        }
    }

    pub(super) fn repel<R: Rng + ?Sized>(
        &self,
        anchor: ArrayView1<f64>,
        proposal: &mut Array1<f64>,
        rng: &mut R,
        stats: &mut RepulsionStats,
    ) {
        if !self.enabled
            || !self.coordinates.feasible(proposal.view())
            || !self.coordinates.feasible(anchor)
        {
            return;
        }
        let point = self.coordinates.describe(proposal.view());
        let anchor = self.coordinates.describe(anchor);
        let Some((separation, anchor_overlaps)) = self.peers.separate(
            point.view(),
            anchor.view(),
            self.coordinates.widths.view(),
            self.coordinates.free_scale,
            self.radius,
            self.weight,
            rng,
        ) else {
            return;
        };
        stats.sample_peer_checks += 1;
        stats.sample_anchor_overlaps += usize::from(anchor_overlaps);
        match separation {
            Separation::Distant => {
                stats.sample_anchor_only_overlaps += usize::from(anchor_overlaps);
            }
            Separation::Constrained => {
                stats.sample_overlaps += 1;
                stats.constrained_repulsions += 1;
            }
            Separation::Moved(descriptor) => {
                stats.sample_overlaps += 1;
                if let Some(moved) =
                    self.physical_repulsion(proposal.view(), point.view(), descriptor.view())
                {
                    *proposal = moved;
                    stats.repelled_proposals += 1;
                } else {
                    stats.constrained_repulsions += 1;
                }
            }
        }
    }

    /// Validate clearance and the step cap in actual parameter coordinates.
    fn physical_repulsion(
        &self,
        proposal: ArrayView1<f64>,
        point: ArrayView1<f64>,
        descriptor: ArrayView1<f64>,
    ) -> Option<Array1<f64>> {
        let distance = self.peers.clearance(point)?;
        let increment = (self.radius - distance) * self.weight.min(1.0);
        let cap = increment + 16.0 * f64::EPSILON;
        let admissible = |candidate: ArrayView1<f64>| {
            let actual = self.coordinates.describe(candidate);
            EuclideanMetric.distance(actual.view(), point) <= cap
                && self
                    .peers
                    .clearance(actual.view())
                    .is_some_and(|d| d > distance)
        };
        let mut candidate = self.coordinates.position(descriptor);
        if admissible(candidate.view()) {
            return Some(candidate);
        }

        // Normalized corrections can round onto a different physical lattice
        // site. Poll physical coordinates without spending another callback or
        // changing the random stream, and validate against the entire cloud.
        candidate.assign(&proposal);
        for (j, &width) in self.coordinates.widths.iter().enumerate() {
            if width <= 0.0 {
                continue;
            }
            let step = (increment / self.coordinates.free_scale) * width;
            for sign in [-1.0, 1.0] {
                candidate[j] = (proposal[j] + sign * step)
                    .clamp(self.coordinates.low[j], self.coordinates.high[j]);
                let actual = self.coordinates.describe(candidate.view());
                if EuclideanMetric.distance(actual.view(), point) > cap {
                    candidate[j] = if candidate[j] > proposal[j] {
                        candidate[j].next_down().max(proposal[j])
                    } else {
                        candidate[j].next_up().min(proposal[j])
                    };
                }
                if admissible(candidate.view()) {
                    return Some(candidate);
                }
            }
            candidate[j] = proposal[j];
        }
        None
    }
}
