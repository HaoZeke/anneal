//! Coverage of evaluated search boundaries, independent of minimum identity.

use std::collections::HashMap;

use eindir_core::Bounds;
use ndarray::{Array1, ArrayView1};

use crate::bias::{BasinBias, Bias, Fingerprint};
use crate::methods::ensemble::HistoryMode;
use crate::shared_bias::SharedDeposits;

use super::BoxEnsembleConfig;

/// Repulsion over the evaluated regions of a finite box.
///
/// Coverage needs no gradient or stationary-point certificate. The descriptor
/// distance is the RMS coordinate displacement divided by each free
/// coordinate's box width; fixed coordinates contribute no distance.
#[derive(Clone, Debug, PartialEq)]
pub struct BoxCoverageConfig {
    /// Coverage-region radius in normalized RMS box distance, independent of
    /// [`BoxEnsembleConfig::identity_tol`].
    pub radius: f64,
    /// Initial deposit height in objective units; zero leaves acceptance unbiased.
    pub height: f64,
    /// Well-tempering factor, finite and greater than one.
    pub well_tempering: f64,
    /// Nonnegative multiplier for foreign deposits.
    pub peer_weight: f64,
    /// Publish local coverage to the other replicas, even without a minimum ledger.
    /// [`BoxEnsembleConfig::shared_deposits`] caps foreign visits per receiving
    /// region at each funded hop boundary; zero disables foreign deposits.
    pub shared: bool,
}

impl Default for BoxCoverageConfig {
    fn default() -> Self {
        Self {
            radius: 0.05,
            height: 0.1,
            well_tempering: 5.0,
            peer_weight: 1.0,
            shared: true,
        }
    }
}

impl BoxCoverageConfig {
    pub(super) fn for_ensemble(config: &BoxEnsembleConfig) -> Self {
        Self {
            shared: matches!(config.history, HistoryMode::Shared) && config.shared_deposits > 0,
            ..Self::default()
        }
    }

    fn validate(&self) {
        assert!(self.radius.is_finite() && self.radius > 0.0, "coverage radius must be finite and positive");
        assert!(self.height.is_finite() && self.height >= 0.0, "coverage height must be finite and nonnegative");
        assert!(self.well_tempering.is_finite() && self.well_tempering > 1.0, "coverage well-tempering must be finite and greater than one");
        assert!(self.peer_weight.is_finite() && self.peer_weight >= 0.0, "coverage peer weight must be finite and nonnegative");
    }
}

/// Search observations and their delivery, not a census of certified minima.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CoverageStats {
    /// Finite feasible initial boundaries and completed trial boundaries,
    /// including rejected trials. Inner line-search probes are not extra visits.
    pub local_observations: usize,
    /// Local visits placed on the in-process exchange, each published once.
    pub published_visits: u64,
    /// Foreign visits applied to recipient biases, excluding self-delivery.
    pub applied_foreign_visits: usize,
    /// Received foreign visits excluded by the per-region, per-boundary cap.
    pub capped_foreign_visits: u64,
    /// Regions held by each chain, including imported regions. These counts
    /// cannot be summed as an ensemble-wide exact-identity census.
    pub per_chain_regions: Vec<usize>,
}

#[derive(Clone)]
struct NormalizedCoordinates {
    low: Array1<f64>,
    high: Array1<f64>,
    widths: Array1<f64>,
    free_scale: f64,
}

impl NormalizedCoordinates {
    fn new(bounds: &Bounds<f64>) -> Self {
        let widths = &bounds.high - &bounds.low;
        assert!(bounds.low.iter().chain(bounds.high.iter()).all(|x| x.is_finite()), "coverage requires finite box bounds");
        assert!(widths.iter().all(|w| w.is_finite() && *w >= 0.0), "coverage requires finite nonnegative box widths");
        let free = widths.iter().filter(|w| **w > 0.0).count();
        Self {
            low: bounds.low.clone(),
            high: bounds.high.clone(),
            widths,
            free_scale: 1.0 / (free.max(1) as f64).sqrt(),
        }
    }

    fn feasible(&self, x: ArrayView1<f64>) -> bool {
        x.len() == self.widths.len()
            && x.iter().zip(self.low.iter().zip(self.high.iter())).all(|(x, (low, high))| x.is_finite() && x >= low && x <= high)
    }
}

impl Fingerprint for NormalizedCoordinates {
    fn describe(&self, x: ArrayView1<f64>) -> Array1<f64> {
        Array1::from_iter(x.iter().zip(self.low.iter().zip(self.widths.iter())).map(|(x, (low, width))| {
            if *width > 0.0 { ((x - low) / width) * self.free_scale } else { 0.0 }
        }))
    }
}

pub(super) struct Coverage {
    coordinates: NormalizedCoordinates,
    biases: Vec<BasinBias<NormalizedCoordinates>>,
    exchange: Option<SharedDeposits>,
    peer_weight: f64,
    foreign_cap: u64,
    stats: CoverageStats,
}

impl Coverage {
    pub(super) fn new(bounds: &Bounds<f64>, replicas: usize, config: &BoxCoverageConfig, foreign_cap: usize) -> Self {
        config.validate();
        let coordinates = NormalizedCoordinates::new(bounds);
        let biases = (0..replicas).map(|_| BasinBias::new(coordinates.clone(), config.radius, config.height, config.well_tempering)).collect();
        let exchange = (config.shared && config.peer_weight > 0.0 && foreign_cap > 0 && replicas > 1).then(|| SharedDeposits::new(replicas));
        Self { coordinates, biases, exchange, peer_weight: config.peer_weight, foreign_cap: foreign_cap as u64, stats: CoverageStats::default() }
    }

    pub(super) fn describe(&self, x: ArrayView1<f64>, value: f64) -> Option<Array1<f64>> {
        (value.is_finite() && self.coordinates.feasible(x)).then(|| self.coordinates.describe(x))
    }

    pub(super) fn potential(&self, replica: usize, descriptor: ArrayView1<f64>) -> f64 {
        self.biases[replica].potential(descriptor)
    }

    /// Read once at a funded hop boundary, before its acceptance decision.
    pub(super) fn hear(&mut self, replica: usize, temperature: f64) {
        let Some(exchange) = &mut self.exchange else { return; };
        let mut applied = HashMap::<usize, u64>::new();
        let bias = &mut self.biases[replica];
        for (descriptor, count) in exchange.drain(replica) {
            let region = bias.index().lookup(descriptor.view()).unwrap_or_else(|| bias.index().n_basins());
            let region_count = applied.entry(region).or_default();
            let admitted = count.min(self.foreign_cap.saturating_sub(*region_count));
            bias.deposit_scaled_n(descriptor.view(), temperature, self.peer_weight, admitted);
            *region_count += admitted;
            self.stats.applied_foreign_visits += admitted as usize;
            self.stats.capped_foreign_visits += count - admitted;
        }
    }

    /// Record only a local search observation, after deciding trial acceptance.
    pub(super) fn observe(&mut self, replica: usize, descriptor: ArrayView1<f64>, temperature: f64) {
        self.biases[replica].deposit(descriptor, temperature);
        self.stats.local_observations += 1;
        if let Some(exchange) = &mut self.exchange {
            exchange.publish(replica, vec![(descriptor.to_owned(), 1)]);
        }
    }

    pub(super) fn finish(mut self) -> CoverageStats {
        self.stats.published_visits = self.exchange.as_ref().map_or(0, |exchange| exchange.counts().0);
        self.stats.per_chain_regions = self.biases.iter().map(|bias| bias.index().n_basins()).collect();
        self.stats
    }
}
