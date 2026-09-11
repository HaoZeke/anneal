//! Coverage of evaluated search boundaries, independent of minimum identity.

use std::collections::HashMap;

use eindir_core::Bounds;
use ndarray::{Array1, ArrayView1};
use rand::Rng;

use crate::bias::{BasinBias, BasinMetric, Bias, EuclideanMetric, Fingerprint};
use crate::methods::ensemble::HistoryMode;
use crate::methods::minima_hopping::EscapeFeedback;
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
        assert!(
            self.radius.is_finite() && self.radius > 0.0,
            "coverage radius must be finite and positive"
        );
        assert!(
            self.height.is_finite() && self.height >= 0.0,
            "coverage height must be finite and nonnegative"
        );
        assert!(
            self.well_tempering.is_finite() && self.well_tempering > 1.0,
            "coverage well-tempering must be finite and greater than one"
        );
        assert!(
            self.peer_weight.is_finite() && self.peer_weight >= 0.0,
            "coverage peer weight must be finite and nonnegative"
        );
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
    /// Uncertified quenches returning into a known region from a paid launch
    /// outside that region; zero-height coverage supplies no escape feedback.
    pub recrossings: usize,
    /// Recrossings whose selected return region contains admitted peer visits.
    pub peer_recrossings: usize,
    /// Recrossings with peer visits but no local arrivals in that selected region.
    /// This is conditional on the actual region map, not a private-chain replay.
    pub peer_only_recrossings: usize,
    /// Recrossing updates that change the bounded escape scale.
    pub escape_updates: usize,
    /// Uncertified trial arrivals outside every held region, before their local
    /// deposit. Initial observations and zero-height coverage do not supply feedback.
    pub novel_arrivals: usize,
    /// Novel-arrival updates that change the bounded escape scale.
    pub novelty_updates: usize,
}

/// Conditional influence of imported coverage heights on terminal acceptance.
///
/// Each comparison holds the actual proposal, temperature, region map and
/// local deposit heights fixed, removing only imported height increments.
/// This is not a replay of a chain with private history: imported centres and
/// their effects on local well-tempering remain part of the conditioning.
/// No objective evaluations or random draws are added for this measurement.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct CoverageDecisionStats {
    /// Terminal acceptance comparisons, including unresolved arithmetic.
    pub comparisons: usize,
    /// Comparisons excluded from influence measures by nonfinite arithmetic.
    pub unresolved: usize,
    /// Trials accepted by the actual biased decision.
    pub accepted: usize,
    /// Comparisons with imported visits in either selected endpoint region.
    pub peer_overlap: usize,
    /// Comparisons with a nonzero difference of imported endpoint heights.
    pub peer_delta_changes: usize,
    /// Comparisons whose acceptance probability changes without that difference.
    pub probability_changes: usize,
    /// Actual comparisons that consume a uniform acceptance draw.
    pub drawn_comparisons: usize,
    /// Different decisions under an actual draw with imported heights removed.
    /// Downhill short-circuit decisions have no draw and do not enter this count.
    pub drawn_disagreements: usize,
    /// Sum of absolute conditional acceptance-probability changes.
    pub probability_change_sum: f64,
    /// Largest absolute conditional acceptance-probability change.
    pub max_probability_change: f64,
    /// Largest absolute difference of imported endpoint heights.
    pub max_abs_peer_delta: f64,
    /// Largest such difference divided by the effective acceptance temperature.
    pub max_abs_peer_delta_over_temperature: f64,
}

#[derive(Clone, Copy, Default)]
struct ForeignWell {
    visits: u64,
    height: f64,
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
        assert!(
            bounds
                .low
                .iter()
                .chain(bounds.high.iter())
                .all(|x| x.is_finite()),
            "coverage requires finite box bounds"
        );
        assert!(
            widths.iter().all(|w| w.is_finite() && *w >= 0.0),
            "coverage requires finite nonnegative box widths"
        );
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
            && x.iter()
                .zip(self.low.iter().zip(self.high.iter()))
                .all(|(x, (low, high))| x.is_finite() && x >= low && x <= high)
    }
}

impl Fingerprint for NormalizedCoordinates {
    fn describe(&self, x: ArrayView1<f64>) -> Array1<f64> {
        Array1::from_iter(x.iter().zip(self.low.iter().zip(self.widths.iter())).map(
            |(x, (low, width))| {
                if *width > 0.0 {
                    ((x - low) / width) * self.free_scale
                } else {
                    0.0
                }
            },
        ))
    }
}

pub(super) struct Coverage {
    coordinates: NormalizedCoordinates,
    biases: Vec<BasinBias<NormalizedCoordinates>>,
    foreign_wells: Vec<Vec<ForeignWell>>,
    exchange: Option<SharedDeposits>,
    peer_weight: f64,
    foreign_cap: u64,
    stats: CoverageStats,
    decisions: CoverageDecisionStats,
}

impl Coverage {
    pub(super) fn new(
        bounds: &Bounds<f64>,
        replicas: usize,
        config: &BoxCoverageConfig,
        foreign_cap: usize,
    ) -> Self {
        config.validate();
        let coordinates = NormalizedCoordinates::new(bounds);
        let biases = (0..replicas)
            .map(|_| {
                let mut bias = BasinBias::new(
                    coordinates.clone(),
                    config.radius,
                    config.height,
                    config.well_tempering,
                );
                // Box coverage uses its explicit fixed-height well-tempering
                // settings, independent of catalog-specific entropy controls.
                bias.entropic = false;
                bias
            })
            .collect();
        let exchange =
            (config.shared && config.peer_weight > 0.0 && foreign_cap > 0 && replicas > 1)
                .then(|| SharedDeposits::new(replicas));
        Self {
            coordinates,
            biases,
            foreign_wells: vec![Vec::new(); replicas],
            exchange,
            peer_weight: config.peer_weight,
            foreign_cap: foreign_cap as u64,
            stats: CoverageStats::default(),
            decisions: CoverageDecisionStats::default(),
        }
    }

    pub(super) fn describe(&self, x: ArrayView1<f64>, value: f64) -> Option<Array1<f64>> {
        (value.is_finite() && self.coordinates.feasible(x)).then(|| self.coordinates.describe(x))
    }

    fn heights(&self, replica: usize, descriptor: ArrayView1<f64>) -> (f64, ForeignWell) {
        let bias = &self.biases[replica];
        bias.index()
            .lookup(descriptor)
            .map_or((0.0, ForeignWell::default()), |region| {
                (
                    bias.well_depth(region),
                    self.foreign_wells[replica]
                        .get(region)
                        .copied()
                        .unwrap_or_default(),
                )
            })
    }

    /// Connect a paid arrival to escape without asserting stationarity.
    /// The existing map is read before recording this trial's local arrival.
    pub(super) fn feedback_from_arrival(
        &mut self,
        replica: usize,
        here: ArrayView1<f64>,
        launch: ArrayView1<f64>,
        trial: ArrayView1<f64>,
        feedback: &mut EscapeFeedback,
    ) {
        let bias = &self.biases[replica];
        if bias.height() == 0.0 {
            return;
        }
        let index = bias.index();
        let Some(region) = index.lookup(trial) else {
            let previous_scale = feedback.escape();
            feedback.observe_coverage_discovery();
            self.stats.novel_arrivals += 1;
            self.stats.novelty_updates += usize::from(feedback.escape() != previous_scale);
            return;
        };
        // Overlapping regions can select different IDs without a departure.
        // Require the launch to lie outside the actual return-region ball.
        if EuclideanMetric.distance(launch, index.centre(region)) <= index.merge_radius() {
            return;
        }
        let local_visits = index.visits(region);
        let peer_visits = self.foreign_wells[replica]
            .get(region)
            .map_or(0, |well| well.visits);
        let previous_scale = feedback.escape();
        feedback.observe_coverage_return(
            index.lookup(here) == Some(region),
            local_visits.saturating_add(peer_visits),
        );
        self.stats.recrossings += 1;
        self.stats.peer_recrossings += usize::from(peer_visits > 0);
        self.stats.peer_only_recrossings += usize::from(peer_visits > 0 && local_visits == 0);
        self.stats.escape_updates += usize::from(feedback.escape() != previous_scale);
    }

    /// Apply the biased Metropolis decision and measure direct peer-height
    /// influence with the same endpoint lookups and original uniform draw.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn accepts<R: Rng + ?Sized>(
        &mut self,
        replica: usize,
        here: ArrayView1<f64>,
        trial: ArrayView1<f64>,
        energy: f64,
        trial_energy: f64,
        temperature: f64,
        rng: &mut R,
    ) -> bool {
        let (v_here, foreign_here) = self.heights(replica, here);
        let (v_trial, foreign_trial) = self.heights(replica, trial);
        let delta = (trial_energy - energy) + (v_trial - v_here);
        let effective_temperature = temperature.max(1e-300);
        let (probability, draw, accepted) = if delta <= 0.0 {
            (1.0, None, true)
        } else {
            let probability = (-delta / effective_temperature).exp();
            let draw = rng.random::<f64>();
            (probability, Some(draw), draw < probability)
        };
        let decisions = &mut self.decisions;
        decisions.comparisons += 1;
        decisions.accepted += usize::from(accepted);
        decisions.drawn_comparisons += usize::from(draw.is_some());

        let peer_delta = foreign_trial.height - foreign_here.height;
        let conditional_delta = delta - peer_delta;
        let conditional_probability = if conditional_delta <= 0.0 {
            1.0
        } else {
            (-conditional_delta / effective_temperature).exp()
        };
        let normalized_peer_delta = peer_delta.abs() / effective_temperature;
        if ![
            energy,
            trial_energy,
            temperature,
            delta,
            peer_delta,
            conditional_delta,
            probability,
            conditional_probability,
            normalized_peer_delta,
        ]
        .iter()
        .all(|value| value.is_finite())
        {
            decisions.unresolved += 1;
            return accepted;
        }
        decisions.peer_overlap += usize::from(foreign_here.visits > 0 || foreign_trial.visits > 0);
        decisions.peer_delta_changes += usize::from(peer_delta != 0.0);
        let change = (probability - conditional_probability).abs();
        decisions.probability_changes += usize::from(change > 0.0);
        decisions.probability_change_sum += change;
        decisions.max_probability_change = decisions.max_probability_change.max(change);
        decisions.max_abs_peer_delta = decisions.max_abs_peer_delta.max(peer_delta.abs());
        decisions.max_abs_peer_delta_over_temperature = decisions
            .max_abs_peer_delta_over_temperature
            .max(normalized_peer_delta);
        if let Some(draw) = draw {
            decisions.drawn_disagreements +=
                usize::from(accepted != (draw < conditional_probability));
        }
        accepted
    }

    /// Read once at a funded hop boundary, before its acceptance decision.
    pub(super) fn hear(&mut self, replica: usize, temperature: f64) {
        let Some(exchange) = &mut self.exchange else {
            return;
        };
        let mut applied = HashMap::<usize, u64>::new();
        let bias = &mut self.biases[replica];
        for (descriptor, count) in exchange.drain(replica) {
            let region = bias
                .index()
                .lookup(descriptor.view())
                .unwrap_or_else(|| bias.index().n_basins());
            let region_count = applied.entry(region).or_default();
            let admitted = count.min(self.foreign_cap.saturating_sub(*region_count));
            if admitted > 0 {
                let before = if region < bias.index().n_basins() {
                    bias.well_depth(region)
                } else {
                    0.0
                };
                bias.deposit_scaled_n(descriptor.view(), temperature, self.peer_weight, admitted);
                let wells = &mut self.foreign_wells[replica];
                wells.resize(bias.index().n_basins(), ForeignWell::default());
                wells[region].height += bias.well_depth(region) - before;
                wells[region].visits += admitted;
            }
            *region_count += admitted;
            self.stats.applied_foreign_visits += admitted as usize;
            self.stats.capped_foreign_visits += count - admitted;
        }
    }

    /// Record only a local search observation, after deciding trial acceptance.
    pub(super) fn observe(
        &mut self,
        replica: usize,
        descriptor: ArrayView1<f64>,
        temperature: f64,
    ) {
        self.biases[replica].deposit(descriptor, temperature);
        self.stats.local_observations += 1;
        if let Some(exchange) = &mut self.exchange {
            exchange.publish(replica, vec![(descriptor.to_owned(), 1)]);
        }
    }

    pub(super) fn finish(mut self) -> (CoverageStats, CoverageDecisionStats) {
        self.stats.published_visits = self
            .exchange
            .as_ref()
            .map_or(0, |exchange| exchange.counts().0);
        self.stats.per_chain_regions = self
            .biases
            .iter()
            .map(|bias| bias.index().n_basins())
            .collect();
        (self.stats, self.decisions)
    }
}
