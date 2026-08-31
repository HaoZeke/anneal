//! Budgeted execution of coordinator-issued transition-search experiments.
//!
//! One worker turns a same-system ride claim into either producer-side
//! stationary structures or classified negative evidence. All potential calls,
//! including the final stationary evaluations needed to construct records, pass
//! through one hard counter. The receiving coordinator independently
//! re-evaluates the structures and certifies the saddle index.

use std::fmt::{self, Display};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use ndarray::{Array1, ArrayView1};

use crate::catalog::euclidean_gradient_norm;
use crate::catalog_rpc::{
    CatalogCandidate, CatalogRideConnection, CatalogRideOutcome, CatalogRideReport, CatalogRideWork,
};
use crate::descriptor_space::DescriptorSpace;
use crate::pes_exploration::{
    ExactStructureWitness, PesExplorationConfig, PesExplorationError, PesNetwork, PesSurface,
    RideModeDirection, discover_cartesian_mode_connection, localized_cartesian_mode,
};
use crate::ride_ledger::{RideDirection, RideFailure};

/// Producer controls for one claimed transition-search experiment.
#[derive(Debug, Clone)]
pub struct CatalogRideExecutionConfig {
    /// rgmin, rgsaddle, IRC, and receiving-index numerical controls.
    pub exploration: PesExplorationConfig,
    /// Gaussian localization radius in descriptor length units.
    pub localization_radius: f64,
    /// Hard number of producer PES evaluations permitted for the experiment.
    pub maximum_evaluations: u64,
    /// First producer event identity assigned to returned stationary records.
    pub producer_event_sequence: u64,
    /// Producer cumulative PES counter before this experiment starts.
    pub producer_charged_work: u64,
}

#[derive(Debug)]
enum BudgetedSurfaceError {
    Exhausted,
    Surface(String),
}

impl Display for BudgetedSurfaceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exhausted => formatter.write_str("ride PES-call budget exhausted"),
            Self::Surface(message) => formatter.write_str(message),
        }
    }
}

struct BudgetedSurface<'a, S: ?Sized> {
    inner: &'a S,
    maximum: u64,
    evaluations: AtomicU64,
    exhausted: AtomicBool,
}

impl<'a, S: ?Sized> BudgetedSurface<'a, S> {
    fn new(inner: &'a S, maximum: u64) -> Self {
        Self {
            inner,
            maximum,
            evaluations: AtomicU64::new(0),
            exhausted: AtomicBool::new(false),
        }
    }

    fn evaluations(&self) -> u64 {
        self.evaluations.load(Ordering::Relaxed)
    }

    fn exhausted(&self) -> bool {
        self.exhausted.load(Ordering::Relaxed)
    }
}

impl<S> PesSurface for BudgetedSurface<'_, S>
where
    S: PesSurface + ?Sized,
{
    type Error = BudgetedSurfaceError;

    fn evaluate(
        &self,
        coordinates: ArrayView1<'_, f64>,
    ) -> Result<(f64, Array1<f64>), Self::Error> {
        let reservation =
            self.evaluations
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |evaluations| {
                    if evaluations < self.maximum {
                        Some(evaluations + 1)
                    } else {
                        None
                    }
                });
        if reservation.is_err() {
            self.exhausted.store(true, Ordering::Relaxed);
            return Err(BudgetedSurfaceError::Exhausted);
        }
        self.inner
            .evaluate(coordinates)
            .map_err(|error| BudgetedSurfaceError::Surface(error.to_string()))
    }
}

fn failed_report(work: u64, charged_evaluations: u64, failure: RideFailure) -> CatalogRideReport {
    CatalogRideReport {
        work,
        charged_evaluations,
        outcome: CatalogRideOutcome::Failed(failure),
    }
}

fn classify_failure(error: PesExplorationError, budget_exhausted: bool) -> RideFailure {
    if budget_exhausted {
        return RideFailure::BudgetExhausted;
    }
    match error {
        PesExplorationError::QuenchNotConverged { .. } => RideFailure::QuenchNotConverged,
        PesExplorationError::SaddleNotConverged { .. } => RideFailure::SaddleNotConverged,
        PesExplorationError::ActivationNotEscaped { .. } => RideFailure::ActivationNotEscaped,
        PesExplorationError::MinimumModeLostCurvature { .. } => {
            RideFailure::MinimumModeLostCurvature
        }
        PesExplorationError::NotFirstOrder {
            negative_modes: 0, ..
        } => RideFailure::NoNegativeMode,
        PesExplorationError::NotFirstOrder { .. } => RideFailure::HigherIndex,
        PesExplorationError::CollapsedConnection => RideFailure::CollapsedConnection,
        PesExplorationError::DisconnectedConnection => RideFailure::DisconnectedConnection,
        PesExplorationError::InvalidShape(_)
        | PesExplorationError::InvalidConfig(_)
        | PesExplorationError::Surface(_)
        | PesExplorationError::InvalidEvaluation(_)
        | PesExplorationError::Descriptor(_)
        | PesExplorationError::Saddle(_) => RideFailure::Surface,
    }
}

#[allow(clippy::too_many_arguments)]
fn stationary_candidate<S>(
    surface: &BudgetedSurface<'_, S>,
    descriptor_space: &DescriptorSpace,
    coordinates: ArrayView1<'_, f64>,
    species: &[u32],
    producer_replica: u32,
    cell: Option<[f64; 9]>,
    event_sequence: u64,
    seed: u64,
    producer_charged_work: u64,
) -> Result<CatalogCandidate, PesExplorationError>
where
    S: PesSurface + ?Sized,
{
    let (energy, gradient) = surface
        .evaluate(coordinates)
        .map_err(|error| PesExplorationError::Surface(error.to_string()))?;
    if !energy.is_finite() {
        return Err(PesExplorationError::InvalidEvaluation("a nonfinite energy"));
    }
    if gradient.len() != coordinates.len() {
        return Err(PesExplorationError::InvalidEvaluation(
            "a gradient with the wrong dimension",
        ));
    }
    if gradient.iter().any(|component| !component.is_finite()) {
        return Err(PesExplorationError::InvalidEvaluation(
            "a nonfinite gradient",
        ));
    }
    let descriptor = descriptor_space.describe(coordinates, Some(species))?;
    let charged_work = producer_charged_work
        .checked_add(surface.evaluations())
        .ok_or(PesExplorationError::InvalidConfig(
            "producer charged-work counter",
        ))?;
    let gradient_norm = euclidean_gradient_norm(gradient.as_slice().ok_or(
        PesExplorationError::InvalidEvaluation("a noncontiguous gradient"),
    )?);
    Ok(CatalogCandidate {
        producer_replica,
        coordinates: coordinates.to_vec(),
        cell,
        energy,
        forces: gradient.iter().map(|component| -*component).collect(),
        gradient_norm,
        descriptor: descriptor.values().to_vec(),
        descriptor_schema_version: descriptor.schema_version(),
        quench_converged: true,
        charged_work,
        event_sequence,
        seed,
        census_basin: None,
    })
}

fn direction(direction: RideDirection) -> RideModeDirection {
    match direction {
        RideDirection::Negative => RideModeDirection::Negative,
        RideDirection::Positive => RideModeDirection::Positive,
    }
}

/// Return the certified endpoint that is not the coordinator-issued source.
///
/// A usable one-sided ride contains exactly one endpoint equivalent to its
/// claimed source. Reports with the wrong work identity, no source endpoint,
/// or two source-equivalent endpoints do not define a connected proposal.
pub fn connected_destination<W>(
    work: &CatalogRideWork,
    report: &CatalogRideReport,
    witness: &W,
) -> Option<CatalogCandidate>
where
    W: ExactStructureWitness + ?Sized,
{
    if report.work != work.order.id {
        return None;
    }
    let CatalogRideOutcome::Certified(connection) = &report.outcome else {
        return None;
    };
    let source = ArrayView1::from(work.source.coordinates.as_slice());
    let source_endpoint = connection.endpoints.each_ref().map(|endpoint| {
        witness.equivalent(source, ArrayView1::from(endpoint.coordinates.as_slice()))
    });
    match source_endpoint {
        [true, false] => Some(connection.endpoints[1].clone()),
        [false, true] => Some(connection.endpoints[0].clone()),
        [false, false] | [true, true] => None,
    }
}

/// Execute one exclusive claim and construct evidence for coordinator review.
///
/// The work source, local-environment target, ranked Gaussian mode, solver, and
/// sign all come from the coordinator claim. The returned cost includes every
/// producer evaluation and never exceeds `config.maximum_evaluations`.
#[allow(clippy::too_many_arguments)]
pub fn execute_catalog_ride<S, W>(
    surface: &S,
    descriptor_space: &DescriptorSpace,
    work: &CatalogRideWork,
    species: &[u32],
    masses: ArrayView1<'_, f64>,
    frozen_atoms: &[bool],
    config: &CatalogRideExecutionConfig,
    witness: &W,
) -> CatalogRideReport
where
    S: PesSurface + ?Sized,
    W: ExactStructureWitness + ?Sized,
{
    let counted = BudgetedSurface::new(surface, config.maximum_evaluations);
    let work_id = work.order.id;
    let Some(geometry) = descriptor_space.geometry() else {
        return failed_report(work_id, 0, RideFailure::Surface);
    };
    let Ok(representative_atom) = usize::try_from(work.order.representative_atom) else {
        return failed_report(work_id, 0, RideFailure::Surface);
    };
    let source = ArrayView1::from(work.source.coordinates.as_slice());
    let mode = match localized_cartesian_mode(
        source,
        representative_atom,
        frozen_atoms,
        geometry,
        config.localization_radius,
        work.order.seed,
        work.order.arm.mode_rank,
        direction(work.order.arm.direction),
    ) {
        Ok(mode) => mode,
        Err(error) => {
            return failed_report(work_id, 0, classify_failure(error, false));
        }
    };
    let mut exploration = config.exploration.clone();
    exploration.ride_method = work.order.arm.method;
    let mut network = PesNetwork::new();
    let connection = match discover_cartesian_mode_connection(
        &counted,
        descriptor_space,
        &mut network,
        source,
        masses,
        frozen_atoms,
        mode.view(),
        Some(species),
        &exploration,
        witness,
    ) {
        Ok(connection) => connection,
        Err(error) => {
            return failed_report(
                work_id,
                counted.evaluations(),
                classify_failure(error, counted.exhausted()),
            );
        }
    };
    if connection
        .irc_at_minimum
        .iter()
        .any(|at_minimum| !at_minimum)
    {
        return failed_report(work_id, counted.evaluations(), RideFailure::IrcNotConverged);
    }

    let records = [
        connection.saddle_coordinates.view(),
        network.minima()[connection.endpoints[0]].coordinates.view(),
        network.minima()[connection.endpoints[1]].coordinates.view(),
    ];
    let mut candidates = Vec::with_capacity(records.len());
    for (offset, coordinates) in records.into_iter().enumerate() {
        let Some(event_sequence) = config
            .producer_event_sequence
            .checked_add(u64::try_from(offset).unwrap_or(u64::MAX))
        else {
            return failed_report(work_id, counted.evaluations(), RideFailure::Surface);
        };
        match stationary_candidate(
            &counted,
            descriptor_space,
            coordinates,
            species,
            work.order.replica,
            work.source.cell,
            event_sequence,
            work.order.seed,
            config.producer_charged_work,
        ) {
            Ok(candidate) => candidates.push(candidate),
            Err(error) => {
                return failed_report(
                    work_id,
                    counted.evaluations(),
                    classify_failure(error, counted.exhausted()),
                );
            }
        }
    }
    let endpoints = [candidates.remove(1), candidates.remove(1)];
    CatalogRideReport {
        work: work_id,
        charged_evaluations: counted.evaluations(),
        outcome: CatalogRideOutcome::Certified(CatalogRideConnection {
            saddle: candidates.remove(0),
            endpoints,
        }),
    }
}
