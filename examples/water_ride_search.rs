//! Charged GFN2-water minimum--saddle--minimum exploration.
//!
//! One independently budgeted worker quenches a flexible water cluster,
//! registers the receiving-validated minimum with its isolated coordinator,
//! and executes same-PES ride claims until the budget or finite portfolio is
//! exhausted. The universal descriptor ranks novelty; species-aware IRA
//! remains the exact stationary-structure witness.
//!
//! Usage: `water_ride_search [molecules] [charged-budget] [seed]`.

mod common;

use std::collections::HashSet;
use std::io;
use std::net::SocketAddr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use anneal_core::allocate::{ChargedDiscoveryAllocator, FlooredThompson};
use anneal_core::catalog::euclidean_gradient_norm;
use anneal_core::catalog::molecular::{
    MAX_GRADIENT_NORM, MolecularCatalogPresetError, component_gradient_tolerance, descriptor_space,
    engine_binary_digest, length_scale, reference_coordinates, system_signature, water_groups,
    water_species,
};
use anneal_core::catalog_rpc::client::{CatalogClient, ClientConfig};
use anneal_core::catalog_rpc::{
    CatalogCandidate, CatalogIdentity, CatalogRideOutcome, CatalogRideReport,
};
use anneal_core::cooperative_search::ledger::ChargeKind;
use anneal_core::discovery_roster::DiscoveryRole;
use anneal_core::methods::cluster_hopping::{
    Config as ClusterConfig, SoapProposalMode, repack_rigid_groups,
};
use anneal_core::methods::minima_hopping::EscapeFeedback;
use anneal_core::pes_exploration::{
    PesExplorationConfig, PesSurface, RideMethod, quench_minimum_with_norm,
};
use anneal_core::ride_execution::{CatalogRideExecutionConfig, execute_catalog_ride};
use anneal_core::shape::IraStructureWitness;
use anneal_core::source_escape::{SourceEscapeConfig, SourceEscapeOutcome, quench_source_escape};
use common::rgpot_eindir::{RgpotObjective, emit_engine_manifest};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde_json::json;

const WATER_BOX: [f64; 9] = [60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 60.0];
const OXYGEN_MASS: f64 = 15.999;
const HYDROGEN_MASS: f64 = 1.008;
const EXACT_STRUCTURE_RADIUS: f64 = 1e-4;
const RIDE_DISCOVERY_ARM: usize = 0;
const SOURCE_ESCAPE_ARM: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WaterDiscoveryPolicy {
    Adaptive,
    RidgeOnly,
    BasinEscapeOnly,
}

impl WaterDiscoveryPolicy {
    fn from_environment() -> Result<Self, io::Error> {
        match std::env::var("CATALOG_DISCOVERY_POLICY")
            .unwrap_or_else(|_| "adaptive".into())
            .as_str()
        {
            "adaptive" => Ok(Self::Adaptive),
            "ridge_only" => Ok(Self::RidgeOnly),
            "basin_escape_only" => Ok(Self::BasinEscapeOnly),
            value => Err(io::Error::other(format!(
                "unknown CATALOG_DISCOVERY_POLICY {value:?}"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Adaptive => "adaptive",
            Self::RidgeOnly => "ridge_only",
            Self::BasinEscapeOnly => "basin_escape_only",
        }
    }
}

fn discovery_role_name(role: DiscoveryRole) -> &'static str {
    match role {
        DiscoveryRole::BasinEscape => "basin_escape",
        DiscoveryRole::SaddleRide => "saddle_ride",
    }
}

struct WaterSurface {
    objective: Mutex<RgpotObjective>,
    evaluations: AtomicU64,
}

impl WaterSurface {
    fn new(atomic_numbers: &[i32]) -> Self {
        Self {
            objective: Mutex::new(RgpotObjective::xtb(atomic_numbers, WATER_BOX)),
            evaluations: AtomicU64::new(0),
        }
    }

    fn evaluations(&self) -> u64 {
        self.evaluations.load(Ordering::Relaxed)
    }
}

impl PesSurface for WaterSurface {
    type Error = String;

    fn evaluate(
        &self,
        coordinates: ArrayView1<'_, f64>,
    ) -> Result<(f64, Array1<f64>), Self::Error> {
        let coordinates = coordinates
            .as_slice()
            .ok_or_else(|| "water coordinates must be contiguous".to_owned())?;
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        let fresh = self
            .objective
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .fresh_evaluation(coordinates)?;
        Ok((
            fresh.energy,
            Array1::from_iter(fresh.forces.into_iter().map(|force| -force)),
        ))
    }
}

fn required_environment(name: &str) -> Result<String, io::Error> {
    std::env::var(name).map_err(|_| io::Error::other(format!("{name} is required")))
}

fn next_sequence(sequence: &mut u64) -> Result<u64, io::Error> {
    let current = *sequence;
    *sequence = sequence
        .checked_add(1)
        .ok_or_else(|| io::Error::other("event sequence overflow"))?;
    Ok(current)
}

fn record_charge(
    client: &mut CatalogClient,
    event_sequence: &mut u64,
    charged: &mut u64,
    budget: u64,
    kind: ChargeKind,
    calls: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    if calls == 0 {
        return Ok(());
    }
    let cumulative = charged
        .checked_add(calls)
        .ok_or_else(|| io::Error::other("charged counter overflow"))?;
    if cumulative > budget {
        return Err(io::Error::other(format!(
            "charged work {cumulative} exceeds replica budget {budget}"
        ))
        .into());
    }
    client.record_ledger_event(next_sequence(event_sequence)?, kind, calls, cumulative)?;
    *charged = cumulative;
    Ok(())
}

fn ride_config(
    length_scale: f64,
    coordinate_dim: usize,
) -> Result<PesExplorationConfig, MolecularCatalogPresetError> {
    let stationary_component_tolerance = 0.5 * component_gradient_tolerance(coordinate_dim)?;
    Ok(PesExplorationConfig {
        ride_method: RideMethod::Dimer,
        quench_steps: 2_000,
        saddle_steps: 1_000,
        minimum_mode_force_tolerance: 5e-2,
        irc_steps: 200,
        prfo_steps: 300,
        quench_gradient_tolerance: stationary_component_tolerance,
        quench_gradient_norm_tolerance: Some(MAX_GRADIENT_NORM),
        saddle_force_tolerance: stationary_component_tolerance,
        saddle_displacement: 0.1 * length_scale,
        negative_curvature_tolerance: 1e-6,
        hessian_step: 1e-4 * length_scale,
        maximum_move: 0.2 * length_scale,
        irc_step: 0.1 * length_scale,
        irc_force_tolerance: 0.05,
        certify_degenerate_rearrangements: true,
        refine_with_prfo: true,
        ..PesExplorationConfig::default()
    })
}

fn failure_name(outcome: &CatalogRideOutcome) -> Option<String> {
    match outcome {
        CatalogRideOutcome::Certified(_) => None,
        CatalogRideOutcome::Unresolved(evidence) => Some(format!("{:?}", evidence.failure)),
        CatalogRideOutcome::Failed(failure) => Some(format!("{failure:?}")),
    }
}

fn stationary_evidence(
    outcome: &CatalogRideOutcome,
) -> (Option<f64>, Option<f64>, Vec<f64>, Vec<f64>) {
    match outcome {
        CatalogRideOutcome::Certified(connection) => (
            Some(connection.saddle.energy),
            Some(connection.saddle.gradient_norm),
            connection
                .endpoints
                .iter()
                .map(|endpoint| endpoint.energy)
                .collect(),
            connection
                .endpoints
                .iter()
                .map(|endpoint| endpoint.gradient_norm)
                .collect(),
        ),
        CatalogRideOutcome::Unresolved(evidence) => (
            Some(evidence.saddle.energy),
            Some(evidence.saddle.gradient_norm),
            Vec::new(),
            Vec::new(),
        ),
        CatalogRideOutcome::Failed(_) => (None, None, Vec::new(), Vec::new()),
    }
}

fn minimum_pair_distance(coordinates: &[f64]) -> f64 {
    let atom_count = coordinates.len() / 3;
    (0..atom_count)
        .flat_map(|first| (first + 1..atom_count).map(move |second| (first, second)))
        .map(|(first, second)| {
            (0..3)
                .map(|axis| {
                    let delta = coordinates[3 * first + axis] - coordinates[3 * second + axis];
                    delta * delta
                })
                .sum::<f64>()
                .sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = std::env::args().collect::<Vec<_>>();
    let molecule_count = arguments
        .get(1)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(2);
    let budget = arguments
        .get(2)
        .map(|value| value.parse::<u64>())
        .transpose()?
        .unwrap_or(20_000);
    let seed = arguments
        .get(3)
        .map(|value| value.parse::<u64>())
        .transpose()?
        .unwrap_or(1);
    if budget == 0 {
        return Err(io::Error::other("charged budget must be positive").into());
    }
    let ride_budget_cap = std::env::var("CATALOG_RIDE_BUDGET")
        .ok()
        .map(|value| value.parse::<u64>())
        .transpose()?
        .unwrap_or(5_000)
        .max(1);
    let escape_budget_cap = std::env::var("CATALOG_ESCAPE_BUDGET")
        .ok()
        .map(|value| value.parse::<u64>())
        .transpose()?
        .unwrap_or(1_000)
        .max(1);
    let discovery_policy = WaterDiscoveryPolicy::from_environment()?;

    let campaign = required_environment("CATALOG_CAMPAIGN")?;
    let ensemble = required_environment("CATALOG_ENSEMBLE")?;
    let replica = required_environment("CATALOG_REPLICA")?.parse::<u32>()?;
    let endpoint = required_environment("CATALOG_RPC")?.parse::<SocketAddr>()?;
    let engine_path = required_environment("RGPOT_XTB_ENGINE")?;
    let engine_bytes = std::fs::read(&engine_path)?;
    if engine_bytes.is_empty() {
        return Err(io::Error::other("RGPOT_XTB_ENGINE is empty").into());
    }

    let species = water_species(molecule_count)?;
    let atomic_numbers = species
        .iter()
        .map(|&atomic_number| i32::try_from(atomic_number))
        .collect::<Result<Vec<_>, _>>()?;
    let groups = water_groups(molecule_count)?;
    let reference = reference_coordinates(molecule_count)?;
    let scale = length_scale(&species)?;
    let descriptor = descriptor_space(&species)?;
    let signature = system_signature(molecule_count, engine_binary_digest(&engine_bytes))?;
    let signature_digest = signature.digest();
    let coordinate_dim = reference.len();
    let receiver_reserve = u64::try_from(
        coordinate_dim
            .checked_mul(2)
            .and_then(|evaluations| evaluations.checked_add(3))
            .ok_or_else(|| io::Error::other("receiver certification cost overflow"))?,
    )?;
    let masses = Array1::from_iter(species.iter().map(|atomic_number| match atomic_number {
        8 => OXYGEN_MASS,
        1 => HYDROGEN_MASS,
        _ => unreachable!("water_species emits only oxygen and hydrogen"),
    }));
    let frozen = vec![false; species.len()];
    let witness = IraStructureWitness {
        kmax_factor: 1.8,
        radius: EXACT_STRUCTURE_RADIUS,
    };
    let exploration = ride_config(scale, coordinate_dim)?;
    let localization_radius = std::env::var("CATALOG_RIDE_LOCAL_RADIUS")
        .ok()
        .map(|value| value.parse::<f64>())
        .transpose()?
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(1.5 * scale);

    let surface = WaterSurface::new(&atomic_numbers);
    emit_engine_manifest("xtb");
    let mut rng = StdRng::seed_from_u64(seed);
    let start = repack_rigid_groups(
        ArrayView1::from(reference.as_slice()),
        &groups,
        scale,
        &mut rng,
    );
    let quench_tolerance = exploration.quench_gradient_tolerance;
    let minimum = quench_minimum_with_norm(
        &surface,
        start.view(),
        exploration.quench_steps,
        quench_tolerance,
        MAX_GRADIENT_NORM,
    )?;
    let minimum_energy = minimum.energy;
    let minimum_gradient = minimum.gradient.clone();
    let producer_initial_calls = surface.evaluations();
    if producer_initial_calls
        .checked_add(1)
        .is_none_or(|charged| charged > budget)
    {
        return Err(io::Error::other("initial minimum exceeds charged budget").into());
    }
    let minimum_descriptor = descriptor.describe(minimum.coordinates.view(), Some(&species))?;
    let initial_candidate = CatalogCandidate {
        producer_replica: replica,
        coordinates: minimum.coordinates.to_vec(),
        cell: None,
        energy: minimum_energy,
        forces: minimum_gradient.iter().map(|gradient| -*gradient).collect(),
        gradient_norm: euclidean_gradient_norm(
            minimum_gradient
                .as_slice()
                .ok_or_else(|| io::Error::other("minimum gradient must be contiguous"))?,
        ),
        descriptor: minimum_descriptor.values().to_vec(),
        descriptor_schema_version: minimum_descriptor.schema_version(),
        quench_converged: true,
        charged_work: producer_initial_calls,
        event_sequence: 1,
        seed,
        census_basin: None,
    };

    let identity = CatalogIdentity {
        campaign: campaign.clone(),
        ensemble: ensemble.clone(),
        replica,
        signature_digest,
    };
    let mut client = CatalogClient::connect(endpoint, identity, ClientConfig::default())?;
    let mut event_sequence = 1_u64;
    println!(
        "{}",
        json!({
            "kind": "water_initial_proposal",
            "energy": initial_candidate.energy,
            "gradient_norm": initial_candidate.gradient_norm,
            "maximum_gradient": minimum.max_gradient,
            "minimum_pair_distance": minimum_pair_distance(&initial_candidate.coordinates),
            "descriptor_dim": initial_candidate.descriptor.len(),
            "descriptor_schema_version": initial_candidate.descriptor_schema_version,
            "producer_calls": producer_initial_calls,
        })
    );
    let offer = client.offer_candidate(next_sequence(&mut event_sequence)?, initial_candidate)?;
    let mut charged = 0_u64;
    record_charge(
        &mut client,
        &mut event_sequence,
        &mut charged,
        budget,
        ChargeKind::AcceptedQuench,
        producer_initial_calls,
    )?;
    record_charge(
        &mut client,
        &mut event_sequence,
        &mut charged,
        budget,
        ChargeKind::FreshValidation,
        1,
    )?;
    println!(
        "{}",
        json!({
            "kind": "water_initial_minimum",
            "molecules": molecule_count,
            "replica": replica,
            "seed": seed,
            "energy": minimum_energy,
            "gradient_norm": euclidean_gradient_norm(minimum_gradient.as_slice().unwrap()),
            "maximum_gradient": minimum.max_gradient,
            "producer_calls": producer_initial_calls,
            "receiver_calls": 1,
            "charged": charged,
            "basin": offer.catalog.as_ref().map(|mutation| mutation.basin_id),
            "catalog_mutation": offer.catalog.as_ref().map(|mutation| mutation.kind.code()),
        })
    );

    let mut known_basins = HashSet::new();
    let mut live_basin = offer.catalog.as_ref().map(|mutation| mutation.basin_id);
    if let Some(basin) = live_basin {
        known_basins.insert(basin);
    }
    let mut live_coordinates = minimum.coordinates.clone();
    let mut live_energy = minimum_energy;
    let mut hopping = ClusterConfig::recommended_molecular(species.clone(), groups.clone(), 1.0);
    hopping.soap_mode = SoapProposalMode::Rigid;
    hopping.minima_hopping = true;
    hopping.record_gradient = MAX_GRADIENT_NORM;
    let escape_moves = hopping.move_library.kernels(&hopping);
    let mut mechanism_allocator = ChargedDiscoveryAllocator::new(2);
    let mut move_allocator = FlooredThompson::new(escape_moves.len());
    let mut escape_feedback = EscapeFeedback::new(1.0, hopping.temperature.max(1e-6));
    if let Some(basin) = live_basin {
        escape_feedback.observe(None, basin as usize);
    }
    let mut ride_available = true;
    let mut producer_event_sequence = 2_u64;
    let mut attempts = 0_u64;
    let mut source_attempts = 0_u64;
    let mut source_discoveries = 0_u64;
    let mut certified = 0_u64;
    let mut novel_saddles = 0_u64;
    let mut degenerate_rearrangements = 0_u64;
    let mut novel_edges = 0_u64;
    let mut reported_producer_calls = 0_u64;
    let termination;
    loop {
        let remaining = budget.saturating_sub(charged);
        if remaining <= 1 {
            termination = "validation_reserve";
            break;
        }
        let ride_feasible = ride_available && remaining > receiver_reserve;
        if discovery_policy == WaterDiscoveryPolicy::RidgeOnly && !ride_feasible {
            termination = "ride_unavailable";
            break;
        }
        let live_descriptor = descriptor.describe(live_coordinates.view(), Some(&species))?;
        let shared_policy = client.policy_state(
            next_sequence(&mut event_sequence)?,
            live_descriptor.values().to_vec(),
            live_energy,
        )?;
        let mechanism = match discovery_policy {
            WaterDiscoveryPolicy::Adaptive
                if ride_feasible && shared_policy.discovery_role == DiscoveryRole::SaddleRide =>
            {
                RIDE_DISCOVERY_ARM
            }
            WaterDiscoveryPolicy::Adaptive => SOURCE_ESCAPE_ARM,
            WaterDiscoveryPolicy::RidgeOnly => RIDE_DISCOVERY_ARM,
            WaterDiscoveryPolicy::BasinEscapeOnly => SOURCE_ESCAPE_ARM,
        };
        if mechanism == RIDE_DISCOVERY_ARM {
            let maximum_evaluations = (remaining - receiver_reserve).min(ride_budget_cap);
            let Some(work) = client.claim_ride(
                next_sequence(&mut event_sequence)?,
                seed ^ attempts.wrapping_mul(0x9e37_79b9_7f4a_7c15),
            )?
            else {
                ride_available = false;
                if discovery_policy == WaterDiscoveryPolicy::RidgeOnly {
                    termination = "ride_claim_unavailable";
                    break;
                }
                continue;
            };
            let execution = CatalogRideExecutionConfig {
                exploration: exploration.clone(),
                localization_radius,
                maximum_evaluations,
                producer_event_sequence,
                producer_charged_work: charged,
            };
            producer_event_sequence = producer_event_sequence
                .checked_add(3)
                .ok_or_else(|| io::Error::other("producer event sequence overflow"))?;
            let report: CatalogRideReport = execute_catalog_ride(
                &surface,
                &descriptor,
                &work,
                &species,
                masses.view(),
                &frozen,
                &execution,
                &witness,
            );
            let producer_calls = report.charged_evaluations;
            let failure = failure_name(&report.outcome);
            let (saddle_energy, saddle_gradient_norm, endpoint_energies, endpoint_gradient_norms) =
                stationary_evidence(&report.outcome);
            let credit = client.report_ride(next_sequence(&mut event_sequence)?, report)?;
            if credit.total_charged_evaluations < producer_calls {
                return Err(io::Error::other("coordinator ride credit lost producer calls").into());
            }
            let receiver_calls = credit.total_charged_evaluations - producer_calls;
            reported_producer_calls = reported_producer_calls
                .checked_add(producer_calls)
                .ok_or_else(|| io::Error::other("producer call counter overflow"))?;
            record_charge(
                &mut client,
                &mut event_sequence,
                &mut charged,
                budget,
                ChargeKind::SaddleRide,
                producer_calls,
            )?;
            record_charge(
                &mut client,
                &mut event_sequence,
                &mut charged,
                budget,
                ChargeKind::FreshValidation,
                receiver_calls,
            )?;
            mechanism_allocator.update(
                RIDE_DISCOVERY_ARM,
                u32::from(credit.novel_saddle || credit.novel_edge),
                credit.total_charged_evaluations,
            );
            attempts += 1;
            certified += u64::from(credit.certified_connection);
            novel_saddles += u64::from(credit.novel_saddle);
            degenerate_rearrangements += u64::from(credit.degenerate_rearrangement);
            novel_edges += u64::from(credit.novel_edge);
            println!(
                "{}",
                json!({
                    "kind": "water_ride_report",
                    "work": work.order.id,
                    "source_basin": work.order.arm.source_basin,
                    "environment_class": work.order.arm.environment_class,
                    "representative_atom": work.order.representative_atom,
                    "mode_rank": work.order.arm.mode_rank,
                    "direction": format!("{:?}", work.order.arm.direction),
                    "method": format!("{:?}", work.order.arm.method),
                    "attempt": work.order.attempt,
                    "producer_calls": producer_calls,
                    "receiver_calls": receiver_calls,
                    "total_calls": credit.total_charged_evaluations,
                    "certified": credit.certified_connection,
                    "novel_saddle": credit.novel_saddle,
                    "degenerate_rearrangement": credit.degenerate_rearrangement,
                    "novel_edge": credit.novel_edge,
                    "discovery_policy": discovery_policy.as_str(),
                    "assigned_role": discovery_role_name(shared_policy.discovery_role),
                    "discovery_epoch": shared_policy.discovery_epoch,
                    "basin_unseen_mass_upper": shared_policy.basin_unseen_mass_upper,
                    "saddle_unseen_mass_upper": shared_policy.saddle_unseen_mass_upper,
                    "basin_discovery_attempts": shared_policy.basin_discovery_attempts,
                    "basin_discovery_charged": shared_policy.basin_discovery_charged,
                    "saddle_discovery_attempts": shared_policy.saddle_discovery_attempts,
                    "saddle_discovery_charged": shared_policy.saddle_discovery_charged,
                    "saddle_coverage_saturated": shared_policy.saddle_coverage_saturated,
                    "failure": failure,
                    "saddle_energy": saddle_energy,
                    "saddle_gradient_norm": saddle_gradient_norm,
                    "endpoint_energies": endpoint_energies,
                    "endpoint_gradient_norms": endpoint_gradient_norms,
                    "charged": charged,
                })
            );
            continue;
        }

        let maximum_evaluations = (remaining - 1).min(escape_budget_cap);
        let move_index = move_allocator
            .pulls()
            .iter()
            .position(|&pulls| pulls == 0)
            .unwrap_or_else(|| move_allocator.select(&mut rng));
        let move_name = escape_moves[move_index].name();
        let proposal = escape_moves[move_index].propose_scaled(
            live_coordinates.view(),
            hopping.temperature,
            escape_feedback.escape(),
            &mut rng,
        );
        let escape_config = SourceEscapeConfig {
            maximum_evaluations,
            quench_steps: exploration.quench_steps,
            gradient_tolerance: exploration.quench_gradient_tolerance,
            gradient_norm_tolerance: MAX_GRADIENT_NORM,
        };
        let outcome = quench_source_escape(&surface, proposal.view(), &escape_config);
        source_attempts += 1;
        match outcome {
            SourceEscapeOutcome::Failed(failure) => {
                if failure.charged_evaluations == 0 {
                    return Err(io::Error::other(format!(
                        "source escape made no PES progress: {}",
                        failure.error
                    ))
                    .into());
                }
                reported_producer_calls = reported_producer_calls
                    .checked_add(failure.charged_evaluations)
                    .ok_or_else(|| io::Error::other("producer call counter overflow"))?;
                record_charge(
                    &mut client,
                    &mut event_sequence,
                    &mut charged,
                    budget,
                    ChargeKind::BasinEscape,
                    failure.charged_evaluations,
                )?;
                mechanism_allocator.update(SOURCE_ESCAPE_ARM, 0, failure.charged_evaluations);
                move_allocator.update(move_index, false);
                println!(
                    "{}",
                    json!({
                        "kind": "water_source_escape",
                        "attempt": source_attempts,
                        "move": move_name,
                        "escape_scale": escape_feedback.escape(),
                        "producer_calls": failure.charged_evaluations,
                        "converged": false,
                        "discovery_policy": discovery_policy.as_str(),
                        "assigned_role": discovery_role_name(shared_policy.discovery_role),
                        "discovery_epoch": shared_policy.discovery_epoch,
                        "basin_unseen_mass_upper": shared_policy.basin_unseen_mass_upper,
                        "saddle_unseen_mass_upper": shared_policy.saddle_unseen_mass_upper,
                        "basin_discovery_attempts": shared_policy.basin_discovery_attempts,
                        "basin_discovery_charged": shared_policy.basin_discovery_charged,
                        "saddle_discovery_attempts": shared_policy.saddle_discovery_attempts,
                        "saddle_discovery_charged": shared_policy.saddle_discovery_charged,
                        "saddle_coverage_saturated": shared_policy.saddle_coverage_saturated,
                        "error": failure.error,
                        "charged": charged,
                    })
                );
            }
            SourceEscapeOutcome::Converged(record) => {
                let producer_calls = record.charged_evaluations;
                let candidate_sequence = producer_event_sequence;
                producer_event_sequence = producer_event_sequence
                    .checked_add(1)
                    .ok_or_else(|| io::Error::other("producer event sequence overflow"))?;
                let minimum_descriptor =
                    descriptor.describe(record.minimum.coordinates.view(), Some(&species))?;
                let candidate = CatalogCandidate {
                    producer_replica: replica,
                    coordinates: record.minimum.coordinates.to_vec(),
                    cell: None,
                    energy: record.minimum.energy,
                    forces: record
                        .minimum
                        .gradient
                        .iter()
                        .map(|gradient| -*gradient)
                        .collect(),
                    gradient_norm: euclidean_gradient_norm(
                        record.minimum.gradient.as_slice().ok_or_else(|| {
                            io::Error::other("escape gradient must be contiguous")
                        })?,
                    ),
                    descriptor: minimum_descriptor.values().to_vec(),
                    descriptor_schema_version: minimum_descriptor.schema_version(),
                    quench_converged: true,
                    charged_work: charged
                        .checked_add(producer_calls)
                        .ok_or_else(|| io::Error::other("charged counter overflow"))?,
                    event_sequence: candidate_sequence,
                    seed: seed ^ source_attempts.wrapping_mul(0xd1b5_4a32_d192_ed03),
                    census_basin: None,
                };
                let offer =
                    client.offer_candidate(next_sequence(&mut event_sequence)?, candidate)?;
                reported_producer_calls = reported_producer_calls
                    .checked_add(producer_calls)
                    .ok_or_else(|| io::Error::other("producer call counter overflow"))?;
                record_charge(
                    &mut client,
                    &mut event_sequence,
                    &mut charged,
                    budget,
                    ChargeKind::BasinEscape,
                    producer_calls,
                )?;
                record_charge(
                    &mut client,
                    &mut event_sequence,
                    &mut charged,
                    budget,
                    ChargeKind::FreshValidation,
                    1,
                )?;
                let reached_basin = offer.catalog.as_ref().map(|mutation| mutation.basin_id);
                if let Some(basin) = reached_basin {
                    known_basins.insert(basin);
                }
                let discovered = offer
                    .catalog
                    .as_ref()
                    .is_some_and(|mutation| mutation.new_basin);
                source_discoveries += u64::from(discovered);
                mechanism_allocator.update(
                    SOURCE_ESCAPE_ARM,
                    u32::from(discovered),
                    producer_calls + 1,
                );
                move_allocator.update(move_index, discovered);
                let visit = reached_basin.map(|basin| {
                    escape_feedback.observe(live_basin.map(|value| value as usize), basin as usize)
                });
                let accepted = reached_basin.is_some()
                    && escape_feedback.accept(record.minimum.energy - live_energy);
                if accepted {
                    live_basin = reached_basin;
                    live_energy = record.minimum.energy;
                    live_coordinates = record.minimum.coordinates.clone();
                }
                if discovered {
                    ride_available = true;
                }
                println!(
                    "{}",
                    json!({
                        "kind": "water_source_escape",
                        "attempt": source_attempts,
                        "move": move_name,
                        "escape_scale": escape_feedback.escape(),
                        "acceptance_threshold": escape_feedback.threshold(),
                        "producer_calls": producer_calls,
                        "receiver_calls": 1,
                        "converged": true,
                        "energy": record.minimum.energy,
                        "gradient_norm": record.minimum.gradient.dot(&record.minimum.gradient).sqrt(),
                        "basin": reached_basin,
                        "visit": visit.map(|value| format!("{value:?}")),
                        "new_basin": discovered,
                        "discovery_policy": discovery_policy.as_str(),
                        "assigned_role": discovery_role_name(shared_policy.discovery_role),
                        "discovery_epoch": shared_policy.discovery_epoch,
                        "basin_unseen_mass_upper": shared_policy.basin_unseen_mass_upper,
                        "saddle_unseen_mass_upper": shared_policy.saddle_unseen_mass_upper,
                        "basin_discovery_attempts": shared_policy.basin_discovery_attempts,
                        "basin_discovery_charged": shared_policy.basin_discovery_charged,
                        "saddle_discovery_attempts": shared_policy.saddle_discovery_attempts,
                        "saddle_discovery_charged": shared_policy.saddle_discovery_charged,
                        "saddle_coverage_saturated": shared_policy.saddle_coverage_saturated,
                        "adopted": accepted,
                        "catalog_mutation": offer.catalog.as_ref().map(|mutation| mutation.kind.code()),
                        "charged": charged,
                    })
                );
            }
        }
    }

    let snapshot = client.snapshot(next_sequence(&mut event_sequence)?)?;
    let expected_surface_evaluations = producer_initial_calls
        .checked_add(reported_producer_calls)
        .ok_or_else(|| io::Error::other("producer call counter overflow"))?;
    if surface.evaluations() != expected_surface_evaluations {
        return Err(io::Error::other(format!(
            "surface evaluated {} times but reports account for {expected_surface_evaluations}",
            surface.evaluations()
        ))
        .into());
    }
    println!(
        "{}",
        json!({
            "kind": "water_ride_summary",
            "campaign": campaign,
            "ensemble": ensemble,
            "replica": replica,
            "molecules": molecule_count,
            "discovery_policy": discovery_policy.as_str(),
            "attempts": attempts,
            "source_attempts": source_attempts,
            "source_discoveries": source_discoveries,
            "certified": certified,
            "novel_saddles": novel_saddles,
            "degenerate_rearrangements": degenerate_rearrangements,
            "novel_edges": novel_edges,
            "mechanism_pulls": mechanism_allocator.pulls(),
            "mechanism_discovery_rates": mechanism_allocator.rates(),
            "move_pulls": move_allocator.pulls(),
            "move_success_rates": move_allocator.rates(),
            "known_basins": known_basins.len(),
            "charged": charged,
            "budget": budget,
            "producer_surface_evaluations": surface.evaluations(),
            "coordinator_charged": snapshot.aggregate_charged,
            "catalog_entries": snapshot.active_entries,
            "census_visits": snapshot.census_visits,
            "termination": termination,
            "accounted_producer_evaluations": expected_surface_evaluations,
        })
    );
    Ok(())
}
