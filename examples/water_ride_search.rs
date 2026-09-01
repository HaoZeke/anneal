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

use std::io;
use std::net::SocketAddr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use anneal_core::catalog::euclidean_gradient_norm;
use anneal_core::catalog::molecular::{
    MolecularCatalogPresetError, component_gradient_tolerance, descriptor_space,
    engine_binary_digest, length_scale, reference_coordinates, system_signature, water_groups,
    water_species,
};
use anneal_core::catalog_rpc::client::{CatalogClient, ClientConfig};
use anneal_core::catalog_rpc::{
    CatalogCandidate, CatalogIdentity, CatalogRideOutcome, CatalogRideReport,
};
use anneal_core::cooperative_search::ledger::ChargeKind;
use anneal_core::methods::cluster_hopping::repack_rigid_groups;
use anneal_core::pes_exploration::{PesExplorationConfig, PesSurface, RideMethod, quench_minimum};
use anneal_core::ride_execution::{CatalogRideExecutionConfig, execute_catalog_ride};
use anneal_core::shape::IraStructureWitness;
use common::rgpot_eindir::{RgpotObjective, emit_engine_manifest};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde_json::json;

const WATER_BOX: [f64; 9] = [60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 60.0];
const OXYGEN_MASS: f64 = 15.999;
const HYDROGEN_MASS: f64 = 1.008;
const EXACT_STRUCTURE_RADIUS: f64 = 1e-4;

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
    let stationary_component_tolerance = component_gradient_tolerance(coordinate_dim)?;
    Ok(PesExplorationConfig {
        ride_method: RideMethod::Dimer,
        quench_steps: 2_000,
        saddle_steps: 1_000,
        minimum_mode_force_tolerance: 5e-2,
        irc_steps: 200,
        prfo_steps: 300,
        quench_gradient_tolerance: stationary_component_tolerance,
        saddle_force_tolerance: stationary_component_tolerance,
        saddle_displacement: 0.1 * length_scale,
        negative_curvature_tolerance: 1e-6,
        hessian_step: 1e-4 * length_scale,
        maximum_move: 0.2 * length_scale,
        irc_step: 0.1 * length_scale,
        irc_force_tolerance: 0.05,
        refine_with_prfo: true,
        ..PesExplorationConfig::default()
    })
}

fn failure_name(outcome: &CatalogRideOutcome) -> Option<String> {
    match outcome {
        CatalogRideOutcome::Certified(_) => None,
        CatalogRideOutcome::Failed(failure) => Some(format!("{failure:?}")),
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
    let minimum = quench_minimum(
        &surface,
        start.view(),
        exploration.quench_steps,
        quench_tolerance,
    )?;
    let (minimum_energy, minimum_gradient) = surface.evaluate(minimum.coordinates.view())?;
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

    let mut producer_event_sequence = 2_u64;
    let mut attempts = 0_u64;
    let mut certified = 0_u64;
    let mut novel_edges = 0_u64;
    let mut reported_producer_calls = 0_u64;
    let termination;
    loop {
        let remaining = budget.saturating_sub(charged);
        if remaining <= receiver_reserve {
            termination = "receiver_reserve";
            break;
        }
        let maximum_evaluations = (remaining - receiver_reserve).min(ride_budget_cap);
        let Some(work) = client.claim_ride(
            next_sequence(&mut event_sequence)?,
            seed ^ attempts.wrapping_mul(0x9e37_79b9_7f4a_7c15),
        )?
        else {
            termination = "portfolio_exhausted";
            break;
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
            ChargeKind::AuxiliaryEvaluation,
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
        attempts += 1;
        certified += u64::from(credit.certified_connection);
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
                "novel_edge": credit.novel_edge,
                "failure": failure,
                "charged": charged,
            })
        );
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
            "attempts": attempts,
            "certified": certified,
            "novel_edges": novel_edges,
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
