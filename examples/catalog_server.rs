//! Isolated scientific catalog coordinator.
//!
//! Usage: `catalog_server <addr> <n> <capacity> <census-radius> <total-work>
//! <campaign> <ensemble> [replicas] [state-directory] [minimum-probes]
//! [maximum-region-distance] [Dirichlet-concentration] [diffusion-steps]`,
//! where replicas defaults to `0,1,2,3` and the state directory enables
//! restart-safe request replay.
//!
//! `CATALOG_SYSTEM` selects the preset. Unset or `lj` is reduced-unit LJ
//! (`n` is the site count). `CATALOG_SYSTEM=gfn2-water` selects the
//! GFN2-xTB universal-descriptor coordinator (`n` is the molecule count).
//! That arm requires `RGPOT_XTB_ENGINE` so the system signature hashes
//! the loaded engine. Feature `rgpot-ex` provides receiving-side energy and
//! force validation through that exact handle.

#[cfg(feature = "rgpot-ex")]
mod common;

use anneal_core::catalog::lj::{
    CALIBRATION_IRA_TOLERANCE, descriptor_space, fresh_evaluation, reference_coordinates,
    system_signature, validator_config,
};
use anneal_core::catalog_rpc::server::{CatalogServer, ServerConfig};
use anneal_core::shape::IraStructureWitness;
use anneal_core::transition_graph::AttractionRegionConfig;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if let Some(cfg) = anneal_core::campaign::CampaignConfig::bootstrap()? {
        print!("{}", cfg.banner());
    }
    match catalog_system()?.as_str() {
        "lj" => run_lj_coordinator(),
        "gfn2-water" => run_gfn2_water_coordinator(),
        other => Err(format!("CATALOG_SYSTEM must be lj or gfn2-water, not {other}").into()),
    }
}

fn catalog_system() -> Result<String, Box<dyn std::error::Error>> {
    match std::env::var("CATALOG_SYSTEM") {
        Err(std::env::VarError::NotPresent) => Ok("lj".into()),
        Ok(value) if value.is_empty() => Ok("lj".into()),
        Ok(value) => Ok(value),
        Err(error) => Err(error.into()),
    }
}

fn run_lj_coordinator() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let addr = required(&args, 1, "addr")?;
    let n_points = parse::<usize>(&args, 2, "n")?;
    let capacity = parse::<usize>(&args, 3, "capacity")?;
    let census_radius = parse::<f64>(&args, 4, "census-radius")?;
    let total_work = parse::<u64>(&args, 5, "total-work")?;
    let campaign = required(&args, 6, "campaign")?;
    let ensemble = required(&args, 7, "ensemble")?;
    let replicas = args
        .get(8)
        .map(String::as_str)
        .unwrap_or("0,1,2,3")
        .split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?;
    let minimum_probes = optional_parse(&args, 10, 8_u64)?;
    let maximum_distance = optional_parse(&args, 11, 0.35_f64)?;
    let concentration = optional_parse(&args, 12, 0.5_f64)?;
    let diffusion_steps = optional_parse(&args, 13, 2_usize)?;

    let signature = system_signature(n_points)?;
    let digest = signature.digest();
    let descriptor = descriptor_space();
    let reference = reference_coordinates(n_points)?;
    let descriptor_dim = descriptor
        .describe(
            ndarray::ArrayView1::from(&reference),
            Some(&signature.atomic_numbers),
        )?
        .values()
        .len();
    let mut config = ServerConfig::new(campaign, ensemble, digest, replicas)?
        .with_scientific_state(
            signature,
            descriptor,
            validator_config(&reference, descriptor_dim)?,
            capacity,
            census_radius,
            total_work,
            move |coordinates| fresh_evaluation(n_points, coordinates),
        )?
        .with_exact_structure_witness(IraStructureWitness {
            kmax_factor: 1.8,
            radius: CALIBRATION_IRA_TOLERANCE,
        })?
        .with_attraction_region_config(AttractionRegionConfig {
            probe_action: "probe".into(),
            concentration,
            diffusion_steps,
            maximum_distance,
            minimum_probes,
        })?;
    if let Some(directory) = args.get(9) {
        config = config.with_state_directory(directory)?;
    }
    park_coordinator(CatalogServer::start(addr, config)?)
}

/// GFN2-xTB universal-descriptor coordinator. `n` is the molecule count.
///
/// `RGPOT_XTB_ENGINE` must name the loaded `libxtb_engine.so` so the
/// system signature hashes the same binary used by receiving-side evaluation.
fn run_gfn2_water_coordinator() -> Result<(), Box<dyn std::error::Error>> {
    use anneal_core::catalog::FreshEvaluation;
    use anneal_core::catalog::molecular::{
        descriptor_space, engine_binary_digest, reference_coordinates, system_signature,
        validator_config, water_species,
    };

    let args = std::env::args().collect::<Vec<_>>();
    let addr = required(&args, 1, "addr")?;
    let n_molecules = parse::<usize>(&args, 2, "n")?;
    let capacity = parse::<usize>(&args, 3, "capacity")?;
    let census_radius = parse::<f64>(&args, 4, "census-radius")?;
    let total_work = parse::<u64>(&args, 5, "total-work")?;
    let campaign = required(&args, 6, "campaign")?;
    let ensemble = required(&args, 7, "ensemble")?;
    let replicas = args
        .get(8)
        .map(String::as_str)
        .unwrap_or("0,1,2,3")
        .split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?;
    let minimum_probes = optional_parse(&args, 10, 8_u64)?;
    let maximum_distance = optional_parse(&args, 11, 0.35_f64)?;
    let concentration = optional_parse(&args, 12, 0.5_f64)?;
    let diffusion_steps = optional_parse(&args, 13, 2_usize)?;

    let engine_path = std::env::var("RGPOT_XTB_ENGINE").map_err(
        |_| "CATALOG_SYSTEM=gfn2-water requires RGPOT_XTB_ENGINE (path of libxtb_engine.so)",
    )?;
    let engine_bytes = std::fs::read(&engine_path)
        .map_err(|error| format!("read RGPOT_XTB_ENGINE {engine_path}: {error}"))?;
    if engine_bytes.is_empty() {
        return Err("RGPOT_XTB_ENGINE must not be an empty file".into());
    }
    let digest = engine_binary_digest(&engine_bytes);

    let species = water_species(n_molecules)?;
    let reference = reference_coordinates(n_molecules)?;
    let descriptor = descriptor_space(&species)?;
    let descriptor_dim = descriptor
        .describe(
            ndarray::ArrayView1::from(reference.as_slice()),
            Some(&species),
        )?
        .values()
        .len();
    let signature = system_signature(n_molecules, digest)?;
    let evaluate: Box<dyn Fn(&[f64]) -> Result<FreshEvaluation, String> + Send + Sync + 'static> = {
        #[cfg(feature = "rgpot-ex")]
        {
            use common::rgpot_eindir::RgpotObjective;
            use std::sync::{Arc, Mutex};

            let atomic_numbers = species
                .iter()
                .map(|&atomic_number| atomic_number as i32)
                .collect::<Vec<_>>();
            let objective = Arc::new(Mutex::new(RgpotObjective::xtb(
                &atomic_numbers,
                [60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 60.0],
            )));
            Box::new(move |coordinates| {
                objective
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .fresh_evaluation(coordinates)
            })
        }
        #[cfg(not(feature = "rgpot-ex"))]
        {
            Box::new(|_| {
                Err("GFN2-xTB coordinator requires feature rgpot-ex and a loaded engine".into())
            })
        }
    };

    let mut config = ServerConfig::new(campaign, ensemble, signature.digest(), replicas)?
        .with_scientific_state(
            signature,
            descriptor,
            validator_config(&reference, descriptor_dim)?,
            capacity,
            census_radius,
            total_work,
            move |coordinates| evaluate(coordinates),
        )?
        .with_exact_structure_witness(IraStructureWitness {
            kmax_factor: 1.8,
            radius: 1e-4,
        })?
        .with_attraction_region_config(AttractionRegionConfig {
            probe_action: "probe".into(),
            concentration,
            diffusion_steps,
            maximum_distance,
            minimum_probes,
        })?;
    if let Some(directory) = args.get(9) {
        config = config.with_state_directory(directory)?;
    }
    park_coordinator(CatalogServer::start(addr, config)?)
}

fn park_coordinator(server: CatalogServer) -> Result<(), Box<dyn std::error::Error>> {
    let header = server.header();
    println!(
        "{{\"kind\":\"catalog_server_header\",\"campaign\":\"{}\",\"ensemble\":\"{}\",\"addr\":\"{}\",\"replicas\":{:?},\"initial_snapshot_version\":{},\"empty_state_proof\":{}}}",
        header.campaign,
        header.ensemble,
        server.addr(),
        header.replicas,
        header.initial_snapshot_version,
        header.empty_state_proof
    );
    std::io::stdout().flush()?;
    loop {
        std::thread::park();
    }
}

fn required<'a>(
    args: &'a [String],
    index: usize,
    name: &str,
) -> Result<&'a str, Box<dyn std::error::Error>> {
    args.get(index)
        .map(String::as_str)
        .ok_or_else(|| format!("missing required argument {name}").into())
}

fn parse<T>(args: &[String], index: usize, name: &str) -> Result<T, Box<dyn std::error::Error>>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + 'static,
{
    Ok(required(args, index, name)?.parse()?)
}

fn optional_parse<T>(
    args: &[String],
    index: usize,
    default: T,
) -> Result<T, Box<dyn std::error::Error>>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + 'static,
{
    args.get(index)
        .map(|value| value.parse::<T>())
        .transpose()
        .map(|value| value.unwrap_or(default))
        .map_err(Into::into)
}
