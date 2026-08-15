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
//! GFN2-xTB leftover SOAP identity (`n` is the molecule count) and the
//! refusing GFN2 evaluator. Leftover SOAP is not a `DescriptorSpace`, so
//! that mode does not start a coordinator.

use anneal_core::catalog::lj::{
    descriptor_space, fresh_evaluation, reference_coordinates, system_signature, validator_config,
};
use anneal_core::catalog_rpc::server::{CatalogServer, ServerConfig};
use anneal_core::transition_graph::AttractionRegionConfig;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match catalog_system()?.as_str() {
        "lj" => run_lj_coordinator(),
        "gfn2-water" => refuse_gfn2_water_coordinator(),
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
    let server = CatalogServer::start(addr, config)?;
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

/// GFN2-xTB leftover SOAP and the refusing evaluator. This is not a
/// coordinator: leftover cannot be a `DescriptorSpace`, and a visit
/// cannot be fresh-evaluated without a loaded rgpot engine.
fn refuse_gfn2_water_coordinator() -> Result<(), Box<dyn std::error::Error>> {
    use anneal_core::catalog::molecular::{
        fresh_evaluation as water_fresh_evaluation, leftover_descriptor_dim, leftover_values,
        reference_coordinates, system_signature, validator_config, water_species,
    };

    let args = std::env::args().collect::<Vec<_>>();
    let n_molecules = parse::<usize>(&args, 2, "n")?;
    let species = water_species(n_molecules)?;
    let reference = reference_coordinates(n_molecules)?;
    let leftover = leftover_values(&reference, &species)?;
    let leftover_dim = leftover_descriptor_dim(&species)?;
    if leftover.len() != leftover_dim {
        return Err("water leftover length disagrees with leftover_descriptor_dim".into());
    }
    let _validator = validator_config(&reference, leftover_dim)?;

    // Identity constructor for this system. The digest argument is the
    // SHA-256 of the loaded libxtb_engine.so and is not invented here.
    let _identity = system_signature;
    if water_fresh_evaluation(n_molecules, &reference).is_ok() {
        return Err(
            "GFN2-xTB catalog evaluation invented an energy without a loaded rgpot engine".into(),
        );
    }

    // Type error if leftover is passed to with_scientific_state:
    //
    //   error[E0308]: mismatched types
    //     leftover
    //     ^^^^^^^^ expected `DescriptorSpace`, found `Vec<f64>`
    //     expected struct `anneal_core::descriptor_space::DescriptorSpace`
    //        found struct `Vec<f64>`
    //
    // leftover_values as the descriptor argument:
    //
    //   leftover_values
    //   ^^^^^^^^^^^^^^^ expected `DescriptorSpace`, found fn item
    //   expected struct `DescriptorSpace`
    //      found fn item `fn(&[f64], &[u32]) -> Result<Vec<f64>, MolecularCatalogPresetError>`
    //
    // DescriptorBlockKind is SoapMean | SoapVariance | AceNu3Mean. There
    // is no leftover aggregation, so describe() cannot be leftover SOAP.
    // A SoapMean space named jcc-water-soap-leftover would be a different
    // descriptor than stacked p_i - mu_z.
    //
    // Census radius is a CLI input and is unused here; this arm does not
    // invent one.
    Err(format!(
        "CATALOG_SYSTEM=gfn2-water cannot start a coordinator: leftover SOAP is Vec<f64> \
         (len {leftover_dim}), not DescriptorSpace (E0308: expected DescriptorSpace, found \
         Vec<f64>). DescriptorBlockKind is SoapMean | SoapVariance | AceNu3Mean. GFN2 visits \
         cannot be validated: fresh_evaluation refuses without a loaded rgpot engine handle."
    )
    .into())
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
