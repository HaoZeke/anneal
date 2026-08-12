//! Isolated scientific catalog coordinator for a reduced-unit LJ ensemble.
//!
//! Usage: `catalog_server <addr> <n> <capacity> <census-radius> <total-work>
//! <campaign> <ensemble> [replicas] [state-directory]`, where replicas defaults
//! to `0,1,2,3` and the state directory enables restart-safe request replay.

use anneal_core::catalog::lj::{
    descriptor_space, fresh_evaluation, reference_coordinates, system_signature, validator_config,
};
use anneal_core::catalog_rpc::server::{CatalogServer, ServerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
        )?;
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
