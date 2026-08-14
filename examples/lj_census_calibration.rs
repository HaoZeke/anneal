//! Generate development-only same-minimum descriptor pairs for LJ census calibration.

use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

use anneal_core::catalog::lj::{
    CALIBRATION_GRADIENT_TOLERANCE, accepts_repeated_quench, descriptor_space,
    discovered_minimum_id, perturb_reference, system_signature,
};
use anneal_core::methods::cluster_hopping::{Config, random_cluster};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use anneal_core::shape::match_shapes;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;

struct QuenchEvidence {
    state: Array1<f64>,
    energy: f64,
    gradient_norm: f64,
    evaluations: usize,
}

const SOURCE_DENSITY: f64 = 0.7;

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(2 * bytes.len());
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

fn descriptor_schema_json(
    name: &str,
    version: u32,
    hyperparameters: &BTreeMap<String, String>,
    species_channels: &[u32],
) -> String {
    let parameters = hyperparameters
        .iter()
        .map(|(key, value)| format!("\"{key}\":\"{value}\""))
        .collect::<Vec<_>>()
        .join(",");
    let channels = species_channels
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"schema\":\"{name}\",\"version\":{version},\"hyperparameters\":{{{parameters}}},\"species_channels\":[{channels}]}}"
    )
}

fn quench(
    potential: &PairPotential,
    start: ArrayView1<f64>,
) -> Result<QuenchEvidence, Box<dyn Error>> {
    let mut optimizer = WarmLbfgs::default();
    let (energy, state, evaluations) = optimizer.minimize(start, 2_000, |coordinates| {
        Some(potential.value_and_gradient(coordinates))
    });
    let (fresh_energy, gradient) = potential.value_and_gradient(state.view());
    let gradient_norm = gradient
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if (energy - fresh_energy).abs() > 1e-10 {
        return Err("quench and fresh energies disagree".into());
    }
    Ok(QuenchEvidence {
        state,
        energy: fresh_energy,
        gradient_norm,
        evaluations,
    })
}

fn required<T: std::str::FromStr>(args: &[String], index: usize, name: &str) -> Result<T, String> {
    args.get(index)
        .ok_or_else(|| format!("missing {name}"))?
        .parse()
        .map_err(|_| format!("invalid {name}"))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 7 {
        return Err(format!(
            "usage: {} N PAIRS BASE_SEED SIGMA PAIRS_JSONL SIGNATURE_JSON",
            args.first().map_or("lj_census_calibration", String::as_str)
        )
        .into());
    }
    let n_points: usize = required(&args, 1, "N")?;
    let pair_count: usize = required(&args, 2, "pair count")?;
    let base_seed: u64 = required(&args, 3, "base seed")?;
    let perturbation_sigma: f64 = required(&args, 4, "perturbation sigma")?;
    if pair_count < 100 {
        return Err("pair count must be at least 100".into());
    }

    let signature = system_signature(n_points)?;
    let signature_digest = hex(&signature.digest());
    let schema_json = descriptor_schema_json(
        &signature.descriptor.schema,
        signature.descriptor.version,
        &signature.descriptor.hyperparameters,
        &signature.descriptor.species_channels,
    );
    let descriptor = descriptor_space();
    let potential = PairPotential::lennard_jones(n_points);
    let config = Config::recommended(n_points);
    let pair_path = Path::new(&args[5]);
    let manifest_path = Path::new(&args[6]);
    if let Some(parent) = pair_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = manifest_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut pairs = BufWriter::new(fs::File::create(pair_path)?);
    let mut accepted = 0usize;
    let mut attempt = 0u64;
    let mut source_rejections = 0u64;
    let mut quench_rejections = 0u64;
    let mut identity_rejections = 0u64;
    let maximum_attempts = u64::try_from(pair_count)?.saturating_mul(100);
    while accepted < pair_count && attempt < maximum_attempts {
        let source_seed = base_seed
            .checked_add(3 * attempt)
            .ok_or("source seed overflow")?;
        let left_seed = source_seed
            .checked_add(1)
            .ok_or("left perturbation seed overflow")?;
        let right_seed = left_seed
            .checked_add(1)
            .ok_or("right perturbation seed overflow")?;
        attempt += 1;
        let mut source_rng = rand::rngs::StdRng::seed_from_u64(source_seed);
        let source_start = random_cluster(
            n_points,
            SOURCE_DENSITY,
            config.min_separation,
            &mut source_rng,
        );
        if source_start.len() != 3 * n_points {
            source_rejections += 1;
            continue;
        }
        let source = match quench(&potential, source_start.view()) {
            Ok(source)
                if source.energy.is_finite()
                    && source.gradient_norm <= CALIBRATION_GRADIENT_TOLERANCE =>
            {
                source
            }
            _ => {
                source_rejections += 1;
                continue;
            }
        };
        let left_start = perturb_reference(
            source.state.as_slice().expect("source is contiguous"),
            n_points,
            left_seed,
            perturbation_sigma,
        )?;
        let right_start = perturb_reference(
            source.state.as_slice().expect("source is contiguous"),
            n_points,
            right_seed,
            perturbation_sigma,
        )?;
        let (left, right) = match (
            quench(&potential, ArrayView1::from(&left_start)),
            quench(&potential, ArrayView1::from(&right_start)),
        ) {
            (Ok(left), Ok(right)) => (left, right),
            _ => {
                quench_rejections += 1;
                continue;
            }
        };
        let left_ira_distance = match_shapes(source.state.view(), left.state.view(), 1.8)?.distance;
        let right_ira_distance =
            match_shapes(source.state.view(), right.state.view(), 1.8)?.distance;
        if !accepts_repeated_quench(
            source.energy,
            left.energy,
            left.gradient_norm,
            left_ira_distance,
        ) || !accepts_repeated_quench(
            source.energy,
            right.energy,
            right.gradient_norm,
            right_ira_distance,
        ) {
            identity_rejections += 1;
            continue;
        }
        let left_descriptor =
            descriptor.describe(left.state.view(), Some(&signature.atomic_numbers))?;
        let right_descriptor =
            descriptor.describe(right.state.view(), Some(&signature.atomic_numbers))?;
        let distance = left_descriptor.distance(&right_descriptor)?;
        let pair_id = format!("lj{n_points}-pair-{accepted:04}");
        let minimum_id = discovered_minimum_id(n_points, source_seed, source.energy);
        writeln!(
            pairs,
            concat!(
                "{{\"pair_id\":\"{}\",",
                "\"left_configuration_id\":\"seed-{}\",\"right_configuration_id\":\"seed-{}\",",
                "\"left_minimum_id\":\"{}\",\"right_minimum_id\":\"{}\",",
                "\"left_seed\":{},\"right_seed\":{},\"perturbation_sigma\":{:.17e},",
                "\"left_quench_evaluations\":{},\"right_quench_evaluations\":{},",
                "\"left_energy\":{:.17e},\"right_energy\":{:.17e},",
                "\"left_gradient_norm\":{:.17e},\"right_gradient_norm\":{:.17e},",
                "\"left_ira_distance\":{:.17e},\"right_ira_distance\":{:.17e},",
                "\"signature_digest\":\"{}\",\"descriptor_schema\":{},\"distance\":{:.17e}}}"
            ),
            pair_id,
            left_seed,
            right_seed,
            minimum_id,
            minimum_id,
            left_seed,
            right_seed,
            perturbation_sigma,
            left.evaluations,
            right.evaluations,
            left.energy,
            right.energy,
            left.gradient_norm,
            right.gradient_norm,
            left_ira_distance,
            right_ira_distance,
            signature_digest,
            schema_json,
            distance,
        )?;
        accepted += 1;
    }
    pairs.flush()?;
    if accepted != pair_count {
        return Err(format!(
            concat!(
                "accepted {accepted} of {pair_count} pairs in {attempt} deterministic attempts; ",
                "source_rejections={source_rejections} quench_rejections={quench_rejections} ",
                "identity_rejections={identity_rejections}"
            )
        )
        .into());
    }

    let mut manifest = BufWriter::new(fs::File::create(manifest_path)?);
    writeln!(
        manifest,
        concat!(
            "{{\n  \"artifact_schema_version\": 1,\n  \"system\": \"lj{}\",",
            "\n  \"signature_digest\": \"{}\",\n  \"descriptor_schema\": {},",
            "\n  \"source_policy\": \"seeded-random-cluster-quench-v1\",",
            "\n  \"source_density\": {:.17e},\n  \"source_min_separation\": {:.17e},",
            "\n  \"pair_count\": {},",
            "\n  \"identity_contract\": {{\"energy_abs_tolerance\": 1e-7,",
            "\"gradient_norm_tolerance\": 1e-5,\"ira_distance_tolerance\": 1e-4}},",
            "\n  \"attempt_count\": {},\n  \"source_rejections\": {},",
            "\n  \"quench_rejections\": {},",
            "\n  \"identity_rejections\": {},\n  \"base_seed\": {},",
            "\n  \"perturbation_sigma\": {:.17e}\n}}"
        ),
        n_points,
        signature_digest,
        schema_json,
        SOURCE_DENSITY,
        config.min_separation,
        pair_count,
        attempt,
        source_rejections,
        quench_rejections,
        identity_rejections,
        base_seed,
        perturbation_sigma,
    )?;
    manifest.flush()?;
    println!(
        "CALIBRATION_OK lj{n_points} pairs={accepted} attempts={attempt} signature={signature_digest}"
    );
    Ok(())
}
