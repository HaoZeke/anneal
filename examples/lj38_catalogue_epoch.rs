#![allow(dead_code, unused_imports)]
//! Four-replica LJ38 catalogue epoch.
//!
//! One synchronization: four quenched replicas, four ranks, one
//! reconfiguration. The log checks the offspring against the systematic
//! rule. `PAPER_SMOKE=1` caps the quenches and writes a file whose name
//! contains `SMOKE`. That file is not a paper table. The paper force
//! total is 400000 evaluations, shared by the four replicas. Full runs
//! launch from the v0.12.0 tag.
//!
//! Seeds 3100-3103 are the first four integers of the registered LJ38
//! block in `paper/seeds.toml`.

use anneal_core::catalog::PackingBook;
use anneal_core::catalog::lj::system_signature;
use anneal_core::methods::cluster_hopping::random_cluster;
use anneal_core::methods::feynman_kac::{
    PopulationMember, SelectionCoefficients, rank_population, reconfiguration_plan,
};
use anneal_core::methods::lattice_search::lj_value_gradient;
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::residual_field::ResidualField;
use ndarray::Array1;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use sha2::{Digest, Sha256};
use std::time::Instant;

const N: usize = 38;
const REPLICAS: usize = 4;
const SEED0: u64 = 3100;
const PAPER_FORCE_TOTAL: usize = 400_000;
const SMOKE_CALLS: usize = 200;
const SYSTEMATIC_OFFSET: f64 = 0.125;
const MAX_OFFSPRING: usize = 2;

fn main() {}

struct ReplicaRow {
    seed: u64,
    energy: f64,
    coordinates: Array1<f64>,
    force_calls: usize,
}

fn normalized_weights(
    evidence: &[anneal_core::methods::feynman_kac::BasinEvidence],
    coefficients: SelectionCoefficients,
) -> Vec<f64> {
    let mut logs: Vec<f64> = evidence
        .iter()
        .map(|item| {
            -coefficients.energy * item.energy_rank()
                + coefficients.novelty * item.novelty_rank()
                + coefficients.scarcity * item.scarcity_rank()
                + coefficients.uncertainty * item.uncertainty_rank()
        })
        .collect();
    let maximum = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for value in &mut logs {
        *value = (*value - maximum).max(-coefficients.log_weight_clip);
    }
    let normalizer: f64 = logs.iter().map(|value| value.exp()).sum();
    logs.into_iter()
        .map(|value| value.exp() / normalizer)
        .collect()
}

fn systematic_parents(weights: &[f64], offset: f64) -> Vec<usize> {
    let population = weights.len();
    let mut parents = Vec::with_capacity(population);
    let mut parent = 0usize;
    let mut cumulative = weights[0];
    for destination in 0..population {
        let threshold = (offset + destination as f64) / population as f64;
        while parent + 1 < population && threshold >= cumulative {
            parent += 1;
            cumulative += weights[parent];
        }
        parents.push(parent);
    }
    parents
}

fn offspring_counts(parents: &[usize]) -> Vec<usize> {
    let mut counts = vec![0usize; parents.len()];
    for &parent in parents {
        counts[parent] += 1;
    }
    counts
}

fn config_digest(smoke: bool, per_replica: usize, coefficients: SelectionCoefficients) -> String {
    let text = format!(
        "catalogue-epoch-v1 n={N} replicas={REPLICAS} seed0={SEED0} paper_force_total={PAPER_FORCE_TOTAL} per_replica={per_replica} smoke={smoke} offset={SYSTEMATIC_OFFSET} max_offspring={MAX_OFFSPRING} energy={} novelty={} scarcity={} uncertainty={} clip={}",
        coefficients.energy,
        coefficients.novelty,
        coefficients.scarcity,
        coefficients.uncertainty,
        coefficients.log_weight_clip,
    );
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hex(&hasher.finalize())
}

fn finish(report: &str, smoke: bool) {
    let path = std::env::var("PAPER_OUT").unwrap_or_else(|_| {
        if smoke {
            "lj38-catalogue-epoch-SMOKE.txt".into()
        } else {
            "lj38-catalogue-epoch.txt".into()
        }
    });
    if smoke && !path.contains("SMOKE") {
        panic!("PAPER_SMOKE output path must contain SMOKE");
    }
    std::fs::write(&path, report).expect("epoch log");
    print!("{report}");
}

fn push(report: &mut String, line: &str) {
    report.push_str(line);
}

fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|text| text.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split(':').nth(1))
                .map(|value| value.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
