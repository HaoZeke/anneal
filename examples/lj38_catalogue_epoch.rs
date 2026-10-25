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

fn main() {
    let smoke = std::env::var("PAPER_SMOKE").ok().as_deref() == Some("1");
    let per_replica = if smoke {
        SMOKE_CALLS
    } else {
        PAPER_FORCE_TOTAL / REPLICAS
    };
    let mut report = String::new();
    push(
        &mut report,
        &format!(
            "label: {}\ndriver: lj38_catalogue_epoch\ncommit: {}\ncpu_model: {}\n",
            if smoke { "smoke" } else { "untagged" },
            git_commit(),
            cpu_model(),
        ),
    );
    let signature = system_signature(N).expect("LJ38 signature");
    let coefficients = SelectionCoefficients::default();
    let config_digest = config_digest(smoke, per_replica, coefficients);
    push(
        &mut report,
        &format!(
            "engine_kind: {}\nengine_digest: {}\nsystem_digest: {}\nconfig_digest: {}\n",
            signature.engine.kind,
            hex(&signature.engine.config_digest),
            hex(&signature.digest()),
            config_digest,
        ),
    );
    push(
        &mut report,
        &format!(
            "n: {N}\nreplicas: {REPLICAS}\nseed_start: {SEED0}\nseed_end: {}\npaper_force_total: {PAPER_FORCE_TOTAL}\nper_replica_cap: {per_replica}\nsystematic_offset: {SYSTEMATIC_OFFSET}\nmax_offspring: {MAX_OFFSPRING}\nlog_w: -r_E + 0.8 r_N + 0.6 r_S + 0.4 r_U\nclip_nats: {}\n",
            SEED0 + REPLICAS as u64 - 1,
            coefficients.log_weight_clip,
        ),
    );

    let started = Instant::now();
    let mut calls = 0usize;
    let mut book = PackingBook::default();
    let mut field = ResidualField::new();
    let mut rows: Vec<ReplicaRow> = Vec::with_capacity(REPLICAS);
    for replica in 0..REPLICAS {
        let seed = SEED0 + replica as u64;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut point = random_cluster(N, 0.7, 0.5, &mut rng);
        let mut best_energy = f64::INFINITY;
        let mut best_point = point.clone();
        let budget = per_replica;
        let mut spent = 0usize;
        let mut solver = WarmLbfgs::default();
        while spent < budget {
            let remaining = budget - spent;
            solver.forget();
            let (energy, quenched, used) = solver.minimize(point.view(), remaining, |x| {
                if spent >= budget {
                    return None;
                }
                spent += 1;
                calls += 1;
                let coords = x.as_slice().expect("LJ38 coordinates are contiguous");
                let mut gradient = vec![0.0; coords.len()];
                let value = lj_value_gradient(coords, &mut gradient);
                Some((value, Array1::from_vec(gradient)))
            });
            if !energy.is_finite() {
                finish(&report, smoke);
                panic!("replica {replica} quench energy is not finite");
            }
            if energy < best_energy {
                best_energy = energy;
                best_point = quenched;
            }
            if used == 0 || spent >= budget {
                break;
            }
            point = best_point.clone();
            for value in point.iter_mut() {
                *value += 0.05 * (rng.random::<f64>() - 0.5);
            }
        }
        book.observe(best_point.as_slice().expect("best point"));
        field.observe(replica, best_energy);
        if replica > 0 {
            field.edge(0, replica);
        }
        rows.push(ReplicaRow {
            seed,
            energy: best_energy,
            coordinates: best_point,
            force_calls: spent,
        });
    }
    let quench_wall = started.elapsed();

    let mut members = Vec::with_capacity(REPLICAS);
    for (replica, row) in rows.iter().enumerate() {
        let histogram = book.histogram(row.coordinates.as_slice().expect("row"));
        let visits = histogram
            .as_ref()
            .and_then(|histogram| book.family_of(histogram))
            .map(|family| book.visits(family))
            .filter(|visits| *visits > 0)
            .unwrap_or(1);
        let novelty = histogram
            .as_ref()
            .map(|histogram| book.novelty(histogram))
            .unwrap_or(0.0);
        let uncertainty = field.score(replica);
        push(
            &mut report,
            &format!(
                "replica: {replica} seed: {} energy: {:.8} visits: {visits} novelty: {novelty:.8} uncertainty: {uncertainty:.12} force_calls: {}\n",
                row.seed, row.energy, row.force_calls,
            ),
        );
        members.push(
            PopulationMember::new(
                replica as u32,
                row.energy,
                novelty,
                visits as f64,
                uncertainty,
            )
            .expect("replica evidence"),
        );
    }

    let ranked = rank_population(&members).expect("four distinct replicas");
    let evidence: Vec<_> = ranked.iter().map(|member| member.evidence()).collect();
    let plan = reconfiguration_plan(&evidence, coefficients, SYSTEMATIC_OFFSET, MAX_OFFSPRING)
        .expect("reconfiguration");
    let expected = normalized_weights(&evidence, coefficients);
    let weights_match = plan
        .weights()
        .iter()
        .zip(&expected)
        .all(|(got, want)| (got - want).abs() < 1e-12);
    let uncapped = systematic_parents(plan.weights(), SYSTEMATIC_OFFSET);
    let parents_match = if *offspring_counts(&uncapped).iter().max().unwrap_or(&0) <= MAX_OFFSPRING
    {
        plan.parents() == uncapped
    } else {
        plan.diagnostics().max_family_size <= MAX_OFFSPRING
    };
    let counts = offspring_counts(plan.parents());
    for replica in 0..REPLICAS {
        push(
            &mut report,
            &format!(
                "rank: {replica} r_E: {:.6} r_N: {:.6} r_S: {:.6} r_U: {:.6} weight: {:.8} parent: {} offspring: {}\n",
                evidence[replica].energy_rank(),
                evidence[replica].novelty_rank(),
                evidence[replica].scarcity_rank(),
                evidence[replica].uncertainty_rank(),
                plan.weights()[replica],
                plan.parents()[replica],
                counts[replica],
            ),
        );
    }
    let wall = started.elapsed();
    let calls_per_second = calls as f64 / quench_wall.as_secs_f64().max(1e-9);
    let estimate_seconds = PAPER_FORCE_TOTAL as f64 / calls_per_second;
    push(
        &mut report,
        &format!(
            "force_calls: {calls}\nwall_seconds: {wall:.6}\nquench_wall_seconds: {quench:.6}\nforce_calls_per_second: {calls_per_second:.4}\nestimated_paper_wall_seconds: {estimate_seconds:.4}\nestimated_paper_wall_note: scales the smoke quench, which includes four packing observations; a hop is extra\nweights_match: {weights_match}\nparents_match: {parents_match}\nmax_family_size: {max_family}\nunique_parents: {unique_parents}\n",
            wall = wall.as_secs_f64(),
            quench = quench_wall.as_secs_f64(),
            max_family = plan.diagnostics().max_family_size,
            unique_parents = plan.diagnostics().unique_parents,
        ),
    );
    finish(&report, smoke);
    if !(weights_match && parents_match) {
        std::process::exit(1);
    }
}

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
