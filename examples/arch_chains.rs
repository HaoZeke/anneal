// Copyright (c) anneal developers
//! One seed of a chain ensemble on Au55 or rigid TIP4P (H2O)20.
//!
//! ```text
//! arch_chains run <au55|tip4p20> <comm|indep|single> <seed> <revision>
//! arch_chains probe <au55|tip4p20> <comm|indep|single> <seed> <budget>
//! ```
//!
//! `run` spends the paper budget. `probe` spends the given budget and is
//! not a campaign row. Both print one JSON object on stdout.
//!
//! The search is `methods::ensemble::run_ensemble`. `comm` is four replicas
//! with one shared minimum history. `indep` is four replicas with a private
//! history each. `single` is one replica and the whole budget. The aggregate
//! budget is split by `EnsembleConfig::budgets`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use anneal_core::bias::{Fingerprint, SortedPairs};
use anneal_core::descriptor_space::{DescriptorGeometry, universal_descriptor_space};
use anneal_core::methods::cluster_hopping::{Config, random_cluster_in_radius};
use anneal_core::methods::ensemble::{
    EnsembleConfig, EnsembleProblem, EnsembleReport, HistoryMode, ObjectiveFactory, SameFamily,
    StartFactory, run_ensemble,
};
use anneal_core::methods::minima_hopping::HistoryMembership;
use anneal_core::pes_exploration::StructureContext;
use anneal_core::potentials::{
    GminTip4p, Gupta, gmin_tip4p_sites, gmin_water_geometry, random_tip4p_cluster,
};
use ndarray::ArrayView1;

const K: usize = 4;
const CHECKPOINT: usize = 1_000;
const AU55_REFERENCE: f64 = -191.03085280;
const TIP4P20_REFERENCE: f64 = -872.9887544952;
const AU55_BUDGET: usize = 1_000_000;
const TIP4P20_BUDGET: usize = 2_000_000;
const PAIR_RADIUS: f64 = 0.02;
const WATER_SITE_TOL: f64 = 0.05;

#[derive(Clone, Copy)]
enum System {
    Au55,
    Tip4p20,
}

#[derive(Clone, Copy)]
enum Arm {
    Comm,
    Indep,
    Single,
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let status = match args.first().map(String::as_str) {
        Some("run") => cmd_run(&args),
        Some("probe") => cmd_probe(&args),
        _ => {
            eprintln!("usage: arch_chains run|probe ...");
            2
        }
    };
    std::process::exit(status);
}

fn cmd_run(args: &[String]) -> i32 {
    let Some(system) = args.get(1).and_then(|text| parse_system(text)) else {
        eprintln!("run needs au55 or tip4p20");
        return 2;
    };
    let Some(arm) = args.get(2).and_then(|text| parse_arm(text)) else {
        eprintln!("run needs comm, indep, or single");
        return 2;
    };
    let Some(seed) = args.get(3).and_then(|text| text.parse::<u64>().ok()) else {
        eprintln!("run needs a seed");
        return 2;
    };
    let Some(revision) = args.get(4) else {
        eprintln!("run needs a revision");
        return 2;
    };
    if revision.contains('/') || revision.contains(' ') {
        eprintln!("revision must be a single token");
        return 2;
    }
    match execute(system, arm, seed, paper_budget(system), revision) {
        Ok(line) => {
            println!("{line}");
            0
        }
        Err(error) => {
            eprintln!("{error}");
            1
        }
    }
}

fn cmd_probe(args: &[String]) -> i32 {
    let Some(system) = args.get(1).and_then(|text| parse_system(text)) else {
        eprintln!("probe needs au55 or tip4p20");
        return 2;
    };
    let Some(arm) = args.get(2).and_then(|text| parse_arm(text)) else {
        eprintln!("probe needs comm, indep, or single");
        return 2;
    };
    let Some(seed) = args.get(3).and_then(|text| text.parse::<u64>().ok()) else {
        eprintln!("probe needs a seed");
        return 2;
    };
    let Some(budget) = args.get(4).and_then(|text| text.parse::<usize>().ok()) else {
        eprintln!("probe needs a budget");
        return 2;
    };
    if budget < K {
        eprintln!("probe budget must cover every replica");
        return 2;
    }
    match execute(system, arm, seed, budget, "probe") {
        Ok(line) => {
            println!("{line}");
            0
        }
        Err(error) => {
            eprintln!("{error}");
            1
        }
    }
}

fn parse_system(text: &str) -> Option<System> {
    match text {
        "au55" => Some(System::Au55),
        "tip4p20" => Some(System::Tip4p20),
        _ => None,
    }
}

fn parse_arm(text: &str) -> Option<Arm> {
    match text {
        "comm" => Some(Arm::Comm),
        "indep" => Some(Arm::Indep),
        "single" => Some(Arm::Single),
        _ => None,
    }
}

fn paper_budget(system: System) -> usize {
    match system {
        System::Au55 => AU55_BUDGET,
        System::Tip4p20 => TIP4P20_BUDGET,
    }
}

fn system_name(system: System) -> &'static str {
    match system {
        System::Au55 => "au55",
        System::Tip4p20 => "tip4p20",
    }
}

fn arm_name(arm: Arm) -> &'static str {
    match arm {
        Arm::Comm => "comm",
        Arm::Indep => "indep",
        Arm::Single => "single",
    }
}

fn reference_of(system: System) -> f64 {
    match system {
        System::Au55 => AU55_REFERENCE,
        System::Tip4p20 => TIP4P20_REFERENCE,
    }
}

fn ensemble_config(arm: Arm, budget: usize, reference: f64) -> EnsembleConfig {
    let replicas = match arm {
        Arm::Comm | Arm::Indep => K,
        Arm::Single => 1,
    };
    let history = match arm {
        Arm::Comm => HistoryMode::Shared,
        Arm::Indep | Arm::Single => HistoryMode::Private,
    };
    EnsembleConfig {
        replicas,
        budget,
        history,
        membership: HistoryMembership::Accepted,
        shared_bias: None,
        gossip: None,
        two_choice_stall: None,
        checkpoint_interval: CHECKPOINT,
        target: Some(reference + anneal_core::DEFAULT_HIT_TOL),
    }
}

struct Sample {
    report: EnsembleReport,
    wall_s: f64,
    cpu_s: f64,
    rss_anon_kb: u64,
}

fn sample_run(body: impl FnOnce() -> Result<EnsembleReport, String>) -> Result<Sample, String> {
    let cpu0 = cpu_seconds();
    let wall0 = Instant::now();
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(rss_anon_kb()));
    let stop_flag = Arc::clone(&stop);
    let peak_flag = Arc::clone(&peak);
    let sampler = thread::spawn(move || {
        while !stop_flag.load(Ordering::Relaxed) {
            let current = rss_anon_kb();
            peak_flag.fetch_max(current, Ordering::Relaxed);
            thread::sleep(Duration::from_millis(20));
        }
    });
    let report = body();
    stop.store(true, Ordering::Relaxed);
    let _ = sampler.join();
    let report = report?;
    peak.fetch_max(rss_anon_kb(), Ordering::Relaxed);
    Ok(Sample {
        report,
        wall_s: wall0.elapsed().as_secs_f64(),
        cpu_s: (cpu_seconds() - cpu0).max(0.0),
        rss_anon_kb: peak.load(Ordering::Relaxed),
    })
}

fn execute(
    system: System,
    arm: Arm,
    seed: u64,
    budget: usize,
    revision: &str,
) -> Result<String, String> {
    let sample = match system {
        System::Au55 => sample_run(|| run_au55(arm, seed, budget))?,
        System::Tip4p20 => sample_run(|| run_tip4p20(arm, seed, budget))?,
    };
    Ok(row_json(system, arm, seed, budget, revision, &sample))
}

fn run_au55(arm: Arm, seed: u64, budget: usize) -> Result<EnsembleReport, String> {
    let n = 55;
    let species = Gupta::copper_gold_species(0, n);
    let cfg = Config::communicating(n).with_species(species.clone(), None);
    let ens = ensemble_config(arm, budget, AU55_REFERENCE);
    ens.validate(&cfg)?;
    let descriptor =
        universal_descriptor_space(DescriptorGeometry::finite(cfg.length_scale).expect("scale"));
    let context = StructureContext::new(Some(species), None, Some("au55".into()));
    let pairs = SortedPairs { n_points: n };
    let witness = move |left: ArrayView1<'_, f64>, right: ArrayView1<'_, f64>| -> bool {
        if left.len() != 3 * n || right.len() != 3 * n {
            return false;
        }
        let l = pairs.describe(left);
        let r = pairs.describe(right);
        l.iter()
            .zip(r.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
            < PAIR_RADIUS
    };
    let objective: ObjectiveFactory<'_> = &|_| {
        let potential = Gupta::copper_gold(0, n);
        Box::new(move |x: ArrayView1<f64>| potential.value_and_gradient(x))
    };
    let start: StartFactory<'_> =
        &|_, rng| random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, rng);
    let same_family: SameFamily<'_> = &|_, _| false;
    let problem = EnsembleProblem {
        objective,
        start,
        descriptor: &descriptor,
        context: &context,
        witness: &witness,
        same_family,
        certificate: cfg.record_gradient,
        polish_below: 10.0 * cfg.record_gradient,
        callbacks_per_objective: 1,
    };
    run_ensemble(&cfg, &ens, seed, &problem)
}

fn run_tip4p20(arm: Arm, seed: u64, budget: usize) -> Result<EnsembleReport, String> {
    let n = 20;
    let mut cfg = Config::communicating_tip4p(n);
    cfg.rigid_body_geometry = Some(gmin_water_geometry());
    let ens = ensemble_config(arm, budget, TIP4P20_REFERENCE);
    ens.validate(&cfg)?;
    let descriptor = universal_descriptor_space(DescriptorGeometry::finite(1.0).expect("scale"));
    let context = StructureContext::new(None, None, Some("tip4p20".into()));
    let witness = move |left: ArrayView1<'_, f64>, right: ArrayView1<'_, f64>| -> bool {
        tip4p_same(
            left.as_slice().unwrap_or(&[]),
            right.as_slice().unwrap_or(&[]),
            n,
            WATER_SITE_TOL,
        )
    };
    let objective: ObjectiveFactory<'_> = &|_| {
        let potential = GminTip4p::new(n);
        Box::new(move |x: ArrayView1<f64>| potential.value_and_gradient(x))
    };
    let start: StartFactory<'_> = &|_, rng| random_tip4p_cluster(n, rng);
    let same_family: SameFamily<'_> = &|_, _| false;
    let problem = EnsembleProblem {
        objective,
        start,
        descriptor: &descriptor,
        context: &context,
        witness: &witness,
        same_family,
        certificate: tip4p_certificate(6 * n, cfg.record_gradient),
        polish_below: TIP4P_POLISH_BELOW,
        callbacks_per_objective: 1,
    };
    run_ensemble(&cfg, &ens, seed, &problem)
}

/// Euclidean norm matching a rigid root-mean-square force of `record`.
///
/// The ensemble quench certifies on the Euclidean norm. A rigid chain
/// records when the root-mean-square force is at or below the configured
/// record gradient. Half of that equivalent leaves the hop's comparison
/// inside the record window.
fn tip4p_certificate(dimensions: usize, record: f64) -> f64 {
    0.5 * record * (dimensions as f64).sqrt()
}

const TIP4P_POLISH_BELOW: f64 = 1.0e6;

fn row_json(
    system: System,
    arm: Arm,
    seed: u64,
    budget: usize,
    revision: &str,
    sample: &Sample,
) -> String {
    let report = &sample.report;
    let reference = reference_of(system);
    let hit = anneal_core::energy_hits(report.best, reference, anneal_core::DEFAULT_HIT_TOL);
    let calls = report.aggregate_charged;
    let calls_to_hit = if hit {
        report.first_target_calls.unwrap_or(calls)
    } else {
        budget.saturating_add(1)
    };
    let replica_charged: usize = report.replicas.iter().map(|row| row.charged).sum();
    let history_obs: usize = report.replicas.iter().map(|row| row.history_cost.0).sum();
    let history_refusals: usize = report.replicas.iter().map(|row| row.history_cost.1).sum();
    let shared_deposits: usize = report
        .replicas
        .iter()
        .map(|row| row.outcome.shared_deposits)
        .sum();
    let gossip_rounds: usize = report
        .replicas
        .iter()
        .map(|row| row.outcome.gossip_rounds)
        .sum();
    let two_choice: usize = report
        .replicas
        .iter()
        .map(|row| row.two_choice_restarts)
        .sum();
    let bias_published: u64 = report.replicas.iter().map(|row| row.bias_published).sum();
    let (history_minima, history_accepted, history_visits) = report
        .histories
        .iter()
        .fold((0usize, 0usize, 0u64), |acc, row| {
            (acc.0 + row.0, acc.1 + row.1, acc.2 + row.2)
        });
    serde_json::json!({
        "system": system_name(system),
        "arm": arm_name(arm),
        "seed": seed,
        "chains": report.replicas.len(),
        "calls": calls,
        "budget": budget,
        "hit": usize::from(hit),
        "calls_to_hit": calls_to_hit,
        "best_energy": report.best,
        "wall_s": sample.wall_s,
        "cpu_s": sample.cpu_s,
        "rss_anon_kb": sample.rss_anon_kb,
        "ledger_aggregate": calls,
        "ledger_replica_sum": replica_charged,
        "ledger_history_obs": history_obs,
        "ledger_history_refusals": history_refusals,
        "ledger_history_minima": history_minima,
        "ledger_history_accepted": history_accepted,
        "ledger_history_visits": history_visits,
        "ledger_bias_published": bias_published,
        "ledger_exchange_published": report.exchange.0,
        "ledger_exchange_delivered": report.exchange.1,
        "ledger_shared_deposits": shared_deposits,
        "ledger_gossip_rounds": gossip_rounds,
        "ledger_two_choice_restarts": two_choice,
        "ledger_sum": calls,
        "history_stores": report.histories.len(),
        "revision": revision,
        "nproc": nproc_count(),
        "cpu_model": cpu_model(),
    })
    .to_string()
}

fn cpu_seconds() -> f64 {
    let Ok(text) = std::fs::read_to_string("/proc/self/stat") else {
        return 0.0;
    };
    let Some((_, rest)) = text.rsplit_once(')') else {
        return 0.0;
    };
    let mut fields = rest.split_whitespace();
    // Field 3 is the first token after comm. utime is field 14, stime field 15.
    let utime: f64 = fields
        .nth(11)
        .and_then(|text| text.parse().ok())
        .unwrap_or(0.0);
    let stime: f64 = fields
        .next()
        .and_then(|text| text.parse().ok())
        .unwrap_or(0.0);
    (utime + stime) / clock_ticks()
}

fn clock_ticks() -> f64 {
    let ticks = libc_clk_tck();
    if ticks > 0 { ticks as f64 } else { 100.0 }
}

fn libc_clk_tck() -> i64 {
    const SC_CLK_TCK: i32 = 2;
    let ticks = libc_sysconf(SC_CLK_TCK);
    if ticks > 0 { ticks } else { 100 }
}

fn rss_anon_kb() -> u64 {
    let Ok(text) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("RssAnon:") else {
            continue;
        };
        return rest
            .split_whitespace()
            .next()
            .and_then(|text| text.parse().ok())
            .unwrap_or(0);
    }
    0
}

fn nproc_count() -> usize {
    // The process is pinned, so the affinity mask is one CPU. The row
    // records the machine's online processors.
    const SC_NPROCESSORS_ONLN: i32 = 84;
    let online = libc_sysconf(SC_NPROCESSORS_ONLN);
    if online > 0 { online as usize } else { 1 }
}

fn libc_sysconf(name: i32) -> i64 {
    unsafe extern "C" {
        fn sysconf(name: i32) -> i64;
    }
    unsafe { sysconf(name) }
}

fn cpu_model() -> String {
    let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") else {
        return "unknown".into();
    };
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("model name") else {
            continue;
        };
        let Some((_, name)) = rest.split_once(':') else {
            continue;
        };
        return name.trim().replace(',', " ");
    }
    "unknown".into()
}

fn tip4p_same(left: &[f64], right: &[f64], n: usize, tol: f64) -> bool {
    let Some(a) = lab_waters(left, n) else {
        return false;
    };
    let Some(b) = lab_waters(right, n) else {
        return false;
    };
    let tol2 = tol * tol;
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        for j in 0..n {
            if site_close(a[i][0], b[j][0], tol2) && hydrogens_close(a[i], b[j], tol2) {
                adj[i].push(j);
            }
        }
    }
    perfect_matching(&adj, n)
}

fn lab_waters(x: &[f64], n: usize) -> Option<Vec<[[f64; 3]; 3]>> {
    if x.len() != 6 * n {
        return None;
    }
    let body = gmin_tip4p_sites();
    let mut waters = Vec::with_capacity(n);
    for i in 0..n {
        let centre = [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
        let omega = [x[3 * n + 3 * i], x[3 * n + 3 * i + 1], x[3 * n + 3 * i + 2]];
        if !centre
            .iter()
            .chain(omega.iter())
            .all(|value| value.is_finite())
        {
            return None;
        }
        let mut mol = [[0.0; 3]; 3];
        for site in 0..3 {
            let placed = rotate(omega, body[site]);
            mol[site] = [
                centre[0] + placed[0],
                centre[1] + placed[1],
                centre[2] + placed[2],
            ];
        }
        waters.push(mol);
    }
    Some(waters)
}

fn rotate(omega: [f64; 3], vector: [f64; 3]) -> [f64; 3] {
    let theta = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();
    if theta < 1.0e-12 {
        return vector;
    }
    let axis = [omega[0] / theta, omega[1] / theta, omega[2] / theta];
    let (sin, cos) = theta.sin_cos();
    let cross = [
        axis[1] * vector[2] - axis[2] * vector[1],
        axis[2] * vector[0] - axis[0] * vector[2],
        axis[0] * vector[1] - axis[1] * vector[0],
    ];
    let dot = axis[0] * vector[0] + axis[1] * vector[1] + axis[2] * vector[2];
    [
        vector[0] * cos + cross[0] * sin + axis[0] * dot * (1.0 - cos),
        vector[1] * cos + cross[1] * sin + axis[1] * dot * (1.0 - cos),
        vector[2] * cos + cross[2] * sin + axis[2] * dot * (1.0 - cos),
    ]
}

fn site_close(a: [f64; 3], b: [f64; 3], tol2: f64) -> bool {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    d0 * d0 + d1 * d1 + d2 * d2 <= tol2
}

fn hydrogens_close(left: [[f64; 3]; 3], right: [[f64; 3]; 3], tol2: f64) -> bool {
    (site_close(left[1], right[1], tol2) && site_close(left[2], right[2], tol2))
        || (site_close(left[1], right[2], tol2) && site_close(left[2], right[1], tol2))
}

fn perfect_matching(adj: &[Vec<usize>], n: usize) -> bool {
    let mut assigned = vec![None; n];
    for start in 0..n {
        let mut seen = vec![false; n];
        if !augment(start, adj, &mut seen, &mut assigned) {
            return false;
        }
    }
    true
}

fn augment(
    left: usize,
    adj: &[Vec<usize>],
    seen: &mut [bool],
    assigned: &mut [Option<usize>],
) -> bool {
    for &right in &adj[left] {
        if seen[right] {
            continue;
        }
        seen[right] = true;
        if assigned[right].is_none_or(|owner| augment(owner, adj, seen, assigned)) {
            assigned[right] = Some(left);
            return true;
        }
    }
    false
}
