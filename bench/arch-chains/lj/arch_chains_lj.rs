//! Single-run driver for the communicating vs independent vs single-chain LJ campaign.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anneal_core::bias::{Fingerprint, SortedPairs};
use anneal_core::catalog::lj;
use anneal_core::methods::cluster_hopping::{Config, random_cluster_in_radius};
use anneal_core::methods::ensemble::{
    EnsembleConfig, EnsembleProblem, HistoryMode, ObjectiveFactory, StartFactory, run_ensemble,
};
use anneal_core::methods::minima_hopping::{HistoryMembership, SerializedWitness};
use anneal_core::minima_db::{MinimaCorpus, MinimaSet, MinimaUnits};
use anneal_core::pes_exploration::StructureContext;
use anneal_core::potentials::PairPotential;
use anneal_core::{SEARCH_HIT_TOL, energy_hits};
use ndarray::ArrayView1;

fn reference_energy(n: usize) -> f64 {
    match n {
        38 => -173.928427,
        75 => -397.492331,
        98 => -543.665361,
        other => panic!("unsupported LJ size {other}"),
    }
}

fn read_proc_status_kib() -> (u64, u64) {
    let Ok(text) = fs::read_to_string("/proc/self/status") else {
        return (0, 0);
    };
    let mut rss_anon = 0u64;
    let mut vm_hwm = 0u64;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("RssAnon:") {
            if let Some(val) = rest.split_whitespace().next().and_then(|s| s.parse().ok()) {
                rss_anon = val;
            }
        } else if let Some(rest) = line.strip_prefix("VmHWM:") {
            if let Some(val) = rest.split_whitespace().next().and_then(|s| s.parse().ok()) {
                vm_hwm = val;
            }
        }
    }
    (rss_anon, vm_hwm)
}

#[cfg(feature = "bank-rpc")]
fn process_cpu_seconds() -> f64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if rc != 0 {
        return 0.0;
    }
    let user = usage.ru_utime.tv_sec as f64 + usage.ru_utime.tv_usec as f64 * 1e-6;
    let sys = usage.ru_stime.tv_sec as f64 + usage.ru_stime.tv_usec as f64 * 1e-6;
    user + sys
}

#[cfg(not(feature = "bank-rpc"))]
fn process_cpu_seconds() -> f64 {
    let Ok(text) = fs::read_to_string("/proc/self/stat") else {
        return 0.0;
    };
    let Some(rparen) = text.rfind(')') else {
        return 0.0;
    };
    let fields: Vec<&str> = text[rparen + 2..].split_whitespace().collect();
    if fields.len() < 13 {
        return 0.0;
    }
    let utime: f64 = fields[11].parse().unwrap_or(0.0);
    let stime: f64 = fields[12].parse().unwrap_or(0.0);
    (utime + stime) / 100.0
}

fn main() {
    anneal_core::install_rgpot_kernels!();

    let args: Vec<String> = std::env::args().collect();
    let n: usize = args
        .get(1)
        .expect("usage: arch_chains_lj <n> <budget> <arm> <seed> <corpus_dir>")
        .parse()
        .expect("invalid n");
    let budget: usize = args.get(2).expect("missing budget").parse().expect("invalid budget");
    let arm = args.get(3).expect("missing arm").as_str();
    let seed: u64 = args.get(4).expect("missing seed").parse().expect("invalid seed");
    let corpus_dir = PathBuf::from(args.get(5).expect("missing corpus_dir"));

    let (replicas, history) = match arm {
        "comm" => (4usize, HistoryMode::Shared),
        "indep" => (4usize, HistoryMode::Private),
        "single" => (1usize, HistoryMode::Private),
        other => panic!("unknown arm {other:?}; expected comm, indep, or single"),
    };

    let reference = reference_energy(n);
    let cfg = Config::communicating(n);
    let ens = EnsembleConfig {
        replicas,
        budget,
        history,
        membership: HistoryMembership::Accepted,
        shared_bias: None,
        gossip: None,
        two_choice_stall: None,
        checkpoint_interval: 1000,
        target: Some(reference + SEARCH_HIT_TOL),
    };
    ens.validate(&cfg).expect("valid ensemble config");

    let descriptor = lj::descriptor_space();
    let fingerprint = SortedPairs { n_points: n };
    let ira_radius = lj::CALIBRATION_IRA_TOLERANCE;
    let witness = SerializedWitness(Mutex::new(
        move |left: ArrayView1<'_, f64>, right: ArrayView1<'_, f64>| {
            let l = fingerprint.describe(left);
            let r = fingerprint.describe(right);
            l.iter()
                .zip(r.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt()
                < ira_radius
        },
    ));
    let species = vec![18u32; n];
    let context = StructureContext::new(Some(species.clone()), None, Some(format!("lj-reduced-n{n}")));
    let objective: ObjectiveFactory<'_> = &|_| {
        let potential = PairPotential::lennard_jones(n);
        Box::new(move |x: ArrayView1<'_, f64>| potential.value_and_gradient(x))
    };
    let start: StartFactory<'_> =
        &|_, rng| random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, rng);
    let same_family = |a: &[f64], b: &[f64]| !anneal_core::catalog::different_packing_family(a, b);
    let problem = EnsembleProblem {
        objective,
        start,
        descriptor: &descriptor,
        context: &context,
        witness: &witness,
        same_family: &same_family,
        certificate: 1e-5,
        polish_below: 1e-3,
        callbacks_per_objective: 1,
    };

    let running = Arc::new(AtomicBool::new(true));
    let peak_rss_anon = Arc::new(AtomicU64::new(0));
    let peak_vm_hwm = Arc::new(AtomicU64::new(0));
    let (init_rss, init_hwm) = read_proc_status_kib();
    peak_rss_anon.store(init_rss, Ordering::Relaxed);
    peak_vm_hwm.store(init_hwm, Ordering::Relaxed);

    let watcher_flag = Arc::clone(&running);
    let watcher_rss = Arc::clone(&peak_rss_anon);
    let watcher_hwm = Arc::clone(&peak_vm_hwm);
    let watcher = thread::spawn(move || {
        while watcher_flag.load(Ordering::Relaxed) {
            let (rss, hwm) = read_proc_status_kib();
            watcher_rss.fetch_max(rss, Ordering::Relaxed);
            watcher_hwm.fetch_max(hwm, Ordering::Relaxed);
            thread::sleep(Duration::from_millis(10));
        }
    });

    let cpu_before = process_cpu_seconds();
    let wall_before = Instant::now();
    let report = run_ensemble(&cfg, &ens, seed, &problem).expect("ensemble run succeeded");
    let wall_s = wall_before.elapsed().as_secs_f64();
    let cpu_s = (process_cpu_seconds() - cpu_before).max(0.0);

    running.store(false, Ordering::Relaxed);
    let _ = watcher.join();
    let (end_rss, end_hwm) = read_proc_status_kib();
    let peak_rss_anon_kib = peak_rss_anon.load(Ordering::Relaxed).max(end_rss);
    let vm_hwm_kib = peak_vm_hwm.load(Ordering::Relaxed).max(end_hwm);

    let mut replica_charged_sum = 0usize;
    let mut outcome_charged_sum = 0usize;
    let mut hops_sum = 0usize;
    let mut screened_out_sum = 0usize;
    let mut returned_sum = 0usize;
    let mut basins_sum = 0usize;
    let mut history_obs_sum = 0usize;
    let mut history_new_sum = 0usize;
    let mut history_refused_sum = 0usize;
    let mut history_s_sum = 0.0f64;
    let mut orbits_sum = 0usize;
    let mut symmetrised_sum = 0usize;
    let mut continuous_sym_sum = 0usize;
    let mut visit_same_sum = 0usize;
    let mut visit_known_sum = 0usize;
    let mut visit_new_sum = 0usize;

    let verifier = PairPotential::lennard_jones(n);
    let mut corpus_entries: Vec<(f64, ArrayView1<'_, f64>)> = Vec::new();
    for rep in &report.replicas {
        replica_charged_sum += rep.charged;
        outcome_charged_sum += rep.outcome.charged;
        hops_sum += rep.outcome.hops;
        screened_out_sum += rep.outcome.screened_out;
        returned_sum += rep.outcome.returned;
        basins_sum += rep.outcome.basins;
        history_obs_sum += rep.outcome.history_visits.0;
        history_new_sum += rep.outcome.history_visits.1;
        history_refused_sum += rep.history_cost.1;
        history_s_sum += rep.history_cost.2;
        orbits_sum += rep.outcome.orbits.0;
        symmetrised_sum += rep.outcome.symmetrised.0;
        continuous_sym_sum += rep.outcome.continuous_symmetry.0;
        visit_same_sum += rep.outcome.visit_counts.0;
        visit_known_sum += rep.outcome.visit_counts.1;
        visit_new_sum += rep.outcome.visit_counts.2;

        if let Some(best_x) = rep.outcome.best_state.as_ref() {
            let (e, g) = verifier.value_and_gradient(best_x.view());
            let gmax = g.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            assert!(
                e.is_finite() && (e - rep.outcome.best).abs() < 1e-6,
                "replica {} energy mismatch: reported {} vs {}",
                rep.replica,
                rep.outcome.best,
                e
            );
            assert!(
                g.iter().all(|v| v.is_finite()) && gmax < 1e-3,
                "replica {} non-stationary best gradient {gmax:.2e}",
                rep.replica
            );
            corpus_entries.push((rep.outcome.best, best_x.view()));
        }
        if let Some(final_x) = rep.outcome.final_state.as_ref()
            && rep.outcome.final_energy.is_finite()
        {
            corpus_entries.push((rep.outcome.final_energy, final_x.view()));
        }
    }

    assert_eq!(
        report.aggregate_charged, replica_charged_sum,
        "aggregate_charged disagrees with sum(replica.charged)"
    );
    assert_eq!(
        replica_charged_sum, outcome_charged_sum,
        "sum(replica.charged) disagrees with sum(outcome.charged)"
    );

    let sig = lj::system_signature(n).expect("valid lj system signature");
    let corpus = MinimaCorpus::open(&corpus_dir, sig.digest()).expect("open minima corpus");
    let set = MinimaSet {
        system: format!("lj{n}-{arm}"),
        temperature: cfg.temperature,
        seed,
    };
    let units = MinimaUnits {
        length: "sigma".into(),
        energy: "epsilon".into(),
    };
    let corpus_appended = corpus
        .record(
            &set,
            &species,
            &units,
            &corpus_entries,
            1e-7,
            serde_json::json!({
                "arm": arm,
                "replicas": replicas,
                "budget": budget,
            }),
        )
        .expect("record minima corpus");

    let hit = energy_hits(report.best, reference, SEARCH_HIT_TOL);
    assert_eq!(hit, report.solved(ens.target));
    let calls_to_hit = if hit {
        report.first_target_calls.unwrap_or(report.aggregate_charged)
    } else {
        budget + 1
    };

    let history_minima_total: usize = report.histories.iter().map(|h| h.0).sum();
    let history_accepted_total: usize = report.histories.iter().map(|h| h.1).sum();
    let history_visits_total: u64 = report.histories.iter().map(|h| h.2).sum();

    let payload = serde_json::json!({
        "system": format!("lj{n}"),
        "n_points": n,
        "arm": arm,
        "replicas": replicas,
        "seed": seed,
        "budget": budget,
        "reference": reference,
        "best_energy": report.best,
        "hit": hit,
        "first_target_calls": report.first_target_calls,
        "calls_to_hit": calls_to_hit,
        "aggregate_charged": report.aggregate_charged,
        "ledger_sum": replica_charged_sum,
        "hops": hops_sum,
        "screened_out": screened_out_sum,
        "returned": returned_sum,
        "basins": basins_sum,
        "history_minima": history_minima_total,
        "history_accepted": history_accepted_total,
        "history_visits": history_visits_total,
        "history_obs": history_obs_sum,
        "history_new": history_new_sum,
        "history_refused": history_refused_sum,
        "history_s": history_s_sum,
        "orbits_count": orbits_sum,
        "symmetrised_count": symmetrised_sum,
        "continuous_sym_count": continuous_sym_sum,
        "visit_same": visit_same_sum,
        "visit_known": visit_known_sum,
        "visit_new": visit_new_sum,
        "corpus_appended": corpus_appended,
        "wall_s": wall_s,
        "cpu_s": cpu_s,
        "peak_rss_anon_kib": peak_rss_anon_kib,
        "vm_hwm_kib": vm_hwm_kib,
    });
    println!("RESULT_JSON:{payload}");
}
