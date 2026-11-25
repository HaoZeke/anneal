//! Architecture scaling and warm-start benchmark driver for LJ75.
//!
//! Uses `Config::communicating(75)`, `run_ensemble` (`src/methods/ensemble.rs`),
//! `PairPotential::lennard_jones(75)`, and `MinimaCorpus` (`src/minima_db.rs`)
//! without modifying `src/`.

use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anneal_core::bias::{Fingerprint, SortedPairs};
use anneal_core::catalog::different_packing_family;
use anneal_core::catalog::lj::{CALIBRATION_IRA_TOLERANCE, descriptor_space, system_signature};
use anneal_core::methods::cluster_hopping::{
    Config, Ledger, QuenchStatus, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::ensemble::{
    EnsembleConfig, EnsembleProblem, HistoryMode, ObjectiveFactory, StartFactory, run_ensemble,
};
use anneal_core::methods::minima_hopping::{HistoryMembership, SerializedWitness};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::minima_db::{MinimaCorpus, MinimaSet, MinimaUnits};
use anneal_core::pes_exploration::StructureContext;
use anneal_core::potentials::PairPotential;
use anneal_core::{DEFAULT_HIT_TOL, energy_hits};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Serialize;

const N_ATOMS: usize = 75;
const LJ75_REFERENCE: f64 = -397.492331;
const HIT_TOL: f64 = DEFAULT_HIT_TOL;
const CONVERGED_FMAX: f64 = 1.0e-3;
const CHECKPOINT_INTERVAL: usize = 1_000;

#[derive(Serialize)]
struct BenchRow {
    system: String,
    arm: String,
    seed: u64,
    workers: usize,
    total_oracle_calls: usize,
    call_budget: usize,
    hit: u8,
    calls_to_hit: usize,
    ckpt_calls_to_hit: Option<usize>,
    wall_to_hit_s: Option<f64>,
    best_energy: f64,
    wall_s: f64,
    cpu_s: f64,
    peak_rss_anon_kib: u64,
    replica_charged: String,
    hops: usize,
    basins: usize,
    screened_out: usize,
    returned: usize,
    history_obs: usize,
    history_new: usize,
    ledger_sum: usize,
    evals_observed: usize,
    warm_seed_energy: Option<f64>,
    corpus_recorded: Option<usize>,
}

fn read_proc_cpu_ticks() -> u64 {
    let Ok(text) = fs::read_to_string("/proc/self/stat") else {
        return 0;
    };
    let Some(rparen) = text.rfind(')') else {
        return 0;
    };
    let rest: Vec<&str> = text[rparen + 2..].split_whitespace().collect();
    if rest.len() < 13 {
        return 0;
    }
    let utime: u64 = rest[11].parse().unwrap_or(0);
    let stime: u64 = rest[12].parse().unwrap_or(0);
    utime + stime
}

fn read_rss_anon_kib() -> u64 {
    let Ok(text) = fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("RssAnon:") {
            let num = rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0);
            return num;
        }
    }
    0
}

struct RssSampler {
    stop: Arc<AtomicBool>,
    peak_kib: Arc<AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl RssSampler {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peak_kib = Arc::new(AtomicU64::new(read_rss_anon_kib()));
        let stop_bg = Arc::clone(&stop);
        let peak_bg = Arc::clone(&peak_kib);
        let handle = std::thread::spawn(move || {
            while !stop_bg.load(Ordering::Relaxed) {
                let cur = read_rss_anon_kib();
                peak_bg.fetch_max(cur, Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(10));
            }
        });
        Self {
            stop,
            peak_kib,
            handle: Some(handle),
        }
    }

    fn finish(mut self) -> u64 {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        let cur = read_rss_anon_kib();
        self.peak_kib.fetch_max(cur, Ordering::Relaxed);
        self.peak_kib.load(Ordering::Relaxed)
    }
}

fn fmax_of(g: &Array1<f64>) -> f64 {
    g.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()))
}

fn run_scaling(workers: usize, seed: u64, budget: usize) -> BenchRow {
    let cfg = Config::communicating(N_ATOMS);
    let target = LJ75_REFERENCE + HIT_TOL;
    let ens = EnsembleConfig {
        replicas: workers,
        budget,
        history: HistoryMode::Shared,
        membership: HistoryMembership::Accepted,
        shared_bias: None,
        gossip: None,
        two_choice_stall: None,
        checkpoint_interval: CHECKPOINT_INTERVAL,
        target: Some(target),
    };
    ens.validate(&cfg)
        .unwrap_or_else(|e| panic!("invalid ensemble config: {e}"));

    let descriptor = descriptor_space();
    let context = StructureContext::new(
        Some(vec![18; N_ATOMS]),
        None,
        Some(format!("lj-reduced-n{N_ATOMS}")),
    );
    let fingerprint = SortedPairs { n_points: N_ATOMS };
    let pair_cache: std::cell::RefCell<
        std::collections::HashMap<u64, Vec<(Vec<u64>, Arc<Vec<f64>>)>>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
    let witness = SerializedWitness(Mutex::new(
        move |left: ArrayView1<'_, f64>, right: ArrayView1<'_, f64>| {
            let get_desc = |v: ArrayView1<'_, f64>| -> Arc<Vec<f64>> {
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                for &x in v.iter() {
                    h ^= x.to_bits();
                    h = h.wrapping_mul(0x0100_0000_01b3);
                }
                {
                    let cache = pair_cache.borrow();
                    if let Some(bucket) = cache.get(&h) {
                        for (coords, desc) in bucket {
                            if coords.len() == v.len()
                                && coords
                                    .iter()
                                    .zip(v.iter())
                                    .all(|(&a, &b)| a == b.to_bits())
                            {
                                return Arc::clone(desc);
                            }
                        }
                    }
                }
                let coords: Vec<u64> = v.iter().map(|x| x.to_bits()).collect();
                let desc = Arc::new(fingerprint.describe(v).to_vec());
                pair_cache
                    .borrow_mut()
                    .entry(h)
                    .or_default()
                    .push((coords, Arc::clone(&desc)));
                desc
            };
            let l = get_desc(left);
            let r = get_desc(right);
            let tol2 = CALIBRATION_IRA_TOLERANCE * CALIBRATION_IRA_TOLERANCE;
            // Early reject from tail/middle elements before sequential scan.
            let n = l.len();
            if n > 0 {
                let d_last = l[n - 1] - r[n - 1];
                if d_last * d_last >= tol2 {
                    return false;
                }
                let d_mid = l[n / 2] - r[n / 2];
                if d_mid * d_mid >= tol2 {
                    return false;
                }
            }
            let mut sum2 = 0.0_f64;
            for (a, b) in l.iter().zip(r.iter()) {
                let d = a - b;
                sum2 += d * d;
                if sum2 >= tol2 {
                    return false;
                }
            }
            true
        },
    ));
    let same_family = |a: &[f64], b: &[f64]| !different_packing_family(a, b);

    let evals_observed = AtomicUsize::new(0);
    let first_hit_ns = AtomicU64::new(u64::MAX);
    let first_hit_evals = AtomicUsize::new(usize::MAX);
    let potential = PairPotential::lennard_jones(N_ATOMS);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .unwrap_or_else(|e| panic!("rayon pool failed: {e}"));

    let rss = RssSampler::start();
    let cpu_before = read_proc_cpu_ticks();
    let started = Instant::now();

    let objective: ObjectiveFactory<'_> = &|_| {
        let evals_observed = &evals_observed;
        let first_hit_ns = &first_hit_ns;
        let first_hit_evals = &first_hit_evals;
        let potential = &potential;
        Box::new(move |x: ArrayView1<f64>| {
            let idx = evals_observed.fetch_add(1, Ordering::Relaxed) + 1;
            let (e, g) = potential.value_and_gradient(x);
            if first_hit_ns.load(Ordering::Relaxed) == u64::MAX
                && energy_hits(e, LJ75_REFERENCE, HIT_TOL)
                && fmax_of(&g) <= CONVERGED_FMAX
            {
                let ns = started.elapsed().as_nanos().min(u128::from(u64::MAX - 1)) as u64;
                if first_hit_ns
                    .compare_exchange(u64::MAX, ns, Ordering::SeqCst, Ordering::Relaxed)
                    .is_ok()
                {
                    let _ = first_hit_evals.compare_exchange(
                        usize::MAX,
                        idx,
                        Ordering::SeqCst,
                        Ordering::Relaxed,
                    );
                }
            }
            (e, g)
        })
    };
    let start: StartFactory<'_> =
        &|_, rng| random_cluster_in_radius(N_ATOMS, cfg.start_radius(), cfg.min_separation, rng);

    let problem = EnsembleProblem {
        objective,
        start,
        descriptor: &descriptor,
        context: &context,
        witness: &witness,
        same_family: &same_family,
        certificate: 1.0e-5,
        polish_below: CONVERGED_FMAX,
        callbacks_per_objective: 1,
    };

    let report = pool.install(|| {
        run_ensemble(&cfg, &ens, seed, &problem)
            .unwrap_or_else(|e| panic!("run_ensemble failed: {e}"))
    });

    let wall_s = started.elapsed().as_secs_f64();
    let cpu_after = read_proc_cpu_ticks();
    let peak_rss_anon_kib = rss.finish();
    let cpu_s = cpu_after.saturating_sub(cpu_before) as f64 / 100.0;

    let hit_bool = energy_hits(report.best, LJ75_REFERENCE, HIT_TOL);
    let hit = u8::from(hit_bool);
    let observed = evals_observed.load(Ordering::SeqCst);
    let obj_hit_evals = first_hit_evals.load(Ordering::SeqCst);
    let obj_hit_ns = first_hit_ns.load(Ordering::SeqCst);
    let calls_to_hit = if hit_bool {
        if obj_hit_evals != usize::MAX {
            obj_hit_evals
        } else {
            report.first_target_calls.unwrap_or(report.aggregate_charged)
        }
    } else {
        budget + 1
    };
    let wall_to_hit_s = if hit_bool && obj_hit_ns != u64::MAX {
        Some(obj_hit_ns as f64 / 1.0e9)
    } else {
        None
    };

    let replica_charged_vec: Vec<usize> = report.replicas.iter().map(|r| r.charged).collect();
    let ledger_sum: usize = replica_charged_vec.iter().sum();
    assert_eq!(
        ledger_sum, report.aggregate_charged,
        "ledger sum must equal aggregate charged"
    );

    let replica_charged = replica_charged_vec
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(";");
    let hops: usize = report.replicas.iter().map(|r| r.outcome.hops).sum();
    let basins: usize = report.replicas.iter().map(|r| r.outcome.basins).sum();
    let screened_out: usize = report.replicas.iter().map(|r| r.outcome.screened_out).sum();
    let returned: usize = report.replicas.iter().map(|r| r.outcome.returned).sum();
    let history_obs: usize = report
        .replicas
        .iter()
        .map(|r| r.outcome.history_visits.0)
        .sum();
    let history_new: usize = report
        .replicas
        .iter()
        .map(|r| r.outcome.history_visits.1)
        .sum();

    BenchRow {
        system: "lj75".to_string(),
        arm: format!("comm-p{workers}"),
        seed,
        workers,
        total_oracle_calls: report.aggregate_charged,
        call_budget: budget,
        hit,
        calls_to_hit,
        ckpt_calls_to_hit: report.first_target_calls,
        wall_to_hit_s,
        best_energy: report.best,
        wall_s,
        cpu_s,
        peak_rss_anon_kib,
        replica_charged,
        hops,
        basins,
        screened_out,
        returned,
        history_obs,
        history_new,
        ledger_sum,
        evals_observed: observed,
        warm_seed_energy: None,
        corpus_recorded: None,
    }
}

fn select_warm_start(corpus_dir: &Path, cfg: &Config) -> (f64, Array1<f64>) {
    let sig = system_signature(N_ATOMS)
        .expect("LJ75 signature")
        .digest();
    let corpus = MinimaCorpus::open(corpus_dir, sig).expect("open minima corpus");
    let minima = corpus
        .minima(&format!("lj{N_ATOMS}"), cfg.temperature)
        .expect("read minima corpus");
    let chosen = minima
        .into_iter()
        .find(|m| !energy_hits(m.energy, LJ75_REFERENCE, HIT_TOL))
        .expect("corpus has at least one sub-target minimum");
    assert_eq!(chosen.coordinates.len(), 3 * N_ATOMS);
    (chosen.energy, Array1::from(chosen.coordinates))
}

fn run_single_chain(
    arm: &str,
    seed: u64,
    budget: usize,
    corpus_dir: Option<&Path>,
    record_corpus: bool,
) -> BenchRow {
    let cfg = Config::communicating(N_ATOMS);
    let potential = PairPotential::lennard_jones(N_ATOMS);

    let (warm_seed_energy, start) = match (arm, corpus_dir) {
        ("warm", Some(dir)) => {
            let (e, coords) = select_warm_start(dir, &cfg);
            (Some(e), coords)
        }
        _ => {
            let mut start_rng = StdRng::seed_from_u64(seed);
            let coords =
                random_cluster_in_radius(N_ATOMS, cfg.start_radius(), cfg.min_separation, &mut start_rng);
            (None, coords)
        }
    };

    let mut ledger = Ledger::new(budget);
    if !record_corpus {
        ledger.stop_below(LJ75_REFERENCE + HIT_TOL);
    }
    let mut optimiser = WarmLbfgs::default();
    let evals_observed = AtomicUsize::new(0);
    let first_hit_ns = AtomicU64::new(u64::MAX);
    let first_hit_evals = AtomicUsize::new(usize::MAX);

    let rss = RssSampler::start();
    let cpu_before = read_proc_cpu_ticks();
    let started = Instant::now();

    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            if !led.charge() {
                return None;
            }
            let idx = evals_observed.fetch_add(1, Ordering::Relaxed) + 1;
            let (e, g) = potential.value_and_gradient(v);
            if first_hit_ns.load(Ordering::Relaxed) == u64::MAX
                && energy_hits(e, LJ75_REFERENCE, HIT_TOL)
                && fmax_of(&g) <= CONVERGED_FMAX
            {
                let ns = started.elapsed().as_nanos().min(u128::from(u64::MAX - 1)) as u64;
                if first_hit_ns
                    .compare_exchange(u64::MAX, ns, Ordering::SeqCst, Ordering::Relaxed)
                    .is_ok()
                {
                    let _ = first_hit_evals.compare_exchange(
                        usize::MAX,
                        idx,
                        Ordering::SeqCst,
                        Ordering::Relaxed,
                    );
                }
            }
            Some((e, g))
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        evals_observed.fetch_add(1, Ordering::Relaxed);
        Some(potential.value_and_gradient(v).1)
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let outcome = run_with_gradient(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start");

    let wall_s = started.elapsed().as_secs_f64();
    let cpu_after = read_proc_cpu_ticks();
    let peak_rss_anon_kib = rss.finish();
    let cpu_s = cpu_after.saturating_sub(cpu_before) as f64 / 100.0;

    let corpus_recorded = if record_corpus {
        let dir = corpus_dir.expect("corpus dir required");
        let sig = system_signature(N_ATOMS)
            .expect("LJ75 signature")
            .digest();
        let set = MinimaSet {
            system: format!("lj{N_ATOMS}"),
            temperature: cfg.temperature,
            seed,
        };
        let mut entries: Vec<(f64, ArrayView1<f64>)> = ledger
            .quench_boundaries()
            .iter()
            .filter(|b| b.status() == QuenchStatus::Validated)
            .map(|b| (b.energy(), b.state()))
            .collect();
        if let Some(best_state) = outcome.best_state.as_ref() {
            entries.push((outcome.best, best_state.view()));
        }
        let corpus = MinimaCorpus::open(dir, sig).expect("open minima corpus");
        let stored = corpus
            .record(
                &set,
                &[],
                &MinimaUnits {
                    length: "sigma".into(),
                    energy: "epsilon".into(),
                },
                &entries,
                1.0e-6,
                serde_json::json!({
                    "driver": "arch_lj75_bench",
                    "preset": "Config::communicating",
                    "budget": budget,
                }),
            )
            .expect("record minima corpus");
        Some(stored)
    } else {
        None
    };

    let hit_bool = energy_hits(outcome.best, LJ75_REFERENCE, HIT_TOL);
    let hit = u8::from(hit_bool);
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, e)| energy_hits(*e, LJ75_REFERENCE, HIT_TOL))
        .map(|(_, spent, _, _)| *spent);
    let calls_to_hit = if hit_bool {
        at_hit.unwrap_or(outcome.charged)
    } else {
        budget + 1
    };
    let obj_hit_ns = first_hit_ns.load(Ordering::Relaxed);
    let wall_to_hit_s = if hit_bool {
        if obj_hit_ns != u64::MAX {
            Some(obj_hit_ns as f64 / 1.0e9)
        } else {
            Some(wall_s)
        }
    } else {
        None
    };

    BenchRow {
        system: "lj75".to_string(),
        arm: arm.to_string(),
        seed,
        workers: 1,
        total_oracle_calls: outcome.charged,
        call_budget: budget,
        hit,
        calls_to_hit,
        ckpt_calls_to_hit: at_hit,
        wall_to_hit_s,
        best_energy: outcome.best,
        wall_s,
        cpu_s,
        peak_rss_anon_kib,
        replica_charged: outcome.charged.to_string(),
        hops: outcome.hops,
        basins: outcome.basins,
        screened_out: outcome.screened_out,
        returned: outcome.returned,
        history_obs: outcome.history_visits.0,
        history_new: outcome.history_visits.1,
        ledger_sum: outcome.charged,
        evals_observed: evals_observed.load(Ordering::Relaxed),
        warm_seed_energy,
        corpus_recorded,
    }
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    let row = match cmd {
        "scaling" => {
            let workers: usize = args[1].parse().expect("workers");
            let seed: u64 = args[2].parse().expect("seed");
            let budget: usize = args[3].parse().expect("budget");
            run_scaling(workers, seed, budget)
        }
        "build-corpus" => {
            let dir = Path::new(&args[1]);
            let seed: u64 = args[2].parse().expect("seed");
            let budget: usize = args[3].parse().expect("budget");
            run_single_chain("corpus", seed, budget, Some(dir), true)
        }
        "single" => {
            let arm = args[1].as_str();
            let seed: u64 = args[2].parse().expect("seed");
            let budget: usize = args[3].parse().expect("budget");
            let dir = args.get(4).map(|s| Path::new(s.as_str()));
            run_single_chain(arm, seed, budget, dir, false)
        }
        _ => {
            eprintln!("usage: arch_lj75_bench <scaling W SEED BUDGET | build-corpus DIR SEED BUDGET | single cold|warm SEED BUDGET [DIR]>");
            std::process::exit(2);
        }
    };
    println!("{}", serde_json::to_string(&row).expect("serialize row"));
}
