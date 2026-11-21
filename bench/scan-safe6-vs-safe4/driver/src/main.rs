// Copyright (c) anneal developers
//! One registered cell of the scan-safe6 against scan-safe4 gate.
//!
//! ```text
//! scan-safe6-vs-safe4 budget <cell>
//! scan-safe6-vs-safe4 run <cell> <seed>
//! ```
//!
//! The search is `Config::recommended` at the budgets and references
//! named in the gate notes. Gold uses the basin stride of the metal
//! driver. `H2H_*` and `ANNEAL_*` variables are refused.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{Gupta, PairPotential};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use sha2::{Digest, Sha256};
use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

const HIT_TOL: f64 = anneal_core::DEFAULT_HIT_TOL;

#[derive(Clone, Copy)]
struct MemoryPeak {
    rss_anon_kb: u64,
    vm_hwm_kb: u64,
}

struct MemorySampler {
    stop: Arc<AtomicBool>,
    worker: thread::JoinHandle<MemoryPeak>,
}

impl MemorySampler {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let worker = thread::spawn(move || {
            let mut peak = MemoryPeak {
                rss_anon_kb: 0,
                vm_hwm_kb: 0,
            };
            loop {
                let sample = read_self_status();
                peak.rss_anon_kb = peak.rss_anon_kb.max(sample.rss_anon_kb);
                peak.vm_hwm_kb = peak.vm_hwm_kb.max(sample.vm_hwm_kb);
                if worker_stop.load(Ordering::Acquire) {
                    return peak;
                }
                thread::sleep(Duration::from_millis(1));
            }
        });
        Self { stop, worker }
    }

    fn finish(self) -> MemoryPeak {
        self.stop.store(true, Ordering::Release);
        self.worker.join().expect("memory sampler")
    }
}

fn read_self_status() -> MemoryPeak {
    let text = fs::read_to_string("/proc/self/status").expect("read own process status");
    let mut sample = MemoryPeak {
        rss_anon_kb: 0,
        vm_hwm_kb: 0,
    };
    for line in text.lines() {
        if let Some(value) = line.strip_prefix("RssAnon:") {
            sample.rss_anon_kb = value
                .split_whitespace()
                .next()
                .expect("RssAnon value")
                .parse()
                .expect("numeric RssAnon");
        } else if let Some(value) = line.strip_prefix("VmHWM:") {
            sample.vm_hwm_kb = value
                .split_whitespace()
                .next()
                .expect("VmHWM value")
                .parse()
                .expect("numeric VmHWM");
        }
    }
    sample
}

#[derive(Clone, Copy)]
enum Kind {
    Lj,
    Morse { rho: f64 },
    GuptaAu,
}

#[derive(Clone, Copy)]
struct Spec {
    n: usize,
    budget: usize,
    reference: f64,
    kind: Kind,
}

fn system_spec(name: &str) -> Option<Spec> {
    let spec = match name {
        "lj38" => Spec {
            n: 38,
            budget: 400_000,
            reference: -173.928427,
            kind: Kind::Lj,
        },
        "lj55" => Spec {
            n: 55,
            budget: 1_000_000,
            reference: -279.248470,
            kind: Kind::Lj,
        },
        "lj75" => Spec {
            n: 75,
            budget: 4_000_000,
            reference: -397.492331,
            kind: Kind::Lj,
        },
        "lj98" => Spec {
            n: 98,
            budget: 4_000_000,
            reference: -543.665361,
            kind: Kind::Lj,
        },
        "lj102" => Spec {
            n: 102,
            budget: 4_000_000,
            reference: -569.363652,
            kind: Kind::Lj,
        },
        "lj103" => Spec {
            n: 103,
            budget: 4_000_000,
            reference: -575.766131,
            kind: Kind::Lj,
        },
        "lj104" => Spec {
            n: 104,
            budget: 4_000_000,
            reference: -582.086642,
            kind: Kind::Lj,
        },
        "lj150" => Spec {
            n: 150,
            budget: 4_000_000,
            reference: -893.310258,
            kind: Kind::Lj,
        },
        "au55" => Spec {
            n: 55,
            budget: 1_000_000,
            reference: -191.03085280,
            kind: Kind::GuptaAu,
        },
        "m80r6" => Spec {
            n: 80,
            budget: 4_000_000,
            reference: -378.333471,
            kind: Kind::Morse { rho: 6.0 },
        },
        "m80r14" => Spec {
            n: 80,
            budget: 4_000_000,
            reference: -340.811371,
            kind: Kind::Morse { rho: 14.0 },
        },
        _ => return None,
    };
    Some(spec)
}

fn run_atomic<F>(cfg: &Config, spec: Spec, seed: u64, eval_fn: F) -> (Outcome, f64, f64)
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Sync,
{
    let mut start_rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(
        spec.n,
        cfg.start_radius(),
        cfg.min_separation,
        &mut start_rng,
    );
    let mut ledger = Ledger::new(spec.budget);
    ledger.stop_below(spec.reference + HIT_TOL);
    let mut optimiser = WarmLbfgs::default();
    let pot_ns = AtomicU64::new(0);

    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            if !led.charge() {
                return None;
            }
            let mark = Instant::now();
            let pair = eval_fn(v);
            pot_ns.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
            Some(pair)
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        let mark = Instant::now();
        let gradient = eval_fn(v).1;
        pot_ns.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Some(gradient)
    };

    let mut search_rng = StdRng::seed_from_u64(seed);
    let started = Instant::now();
    let outcome = run_with_gradient(
        cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut search_rng,
    )
    .expect("finite start");
    let wall_s = started.elapsed().as_secs_f64();
    let inside_s = pot_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
    (outcome, wall_s, inside_s)
}

fn calls_to_hit(outcome: &Outcome, spec: Spec) -> (bool, usize) {
    let hit = outcome.best.is_finite() && outcome.best <= spec.reference + HIT_TOL;
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| energy.is_finite() && *energy <= spec.reference + HIT_TOL)
        .map(|(_, spent, _, _)| *spent);
    let calls = if hit {
        match at_hit {
            Some(spent) if spent <= spec.budget => spent,
            _ => outcome.charged.min(spec.budget),
        }
    } else {
        spec.budget
    };
    (hit, calls)
}

fn put_u64(hasher: &mut Sha256, value: u64) {
    hasher.update(value.to_le_bytes());
}

fn trajectory_digest(outcome: &Outcome, calls: usize) -> String {
    let mut hasher = Sha256::new();
    put_u64(&mut hasher, outcome.charged as u64);
    put_u64(&mut hasher, calls as u64);
    put_u64(&mut hasher, outcome.final_energy.to_bits());
    put_u64(&mut hasher, outcome.best.to_bits());
    put_u64(&mut hasher, outcome.accepted_transitions.len() as u64);
    for step in &outcome.accepted_transitions {
        put_u64(&mut hasher, step.hop as u64);
        let action = step.action.as_bytes();
        put_u64(&mut hasher, action.len() as u64);
        hasher.update(action);
        put_u64(&mut hasher, step.from_energy.to_bits());
        put_u64(&mut hasher, step.to_energy.to_bits());
        hasher.update([u8::from(step.adopted)]);
        put_u64(&mut hasher, step.to_state.len() as u64);
        for value in step.to_state.iter() {
            put_u64(&mut hasher, value.to_bits());
        }
    }
    match &outcome.final_state {
        Some(state) => {
            put_u64(&mut hasher, state.len() as u64);
            for value in state.iter() {
                put_u64(&mut hasher, value.to_bits());
            }
        }
        None => put_u64(&mut hasher, 0),
    }
    format!("{:x}", hasher.finalize())
}

fn emit(
    spec: Spec,
    seed: u64,
    hit: bool,
    calls: usize,
    outcome: &Outcome,
    wall_s: f64,
    inside_s: f64,
    nearest_index_calls: u64,
    memory: MemoryPeak,
) {
    let outside_s = (wall_s - inside_s).max(0.0);
    let digest = trajectory_digest(outcome, calls);
    println!(
        "cell,seed,budget,reference,hit,calls,charged,wall,inside_s,outside_s,best,best_bits,final,final_bits,trajectory_sha256,nearest_index_calls,rss_anon_peak_kb,vm_hwm_kb"
    );
    println!(
        "{cell},{seed},{budget},{reference:?},{hit},{calls},{charged},{wall_s:.9},{inside_s:.9},{outside_s:.9},{best:?},{best_bits},{final:?},{final_bits},{digest},{nearest_index_calls},{rss_anon_peak_kb},{vm_hwm_kb}",
        cell = std::env::args().nth(2).unwrap(),
        budget = spec.budget,
        reference = spec.reference,
        hit = usize::from(hit),
        charged = outcome.charged,
        best = outcome.best,
        best_bits = format!("{:016x}", outcome.best.to_bits()),
        final = outcome.final_energy,
        final_bits = format!("{:016x}", outcome.final_energy.to_bits()),
        rss_anon_peak_kb = memory.rss_anon_kb,
        vm_hwm_kb = memory.vm_hwm_kb,
    );
}

fn main() {
    for (key, _) in std::env::vars() {
        if key.starts_with("H2H_") || key.starts_with("ANNEAL_") {
            eprintln!("refusing forbidden environment variable {key}");
            std::process::exit(2);
        }
    }
    anneal_core::install_rgpot_kernels!();

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: scan-safe6-vs-safe4 <run|budget> <cell> [seed]");
        std::process::exit(2);
    }
    match args[0].as_str() {
        "budget" => {
            let cell = args.get(1).expect("budget needs a cell");
            let spec = system_spec(cell).unwrap_or_else(|| {
                eprintln!("unknown cell {cell}");
                std::process::exit(2);
            });
            println!("{}", spec.budget);
            println!("{:?}", spec.reference);
        }
        "run" => {
            let memory = MemorySampler::start();
            let nearest_before = anneal_core::symmetrise::nearest_index_call_count();
            let cell = args.get(1).expect("run needs a cell");
            let seed: u64 = args
                .get(2)
                .expect("run needs a seed")
                .parse()
                .expect("seed");
            let spec = system_spec(cell).unwrap_or_else(|| {
                eprintln!("unknown cell {cell}");
                std::process::exit(2);
            });
            let (outcome, wall_s, inside_s) = match spec.kind {
                Kind::Lj => {
                    let cfg = Config::recommended(spec.n);
                    let pot = PairPotential::lennard_jones(spec.n);
                    run_atomic(&cfg, spec, seed, |v| pot.value_and_gradient(v))
                }
                Kind::Morse { rho } => {
                    let cfg = Config::recommended(spec.n);
                    let pot = PairPotential::morse(spec.n, rho);
                    run_atomic(&cfg, spec, seed, |v| pot.value_and_gradient(v))
                }
                Kind::GuptaAu => {
                    let species = Gupta::copper_gold_species(0, spec.n);
                    let mut cfg = Config::recommended(spec.n).with_species(species, None);
                    cfg.basin_stride = 8;
                    let pot = Gupta::copper_gold(0, spec.n);
                    run_atomic(&cfg, spec, seed, |v| pot.value_and_gradient(v))
                }
            };
            let (hit, calls) = calls_to_hit(&outcome, spec);
            let nearest_index_calls =
                anneal_core::symmetrise::nearest_index_call_count() - nearest_before;
            let memory = memory.finish();
            emit(
                spec,
                seed,
                hit,
                calls,
                &outcome,
                wall_s,
                inside_s,
                nearest_index_calls,
                memory,
            );
        }
        other => {
            eprintln!("unknown command {other}");
            std::process::exit(2);
        }
    }
}
