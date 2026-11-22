// Copyright (c) anneal developers
//! One seed of `Config::recommended` on a Sutton-Chen cluster.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

const HIT_TOL: f64 = anneal_core::DEFAULT_HIT_TOL;

#[repr(C)]
struct ScConfig {
    epsilon: f64,
    c: f64,
    a: f64,
    n: i32,
    m: i32,
    cutoff: f64,
    periodic: i32,
}

struct Family {
    n_exp: i32,
    m_exp: i32,
    c: f64,
}

struct Spec {
    n: usize,
    budget: usize,
    reference: f64,
    family: Family,
}

fn main() {
    std::hint::black_box(rnext2_gate_driver::placement_pad_byte());
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("run");
    let status = match cmd {
        "run" => cmd_run(&args),
        "budget" => cmd_budget(&args),
        _ => {
            eprintln!("unknown command {cmd}");
            2
        }
    };
    std::process::exit(status);
}

fn family(name: &str) -> Option<Family> {
    let (n_exp, m_exp, c) = match name {
        "126" => (12, 6, 144.41),
        "108" => (10, 8, 34.408),
        "96" => (9, 6, 39.432),
        _ => return None,
    };
    Some(Family { n_exp, m_exp, c })
}

fn system_spec(name: &str) -> Option<Spec> {
    let (stem, n) = name.rsplit_once('-')?;
    let stem = stem.strip_prefix("sc")?;
    let n: usize = n.parse().ok()?;
    let family = family(stem)?;
    let (reference, budget) = match (stem, n) {
        ("126", 38) => (-35419.9804, 400_000),
        ("126", 55) => (-52884.6806, 1_000_000),
        ("126", 80) => (-78414.6271, 1_000_000),
        ("108", 38) => (-10117.2454, 400_000),
        ("108", 55) => (-14814.5225, 1_000_000),
        ("108", 80) => (-21819.9952, 1_000_000),
        ("96", 38) => (-8917.7056, 400_000),
        ("96", 55) => (-13217.8963, 1_000_000),
        ("96", 80) => (-19560.8966, 1_000_000),
        _ => return None,
    };
    Some(Spec {
        n,
        budget,
        reference,
        family,
    })
}

fn sutton_chen_candidates() -> Vec<String> {
    let mut candidates = Vec::new();
    for key in ["RGPOT_SC_LIB", "RGPOT_LIBRARY", "RGPOT_RGPOT_ENGINE"] {
        if let Ok(path) = std::env::var(key)
            && !path.is_empty()
            && !candidates.contains(&path)
        {
            candidates.push(path);
        }
    }
    for path in ["librgpot.so", "./librgpot.so"] {
        if !candidates.iter().any(|candidate| candidate == path) {
            candidates.push(path.to_string());
        }
    }
    candidates
}

fn library() -> Result<libloading::Library, String> {
    let mut failures = Vec::new();
    for candidate in sutton_chen_candidates() {
        match unsafe { libloading::Library::new(&candidate) } {
            Ok(library) => {
                let symbol: Result<libloading::Symbol<*mut std::ffi::c_void>, _> =
                    unsafe { library.get(b"rgpot_sutton_chen_energy_forces\0") };
                if symbol.is_ok() {
                    return Ok(library);
                }
                failures.push(format!(
                    "{candidate}: missing rgpot_sutton_chen_energy_forces"
                ));
            }
            Err(error) => failures.push(format!("{candidate}: {error}")),
        }
    }
    Err(format!(
        "rgpot Sutton-Chen library not found ({})",
        failures.join("; ")
    ))
}

fn evaluate(lib: &libloading::Library, family: &Family, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let cfg = ScConfig {
        epsilon: 1.0,
        c: family.c,
        a: 1.0,
        n: family.n_exp,
        m: family.m_exp,
        cutoff: 0.0,
        periodic: 0,
    };
    let mut energy = 0.0;
    let mut forces = vec![0.0; x.len()];
    let positions = x.as_slice().expect("coordinates are contiguous");
    type Eval = unsafe extern "C" fn(
        *const ScConfig,
        i32,
        *const f64,
        *const f64,
        *mut f64,
        *mut f64,
    ) -> i32;
    let func: libloading::Symbol<Eval> = unsafe {
        lib.get(b"rgpot_sutton_chen_energy_forces\0")
            .expect("rgpot_sutton_chen_energy_forces")
    };
    let status = unsafe {
        func(
            &cfg,
            n as i32,
            positions.as_ptr(),
            std::ptr::null(),
            &mut energy,
            forces.as_mut_ptr(),
        )
    };
    assert_eq!(status, 0, "rgpot Sutton-Chen rejected the geometry");
    let gradient = Array1::from_vec(forces.iter().map(|force| -force).collect());
    (energy, gradient)
}

fn cmd_budget(args: &[String]) -> i32 {
    let Some(system) = args.get(1).map(String::as_str) else {
        eprintln!("budget needs a system");
        return 2;
    };
    let Some(spec) = system_spec(system) else {
        eprintln!("unknown system {system}");
        return 2;
    };
    println!("{}", spec.budget);
    println!("{:.10}", spec.reference);
    0
}

fn cmd_run(args: &[String]) -> i32 {
    let Some(system) = args.get(1).map(String::as_str) else {
        eprintln!("run needs a system");
        return 2;
    };
    let Some(seed) = args.get(2).and_then(|v| v.parse::<u64>().ok()) else {
        eprintln!("run needs a seed");
        return 2;
    };
    let Some(spec) = system_spec(system) else {
        eprintln!("unknown system {system}");
        return 2;
    };
    let budget = spec.budget;
    let cfg = Config::recommended(spec.n);
    let mut start_rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(
        spec.n,
        cfg.start_radius(),
        cfg.min_separation,
        &mut start_rng,
    );
    let lib = match library() {
        Ok(lib) => lib,
        Err(error) => {
            eprintln!("{error}");
            return 1;
        }
    };
    let (outcome, wall, pot_s) = search(&lib, &cfg, &spec, &start, seed, budget);
    let hit = anneal_core::energy_hits(outcome.best, spec.reference, HIT_TOL);
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| anneal_core::energy_hits(*energy, spec.reference, HIT_TOL))
        .map(|(_, spent, _, _)| *spent);
    let evals = if hit {
        at_hit.or(hit.then_some(outcome.charged)).unwrap_or(budget)
    } else {
        budget
    };
    let outside = (wall - pot_s).max(0.0);
    println!("system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s");
    println!(
        "{system},anneal,{seed},{budget},{hit},{evals},{},{wall:.6},{:.8},{pot_s:.6},{outside:.6}",
        outcome.charged, outcome.best
    );
    rnext2_gate_driver::print_h2h_mem(outcome.best);
    0
}

fn search(
    lib: &libloading::Library,
    cfg: &Config,
    spec: &Spec,
    start: &Array1<f64>,
    seed: u64,
    budget: usize,
) -> (Outcome, f64, f64) {
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(spec.reference + HIT_TOL);
    let mut optimiser = WarmLbfgs::default();
    let pot_ns = std::sync::atomic::AtomicU64::new(0);
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            if !led.charge() {
                return None;
            }
            let mark = Instant::now();
            let pair = evaluate(lib, &spec.family, v);
            pot_ns.fetch_add(
                mark.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            Some(pair)
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        let mark = Instant::now();
        let gradient = evaluate(lib, &spec.family, v).1;
        pot_ns.fetch_add(
            mark.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        Some(gradient)
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let started = Instant::now();
    let outcome = run_with_gradient(
        cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start");
    let pot_s = pot_ns.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1.0e9;
    (outcome, started.elapsed().as_secs_f64(), pot_s)
}
