// Copyright (c) anneal developers
//! Gate driver for binary-LJ (`h2h_blj` spec) and `search_binary`.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anneal_core::binary_water::{BinaryLennardJonesProblem, search_binary};
use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::BinaryLennardJones;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

const HIT_TOL: f64 = 0.001;
const LABEL_A: u32 = 200;
const LABEL_B: u32 = 201;

#[cfg(feature = "hook-binary-lj")]
static HOOK_POT_NS: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "hook-binary-lj")]
fn timed_binary_lj(
    n_a: usize,
    eps: [f64; 3],
    sig: [f64; 3],
    cutoff: f64,
    positions: &[f64],
    gradient: &mut [f64],
) -> f64 {
    let mark = Instant::now();
    let energy = ::rgpot_pair::binary_lj(n_a, eps, sig, cutoff, positions, gradient);
    HOOK_POT_NS.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
    energy
}

fn install_kernels() {
    #[cfg(feature = "hook-binary-lj")]
    {
        anneal_core::potentials::install_cluster_kernels(
            anneal_core::potentials::ClusterKernels::new(
                ::rgpot_pair::cluster_lj,
                ::rgpot_pair::cambridge_morse,
                ::rgpot_pair::gupta,
                timed_binary_lj,
            ),
        );
    }
    #[cfg(not(feature = "hook-binary-lj"))]
    {
        anneal_core::install_rgpot_kernels!();
    }
}

fn main() {
    install_kernels();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    let status = match cmd {
        "blj" => cmd_blj(&args[1..]),
        "search_binary" => cmd_search_binary(&args[1..]),
        other => {
            eprintln!("usage: driver <blj|search_binary> ... (got {other:?})");
            2
        }
    };
    std::process::exit(status);
}

struct BljSpec {
    n: usize,
    n_a: usize,
    sigma_bb: f64,
    budget: usize,
    reference: f64,
}

fn blj_spec(name: &str) -> Option<BljSpec> {
    let (n, n_a, sigma_bb, reference) = match name {
        "blj105n38" => (38, 8, 1.05, -177.2090846748),
        "blj105n55" => (55, 13, 1.05, -283.1545313810),
        "blj115n38" => (38, 9, 1.15, -187.3676867453),
        "blj115n55" => (55, 16, 1.15, -289.2931853058),
        "blj130n38" => (38, 13, 1.30, -191.1840192602),
        "blj130n55" => (55, 19, 1.30, -306.7337643721),
        "blj130n98" => (98, 41, 1.30, -590.7874132760),
        _ => return None,
    };
    let budget = match n {
        38 => 400_000,
        55 => 1_000_000,
        98 => 4_000_000,
        _ => return None,
    };
    Some(BljSpec {
        n,
        n_a,
        sigma_bb,
        budget,
        reference,
    })
}

fn env_budget() -> Option<usize> {
    let text = std::env::var("H2H_BUDGET").ok()?;
    let budget = text.parse::<usize>().ok()?;
    (budget > 0).then_some(budget)
}

fn cmd_blj(args: &[String]) -> i32 {
    if args.len() != 2 {
        eprintln!("usage: driver blj <system> <seed>");
        return 2;
    }
    let system = args[0].as_str();
    let seed: u64 = match args[1].parse() {
        Ok(v) => v,
        Err(_) => {
            eprintln!("invalid seed {}", args[1]);
            return 2;
        }
    };
    let Some(spec) = blj_spec(system) else {
        eprintln!("unknown BLJ system {system}");
        return 2;
    };
    let budget = env_budget().unwrap_or(spec.budget);
    let mut labels = vec![LABEL_A; spec.n];
    for label in labels.iter_mut().skip(spec.n_a) {
        *label = LABEL_B;
    }
    let cfg = Config::recommended(spec.n).with_species(labels, None);
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(spec.n, cfg.start_radius(), cfg.min_separation, &mut rng);

    let evaluated = BinaryLennardJones::new(
        spec.n_a,
        spec.n - spec.n_a,
        1.0,
        1.0,
        1.0,
        1.0,
        0.5 * (1.0 + spec.sigma_bb),
        spec.sigma_bb,
    );
    let mut ledger = Ledger::new(budget);
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
            let pair = evaluated.value_and_gradient(v);
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
        let gradient = evaluated.value_and_gradient(v).1;
        pot_ns.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Some(gradient)
    };
    let mut search_rng = StdRng::seed_from_u64(seed);
    let started = Instant::now();
    let outcome: Outcome = run_with_gradient(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut search_rng,
    )
    .expect("finite start");
    let wall = started.elapsed().as_secs_f64();
    let pot_s = pot_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
    let outside_s = (wall - pot_s).max(0.0);

    let level = spec.reference + HIT_TOL;
    let hit = outcome.best <= level;
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| *energy <= level)
        .map(|(_, spent, _, _)| *spent);
    let evals = if hit {
        at_hit.or(Some(outcome.charged)).unwrap_or(budget)
    } else {
        budget
    };
    println!(
        "gate_row system={system} driver=h2h_blj seed={seed} budget={budget} hit={} evals_to_hit={evals} charged={} wall_s={wall:.6} best={:.10} pot_s={pot_s:.6} outside_s={outside_s:.6} pot_source=direct",
        u8::from(hit),
        outcome.charged,
        outcome.best,
    );
    0
}

fn read_xyz_rows(path: &Path) -> Vec<f64> {
    let file = File::open(path).expect("coords");
    let mut coords = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line.expect("coords line");
        if line.split_whitespace().count() < 3 {
            continue;
        }
        for token in line.split_whitespace().take(3) {
            coords.push(token.parse::<f64>().expect("coordinate"));
        }
    }
    coords
}

fn cmd_search_binary(args: &[String]) -> i32 {
    if args.len() != 4 {
        eprintln!("usage: driver search_binary <coords-file> <seed> <budget> <target>");
        return 2;
    }
    let coords_path = Path::new(&args[0]);
    let seed: u64 = args[1].parse().expect("seed");
    let budget: usize = args[2].parse().expect("budget");
    let target: f64 = args[3].parse().expect("target");
    let coords = read_xyz_rows(coords_path);
    assert_eq!(coords.len(), 300, "lj100-binary expects 100 atoms (300 coords)");

    let problem = BinaryLennardJonesProblem {
        n_a: 42,
        n_b: 58,
        epsilon_aa: 1.0,
        epsilon_ab: 1.0,
        epsilon_bb: 1.0,
        sigma_aa: 1.0,
        sigma_ab: 1.15,
        sigma_bb: 1.3,
    };

    #[cfg(feature = "hook-binary-lj")]
    HOOK_POT_NS.store(0, Ordering::Relaxed);

    let started = Instant::now();
    let found = search_binary(
        &problem,
        Array1::from_vec(coords.clone()).view(),
        budget,
        target,
        seed,
        false,
    );
    let wall = started.elapsed().as_secs_f64();
    assert!(!found.nve_unavailable, "search_binary requires rgsaddle feature");

    let (pot_s, pot_source) = {
        #[cfg(feature = "hook-binary-lj")]
        {
            (HOOK_POT_NS.load(Ordering::Relaxed) as f64 / 1.0e9, "hook")
        }
        #[cfg(not(feature = "hook-binary-lj"))]
        {
            // Arm A lacks ClusterKernels::binary_lj dispatch and search_binary
            // takes no potential callback, so measure per-eval BinaryLennardJones
            // latency post-run on the returned state.
            let potential = BinaryLennardJones::new(
                problem.n_a,
                problem.n_b,
                problem.epsilon_aa,
                problem.epsilon_ab,
                problem.epsilon_bb,
                problem.sigma_aa,
                problem.sigma_ab,
                problem.sigma_bb,
            );
            let sample = Array1::from_vec(found.state.clone());
            let reps = 2_000usize;
            let mark = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(potential.value_and_gradient(std::hint::black_box(sample.view())));
            }
            let per_eval = mark.elapsed().as_secs_f64() / reps as f64;
            (per_eval * found.charged as f64, "calibrated")
        }
    };
    let outside_s = (wall - pot_s).max(0.0);
    let hit = found.success(target);
    let evals = if hit { found.charged } else { budget };
    println!(
        "gate_row system=lj100-binary driver=search_binary seed={seed} budget={budget} hit={} evals_to_hit={evals} charged={} wall_s={wall:.6} best={:.10} pot_s={pot_s:.6} outside_s={outside_s:.6} pot_source={pot_source}",
        u8::from(hit),
        found.charged,
        found.energy,
    );
    0
}
