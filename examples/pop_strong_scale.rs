//! Strong scaling for the population methods.
//!
//! The Lennard-Jones problem is fixed: atom count, walker count, generation
//! count, quench length, and diffusion Monte Carlo budget. The worker count
//! is the thread pool and takes the values 1, 2, 4, 8, 16, and 32. These
//! runs are scaling measurements. They are not quality rows.

use std::env;
use std::process::ExitCode;
use std::time::Instant;

use anneal_core::methods::dmc_population::{
    run_dmc_population, DEFAULT_BETA0, DEFAULT_STEPS_PER_CONTROL,
};
use anneal_core::methods::population_annealing::{
    population_anneal_threads, population_halving_threads, PopulationAnneal,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Lennard-Jones atoms. Fixed for every worker count.
const ATOMS: usize = 98;
/// Walkers in the roster. Fixed for every worker count.
const WALKERS: usize = 32;
/// Rounds. Fixed for every worker count.
const GENERATIONS: usize = 400;
/// Closing quench iterations for population annealing.
const QUENCH_STEPS: usize = 40;
/// Evaluation budget for diffusion Monte Carlo, gradient included.
const DMC_BUDGET: usize = 20000;

/// Lennard-Jones objective. Proposal batches use the pool installed around
/// the diffusion Monte Carlo run. Gradient polish also uses that pool: the
/// search schedules one wave of elite passes and merges them in order.
struct BatchPotential {
    pot: PairPotential,
}

impl Objective<f64> for BatchPotential {
    fn dim(&self) -> usize {
        3 * self.pot.n_points
    }

    fn bounds(&self) -> &Bounds<f64> {
        self.pot.bounds()
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.pot.eval(x)
    }

    fn eval_batch(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut out = vec![0.0; x.nrows()];
        use rayon::prelude::*;
        out.par_iter_mut().enumerate().for_each(|(i, slot)| {
            *slot = self.pot.eval(x.row(i));
        });
        Array1::from(out)
    }
}

impl Gradient<f64> for BatchPotential {
    fn dim(&self) -> usize {
        3 * self.pot.n_points
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        Gradient::grad(&self.pot, x)
    }
}

struct Run {
    best_energy: f64,
    evaluations: usize,
    decisions: usize,
    n_evals: usize,
    n_grads: usize,
    n_accepts: usize,
    position_fnv: u64,
}

fn position_fnv(position: &Array1<f64>) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for value in position.iter() {
        hash ^= value.to_bits();
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn cpu_model() -> String {
    let text = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("model name") else {
            continue;
        };
        let name = rest.split_once(':').map(|(_, value)| value).unwrap_or("");
        let name = name.replace("(R)", "").replace("(TM)", "");
        return name.split_whitespace().collect::<Vec<_>>().join(" ");
    }
    "unknown".to_string()
}

fn memory_kib() -> u64 {
    let text = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("MemTotal:") else {
            continue;
        };
        return rest
            .split_whitespace()
            .next()
            .unwrap_or("0")
            .parse()
            .unwrap_or(0);
    }
    0
}

fn gpu_name() -> String {
    if std::path::Path::new("/proc/driver/nvidia/version").is_file() {
        "nvidia".to_string()
    } else {
        "none".to_string()
    }
}

fn logical_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn physical_cpus() -> usize {
    let text = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("cpu cores") else {
            continue;
        };
        let count = rest
            .split_once(':')
            .map(|(_, value)| value.trim())
            .unwrap_or("0");
        if let Ok(count) = count.parse::<usize>() {
            if count > 0 {
                return count;
            }
        }
    }
    logical_cpus()
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|item| item == name)
        .and_then(|index| args.get(index + 1))
        .cloned()
}

fn parse_usize(args: &[String], name: &str, default: usize) -> Result<usize, ExitCode> {
    let Some(text) = flag(args, name) else {
        return Ok(default);
    };
    text.parse::<usize>().map_err(|_| {
        eprintln!("expected an integer for {name}");
        ExitCode::from(2)
    })
}

fn run_population(
    workers: usize,
    seed: u64,
    atoms: usize,
    walkers: usize,
    generations: usize,
    quench_steps: usize,
) -> Run {
    let mut cfg = PopulationAnneal::reference(atoms);
    cfg.walkers = walkers;
    cfg.quench_steps = quench_steps;
    let pot = PairPotential::lennard_jones(atoms);
    let mut rng = StdRng::seed_from_u64(seed);
    let report = population_anneal_threads(
        atoms,
        &cfg,
        generations,
        workers,
        &|x| pot.eval(x),
        &|x| {
            let mut optimiser = WarmLbfgs::default();
            let (value, relaxed, calls) =
                optimiser.minimize(x, quench_steps, |v| Some(pot.value_and_gradient(v)));
            value.is_finite().then_some((value, relaxed, calls))
        },
        &mut rng,
    );
    Run {
        best_energy: report.best_energy,
        evaluations: report.evaluations,
        decisions: report.decisions,
        n_evals: report.evaluations,
        n_grads: 0,
        n_accepts: 0,
        position_fnv: 0,
    }
}

fn run_halving(workers: usize, seed: u64, atoms: usize, walkers: usize, generations: usize) -> Run {
    let mut cfg = PopulationAnneal::reference(atoms);
    cfg.walkers = walkers;
    let pot = PairPotential::lennard_jones(atoms);
    let mut rng = StdRng::seed_from_u64(seed);
    let report = population_halving_threads(
        atoms,
        &cfg,
        generations,
        workers,
        &|x| pot.eval(x),
        &mut rng,
    );
    Run {
        best_energy: report.best_energy,
        evaluations: report.evaluations,
        decisions: report.decisions,
        n_evals: report.evaluations,
        n_grads: 0,
        n_accepts: 0,
        position_fnv: 0,
    }
}

fn run_dmc(workers: usize, seed: u64, atoms: usize, walkers: usize, budget: usize) -> Run {
    let pot = PairPotential::lennard_jones(atoms);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers.max(1))
        .build()
        .expect("thread pool");
    let report = pool.install(|| {
        let obj = BatchPotential { pot };
        let mut rng = StdRng::seed_from_u64(seed);
        run_dmc_population(
            &obj,
            Some(&obj),
            budget,
            seed,
            walkers,
            DEFAULT_STEPS_PER_CONTROL,
            DEFAULT_BETA0,
            &mut rng,
        )
    });
    Run {
        best_energy: report.best_val,
        evaluations: report.n_evals.saturating_add(report.n_grads),
        decisions: report.controls,
        n_evals: report.n_evals,
        n_grads: report.n_grads,
        n_accepts: report.n_accepts,
        position_fnv: position_fnv(&report.best_pos),
    }
}

fn probe(atoms: usize, quench_steps: usize) {
    let pot = PairPotential::lennard_jones(atoms);
    let x = Array1::from_elem(3 * atoms, 0.7);
    let start = Instant::now();
    let mut acc = 0.0;
    for _ in 0..200 {
        acc = std::hint::black_box(pot.eval(x.view()));
    }
    let eval_s = start.elapsed().as_secs_f64() / 200.0;
    let mut optimiser = WarmLbfgs::default();
    let start = Instant::now();
    let (value, _, calls) =
        optimiser.minimize(x.view(), quench_steps, |v| Some(pot.value_and_gradient(v)));
    let quench_s = start.elapsed().as_secs_f64();
    println!("probe_eval_s: {eval_s:.9}");
    println!("probe_quench_s: {quench_s:.9}");
    println!("probe_quench_evals: {calls}");
    println!("probe_value: {}", std::hint::black_box(value));
    println!("probe_acc: {}", std::hint::black_box(acc));
}

fn check() -> ExitCode {
    let annealing = |workers: usize| run_population(workers, 9, 8, 4, 2, 3);
    let left = annealing(1);
    let right = annealing(2);
    if left.best_energy.to_bits() != right.best_energy.to_bits()
        || left.evaluations != right.evaluations
    {
        eprintln!("population annealing diverged across thread counts");
        return ExitCode::from(1);
    }
    let halving = |workers: usize| run_halving(workers, 9, 8, 4, 3);
    let left = halving(1);
    let right = halving(4);
    if left.best_energy.to_bits() != right.best_energy.to_bits()
        || left.evaluations != right.evaluations
        || left.decisions != right.decisions
    {
        eprintln!("successive halving diverged across thread counts");
        return ExitCode::from(1);
    }
    let diffusion = |workers: usize| run_dmc(workers, 9, 8, 8, 200);
    let left = diffusion(1);
    let right = diffusion(4);
    if left.best_energy.to_bits() != right.best_energy.to_bits()
        || left.evaluations != right.evaluations
        || left.n_evals != right.n_evals
        || left.n_grads != right.n_grads
        || left.n_accepts != right.n_accepts
        || left.decisions != right.decisions
        || left.position_fnv != right.position_fnv
    {
        eprintln!("diffusion Monte Carlo diverged across thread counts");
        return ExitCode::from(1);
    }
    println!("check: ok");
    ExitCode::SUCCESS
}

/// Full Lennard-Jones problem, serial polish against the parallel pool.
fn prove() -> ExitCode {
    let workers = [1usize, 2, 4, 8];
    println!("prove_problem: lennard_jones atoms {ATOMS} walkers {WALKERS} budget {DMC_BUDGET}");
    println!("prove_workers: 1 2 4 8");
    for seed in 0..6u64 {
        let serial = run_dmc(1, seed, ATOMS, WALKERS, DMC_BUDGET);
        println!(
            "seed {seed} workers 1 energy_bits {} evals {} grads {} accepts {} controls {} pos {position:016x}",
            serial.best_energy.to_bits(),
            serial.n_evals,
            serial.n_grads,
            serial.n_accepts,
            serial.decisions,
            position = serial.position_fnv,
        );
        for threads in workers.iter().copied().skip(1) {
            let parallel = run_dmc(threads, seed, ATOMS, WALKERS, DMC_BUDGET);
            let same = serial.best_energy.to_bits() == parallel.best_energy.to_bits()
                && serial.n_evals == parallel.n_evals
                && serial.n_grads == parallel.n_grads
                && serial.n_accepts == parallel.n_accepts
                && serial.decisions == parallel.decisions
                && serial.position_fnv == parallel.position_fnv;
            println!(
                "seed {seed} workers {threads} match {same} energy_bits {} evals {} grads {} accepts {} controls {} pos {position:016x}",
                parallel.best_energy.to_bits(),
                parallel.n_evals,
                parallel.n_grads,
                parallel.n_accepts,
                parallel.decisions,
                position = parallel.position_fnv,
            );
            if !same {
                eprintln!("diffusion Monte Carlo diverged at seed {seed} workers {threads}");
                return ExitCode::from(1);
            }
        }
    }
    println!("prove: ok");
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let atoms = match parse_usize(&args, "--atoms", ATOMS) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let walkers = match parse_usize(&args, "--walkers", WALKERS) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let generations = match parse_usize(&args, "--generations", GENERATIONS) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let quench_steps = match parse_usize(&args, "--quench-steps", QUENCH_STEPS) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let budget = match parse_usize(&args, "--budget", DMC_BUDGET) {
        Ok(value) => value,
        Err(code) => return code,
    };
    if args.iter().any(|item| item == "--probe") {
        probe(atoms, quench_steps);
        return ExitCode::SUCCESS;
    }
    if args.iter().any(|item| item == "--check") {
        return check();
    }
    if args.iter().any(|item| item == "--prove") {
        return prove();
    }
    let Some(method) = flag(&args, "--method") else {
        eprintln!("missing --method");
        return ExitCode::from(2);
    };
    let workers = match parse_usize(&args, "--workers", 0) {
        Ok(0) => {
            eprintln!("missing --workers");
            return ExitCode::from(2);
        }
        Ok(value) => value,
        Err(code) => return code,
    };
    let repeat = match parse_usize(&args, "--repeat", usize::MAX) {
        Ok(usize::MAX) => {
            eprintln!("missing --repeat");
            return ExitCode::from(2);
        }
        Ok(value) => value,
        Err(code) => return code,
    };
    let seed = repeat as u64;
    let start = Instant::now();
    let run = match method.as_str() {
        "population_annealing" => {
            run_population(workers, seed, atoms, walkers, generations, quench_steps)
        }
        "dmc_population" => run_dmc(workers, seed, atoms, walkers, budget),
        "successive_halving" => run_halving(workers, seed, atoms, walkers, generations),
        _ => {
            eprintln!("unknown method {method}");
            return ExitCode::from(2);
        }
    };
    let wall_s = start.elapsed().as_secs_f64();
    let best = std::hint::black_box(run.best_energy);
    println!("method: {method}");
    println!("scaling: strong");
    println!("workers: {workers}");
    println!("repeat: {repeat}");
    println!("seed: {seed}");
    println!("wall_s: {wall_s:.9}");
    println!("best_energy: {best:.16}");
    println!("best_energy_bits: {}", best.to_bits());
    println!("evaluations: {}", run.evaluations);
    println!("n_evals: {}", run.n_evals);
    println!("n_grads: {}", run.n_grads);
    println!("n_accepts: {}", run.n_accepts);
    println!("best_pos_fnv: {:016x}", run.position_fnv);
    println!("decisions: {}", run.decisions);
    println!("atoms: {atoms}");
    println!("walkers: {walkers}");
    println!("generations: {generations}");
    println!("quench_steps: {quench_steps}");
    println!("dmc_budget: {budget}");
    println!("cpu_model: {}", cpu_model());
    println!("logical_cpus: {}", logical_cpus());
    println!("physical_cpus: {}", physical_cpus());
    println!("memory_kib: {}", memory_kib());
    println!("gpu: {}", gpu_name());
    ExitCode::SUCCESS
}
