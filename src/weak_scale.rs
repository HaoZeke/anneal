//! Weak-scaling schedules for parallel tempering, population annealing,
//! and the diffusion population.
//!
//! Worker counts grow the replica count or the charged population.
//! Atoms per cluster stay fixed, so one energy evaluation keeps the same
//! pair count. Each worker therefore keeps the same scheduled charge.
//! The schedules are the scaling rows. They are not quality budgets.
//!
//! Parallel tempering is timed as one operating-system process per worker.
//! The parent draws every swap. Population annealing and the diffusion
//! population stay on threads: their measured calls are native Lennard-Jones
//! evaluations and do not take the interpreter lock.

use std::hint::black_box;
use std::time::Instant;

use eindir_core::Objective;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::cool::LogCool;
use crate::methods::dmc_population::{
    DEFAULT_BETA0, DEFAULT_STEPS_PER_CONTROL, run_dmc_population,
};
use crate::methods::parallel_tempering::{ParallelTemperingSampler, geometric_ladder};
use crate::methods::population_annealing::{PopulationAnneal, population_anneal_sync};
use crate::potentials::PairPotential;
use crate::variant::boltzmann;

/// Atoms in every Lennard-Jones cluster this schedule evaluates.
pub const ATOMS: usize = 13;
/// Chains one parallel-tempering worker holds.
pub const PT_CHAINS_PER_WORKER: usize = 2;
/// Metropolis steps one chain takes during the timed ladder.
pub const PT_STEPS_PER_CHAIN: usize = 262_144;
/// Inner steps between adjacent swaps.
pub const PT_SWAP_PERIOD: usize = 32_768;
/// Walkers one population-annealing worker holds.
pub const PA_WALKERS_PER_WORKER: usize = 8;
/// Resampling generations of the synchronous population.
pub const PA_GENERATIONS: usize = 32;
/// Metropolis steps one walker takes inside one generation.
pub const PA_STEPS_PER_GENERATION: usize = 2_048;
/// Walkers inside one diffusion population.
pub const DMC_WALKERS: usize = 8;
/// Charged evaluations scheduled for one diffusion population.
pub const DMC_BUDGET: usize = 262_144;

/// One timed weak-scaling cell.
#[derive(Clone, Debug)]
pub struct WeakScaleCell {
    /// `pt`, `population`, or `dmc`.
    pub family: &'static str,
    /// Requested workers.
    pub workers: usize,
    /// Processes or threads that evaluated the parallel section.
    pub pool_threads: usize,
    /// `process` for parallel tempering, `thread` for the populations.
    pub worker_kind: &'static str,
    /// Atoms in one cluster.
    pub atoms: usize,
    /// Chains, population walkers, or diffusion walkers across the cell.
    pub replicas: usize,
    /// Energy evaluations charged.
    pub charged: usize,
    /// Wall time in seconds.
    pub wall_s: f64,
    /// Deepest energy the cell observed.
    pub best: f64,
}

/// Scheduled parallel-tempering evaluations at `workers`.
pub fn pt_charged(workers: usize) -> usize {
    workers
        .saturating_mul(PT_CHAINS_PER_WORKER)
        .saturating_mul(1 + PT_STEPS_PER_CHAIN)
}

/// Scheduled population-annealing evaluations at `workers`.
pub fn population_charged(workers: usize) -> usize {
    workers
        .saturating_mul(PA_WALKERS_PER_WORKER)
        .saturating_mul(1 + PA_GENERATIONS.saturating_mul(PA_STEPS_PER_GENERATION))
}

/// Scheduled diffusion-population evaluations at `workers`.
pub fn dmc_charged(workers: usize) -> usize {
    workers.saturating_mul(DMC_BUDGET)
}

/// Width of a Rayon pool built for `threads` workers.
pub fn pool_width(threads: usize) -> usize {
    let threads = threads.max(1);
    if threads == 1 {
        return 1;
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("rayon pool")
        .current_num_threads()
}

/// Times one weak-scaling cell.
pub fn run_cell(family: &str, workers: usize, seed: u64) -> WeakScaleCell {
    assert!(workers >= 1, "workers must be positive");
    match family {
        "pt" => run_pt(workers, seed),
        "population" => run_population(workers, seed),
        "dmc" => run_dmc(workers, seed),
        _ => panic!("unknown family {family}"),
    }
}

fn run_pt(workers: usize, seed: u64) -> WeakScaleCell {
    let reference = PopulationAnneal::reference(ATOMS);
    let potential = PairPotential::lennard_jones(ATOMS);
    let sampler = boltzmann(potential, reference.temperature_high, reference.step)
        .expect("boltzmann variant");
    let chains = PT_CHAINS_PER_WORKER * workers;
    let temps = geometric_ladder(
        reference.temperature_low,
        reference.temperature_high,
        chains,
    );
    let tempering =
        ParallelTemperingSampler::new(sampler, temps, PT_STEPS_PER_CHAIN, PT_SWAP_PERIOD);
    let cooling = LogCool::new(reference.temperature_high, 2.0);
    let bin = std::env::current_exe().expect("current executable");
    let started = Instant::now();
    let result = tempering.run_processes(&cooling, 1, seed, workers, Some(bin.as_path()));
    let wall_s = started.elapsed().as_secs_f64();
    let best = black_box(result.best_val);
    WeakScaleCell {
        family: "pt",
        workers,
        pool_threads: workers,
        worker_kind: "process",
        atoms: ATOMS,
        replicas: chains,
        charged: pt_charged(workers),
        wall_s,
        best,
    }
}

fn run_population(workers: usize, seed: u64) -> WeakScaleCell {
    let walkers = PA_WALKERS_PER_WORKER * workers;
    let reference = PopulationAnneal::reference(ATOMS);
    let potential = PairPotential::lennard_jones(ATOMS);
    let started = Instant::now();
    let report = population_anneal_sync(
        ATOMS,
        walkers,
        workers,
        PA_GENERATIONS,
        PA_STEPS_PER_GENERATION,
        reference.step,
        reference.temperature_high,
        reference.temperature_low,
        reference.radius,
        reference.min_separation,
        |x| potential.eval(x),
        seed,
    );
    let wall_s = started.elapsed().as_secs_f64();
    let best = black_box(report.best_energy);
    WeakScaleCell {
        family: "population",
        workers,
        pool_threads: pool_width(workers),
        worker_kind: "thread",
        atoms: ATOMS,
        replicas: report.walkers,
        charged: report.charged,
        wall_s,
        best,
    }
}

fn run_dmc(workers: usize, seed: u64) -> WeakScaleCell {
    let started = Instant::now();
    let mut charged = 0usize;
    let mut best = f64::INFINITY;
    std::thread::scope(|scope| {
        let mut joins = Vec::with_capacity(workers);
        for index in 0..workers {
            joins.push(scope.spawn(move || {
                let potential = PairPotential::lennard_jones(ATOMS);
                let mut rng = StdRng::seed_from_u64(seed.wrapping_add(index as u64));
                run_dmc_population(
                    &potential,
                    Option::<&PairPotential>::None,
                    DMC_BUDGET,
                    seed.wrapping_add(index as u64),
                    DMC_WALKERS,
                    DEFAULT_STEPS_PER_CONTROL,
                    DEFAULT_BETA0,
                    &mut rng,
                )
            }));
        }
        for join in joins {
            let result = join.join().expect("diffusion population thread");
            charged += result.n_evals;
            if result.best_val < best {
                best = result.best_val;
            }
        }
    });
    let wall_s = started.elapsed().as_secs_f64();
    let best = black_box(best);
    WeakScaleCell {
        family: "dmc",
        workers,
        pool_threads: workers,
        worker_kind: "thread",
        atoms: ATOMS,
        replicas: DMC_WALKERS * workers,
        charged,
        wall_s,
        best,
    }
}

/// Binary entry for `weak_scale` and for the `pt-worker` child.
pub fn entry() {
    let mut args = std::env::args().skip(1);
    let family = args.next().unwrap_or_default();
    if family == "pt-worker" {
        crate::methods::parallel_tempering::process_worker();
        return;
    }
    let workers = args
        .next()
        .and_then(|text| text.parse::<usize>().ok())
        .unwrap_or(0);
    let seed = args
        .next()
        .and_then(|text| text.parse::<u64>().ok())
        .unwrap_or(0);
    if !matches!(family.as_str(), "pt" | "population" | "dmc") || workers < 1 {
        eprintln!("usage: weak_scale <pt|population|dmc> <workers> <seed>");
        std::process::exit(2);
    }
    let cell = run_cell(&family, workers, seed);
    println!(
        "family={family} workers={} pool_threads={} worker_kind={} atoms={} replicas={} charged={} wall_s={:.9} best={:.16e} seed={seed}",
        cell.workers,
        cell.pool_threads,
        cell.worker_kind,
        cell.atoms,
        cell.replicas,
        cell.charged,
        cell.wall_s,
        cell.best,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduled_charge_per_worker_is_constant() {
        let pt = pt_charged(1);
        let population = population_charged(1);
        let dmc = dmc_charged(1);
        for workers in [1usize, 2, 4, 8, 16] {
            assert_eq!(pt_charged(workers), pt * workers);
            assert_eq!(population_charged(workers), population * workers);
            assert_eq!(dmc_charged(workers), dmc * workers);
        }
    }

    #[test]
    fn diffusion_population_charges_a_finite_best() {
        let potential = PairPotential::lennard_jones(4);
        let mut rng = StdRng::seed_from_u64(3);
        let result = run_dmc_population(
            &potential,
            Option::<&PairPotential>::None,
            40,
            3,
            4,
            DEFAULT_STEPS_PER_CONTROL,
            DEFAULT_BETA0,
            &mut rng,
        );
        assert!(result.best_val.is_finite());
        assert!(result.n_evals > 0);
        assert!(result.n_evals <= 40);
    }

    #[test]
    fn one_worker_process_path_matches_serial_run() {
        let potential = PairPotential::lennard_jones(4);
        let sampler = boltzmann(potential, 0.8, 0.38).expect("variant");
        let temps = geometric_ladder(0.05, 0.8, 2);
        let tempering = ParallelTemperingSampler::new(sampler, temps, 23, 5);
        let cooling = LogCool::new(0.8, 2.0);
        let serial = tempering.run(&cooling, 2, 11);
        let parallel = tempering.run_processes(&cooling, 2, 11, 1, None);
        assert_eq!(serial.best_val.to_bits(), parallel.best_val.to_bits());
        assert_eq!(serial.best_pos, parallel.best_pos);
        assert_eq!(serial.swap_attempts, parallel.swap_attempts);
        assert_eq!(serial.swap_accepts, parallel.swap_accepts);
        assert_eq!(
            serial.log_z_ratio_est.to_bits(),
            parallel.log_z_ratio_est.to_bits()
        );
        assert_eq!(
            serial.log_z_ratio_se.to_bits(),
            parallel.log_z_ratio_se.to_bits()
        );
        assert_eq!(serial.chain_histories.len(), parallel.chain_histories.len());
        for (left, right) in serial
            .chain_histories
            .iter()
            .zip(parallel.chain_histories.iter())
        {
            assert_eq!(left.epochs.len(), right.epochs.len());
            for (epoch_left, epoch_right) in left.epochs.iter().zip(right.epochs.iter()) {
                assert_eq!(epoch_left.accepted, epoch_right.accepted);
                assert_eq!(epoch_left.rejected, epoch_right.rejected);
                assert_eq!(
                    epoch_left.best_val.to_bits(),
                    epoch_right.best_val.to_bits()
                );
            }
        }
    }
}
