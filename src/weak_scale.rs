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
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use eindir_core::{FPair, Objective};
use ndarray::Array1;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::cool::LogCool;
use crate::exchange::{Exchange, MetropolisExchange};
use crate::history::{EpochLine, History, State};
use crate::methods::dmc_population::{
    DEFAULT_BETA0, DEFAULT_STEPS_PER_CONTROL, run_dmc_population,
};
use crate::methods::parallel_tempering::{
    ParallelTemperingSampler, PtResult, estimate_log_z_ratio, geometric_ladder,
};
use crate::methods::population_annealing::{PopulationAnneal, population_anneal_sync};
use crate::potentials::PairPotential;
use crate::sampler::Sampler;
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
    let spec = LjLadder {
        atoms: ATOMS,
        chains_per_worker: PT_CHAINS_PER_WORKER,
        steps: PT_STEPS_PER_CHAIN,
        swap_period: PT_SWAP_PERIOD,
        epochs: 1,
        workers,
        seed,
    };
    let bin = std::env::current_exe().expect("current executable");
    let started = Instant::now();
    let result = lj_ladder_result(&spec, Some(bin.as_path()));
    let wall_s = started.elapsed().as_secs_f64();
    let best = black_box(result.best_val);
    WeakScaleCell {
        family: "pt",
        workers,
        pool_threads: workers,
        worker_kind: "process",
        atoms: ATOMS,
        replicas: spec.chain_count(),
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

/// Lennard-Jones parallel tempering timed or compared across workers.
///
/// `chains_per_worker * workers` is the ladder length. Chain `c` uses
/// `StdRng` seeded with `seed.wrapping_add(c)`. The swap stream uses
/// `seed.wrapping_add(chain_count + 1)`. Both streams are the ones
/// [`crate::ParallelTemperingSampler::run`] draws, in the same order.
#[derive(Clone, Debug)]
pub struct LjLadder {
    /// Atoms in the cluster.
    pub atoms: usize,
    /// Chains owned by one worker.
    pub chains_per_worker: usize,
    /// Metropolis steps on each chain in one epoch.
    pub steps: usize,
    /// Steps between adjacent swap attempts. Must be positive.
    pub swap_period: usize,
    /// Epochs. The paper cell uses one.
    pub epochs: usize,
    /// Workers. One process each when this is the timed path.
    pub workers: usize,
    /// Seed passed to [`crate::ParallelTemperingSampler::run`].
    pub seed: u64,
}

impl LjLadder {
    fn check(&self) {
        assert!(self.atoms >= 2, "Lennard-Jones ladder needs two atoms");
        assert!(self.workers >= 1, "workers must be positive");
        assert!(
            self.chains_per_worker >= 1,
            "each worker keeps at least one chain"
        );
        assert!(self.swap_period >= 1, "swap period must be positive");
        assert!(
            self.chain_count() >= 2,
            "parallel tempering needs two chains"
        );
    }

    fn chain_count(&self) -> usize {
        self.workers.saturating_mul(self.chains_per_worker)
    }
}

/// [`crate::ParallelTemperingSampler::run`] on this Lennard-Jones ladder.
pub fn serial_lj_pt(spec: &LjLadder) -> PtResult {
    spec.check();
    let reference = PopulationAnneal::reference(spec.atoms);
    let potential = PairPotential::lennard_jones(spec.atoms);
    let sampler = boltzmann(potential, reference.temperature_high, reference.step)
        .expect("boltzmann variant");
    let temps = geometric_ladder(
        reference.temperature_low,
        reference.temperature_high,
        spec.chain_count(),
    );
    let tempering = ParallelTemperingSampler::new(sampler, temps, spec.steps, spec.swap_period);
    let cooling = LogCool::new(reference.temperature_high, 2.0);
    tempering.run(&cooling, spec.epochs, spec.seed)
}

/// The same ladder as [`serial_lj_pt`].
///
/// One worker runs on the calling thread. More workers run one process
/// each: worker 0 stays in this process, and the others are re-executions
/// of `worker_bin` with the `pt-worker` argument. The contract with
/// [`serial_lj_pt`], when the caller stop flag is clear, is bit-for-bit on:
///
/// - `best_val` and `best_pos` (first strict improvement of the current
///   energy, chains scanned in index order inside a step)
/// - `swap_attempts` and `swap_accepts`
/// - per-chain per-epoch accepted and rejected counts, ladder temperature,
///   and `best_val` bits
/// - `log_z_ratio_est` and `log_z_ratio_se` bits, which fix the hot-rung
///   energy recorded after every step and after a swap when one falls on
///   that step
///
/// `worker_bin` is required when `workers > 1`.
pub fn lj_ladder_result(spec: &LjLadder, worker_bin: Option<&Path>) -> PtResult {
    spec.check();
    let m = spec.chain_count();
    let reference = PopulationAnneal::reference(spec.atoms);
    let profile = std::env::var_os("ANNEAL_PT_PROFILE").is_some();
    let started_profile = Instant::now();
    let (begin, count) = worker_range(spec, 0);
    // Spawn the other processes while this process builds its own chains.
    // A serial exec loop was a large share of the 8-worker wall.
    let mut spawn_joins = Vec::new();
    if spec.workers > 1 {
        let bin = worker_bin
            .expect("process ladder needs the worker executable")
            .to_path_buf();
        for worker in 1..spec.workers {
            let spec_worker = spec.clone();
            let bin_worker = bin.clone();
            spawn_joins.push(std::thread::spawn(move || {
                let (begin, count) = worker_range(&spec_worker, worker);
                Remote::spawn(&bin_worker, &spec_worker, begin, count)
            }));
        }
    }
    let mut local = WorkerChains::new(spec, begin, count);
    let mut remotes = Vec::with_capacity(spawn_joins.len());
    for join in spawn_joins {
        remotes.push(join.join().expect("pt worker spawn thread"));
    }
    let spawn_s = started_profile.elapsed().as_secs_f64();
    let mut states = vec![blank_state(spec.atoms); m];
    install_local(&mut states, &local);
    for remote in &remotes {
        install_remote(&mut states, remote);
    }
    let mut chain_histories: Vec<History> = states
        .iter()
        .map(|state| History::with_capacity(spec.epochs, state.best.clone()))
        .collect();
    let mut best_val = f64::INFINITY;
    let mut best_pos = states[0].cur.pos.to_vec();
    let mut best_at = (u64::MAX, usize::MAX);
    let mut swap_rng = StdRng::seed_from_u64(spec.seed.wrapping_add(m as u64 + 1));
    let mut swap_attempts = 0;
    let mut swap_accepts = 0;
    let mut hot_energies = Vec::with_capacity(spec.epochs.saturating_mul(spec.steps));
    let temps = geometric_ladder(reference.temperature_low, reference.temperature_high, m);
    let exchange = MetropolisExchange;
    let mut local_s = 0.0;
    let mut read_s = 0.0;
    let mut swap_s = 0.0;

    'epochs: for epoch in 0..spec.epochs {
        if crate::caller_stop::raised() {
            break;
        }
        let mut accepted = vec![0usize; m];
        let mut rejected = vec![0usize; m];
        let mut done = 0usize;
        while done < spec.steps {
            if crate::caller_stop::raised() {
                break 'epochs;
            }
            let chunk = spec.swap_period.min(spec.steps - done);
            let step_base = (epoch as u64)
                .saturating_mul(spec.steps as u64)
                .saturating_add(done as u64);
            let hot_before = hot_energies.len();
            let mut reports = Vec::with_capacity(spec.workers);
            for remote in &mut remotes {
                remote.request_step(epoch, chunk, step_base);
            }
            let local_started = Instant::now();
            reports.push(local.advance(epoch, chunk, step_base, m));
            local_s += local_started.elapsed().as_secs_f64();
            let read_started = Instant::now();
            for remote in &mut remotes {
                reports.push(remote.read_step());
            }
            read_s += read_started.elapsed().as_secs_f64();
            for report in &reports {
                for chain in &report.chains {
                    let index = chain.index;
                    accepted[index] += chain.accepted;
                    rejected[index] += chain.rejected;
                    states[index] = chain.state.clone();
                    if chain.sight_val < best_val
                        || (chain.sight_val == best_val
                            && chain.sight_val.is_finite()
                            && (chain.sight_step, index) < best_at)
                    {
                        best_val = chain.sight_val;
                        best_pos = chain.sight_pos.clone();
                        best_at = (chain.sight_step, index);
                    }
                }
                hot_energies.extend_from_slice(&report.hot);
            }
            assert_eq!(
                hot_energies.len(),
                hot_before + chunk,
                "hot-rung trace missed a step"
            );
            done += chunk;
            let swap_started = Instant::now();
            if m >= 2 && done.is_multiple_of(spec.swap_period) {
                let i = swap_rng.random_range(0..m - 1);
                let alpha = exchange.swap_accept_prob(
                    states[i].cur.val,
                    temps[i],
                    states[i + 1].cur.val,
                    temps[i + 1],
                );
                swap_attempts += 1;
                let draw: f64 = swap_rng.random();
                if draw < alpha {
                    states.swap(i, i + 1);
                    swap_accepts += 1;
                    if (local.begin..local.begin + local.slots.len()).contains(&i)
                        || (local.begin..local.begin + local.slots.len()).contains(&(i + 1))
                    {
                        local.install_from(&states);
                    }
                    for remote in &mut remotes {
                        if remote.owns(i) || remote.owns(i + 1) {
                            remote.install_from(&states);
                        }
                    }
                }
                hot_energies.pop();
                hot_energies.push(states[m - 1].cur.val);
            }
            swap_s += swap_started.elapsed().as_secs_f64();
        }
        for (index, history) in chain_histories.iter_mut().enumerate() {
            history.epochs.push(EpochLine {
                epoch,
                temp: temps[index],
                accepted: accepted[index],
                rejected: rejected[index],
                best_val: states[index].best.val,
            });
        }
    }

    for remote in &mut remotes {
        remote.finish();
    }
    if profile {
        eprintln!(
            "pt_profile workers={} spawn_s={spawn_s:.6} local_s={local_s:.6} read_s={read_s:.6} swap_s={swap_s:.6}",
            spec.workers
        );
    }
    for history in &mut chain_histories {
        history.refresh_stationarity_flags();
    }
    let (log_z_ratio_est, log_z_ratio_se) =
        estimate_log_z_ratio(&hot_energies, temps[0], temps[m - 1]);
    PtResult {
        chain_histories,
        best_pos,
        best_val,
        swap_attempts,
        swap_accepts,
        log_z_ratio_est,
        log_z_ratio_se,
    }
}

/// Binary entry for `weak_scale` and for the `pt-worker` child.
pub fn entry() {
    let mut args = std::env::args().skip(1);
    let family = args.next().unwrap_or_default();
    if family == "pt-worker" {
        pt_worker();
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

fn worker_range(spec: &LjLadder, worker: usize) -> (usize, usize) {
    let begin = worker * spec.chains_per_worker;
    (begin, spec.chains_per_worker)
}

fn blank_state(atoms: usize) -> State {
    let pos = Array1::zeros(3 * atoms);
    let pair = FPair { pos, val: 0.0 };
    State {
        cur: pair.clone(),
        best: pair,
    }
}

fn install_local(states: &mut [State], local: &WorkerChains) {
    for (offset, slot) in local.slots.iter().enumerate() {
        states[local.begin + offset] = slot.state.clone();
    }
}

fn install_remote(states: &mut [State], remote: &Remote) {
    for (offset, state) in remote.states.iter().enumerate() {
        states[remote.begin + offset] = state.clone();
    }
}

struct ChainSlot {
    state: State,
    rng: StdRng,
}

struct WorkerChains {
    begin: usize,
    atoms: usize,
    sampler: crate::variant::BoltzmannVariant<PairPotential>,
    slots: Vec<ChainSlot>,
}

impl WorkerChains {
    fn new(spec: &LjLadder, begin: usize, count: usize) -> Self {
        let reference = PopulationAnneal::reference(spec.atoms);
        let potential = PairPotential::lennard_jones(spec.atoms);
        let sampler = boltzmann(potential, reference.temperature_high, reference.step)
            .expect("boltzmann variant");
        let slots = (0..count)
            .map(|offset| {
                let index = begin + offset;
                let mut rng = StdRng::seed_from_u64(spec.seed.wrapping_add(index as u64));
                let state = sampler.initial_state(&mut rng);
                ChainSlot { state, rng }
            })
            .collect();
        Self {
            begin,
            atoms: spec.atoms,
            sampler,
            slots,
        }
    }

    fn advance(
        &mut self,
        epoch: usize,
        steps: usize,
        step_base: u64,
        n_chains: usize,
    ) -> StepReport {
        let hot = n_chains - 1;
        let mut chains = Vec::with_capacity(self.slots.len());
        let mut hot_trace = Vec::new();
        for (offset, slot) in self.slots.iter_mut().enumerate() {
            let index = self.begin + offset;
            let mut accepted = 0usize;
            let mut rejected = 0usize;
            let mut sight_val = f64::INFINITY;
            let mut sight_step = 0u64;
            let mut sight_pos = vec![0.0; 3 * self.atoms];
            for step in 0..steps {
                if self.sampler.step(&mut slot.state, epoch, &mut slot.rng) {
                    accepted += 1;
                } else {
                    rejected += 1;
                }
                let absolute = step_base + step as u64;
                if slot.state.cur.val < sight_val {
                    sight_val = slot.state.cur.val;
                    sight_step = absolute;
                    sight_pos = slot.state.cur.pos.to_vec();
                }
                if index == hot {
                    hot_trace.push(slot.state.cur.val);
                }
            }
            chains.push(ChainReport {
                index,
                accepted,
                rejected,
                state: slot.state.clone(),
                sight_val,
                sight_step,
                sight_pos,
            });
        }
        StepReport {
            chains,
            hot: hot_trace,
        }
    }

    fn install_from(&mut self, states: &[State]) {
        for (offset, slot) in self.slots.iter_mut().enumerate() {
            slot.state = states[self.begin + offset].clone();
        }
    }
}

struct ChainReport {
    index: usize,
    accepted: usize,
    rejected: usize,
    state: State,
    sight_val: f64,
    sight_step: u64,
    sight_pos: Vec<f64>,
}

struct StepReport {
    chains: Vec<ChainReport>,
    hot: Vec<f64>,
}

struct Remote {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    begin: usize,
    count: usize,
    atoms: usize,
    states: Vec<State>,
}

impl Remote {
    fn spawn(bin: &Path, spec: &LjLadder, begin: usize, count: usize) -> Self {
        let mut child = Command::new(bin)
            .arg("pt-worker")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn pt worker");
        let mut stdin =
            BufWriter::with_capacity(1 << 20, child.stdin.take().expect("pt worker stdin"));
        let mut stdout =
            BufReader::with_capacity(1 << 20, child.stdout.take().expect("pt worker stdout"));
        write_config(&mut stdin, spec, begin, count);
        let states = read_ready(&mut stdout, count, spec.atoms);
        Self {
            child,
            stdin,
            stdout,
            begin,
            count,
            atoms: spec.atoms,
            states,
        }
    }

    fn owns(&self, index: usize) -> bool {
        index >= self.begin && index < self.begin + self.count
    }

    fn request_step(&mut self, epoch: usize, steps: usize, step_base: u64) {
        self.stdin.write_all(&[TAG_STEP]).expect("pt worker step");
        write_u32(&mut self.stdin, epoch as u32);
        write_u32(&mut self.stdin, steps as u32);
        write_u64(&mut self.stdin, step_base);
        self.stdin.flush().expect("pt worker step flush");
    }

    fn read_step(&mut self) -> StepReport {
        let mut chains = Vec::with_capacity(self.count);
        for offset in 0..self.count {
            let accepted = read_u64(&mut self.stdout) as usize;
            let rejected = read_u64(&mut self.stdout) as usize;
            let cur = read_pair(&mut self.stdout, self.atoms);
            let best = read_pair(&mut self.stdout, self.atoms);
            let sight_val = read_f64(&mut self.stdout);
            let sight_step = read_u64(&mut self.stdout);
            let sight_pos = read_vec(&mut self.stdout, 3 * self.atoms);
            let state = State { cur, best };
            self.states[offset] = state.clone();
            chains.push(ChainReport {
                index: self.begin + offset,
                accepted,
                rejected,
                state,
                sight_val,
                sight_step,
                sight_pos,
            });
        }
        let hot_len = read_u32(&mut self.stdout) as usize;
        let hot = read_f64s(&mut self.stdout, hot_len);
        StepReport { chains, hot }
    }

    fn install_from(&mut self, states: &[State]) {
        self.stdin.write_all(&[TAG_LOAD]).expect("pt worker load");
        for offset in 0..self.count {
            let state = &states[self.begin + offset];
            write_pair(&mut self.stdin, &state.cur);
            write_pair(&mut self.stdin, &state.best);
            self.states[offset] = state.clone();
        }
        self.stdin.flush().expect("pt worker load flush");
    }

    fn finish(&mut self) {
        let _ = self.stdin.write_all(&[TAG_DONE]);
        let _ = self.stdin.flush();
        let status = self.child.wait().expect("wait for pt worker");
        assert!(status.success(), "pt worker exited {status}");
    }
}

fn pt_worker() {
    let stdin = std::io::stdin();
    let mut input = BufReader::with_capacity(1 << 20, stdin.lock());
    let (spec, begin, count, n_chains) = read_config(&mut input);
    let mut chains = WorkerChains::new(&spec, begin, count);
    let stdout = std::io::stdout();
    let mut output = BufWriter::with_capacity(1 << 20, stdout.lock());
    write_ready(&mut output, &chains);
    loop {
        let mut tag = [0u8; 1];
        if input.read_exact(&mut tag).is_err() {
            return;
        }
        match tag[0] {
            TAG_STEP => {
                let epoch = read_u32(&mut input) as usize;
                let steps = read_u32(&mut input) as usize;
                let step_base = read_u64(&mut input);
                let report = chains.advance(epoch, steps, step_base, n_chains);
                for chain in &report.chains {
                    write_u64(&mut output, chain.accepted as u64);
                    write_u64(&mut output, chain.rejected as u64);
                    write_pair(&mut output, &chain.state.cur);
                    write_pair(&mut output, &chain.state.best);
                    write_f64(&mut output, chain.sight_val);
                    write_u64(&mut output, chain.sight_step);
                    for value in &chain.sight_pos {
                        write_f64(&mut output, *value);
                    }
                }
                write_u32(&mut output, report.hot.len() as u32);
                write_f64s(&mut output, &report.hot);
                output.flush().expect("pt worker step flush");
            }
            TAG_LOAD => {
                for slot in &mut chains.slots {
                    slot.state.cur = read_pair(&mut input, chains.atoms);
                    slot.state.best = read_pair(&mut input, chains.atoms);
                }
            }
            TAG_DONE => return,
            other => panic!("pt worker received tag {other}"),
        }
    }
}

const TAG_STEP: u8 = 1;
const TAG_LOAD: u8 = 2;
const TAG_DONE: u8 = 3;
const MAGIC: &[u8; 4] = b"PTW1";

fn write_config(w: &mut impl Write, spec: &LjLadder, begin: usize, count: usize) {
    w.write_all(MAGIC).expect("pt config");
    write_u32(w, spec.atoms as u32);
    write_u32(w, begin as u32);
    write_u32(w, count as u32);
    write_u32(w, spec.chain_count() as u32);
    write_u64(w, spec.seed);
    w.flush().expect("pt config flush");
}

fn read_config(r: &mut impl Read) -> (LjLadder, usize, usize, usize) {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).expect("pt config magic");
    assert_eq!(&magic, MAGIC, "pt worker received an unknown header");
    let atoms = read_u32(r) as usize;
    let begin = read_u32(r) as usize;
    let count = read_u32(r) as usize;
    let n_chains = read_u32(r) as usize;
    let seed = read_u64(r);
    let spec = LjLadder {
        atoms,
        chains_per_worker: count.max(1),
        steps: 0,
        swap_period: 1,
        epochs: 0,
        workers: 1,
        seed,
    };
    (spec, begin, count, n_chains)
}

fn write_ready(w: &mut impl Write, chains: &WorkerChains) {
    for slot in &chains.slots {
        write_pair(w, &slot.state.cur);
        write_pair(w, &slot.state.best);
    }
    w.flush().expect("pt ready flush");
}

fn read_ready(r: &mut impl Read, count: usize, atoms: usize) -> Vec<State> {
    (0..count)
        .map(|_| State {
            cur: read_pair(r, atoms),
            best: read_pair(r, atoms),
        })
        .collect()
}

fn write_pair(w: &mut impl Write, pair: &FPair<f64>) {
    write_f64(w, pair.val);
    write_u32(w, pair.pos.len() as u32);
    for value in pair.pos.iter() {
        write_f64(w, *value);
    }
}

fn read_pair(r: &mut impl Read, atoms: usize) -> FPair<f64> {
    let val = read_f64(r);
    let dim = read_u32(r) as usize;
    assert_eq!(
        dim,
        3 * atoms,
        "pt worker sent a configuration of the wrong length"
    );
    FPair {
        pos: Array1::from_vec(read_vec(r, dim)),
        val,
    }
}

fn write_u32(w: &mut impl Write, value: u32) {
    w.write_all(&value.to_le_bytes()).expect("pt write");
}

fn write_u64(w: &mut impl Write, value: u64) {
    w.write_all(&value.to_le_bytes()).expect("pt write");
}

fn write_f64(w: &mut impl Write, value: f64) {
    w.write_all(&value.to_le_bytes()).expect("pt write");
}

fn read_u32(r: &mut impl Read) -> u32 {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).expect("pt read");
    u32::from_le_bytes(buf)
}

fn read_u64(r: &mut impl Read) -> u64 {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).expect("pt read");
    u64::from_le_bytes(buf)
}

fn read_f64(r: &mut impl Read) -> f64 {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).expect("pt read");
    f64::from_le_bytes(buf)
}

fn write_f64s(w: &mut impl Write, values: &[f64]) {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(8));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    w.write_all(&bytes).expect("pt write");
}

fn read_f64s(r: &mut impl Read, len: usize) -> Vec<f64> {
    if len == 0 {
        return Vec::new();
    }
    let mut bytes = vec![0u8; len.saturating_mul(8)];
    r.read_exact(&mut bytes).expect("pt read");
    bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("8 bytes")))
        .collect()
}

fn read_vec(r: &mut impl Read, len: usize) -> Vec<f64> {
    (0..len).map(|_| read_f64(r)).collect()
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
        let spec = LjLadder {
            atoms: 4,
            chains_per_worker: 2,
            steps: 23,
            swap_period: 5,
            epochs: 2,
            workers: 1,
            seed: 11,
        };
        let serial = serial_lj_pt(&spec);
        let parallel = lj_ladder_result(&spec, None);
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
