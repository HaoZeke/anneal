//! Process workers for a Lennard-Jones Boltzmann ladder.
//!
//! [`super::parallel_tempering::ParallelTemperingSampler::run_processes`]
//! is the library entry. The child process is
//! [`process_worker`], started as `pt-worker` on the same executable.

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use eindir_core::FPair;
use ndarray::Array1;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::exchange::{Exchange, MetropolisExchange};
use crate::history::{EpochLine, History, State};
use crate::potentials::PairPotential;
use crate::sampler::Sampler;
use crate::variant::{BoltzmannVariant, boltzmann};

use super::parallel_tempering::{ParallelTemperingSampler, PtResult, estimate_log_z_ratio};

type LjSampler = ParallelTemperingSampler<BoltzmannVariant<PairPotential>, MetropolisExchange>;

struct Job {
    atoms: usize,
    seed: u64,
    steps: usize,
    swap_period: usize,
    epochs: usize,
    workers: usize,
    t_init: f64,
    k0: f64,
    sigma: f64,
    temps: Vec<f64>,
}

impl Job {
    fn from_sampler(sampler: &LjSampler, epochs: usize, seed: u64, workers: usize) -> Self {
        assert!(workers >= 1, "workers must be positive");
        assert!(sampler.temps.len() >= 2, "the ladder needs two chains");
        assert!(sampler.swap_period >= 1, "swap period must be positive");
        assert!(sampler.temps.len() >= workers, "each worker needs a chain");
        Self {
            atoms: sampler.sampler.obj.n_points,
            seed,
            steps: sampler.k_inner,
            swap_period: sampler.swap_period,
            epochs,
            workers,
            t_init: sampler.sampler.cool.t_init,
            k0: sampler.sampler.cool.k0,
            sigma: sampler.sampler.mover.sigma,
            temps: sampler.temps.clone(),
        }
    }

    fn chain_count(&self) -> usize {
        self.temps.len()
    }
}

pub(crate) fn run(
    sampler: &LjSampler,
    epochs: usize,
    seed: u64,
    workers: usize,
    worker_bin: Option<&Path>,
) -> PtResult {
    let job = Job::from_sampler(sampler, epochs, seed, workers);
    let m = job.chain_count();
    let profile = std::env::var_os("ANNEAL_PT_PROFILE").is_some();
    let started_profile = Instant::now();
    let (begin, count) = worker_bounds(m, workers, 0);
    let mut spawn_joins = Vec::new();
    if workers > 1 {
        let bin = worker_bin
            .expect("process ladder needs the worker executable")
            .to_path_buf();
        for worker in 1..workers {
            let job_worker = Job {
                atoms: job.atoms,
                seed: job.seed,
                steps: job.steps,
                swap_period: job.swap_period,
                epochs: job.epochs,
                workers: job.workers,
                t_init: job.t_init,
                k0: job.k0,
                sigma: job.sigma,
                temps: Vec::new(),
            };
            let bin_worker = bin.clone();
            spawn_joins.push(std::thread::spawn(move || {
                let (begin, count) = worker_bounds(m, workers, worker);
                Remote::spawn(&bin_worker, &job_worker, begin, count, m)
            }));
        }
    }
    let mut local = WorkerChains::new(
        job.atoms, job.t_init, job.k0, job.sigma, job.seed, begin, count,
    );
    let mut remotes = Vec::with_capacity(spawn_joins.len());
    for join in spawn_joins {
        remotes.push(join.join().expect("pt worker spawn thread"));
    }
    let spawn_s = started_profile.elapsed().as_secs_f64();
    let mut states = vec![blank_state(job.atoms); m];
    install_local(&mut states, &local);
    for remote in &remotes {
        install_remote(&mut states, remote);
    }
    let mut chain_histories: Vec<History> = states
        .iter()
        .map(|state| History::with_capacity(job.epochs, state.best.clone()))
        .collect();
    let mut best_val = f64::INFINITY;
    let mut best_pos = states[0].cur.pos.to_vec();
    let mut best_at = (u64::MAX, usize::MAX);
    let mut swap_rng = StdRng::seed_from_u64(job.seed.wrapping_add(m as u64 + 1));
    let mut swap_attempts = 0usize;
    let mut swap_accepts = 0usize;
    let mut hot_energies = Vec::with_capacity(job.epochs.saturating_mul(job.steps));
    let exchange = MetropolisExchange;
    let mut local_s = 0.0;
    let mut read_s = 0.0;
    let mut swap_s = 0.0;

    'epochs: for epoch in 0..job.epochs {
        if crate::caller_stop::raised() {
            break;
        }
        let mut accepted = vec![0usize; m];
        let mut rejected = vec![0usize; m];
        let mut done = 0usize;
        while done < job.steps {
            if crate::caller_stop::raised() {
                break 'epochs;
            }
            let chunk = job.swap_period.min(job.steps - done);
            let step_base = (epoch as u64)
                .saturating_mul(job.steps as u64)
                .saturating_add(done as u64);
            let hot_before = hot_energies.len();
            let mut reports = Vec::with_capacity(job.workers);
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
            if m >= 2 && done.is_multiple_of(job.swap_period) {
                let i = swap_rng.random_range(0..m - 1);
                let alpha = exchange.swap_accept_prob(
                    states[i].cur.val,
                    job.temps[i],
                    states[i + 1].cur.val,
                    job.temps[i + 1],
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
                temp: job.temps[index],
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
            "pt_profile workers={workers} spawn_s={spawn_s:.6} local_s={local_s:.6} read_s={read_s:.6} swap_s={swap_s:.6}"
        );
    }
    for history in &mut chain_histories {
        history.refresh_stationarity_flags();
    }
    let (log_z_ratio_est, log_z_ratio_se) =
        estimate_log_z_ratio(&hot_energies, job.temps[0], job.temps[m - 1]);
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

pub(crate) fn process_worker() {
    let stdin = std::io::stdin();
    let mut input = BufReader::with_capacity(1 << 20, stdin.lock());
    let (atoms, begin, count, n_chains, seed, t_init, k0, sigma) = read_config(&mut input);
    let mut chains = WorkerChains::new(atoms, t_init, k0, sigma, seed, begin, count);
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

fn worker_bounds(n: usize, workers: usize, worker: usize) -> (usize, usize) {
    let base = n / workers;
    let extra = n % workers;
    let begin = worker * base + worker.min(extra);
    let count = base + usize::from(worker < extra);
    (begin, count)
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
    sampler: BoltzmannVariant<PairPotential>,
    slots: Vec<ChainSlot>,
}

impl WorkerChains {
    fn new(
        atoms: usize,
        t_init: f64,
        k0: f64,
        sigma: f64,
        seed: u64,
        begin: usize,
        count: usize,
    ) -> Self {
        let potential = PairPotential::lennard_jones(atoms);
        let mut sampler = boltzmann(potential, t_init, sigma).expect("boltzmann variant");
        sampler.cool.k0 = k0;
        let slots = (0..count)
            .map(|offset| {
                let index = begin + offset;
                let mut rng = StdRng::seed_from_u64(seed.wrapping_add(index as u64));
                let state = sampler.initial_state(&mut rng);
                ChainSlot { state, rng }
            })
            .collect();
        Self {
            begin,
            atoms,
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
    fn spawn(bin: &Path, job: &Job, begin: usize, count: usize, n_chains: usize) -> Self {
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
        write_config(&mut stdin, job, begin, count, n_chains);
        let states = read_ready(&mut stdout, count, job.atoms);
        Self {
            child,
            stdin,
            stdout,
            begin,
            count,
            atoms: job.atoms,
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

const TAG_STEP: u8 = 1;
const TAG_LOAD: u8 = 2;
const TAG_DONE: u8 = 3;
const MAGIC: &[u8; 4] = b"PTW2";

fn write_config(w: &mut impl Write, job: &Job, begin: usize, count: usize, n_chains: usize) {
    w.write_all(MAGIC).expect("pt config");
    write_u32(w, job.atoms as u32);
    write_u32(w, begin as u32);
    write_u32(w, count as u32);
    write_u32(w, n_chains as u32);
    write_u64(w, job.seed);
    write_f64(w, job.t_init);
    write_f64(w, job.k0);
    write_f64(w, job.sigma);
    w.flush().expect("pt config flush");
}

fn read_config(r: &mut impl Read) -> (usize, usize, usize, usize, u64, f64, f64, f64) {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).expect("pt config magic");
    assert_eq!(&magic, MAGIC, "pt worker received an unknown header");
    let atoms = read_u32(r) as usize;
    let begin = read_u32(r) as usize;
    let count = read_u32(r) as usize;
    let n_chains = read_u32(r) as usize;
    let seed = read_u64(r);
    let t_init = read_f64(r);
    let k0 = read_f64(r);
    let sigma = read_f64(r);
    (atoms, begin, count, n_chains, seed, t_init, k0, sigma)
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
    assert_eq!(dim, 3 * atoms, "pt worker sent a short configuration");
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
