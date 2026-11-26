// Copyright (c) anneal developers
//! Throughput ladder for Au55 and LJ75 on the shared communicating ensemble.
//!
//! ```text
//! arch_throughput ladder
//! arch_throughput one <system> <seed> <workers>
//! arch_throughput probe <system> <calls> <workers>
//! ```
//!
//! `ladder` and `one` append to `bench/arch-scaling/throughput/rows-throughput.csv`.
//! `probe` prints one timing line and does not write a row. The contract is
//! `bench/arch-scaling/throughput/PROTOCOL.md`.

use anneal_core::descriptor_space::{DescriptorGeometry, universal_descriptor_space};
use anneal_core::methods::cluster_hopping::{Config, random_cluster_in_radius};
use anneal_core::methods::ensemble::{EnsembleConfig, EnsembleProblem, HistoryMode, run_ensemble};
use anneal_core::methods::minima_hopping::{HistoryMembership, SerializedWitness};
use anneal_core::pes_exploration::StructureContext;
use anneal_core::potentials::{Gupta, PairPotential};
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use std::fs::OpenOptions;
use std::io::{ErrorKind, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const SEEDS: std::ops::RangeInclusive<u64> = 2000..=2011;
const WORKERS: [usize; 4] = [1, 2, 4, 8];
const BUDGET: usize = 1_000_000;
const ROWS: &str = "bench/arch-scaling/throughput/rows-throughput.csv";
const PANEL: &str = "scripts/paper_0120/run_panel.py";
const SPEC_FILE: &str = "examples/h2h_metal.rs";
const LJ_FILE: &str = "examples/lj_cluster_search.rs";
const HEADER: &str = "system,arm,seed,workers,wall_s,oracle_calls,call_budget,calls_minus_budget,calls_per_s,ledger_objective,ledger_charged,ledger_ensemble,hit,best_energy,revision,nproc,cpu_model,pinned_cores,oversubscribed";

#[derive(Clone, Copy, PartialEq, Eq)]
enum System {
    Au55,
    Lj75,
}

impl System {
    fn parse(text: &str) -> Option<Self> {
        match text {
            "au55" => Some(Self::Au55),
            "lj75" => Some(Self::Lj75),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Au55 => "au55",
            Self::Lj75 => "lj75",
        }
    }

    fn atoms(self) -> usize {
        match self {
            Self::Au55 => 55,
            Self::Lj75 => 75,
        }
    }

    fn all() -> [Self; 2] {
        [Self::Au55, Self::Lj75]
    }
}

#[derive(Clone)]
enum Surface {
    Gupta(Gupta),
    LennardJones(PairPotential),
}

impl Surface {
    fn value_and_gradient(&self, x: ArrayView1<'_, f64>) -> (f64, Array1<f64>) {
        match self {
            Self::Gupta(potential) => potential.value_and_gradient(x),
            Self::LennardJones(potential) => potential.value_and_gradient(x),
        }
    }
}

fn main() {
    unsafe {
        std::env::set_var("OMP_NUM_THREADS", "1");
        std::env::set_var("RAYON_NUM_THREADS", "1");
        std::env::remove_var("ANNEAL_MIN_DUMP");
    }
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let status = match args.first().map(String::as_str) {
        Some("ladder") => cmd_ladder(),
        Some("one") => cmd_one(&args[1..]),
        Some("probe") => cmd_probe(&args[1..]),
        _ => {
            eprintln!("usage: arch_throughput ladder|one|probe");
            2
        }
    };
    std::process::exit(status);
}

fn cmd_ladder() -> i32 {
    let ctx = match Context::load() {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!("{err}");
            return 1;
        }
    };
    eprintln!(
        "nproc={} cpu={} budget={} revision={} deadline={}",
        ctx.nproc,
        ctx.cpu_model,
        ctx.budget,
        ctx.revision,
        ctx.deadline
            .map(|limit| format!("{}s", limit.as_secs()))
            .unwrap_or_else(|| "none".into())
    );
    let started = Instant::now();
    for (seed, workers, system) in schedule() {
        if let Some(limit) = ctx.deadline
            && started.elapsed() >= limit
        {
            eprintln!(
                "deadline reached before {} seed {seed} workers {workers}",
                system.name()
            );
            break;
        }
        if let Err(err) = run_recorded(&ctx, system, seed, workers) {
            eprintln!("{} seed {seed} workers {workers}: {err}", system.name());
            return 1;
        }
    }
    0
}

fn cmd_one(args: &[String]) -> i32 {
    let Some(system) = args.first().and_then(|text| System::parse(text)) else {
        eprintln!("one needs au55 or lj75");
        return 2;
    };
    let Some(seed) = args.get(1).and_then(|text| text.parse::<u64>().ok()) else {
        eprintln!("one needs a seed");
        return 2;
    };
    let Some(workers) = args.get(2).and_then(|text| text.parse::<usize>().ok()) else {
        eprintln!("one needs a worker count");
        return 2;
    };
    if !SEEDS.contains(&seed) || !WORKERS.contains(&workers) {
        eprintln!("seed or worker count is outside the ladder");
        return 2;
    }
    let ctx = match Context::load() {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!("{err}");
            return 1;
        }
    };
    if let Err(err) = run_recorded(&ctx, system, seed, workers) {
        eprintln!("{err}");
        return 1;
    }
    0
}

fn cmd_probe(args: &[String]) -> i32 {
    let Some(system) = args.first().and_then(|text| System::parse(text)) else {
        eprintln!("probe needs au55 or lj75");
        return 2;
    };
    let Some(calls) = args.get(1).and_then(|text| text.parse::<usize>().ok()) else {
        eprintln!("probe needs a call count");
        return 2;
    };
    let Some(workers) = args.get(2).and_then(|text| text.parse::<usize>().ok()) else {
        eprintln!("probe needs a worker count");
        return 2;
    };
    if calls == 0 || !WORKERS.contains(&workers) {
        eprintln!("probe call count or worker count is invalid");
        return 2;
    }
    let ctx = match Context::load() {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!("{err}");
            return 1;
        }
    };
    match measure(&ctx, system, 2000, workers, calls, false) {
        Ok(row) => {
            println!(
                "probe system={} workers={} calls={} wall_s={:.6} oracle={} calls_per_s={:.3} best={:.8} hit={} oversubscribed={}",
                row.system,
                row.workers,
                calls,
                row.wall_s,
                row.oracle_calls,
                row.calls_per_s,
                row.best_energy,
                u8::from(row.hit),
                u8::from(row.oversubscribed)
            );
            0
        }
        Err(err) => {
            eprintln!("{err}");
            1
        }
    }
}

fn schedule() -> Vec<(u64, usize, System)> {
    let mut jobs = Vec::with_capacity(96);
    for seed in SEEDS {
        let index = (seed - *SEEDS.start()) as usize;
        let workers: Vec<usize> = if index % 2 == 0 {
            WORKERS.to_vec()
        } else {
            WORKERS.iter().rev().copied().collect()
        };
        let systems: Vec<System> = if index % 2 == 0 {
            System::all().to_vec()
        } else {
            System::all().iter().rev().copied().collect()
        };
        for worker in workers {
            for system in systems.iter().copied() {
                jobs.push((seed, worker, system));
            }
        }
    }
    jobs
}

struct Context {
    budget: usize,
    au55_reference: f64,
    lj75_reference: f64,
    nproc: usize,
    cpu_model: String,
    revision: String,
    deadline: Option<Duration>,
}

impl Context {
    fn load() -> Result<Self, String> {
        let au_budget = file_line(PANEL, 53)?;
        let lj_budget = file_line(PANEL, 47)?;
        let lj_reference = file_line(PANEL, 57)?;
        if !au_budget.contains("\"au55\"") || !au_budget.contains("1_000_000") {
            return Err(format!("Au55 budget line moved: {au_budget:?}"));
        }
        if !lj_budget.contains("\"lj75\"") || !lj_budget.contains("4_000_000") {
            return Err(format!("LJ75 paper budget line moved: {lj_budget:?}"));
        }
        if lj_reference.trim() != "LJ75_REFERENCE = -397.492331" {
            return Err(format!("LJ75 reference line moved: {lj_reference:?}"));
        }
        let lj_arm = file_line(LJ_FILE, 1295)?;
        if !lj_arm.contains("75") || !lj_arm.contains("-397.492331") {
            return Err(format!("LJ75 reference arm moved: {lj_arm:?}"));
        }
        let head = file_line(SPEC_FILE, 79)?;
        let atoms = file_line(SPEC_FILE, 80)?;
        let reference_line = file_line(SPEC_FILE, 81)?;
        let kind = file_line(SPEC_FILE, 83)?;
        if !head.contains("\"au55\"")
            || !atoms.contains("55")
            || !reference_line.contains("-191.03085280")
            || !kind.contains("n_copper: 0")
        {
            return Err("Au55 spec lines do not match the expected builder".into());
        }
        let nproc = online_processors()?;
        let listed = cpuinfo_processors()?;
        if nproc != listed {
            return Err(format!("nproc {nproc} disagrees with cpuinfo {listed}"));
        }
        Ok(Self {
            budget: BUDGET,
            au55_reference: -191.03085280,
            lj75_reference: -397.492331,
            nproc,
            cpu_model: cpu_model()?,
            revision: git_revision()?,
            deadline: deadline()?,
        })
    }

    fn reference(&self, system: System) -> f64 {
        match system {
            System::Au55 => self.au55_reference,
            System::Lj75 => self.lj75_reference,
        }
    }
}

struct Row {
    system: &'static str,
    seed: u64,
    workers: usize,
    wall_s: f64,
    oracle_calls: usize,
    call_budget: usize,
    calls_minus_budget: i64,
    calls_per_s: f64,
    ledger_objective: usize,
    ledger_charged: usize,
    ledger_ensemble: usize,
    hit: bool,
    best_energy: f64,
    revision: String,
    nproc: usize,
    cpu_model: String,
    pinned_cores: String,
    oversubscribed: bool,
}

fn run_recorded(ctx: &Context, system: System, seed: u64, workers: usize) -> Result<(), String> {
    if already_stored(system, seed, workers)? {
        eprintln!("skip {} seed {seed} workers {workers}", system.name());
        return Ok(());
    }
    let row = measure(ctx, system, seed, workers, ctx.budget, true)?;
    append_row(&row)?;
    eprintln!(
        "{} seed {seed} workers {workers} wall_s={:.3} calls={} calls_per_s={:.1} hit={} deviation={}",
        row.system,
        row.wall_s,
        row.oracle_calls,
        row.calls_per_s,
        u8::from(row.hit),
        row.calls_minus_budget
    );
    Ok(())
}

fn measure(
    ctx: &Context,
    system: System,
    seed: u64,
    workers: usize,
    budget: usize,
    require_duration: bool,
) -> Result<Row, String> {
    if workers == 0 || budget < workers {
        return Err("every replica needs at least one call".into());
    }
    let requested: Vec<usize> = (0..workers.min(ctx.nproc)).collect();
    let pinned = pin_cores(&requested)?;
    let oversubscribed = workers > ctx.nproc;
    let reference = ctx.reference(system);
    let (cfg, species, surface) = build(system)?;
    if cfg.replicas != 1 {
        return Err("communicating preset is not a single-rung chain".into());
    }
    let ens = EnsembleConfig {
        replicas: workers,
        budget,
        history: HistoryMode::Shared,
        membership: HistoryMembership::Accepted,
        shared_bias: None,
        gossip: None,
        two_choice_stall: None,
        checkpoint_interval: 1,
        target: Some(reference + anneal_core::DEFAULT_HIT_TOL),
    };
    let geometry = DescriptorGeometry::finite(cfg.length_scale).map_err(|err| err.to_string())?;
    let descriptor = universal_descriptor_space(geometry);
    let context = StructureContext::new(species, None, Some(system.name().into()));
    let witness = SerializedWitness(Mutex::new(same_geometry));
    let radius = cfg.start_radius();
    let min_sep = cfg.min_separation;
    let n_atoms = system.atoms();
    let calls = Arc::new(AtomicUsize::new(0));
    let objective = {
        let surface = surface.clone();
        let calls = Arc::clone(&calls);
        move |_: usize| {
            let surface = surface.clone();
            let calls = Arc::clone(&calls);
            Box::new(move |x: ArrayView1<'_, f64>| {
                let pair = surface.value_and_gradient(x);
                calls.fetch_add(1, Ordering::Relaxed);
                pair
            }) as Box<dyn FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>) + Send>
        }
    };
    let start =
        move |_: usize, rng: &mut StdRng| random_cluster_in_radius(n_atoms, radius, min_sep, rng);
    let same_family = |_left: &[f64], _right: &[f64]| false;
    let problem = EnsembleProblem {
        objective: &objective,
        start: &start,
        descriptor: &descriptor,
        context: &context,
        witness: &witness,
        same_family: &same_family,
        certificate: 1e-5,
        polish_below: 1e-3,
        callbacks_per_objective: 1,
    };
    let started = Instant::now();
    let report = run_ensemble(&cfg, &ens, seed, &problem)?;
    let wall_s = started.elapsed().as_secs_f64();
    let ledger_objective = calls.load(Ordering::Relaxed);
    let ledger_ensemble: usize = report.replicas.iter().map(|replica| replica.charged).sum();
    let ledger_charged = report.aggregate_charged;
    if ledger_objective != ledger_charged || ledger_ensemble != ledger_charged {
        return Err(format!(
            "ledgers disagree: objective {ledger_objective} charged {ledger_charged} ensemble {ledger_ensemble}"
        ));
    }
    if require_duration && wall_s < 0.05 {
        return Err(format!("wall {wall_s:.6}s is under 50 ms"));
    }
    let hit = anneal_core::energy_hits(report.best, reference, anneal_core::DEFAULT_HIT_TOL);
    let solved = report.solved(Some(reference + anneal_core::DEFAULT_HIT_TOL));
    if hit != solved {
        return Err("ensemble solved flag disagrees with the hit rule".into());
    }
    let calls_minus_budget = ledger_charged as i64 - ctx.budget as i64;
    Ok(Row {
        system: system.name(),
        seed,
        workers,
        wall_s,
        oracle_calls: ledger_charged,
        call_budget: ctx.budget,
        calls_minus_budget,
        calls_per_s: ledger_charged as f64 / wall_s,
        ledger_objective,
        ledger_charged,
        ledger_ensemble,
        hit,
        best_energy: report.best,
        revision: ctx.revision.clone(),
        nproc: ctx.nproc,
        cpu_model: ctx.cpu_model.clone(),
        pinned_cores: format_cores(&pinned),
        oversubscribed,
    })
}

fn build(system: System) -> Result<(Config, Option<Vec<u32>>, Surface), String> {
    let n_atoms = system.atoms();
    match system {
        System::Au55 => {
            let species = Gupta::copper_gold_species(0, n_atoms);
            let cfg = Config::communicating(n_atoms).with_species(species.clone(), None);
            Ok((
                cfg,
                Some(species),
                Surface::Gupta(Gupta::copper_gold(0, n_atoms)),
            ))
        }
        System::Lj75 => Ok((
            Config::communicating(n_atoms),
            None,
            Surface::LennardJones(PairPotential::lennard_jones(n_atoms)),
        )),
    }
}

fn deadline() -> Result<Option<Duration>, String> {
    match std::env::var("THROUGHPUT_DEADLINE_SECS") {
        Err(_) => Ok(None),
        Ok(text) if text.is_empty() => Ok(None),
        Ok(text) => {
            let seconds: u64 = text
                .parse()
                .map_err(|_| format!("THROUGHPUT_DEADLINE_SECS {text:?} is not an integer"))?;
            Ok(Some(Duration::from_secs(seconds)))
        }
    }
}

fn append_row(row: &Row) -> Result<(), String> {
    let path = Path::new(ROWS);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    let fresh = match std::fs::metadata(path) {
        Ok(meta) => meta.len() == 0,
        Err(err) if err.kind() == ErrorKind::NotFound => true,
        Err(err) => return Err(err.to_string()),
    };
    if !fresh {
        let existing = std::fs::read_to_string(path).map_err(|err| err.to_string())?;
        let head = existing.lines().next().unwrap_or("");
        if head != HEADER {
            return Err("rows-throughput.csv header does not match the protocol".into());
        }
        if existing
            .lines()
            .skip(1)
            .any(|line| row_key(line) == Some((row.system, row.seed, row.workers)))
        {
            return Err(format!(
                "{} seed {} workers {} was stored while this run was in flight",
                row.system, row.seed, row.workers
            ));
        }
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| err.to_string())?;
    if fresh {
        writeln!(file, "{HEADER}").map_err(|err| err.to_string())?;
    }
    writeln!(file, "{}", row.line()).map_err(|err| err.to_string())?;
    file.sync_all().map_err(|err| err.to_string())?;
    Ok(())
}

fn already_stored(system: System, seed: u64, workers: usize) -> Result<bool, String> {
    let path = Path::new(ROWS);
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(false),
        Err(err) => return Err(err.to_string()),
    };
    if text.is_empty() {
        return Ok(false);
    }
    let mut lines = text.lines();
    if lines.next() != Some(HEADER) {
        return Err("rows-throughput.csv header does not match the protocol".into());
    }
    Ok(lines.any(|line| row_key(line) == Some((system.name(), seed, workers))))
}

fn row_key(line: &str) -> Option<(&str, u64, usize)> {
    let mut fields = line.split(',');
    let system = fields.next()?;
    let arm = fields.next()?;
    let seed = fields.next()?.parse().ok()?;
    let workers = fields.next()?.parse().ok()?;
    (arm == "comm").then_some((system, seed, workers))
}

impl Row {
    fn line(&self) -> String {
        format!(
            "{},comm,{},{},{:.6},{},{},{},{:.6},{},{},{},{},{:.10},{},{},{},{},{}",
            self.system,
            self.seed,
            self.workers,
            self.wall_s,
            self.oracle_calls,
            self.call_budget,
            self.calls_minus_budget,
            self.calls_per_s,
            self.ledger_objective,
            self.ledger_charged,
            self.ledger_ensemble,
            u8::from(self.hit),
            self.best_energy,
            self.revision,
            self.nproc,
            csv_field(&self.cpu_model),
            csv_field(&self.pinned_cores),
            u8::from(self.oversubscribed)
        )
    }
}

fn csv_field(text: &str) -> String {
    if text.contains([',', '"', '\n']) {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
}

fn file_line(path: &str, number: usize) -> Result<String, String> {
    let text = std::fs::read_to_string(path).map_err(|err| format!("read {path}: {err}"))?;
    text.lines()
        .nth(number - 1)
        .map(str::to_string)
        .ok_or_else(|| format!("{path} has no line {number}"))
}

fn git_revision() -> Result<String, String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .map_err(|_| "git rev-parse failed".to_string())?;
    if !output.status.success() {
        return Err("git rev-parse failed".into());
    }
    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if text.len() != 40 || !text.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return Err("revision is not a full hash".into());
    }
    Ok(text)
}

fn online_processors() -> Result<usize, String> {
    let value = unsafe { sysconf(84) };
    if value <= 0 {
        return Err("online processor count is unavailable".into());
    }
    usize::try_from(value).map_err(|_| "online processor count does not fit".into())
}

fn cpuinfo_processors() -> Result<usize, String> {
    let text = std::fs::read_to_string("/proc/cpuinfo").map_err(|err| err.to_string())?;
    let count = text
        .lines()
        .filter(|line| line.starts_with("processor"))
        .count();
    if count == 0 {
        return Err("cpuinfo lists no processors".into());
    }
    Ok(count)
}

fn cpu_model() -> Result<String, String> {
    let text = std::fs::read_to_string("/proc/cpuinfo").map_err(|err| err.to_string())?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("model name") {
            let model = rest.trim_start_matches([' ', '\t', ':']).trim();
            if !model.is_empty() {
                return Ok(model.to_string());
            }
        }
    }
    Err("cpuinfo has no model name".into())
}

fn pin_cores(cores: &[usize]) -> Result<Vec<usize>, String> {
    if cores.is_empty() {
        return Err("affinity mask is empty".into());
    }
    let mut mask = [0u8; 128];
    for &cpu in cores {
        if cpu >= 1024 {
            return Err(format!("core {cpu} does not fit the mask"));
        }
        mask[cpu / 8] |= 1 << (cpu % 8);
    }
    let rc = unsafe { sched_setaffinity(0, mask.len(), mask.as_ptr()) };
    if rc != 0 {
        return Err(format!(
            "set affinity failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    let mut got = [0u8; 128];
    let rc = unsafe { sched_getaffinity(0, got.len(), got.as_mut_ptr()) };
    if rc != 0 {
        return Err(format!(
            "get affinity failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    let mut set = Vec::new();
    for cpu in 0..1024 {
        if got[cpu / 8] & (1 << (cpu % 8)) != 0 {
            set.push(cpu);
        }
    }
    if set.is_empty() {
        return Err("affinity mask came back empty".into());
    }
    Ok(set)
}

fn format_cores(cores: &[usize]) -> String {
    if cores.is_empty() {
        return String::new();
    }
    let contiguous = cores
        .windows(2)
        .all(|pair| pair[1] == pair[0].saturating_add(1));
    if contiguous && cores.len() > 1 {
        format!("{}-{}", cores[0], cores[cores.len() - 1])
    } else if cores.len() == 1 {
        cores[0].to_string()
    } else {
        cores
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(";")
    }
}

fn same_geometry(left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
    if left.len() != right.len() || left.len() % 3 != 0 {
        return false;
    }
    let mut a = pair_lengths(left);
    let mut b = pair_lengths(right);
    a.sort_by(|left, right| left.total_cmp(right));
    b.sort_by(|left, right| left.total_cmp(right));
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(left, right)| (left - right).abs() <= 1.0e-3)
}

fn pair_lengths(x: ArrayView1<f64>) -> Vec<f64> {
    let n = x.len() / 3;
    let mut out = Vec::with_capacity(n * n.saturating_sub(1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut sum = 0.0;
            for axis in 0..3 {
                let delta = x[3 * i + axis] - x[3 * j + axis];
                sum += delta * delta;
            }
            out.push(sum.sqrt());
        }
    }
    out
}

unsafe extern "C" {
    fn sysconf(name: i32) -> i64;
    fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u8) -> i32;
    fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut u8) -> i32;
}
