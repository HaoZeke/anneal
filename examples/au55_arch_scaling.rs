// Copyright (c) anneal developers
//! Worker ladder for Gupta Au55 on the shared communicating ensemble.
//!
//! ```text
//! au55_arch_scaling ladder
//! au55_arch_scaling one <seed> <workers>
//! au55_arch_scaling probe <calls> <workers>
//! ```
//!
//! `ladder` and `one` append to `bench/arch-scaling/au55/rows.csv`.
//! `probe` prints one timing line and does not write a row. The contract
//! is `bench/arch-scaling/au55/PROTOCOL.md`.

use std::fs::OpenOptions;
use std::io::{ErrorKind, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use anneal_core::descriptor_space::{DescriptorGeometry, universal_descriptor_space};
use anneal_core::methods::cluster_hopping::{Config, random_cluster_in_radius};
use anneal_core::methods::ensemble::{
    EnsembleConfig, EnsembleProblem, EnsembleReport, HistoryMode, run_ensemble,
};
use anneal_core::methods::minima_hopping::{HistoryMembership, SerializedWitness};
use anneal_core::pes_exploration::StructureContext;
use anneal_core::potentials::Gupta;
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;

const SEEDS: std::ops::RangeInclusive<u64> = 2000..=2011;
const WORKERS: [usize; 5] = [1, 2, 4, 8, 16];
const ROWS: &str = "bench/arch-scaling/au55/rows.csv";
const BUDGET_FILE: &str = "scripts/paper_0120/run_panel.py";
const BUDGET_LINE: usize = 53;
const SPEC_FILE: &str = "examples/h2h_metal.rs";
const HEADER: &str = "system,arm,seed,workers,oracle_calls,call_budget,hit,calls_to_hit,wall_to_hit_s,best_energy,wall_s,cpu_s,peak_rss_anon,ledger_charged,ledger_sum,hist_minima,hist_accepted,hist_visits,hist_observations,hist_refusals,exchange_published,exchange_delivered,bias_published,two_choice_restarts,objective_calls,revision,nproc,cpu_model,pinned_cores,oversubscribed";

fn main() {
    unsafe {
        std::env::set_var("OMP_NUM_THREADS", "1");
        std::env::set_var("RAYON_NUM_THREADS", "1");
        std::env::remove_var("ANNEAL_MIN_DUMP");
    }
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let status = match args.first().map(String::as_str) {
        Some("ladder") => cmd_ladder(&args[1..]),
        Some("one") => cmd_one(&args[1..]),
        Some("probe") => cmd_probe(&args[1..]),
        _ => {
            eprintln!("usage: au55_arch_scaling ladder|one|probe");
            2
        }
    };
    std::process::exit(status);
}

fn cmd_ladder(args: &[String]) -> i32 {
    let only = match parse_only(args) {
        Ok(only) => only,
        Err(err) => {
            eprintln!("{err}");
            return 2;
        }
    };
    let ctx = match Context::load() {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!("{err}");
            return 1;
        }
    };
    eprintln!(
        "nproc={} cpu={} budget={} revision={}",
        ctx.nproc, ctx.cpu_model, ctx.budget, ctx.revision
    );
    for seed in SEEDS {
        for workers in order_for(seed) {
            if !only.is_empty() && !only.contains(&workers) {
                continue;
            }
            if let Err(err) = run_recorded(&ctx, seed, workers) {
                eprintln!("seed {seed} workers {workers}: {err}");
                return 1;
            }
        }
    }
    0
}

fn cmd_one(args: &[String]) -> i32 {
    let Some(seed) = args.first().and_then(|text| text.parse::<u64>().ok()) else {
        eprintln!("one needs a seed");
        return 2;
    };
    let Some(workers) = args.get(1).and_then(|text| text.parse::<usize>().ok()) else {
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
    if let Err(err) = run_recorded(&ctx, seed, workers) {
        eprintln!("{err}");
        return 1;
    }
    0
}

fn cmd_probe(args: &[String]) -> i32 {
    let Some(calls) = args.first().and_then(|text| text.parse::<usize>().ok()) else {
        eprintln!("probe needs a call count");
        return 2;
    };
    let Some(workers) = args.get(1).and_then(|text| text.parse::<usize>().ok()) else {
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
    match measure(&ctx, 2000, workers, calls, false) {
        Ok(row) => {
            println!(
                "probe workers={} calls={} wall_s={:.6} cpu_s={:.6} oracle={} objective={} best={:.8} hit={} peak_rss_anon={} oversubscribed={}",
                row.workers,
                calls,
                row.wall_s,
                row.cpu_s,
                row.oracle_calls,
                row.objective_calls,
                row.best_energy,
                u8::from(row.hit),
                row.peak_rss_anon,
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

fn parse_only(args: &[String]) -> Result<Vec<usize>, String> {
    let mut only = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] != "--only" {
            return Err(format!("unknown ladder argument {}", args[i]));
        }
        let Some(text) = args.get(i + 1) else {
            return Err("--only needs a worker count".into());
        };
        let workers: usize = text
            .parse()
            .map_err(|_| format!("--only {text} is not a count"))?;
        if !WORKERS.contains(&workers) {
            return Err(format!("--only {workers} is outside the ladder"));
        }
        only.push(workers);
        i += 2;
    }
    Ok(only)
}

fn order_for(seed: u64) -> Vec<usize> {
    let start = (seed - *SEEDS.start()) as usize % WORKERS.len();
    WORKERS[start..]
        .iter()
        .chain(WORKERS[..start].iter())
        .copied()
        .collect()
}

struct Context {
    budget: usize,
    n_atoms: usize,
    reference: f64,
    nproc: usize,
    cpu_model: String,
    revision: String,
}

impl Context {
    fn load() -> Result<Self, String> {
        let budget_line = file_line(BUDGET_FILE, BUDGET_LINE)?;
        let after = budget_line
            .split_once(':')
            .map(|(_, rest)| rest)
            .ok_or("budget line has no value")?;
        let digits: String = after.chars().filter(|ch| ch.is_ascii_digit()).collect();
        let budget: usize = digits
            .parse()
            .map_err(|_| format!("budget line {budget_line:?} has no integer"))?;
        if !budget_line.contains("\"au55\"") || budget != 1_000_000 {
            return Err(format!("budget line is not the Au55 entry: {budget_line:?}"));
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
            budget,
            n_atoms: 55,
            reference: -191.03085280,
            nproc,
            cpu_model: cpu_model()?,
            revision: git_revision()?,
        })
    }
}

struct Row {
    seed: u64,
    workers: usize,
    oracle_calls: usize,
    call_budget: usize,
    hit: bool,
    calls_to_hit: usize,
    wall_to_hit_s: Option<f64>,
    best_energy: f64,
    wall_s: f64,
    cpu_s: f64,
    peak_rss_anon: u64,
    ledger_charged: usize,
    ledger_sum: usize,
    hist_minima: usize,
    hist_accepted: usize,
    hist_visits: u64,
    hist_observations: usize,
    hist_refusals: usize,
    exchange_published: u64,
    exchange_delivered: u64,
    bias_published: u64,
    two_choice_restarts: usize,
    objective_calls: usize,
    revision: String,
    nproc: usize,
    cpu_model: String,
    pinned_cores: String,
    oversubscribed: bool,
}

fn run_recorded(ctx: &Context, seed: u64, workers: usize) -> Result<(), String> {
    if already_stored(seed, workers)? {
        eprintln!("skip seed {seed} workers {workers}");
        return Ok(());
    }
    let row = measure(ctx, seed, workers, ctx.budget, true)?;
    append_row(&row)?;
    eprintln!(
        "seed {seed} workers {workers} hit={} calls_to_hit={} wall_s={:.3} wall_to_hit_s={}",
        u8::from(row.hit),
        row.calls_to_hit,
        row.wall_s,
        row.wall_to_hit_s
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "-".into())
    );
    Ok(())
}

fn measure(
    ctx: &Context,
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
    let species = Gupta::copper_gold_species(0, ctx.n_atoms);
    let mut cfg = Config::communicating(ctx.n_atoms).with_species(species.clone(), None);
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
        target: Some(ctx.reference + anneal_core::DEFAULT_HIT_TOL),
    };
    let geometry = DescriptorGeometry::finite(cfg.length_scale)?;
    let descriptor = universal_descriptor_space(geometry);
    let context = StructureContext::new(Some(species), None, Some("au55".into()));
    let witness = SerializedWitness(Mutex::new(same_geometry));
    let gupta = Gupta::copper_gold(0, ctx.n_atoms);
    let radius = cfg.start_radius();
    let min_sep = cfg.min_separation;
    let n_atoms = ctx.n_atoms;
    let peak = Arc::new(Peak::new());
    let sampler = {
        let peak = Arc::clone(&peak);
        std::thread::spawn(move || peak.watch())
    };
    let cpu0 = proc_cpu()?;
    let clock = Arc::new(CallClock::new(budget));
    let objective = |_: usize| {
        let potential = gupta.clone();
        let clock = Arc::clone(&clock);
        Box::new(move |x: ArrayView1<'_, f64>| {
            let pair = potential.value_and_gradient(x);
            clock.stamp();
            pair
        }) as Box<dyn FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>) + Send>
    };
    let start = move |_: usize, rng: &mut StdRng| {
        random_cluster_in_radius(n_atoms, radius, min_sep, rng)
    };
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
    let report = run_ensemble(&cfg, &ens, seed, &problem)?;
    let wall_s = clock.elapsed();
    let cpu_s = (proc_cpu()? - cpu0).max(0.0);
    peak.stop.store(true, Ordering::Relaxed);
    sampler
        .join()
        .map_err(|_| "memory sampler panicked".to_string())?;
    let peak_rss_anon = peak.bytes.load(Ordering::Relaxed).max(rss_anon_bytes());
    check_report(ctx, budget, &report, &clock, wall_s, require_duration)?;
    let hit = anneal_core::energy_hits(report.best, ctx.reference, anneal_core::DEFAULT_HIT_TOL);
    let calls_to_hit = if hit {
        report
            .first_target_calls
            .ok_or("a hit has no aggregate call count")?
    } else {
        ctx.budget + 1
    };
    let wall_to_hit_s = if hit {
        Some(clock.at(calls_to_hit)?)
    } else {
        None
    };
    if let Some(at) = wall_to_hit_s
        && at > wall_s + 1e-6
    {
        return Err(format!("wall to hit {at} is after the run {wall_s}"));
    }
    let ledger_sum: usize = report.replicas.iter().map(|replica| replica.charged).sum();
    if ledger_sum != report.aggregate_charged {
        return Err(format!(
            "replica charged sum {ledger_sum} disagrees with aggregate {}",
            report.aggregate_charged
        ));
    }
    let (hist_minima, hist_accepted, hist_visits) = match report.histories.as_slice() {
        [one] => *one,
        other => {
            return Err(format!(
                "shared history count is {}, expected 1",
                other.len()
            ));
        }
    };
    let hist_observations = report
        .replicas
        .iter()
        .map(|replica| replica.history_cost.0)
        .sum();
    let hist_refusals = report
        .replicas
        .iter()
        .map(|replica| replica.history_cost.1)
        .sum();
    let bias_published = report
        .replicas
        .iter()
        .map(|replica| replica.bias_published)
        .sum();
    let two_choice_restarts = report
        .replicas
        .iter()
        .map(|replica| replica.two_choice_restarts)
        .sum();
    Ok(Row {
        seed,
        workers,
        oracle_calls: report.aggregate_charged,
        call_budget: ctx.budget,
        hit,
        calls_to_hit,
        wall_to_hit_s,
        best_energy: report.best,
        wall_s,
        cpu_s,
        peak_rss_anon,
        ledger_charged: report.aggregate_charged,
        ledger_sum,
        hist_minima,
        hist_accepted,
        hist_visits,
        hist_observations,
        hist_refusals,
        exchange_published: report.exchange.0,
        exchange_delivered: report.exchange.1,
        bias_published,
        two_choice_restarts,
        objective_calls: clock.filled(),
        revision: ctx.revision.clone(),
        nproc: ctx.nproc,
        cpu_model: ctx.cpu_model.clone(),
        pinned_cores: format_cores(&pinned),
        oversubscribed,
    })
}

fn check_report(
    ctx: &Context,
    budget: usize,
    report: &EnsembleReport,
    clock: &CallClock,
    wall_s: f64,
    require_duration: bool,
) -> Result<(), String> {
    let solved = report.solved(Some(ctx.reference + anneal_core::DEFAULT_HIT_TOL));
    let hit = anneal_core::energy_hits(report.best, ctx.reference, anneal_core::DEFAULT_HIT_TOL);
    if hit != solved {
        return Err("ensemble solved flag disagrees with the hit rule".into());
    }
    if clock.filled() != report.aggregate_charged {
        return Err(format!(
            "objective stamps {} disagree with charged {}",
            clock.filled(),
            report.aggregate_charged
        ));
    }
    if report.aggregate_charged > budget {
        return Err(format!(
            "charged {} exceeds the budget {budget}",
            report.aggregate_charged
        ));
    }
    if require_duration && wall_s < 0.05 {
        return Err(format!("wall {wall_s:.6}s is under 50 ms"));
    }
    Ok(())
}

struct CallClock {
    start: Instant,
    next: AtomicUsize,
    nanos: Vec<AtomicU64>,
}

impl CallClock {
    fn new(budget: usize) -> Self {
        let mut nanos = Vec::with_capacity(budget);
        for _ in 0..budget {
            nanos.push(AtomicU64::new(0));
        }
        Self {
            start: Instant::now(),
            next: AtomicUsize::new(0),
            nanos,
        }
    }

    fn stamp(&self) {
        let index = self.next.fetch_add(1, Ordering::Relaxed);
        let elapsed = self.start.elapsed().as_nanos() as u64;
        if let Some(slot) = self.nanos.get(index) {
            slot.store(elapsed.max(1), Ordering::Relaxed);
        }
    }

    fn filled(&self) -> usize {
        self.next.load(Ordering::Relaxed)
    }

    fn elapsed(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    fn at(&self, calls: usize) -> Result<f64, String> {
        if calls == 0 || calls > self.nanos.len() {
            return Err(format!("call index {calls} is outside the stamp buffer"));
        }
        let stored = self.nanos[calls - 1].load(Ordering::Relaxed);
        if stored == 0 {
            return Err(format!("objective call {calls} has no stamp"));
        }
        Ok(stored as f64 / 1.0e9)
    }
}

struct Peak {
    stop: AtomicBool,
    bytes: AtomicU64,
}

impl Peak {
    fn new() -> Self {
        Self {
            stop: AtomicBool::new(false),
            bytes: AtomicU64::new(rss_anon_bytes()),
        }
    }

    fn watch(&self) {
        while !self.stop.load(Ordering::Relaxed) {
            self.bytes.fetch_max(rss_anon_bytes(), Ordering::Relaxed);
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        self.bytes.fetch_max(rss_anon_bytes(), Ordering::Relaxed);
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
            return Err("rows.csv header does not match the protocol".into());
        }
        if existing
            .lines()
            .skip(1)
            .any(|line| row_key(line) == Some((row.seed, row.workers)))
        {
            return Err(format!(
                "seed {} workers {} was stored while this run was in flight",
                row.seed, row.workers
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

fn already_stored(seed: u64, workers: usize) -> Result<bool, String> {
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
        return Err("rows.csv header does not match the protocol".into());
    }
    Ok(lines.any(|line| row_key(line) == Some((seed, workers))))
}

fn row_key(line: &str) -> Option<(u64, usize)> {
    let mut fields = line.split(',');
    let system = fields.next()?;
    let arm = fields.next()?;
    let seed = fields.next()?.parse().ok()?;
    let workers = fields.next()?.parse().ok()?;
    (system == "au55" && arm == "comm").then_some((seed, workers))
}

impl Row {
    fn line(&self) -> String {
        format!(
            "au55,comm,{},{},{},{},{},{},{},{:.10},{:.6},{:.6},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            self.seed,
            self.workers,
            self.oracle_calls,
            self.call_budget,
            u8::from(self.hit),
            self.calls_to_hit,
            self.wall_to_hit_s
                .map(|value| format!("{value:.6}"))
                .unwrap_or_default(),
            self.best_energy,
            self.wall_s,
            self.cpu_s,
            self.peak_rss_anon,
            self.ledger_charged,
            self.ledger_sum,
            self.hist_minima,
            self.hist_accepted,
            self.hist_visits,
            self.hist_observations,
            self.hist_refusals,
            self.exchange_published,
            self.exchange_delivered,
            self.bias_published,
            self.two_choice_restarts,
            self.objective_calls,
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

fn proc_cpu() -> Result<f64, String> {
    let text = std::fs::read_to_string("/proc/self/stat").map_err(|err| err.to_string())?;
    let rest = text
        .rsplit_once(')')
        .map(|(_, rest)| rest)
        .ok_or("process stat has no command field")?;
    let fields: Vec<&str> = rest.split_whitespace().collect();
    let utime: u64 = fields
        .get(11)
        .ok_or("process stat has no user time")?
        .parse()
        .map_err(|_| "user time is not an integer")?;
    let stime: u64 = fields
        .get(12)
        .ok_or("process stat has no system time")?
        .parse()
        .map_err(|_| "system time is not an integer")?;
    let ticks = unsafe { sysconf(2) };
    if ticks <= 0 {
        return Err("clock tick count is unavailable".into());
    }
    Ok((utime + stime) as f64 / ticks as f64)
}

fn rss_anon_bytes() -> u64 {
    let Ok(text) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("RssAnon:") {
            let kb: u64 = rest
                .split_whitespace()
                .next()
                .unwrap_or("0")
                .parse()
                .unwrap_or(0);
            return kb.saturating_mul(1024);
        }
    }
    0
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

unsafe extern "C" {
    fn sysconf(name: i32) -> i64;
    fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u8) -> i32;
    fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut u8) -> i32;
}

impl From<anneal_core::descriptor_space::DescriptorError> for String {
    fn from(err: anneal_core::descriptor_space::DescriptorError) -> Self {
        err.to_string()
    }
}
