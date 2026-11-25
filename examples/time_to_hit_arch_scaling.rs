// Copyright (c) anneal developers
//! Time-to-hit worker scaling for the shared communicating ensemble.
//!
//! Usage:
//! `time_to_hit_arch_scaling <lj38|au38|m55r6> <seed> <workers> <rows.csv>`.

use std::fs::OpenOptions;
use std::io::{ErrorKind, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anneal_core::descriptor_space::{universal_descriptor_space, DescriptorGeometry};
use anneal_core::methods::cluster_hopping::{random_cluster_in_radius, Config};
use anneal_core::methods::ensemble::{
    run_ensemble, EnsembleConfig, EnsembleProblem, EnsembleReport, HistoryMode,
};
use anneal_core::methods::minima_hopping::{HistoryMembership, SerializedWitness};
use anneal_core::pes_exploration::StructureContext;
use anneal_core::potentials::{Gupta, PairPotential};
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;

const WORKERS: [usize; 4] = [1, 2, 4, 8];
const HEADER: &str = "system,arm,seed,workers,oracle_calls,call_budget,hit,calls_to_hit,wall_to_hit_s,best_energy,wall_s,ledger_charged,ledger_sum,objective_calls,revision,nproc,pinned_cores";

#[derive(Clone, Copy)]
enum Kind {
    LennardJones,
    Gold,
    Morse6,
}

#[derive(Clone, Copy)]
struct Spec {
    name: &'static str,
    n: usize,
    budget: usize,
    reference: f64,
    kind: Kind,
}

fn spec(name: &str) -> Option<Spec> {
    match name {
        "lj38" => Some(Spec {
            name: "lj38",
            n: 38,
            budget: 400_000,
            reference: -173.928427,
            kind: Kind::LennardJones,
        }),
        "au38" => Some(Spec {
            name: "au38",
            n: 38,
            budget: 400_000,
            reference: -130.54001708,
            kind: Kind::Gold,
        }),
        "m55r6" => Some(Spec {
            name: "m55r6",
            n: 55,
            budget: 1_000_000,
            reference: -250.286609,
            kind: Kind::Morse6,
        }),
        _ => None,
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
    let result = run(&args);
    if let Err(err) = result {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let system = args.first().ok_or("system is required")?;
    let spec = spec(system).ok_or_else(|| format!("unknown system {system}"))?;
    let seed = args
        .get(1)
        .ok_or("seed is required")?
        .parse::<u64>()
        .map_err(|_| "seed is not an integer")?;
    let workers = args
        .get(2)
        .ok_or("workers is required")?
        .parse::<usize>()
        .map_err(|_| "workers is not an integer")?;
    let rows = args.get(3).ok_or("rows path is required")?;
    if !(2000..=2011).contains(&seed) {
        return Err("seed is outside 2000-2011".into());
    }
    if !WORKERS.contains(&workers) {
        return Err("workers must be 1, 2, 4, or 8".into());
    }
    check_budget_source(spec)?;
    if already_stored(rows, spec.name, seed, workers)? {
        eprintln!("skip {} seed {seed} workers {workers}", spec.name);
        return Ok(());
    }
    let row = measure(spec, seed, workers)?;
    append_row(rows, &row)?;
    eprintln!(
        "done {} seed {seed} workers {workers} hit={} calls={} wall_s={:.6}",
        spec.name,
        u8::from(row.hit),
        row.calls_to_hit,
        row.wall_s
    );
    Ok(())
}

fn check_budget_source(spec: Spec) -> Result<(), String> {
    let text = std::fs::read_to_string("scripts/paper_0120/run_panel.py")
        .map_err(|err| format!("read paper budget source: {err}"))?;
    let expected = "\"lj38\": 400_000";
    if spec.name == "lj38" && !text.contains(&expected) {
        return Err(format!("paper budget source does not contain {expected}"));
    }
    Ok(())
}

struct Row {
    system: &'static str,
    seed: u64,
    workers: usize,
    oracle_calls: usize,
    call_budget: usize,
    hit: bool,
    calls_to_hit: usize,
    wall_to_hit_s: Option<f64>,
    best_energy: f64,
    wall_s: f64,
    ledger_charged: usize,
    ledger_sum: usize,
    objective_calls: usize,
    revision: String,
    nproc: usize,
    pinned_cores: String,
}

fn measure(spec: Spec, seed: u64, workers: usize) -> Result<Row, String> {
    let nproc = online_processors()?;
    if nproc != cpuinfo_processors()? {
        return Err("online processor count disagrees with /proc/cpuinfo".into());
    }
    if workers > nproc {
        return Err(format!("{workers} workers exceed nproc {nproc}"));
    }
    let pinned = pin_cores(&(0..workers).collect::<Vec<_>>())?;
    let species = match spec.kind {
        Kind::Gold => Some(Gupta::copper_gold_species(0, spec.n)),
        _ => None,
    };
    let cfg = match species.clone() {
        Some(values) => Config::communicating(spec.n).with_species(values, None),
        None => Config::communicating(spec.n),
    };
    if cfg.replicas != 1 {
        return Err("communicating preset is not a single-rung chain".into());
    }
    let target = spec.reference + anneal_core::DEFAULT_HIT_TOL;
    let ens = EnsembleConfig {
        replicas: workers,
        budget: spec.budget,
        history: HistoryMode::Shared,
        membership: HistoryMembership::Accepted,
        shared_bias: None,
        gossip: None,
        two_choice_stall: None,
        checkpoint_interval: 1,
        target: Some(target),
    };
    let geometry = DescriptorGeometry::finite(cfg.length_scale).map_err(|err| err.to_string())?;
    let descriptor = universal_descriptor_space(geometry);
    let context = StructureContext::new(species, None, Some(spec.name.into()));
    let witness = SerializedWitness(Mutex::new(same_geometry));
    let radius = cfg.start_radius();
    let min_sep = cfg.min_separation;
    let n_atoms = spec.n;
    let stop = Arc::new(AtomicBool::new(false));
    let _stop_guard = anneal_core::caller_stop::Bind::new(Arc::clone(&stop));
    let clock = Arc::new(CallClock::new());
    let objective = |_: usize| {
        let potential = Potential::new(spec);
        let clock = Arc::clone(&clock);
        let stop = Arc::clone(&stop);
        Box::new(move |x: ArrayView1<'_, f64>| {
            let pair = potential.value_and_gradient(x);
            clock.stamp(pair.0, target, &stop);
            pair
        }) as Box<dyn FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>) + Send>
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
    let report = run_ensemble(&cfg, &ens, seed, &problem)?;
    let wall_s = clock.elapsed();
    check_report(spec, &report, &clock)?;
    let hit_call = clock.hit_call();
    let hit = hit_call.is_some();
    let calls_to_hit = hit_call.unwrap_or(spec.budget + 1);
    let wall_to_hit_s = clock.hit_seconds();
    let ledger_sum: usize = report.replicas.iter().map(|replica| replica.charged).sum();
    Ok(Row {
        system: spec.name,
        seed,
        workers,
        oracle_calls: report.aggregate_charged,
        call_budget: spec.budget,
        hit,
        calls_to_hit,
        wall_to_hit_s,
        best_energy: clock.best(),
        wall_s,
        ledger_charged: report.aggregate_charged,
        ledger_sum,
        objective_calls: clock.calls(),
        revision: git_revision()?,
        nproc,
        pinned_cores: format_cores(&pinned),
    })
}

enum Potential {
    Pair(PairPotential),
    Gold(Gupta),
}

impl Potential {
    fn new(spec: Spec) -> Self {
        match spec.kind {
            Kind::LennardJones => Self::Pair(PairPotential::lennard_jones(spec.n)),
            Kind::Gold => Self::Gold(Gupta::copper_gold(0, spec.n)),
            Kind::Morse6 => Self::Pair(PairPotential::morse(spec.n, 6.0)),
        }
    }

    fn value_and_gradient(&self, x: ArrayView1<'_, f64>) -> (f64, Array1<f64>) {
        match self {
            Self::Pair(potential) => potential.value_and_gradient(x),
            Self::Gold(potential) => potential.value_and_gradient(x),
        }
    }
}

fn check_report(spec: Spec, report: &EnsembleReport, clock: &CallClock) -> Result<(), String> {
    let ledger_sum: usize = report.replicas.iter().map(|replica| replica.charged).sum();
    if ledger_sum != report.aggregate_charged {
        return Err(format!(
            "ledger sum {ledger_sum} disagrees with ensemble {}",
            report.aggregate_charged
        ));
    }
    if clock.calls() != report.aggregate_charged {
        return Err(format!(
            "objective calls {} disagree with ensemble {}",
            clock.calls(),
            report.aggregate_charged
        ));
    }
    if report.aggregate_charged > spec.budget {
        return Err("ensemble exceeded the call budget".into());
    }
    if clock.hit_call().is_some() && report.aggregate_charged >= spec.budget {
        return Err("hit did not stop before the budget".into());
    }
    if clock.elapsed() < 0.05 {
        return Err("run was shorter than 50 ms".into());
    }
    Ok(())
}

struct CallClock {
    start: Instant,
    calls: AtomicUsize,
    hit_call: AtomicUsize,
    hit_nanos: AtomicU64,
    best_bits: Mutex<u64>,
}

impl CallClock {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            calls: AtomicUsize::new(0),
            hit_call: AtomicUsize::new(0),
            hit_nanos: AtomicU64::new(0),
            best_bits: Mutex::new(f64::INFINITY.to_bits()),
        }
    }

    fn stamp(&self, energy: f64, target: f64, stop: &AtomicBool) {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        {
            let mut bits = self.best_bits.lock().expect("best energy");
            if energy < f64::from_bits(*bits) {
                *bits = energy.to_bits();
            }
        }
        if energy <= target
            && self
                .hit_call
                .compare_exchange(0, call, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
        {
            self.hit_nanos
                .store(self.start.elapsed().as_nanos() as u64, Ordering::SeqCst);
            stop.store(true, Ordering::SeqCst);
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn elapsed(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    fn hit_call(&self) -> Option<usize> {
        match self.hit_call.load(Ordering::SeqCst) {
            0 => None,
            value => Some(value),
        }
    }

    fn hit_seconds(&self) -> Option<f64> {
        self.hit_call()
            .map(|_| self.hit_nanos.load(Ordering::SeqCst) as f64 / 1.0e9)
    }

    fn best(&self) -> f64 {
        f64::from_bits(*self.best_bits.lock().expect("best energy"))
    }
}

fn append_row(path: &str, row: &Row) -> Result<(), String> {
    let path = Path::new(path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    let fresh = match std::fs::metadata(path) {
        Ok(meta) => meta.len() == 0,
        Err(err) if err.kind() == ErrorKind::NotFound => true,
        Err(err) => return Err(err.to_string()),
    };
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| err.to_string())?;
    if fresh {
        writeln!(file, "{HEADER}").map_err(|err| err.to_string())?;
    }
    writeln!(
        file,
        "{},comm,{},{},{},{},{},{},{},{:.10},{:.6},{},{},{},{},{},{}",
        row.system,
        row.seed,
        row.workers,
        row.oracle_calls,
        row.call_budget,
        u8::from(row.hit),
        row.calls_to_hit,
        row.wall_to_hit_s
            .map(|value| format!("{value:.6}"))
            .unwrap_or_default(),
        row.best_energy,
        row.wall_s,
        row.ledger_charged,
        row.ledger_sum,
        row.objective_calls,
        row.revision,
        row.nproc,
        csv_field(&row.pinned_cores),
    )
    .map_err(|err| err.to_string())?;
    file.sync_all().map_err(|err| err.to_string())
}

fn already_stored(path: &str, system: &str, seed: u64, workers: usize) -> Result<bool, String> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(false),
        Err(err) => return Err(err.to_string()),
    };
    let mut lines = text.lines();
    if lines.next() != Some(HEADER) {
        return Err("rows header does not match".into());
    }
    Ok(lines.any(|line| {
        let fields: Vec<_> = line.split(',').collect();
        fields.first() == Some(&system)
            && fields.get(2).and_then(|value| value.parse().ok()) == Some(seed)
            && fields.get(3).and_then(|value| value.parse().ok()) == Some(workers)
    }))
}

fn same_geometry(left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
    if left.len() != right.len() || left.len() % 3 != 0 {
        return false;
    }
    let mut a = pair_lengths(left);
    let mut b = pair_lengths(right);
    a.sort_by(f64::total_cmp);
    b.sort_by(f64::total_cmp);
    a.iter()
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

fn csv_field(text: &str) -> String {
    if text.contains([',', '"', '\n']) {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
}

fn git_revision() -> Result<String, String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .map_err(|err| format!("git revision: {err}"))?;
    if !output.status.success() {
        return Err("git revision command failed".into());
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
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
    (count > 0)
        .then_some(count)
        .ok_or_else(|| "no processors in /proc/cpuinfo".into())
}

fn pin_cores(cores: &[usize]) -> Result<Vec<usize>, String> {
    let mut mask = [0u8; 128];
    for &cpu in cores {
        if cpu >= 1024 {
            return Err("core does not fit affinity mask".into());
        }
        mask[cpu / 8] |= 1 << (cpu % 8);
    }
    if unsafe { sched_setaffinity(0, mask.len(), mask.as_ptr()) } != 0 {
        return Err(format!(
            "set affinity failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    let mut got = [0u8; 128];
    if unsafe { sched_getaffinity(0, got.len(), got.as_mut_ptr()) } != 0 {
        return Err(format!(
            "get affinity failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok((0..1024)
        .filter(|cpu| got[cpu / 8] & (1 << (cpu % 8)) != 0)
        .collect())
}

fn format_cores(cores: &[usize]) -> String {
    if cores.len() == 1 {
        cores[0].to_string()
    } else {
        format!("{}-{}", cores[0], cores[cores.len() - 1])
    }
}

unsafe extern "C" {
    fn sysconf(name: i32) -> i64;
    fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u8) -> i32;
    fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut u8) -> i32;
}
