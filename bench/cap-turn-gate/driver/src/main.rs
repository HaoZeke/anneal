//! Unified single-process gate driver for `Config::cap_twist`.
//!
//! Runs Lennard-Jones, binary Lennard-Jones, Cambridge Gupta gold, Morse, and
//! GMIN TIP4P water with identical preset mechanics to the stock examples,
//! while sampling peak in-process `RssAnon` and `VmHWM` from
//! `/proc/self/status` and emitting `cap_twist` counters.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::cluster_search::search_from;
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{
    BinaryLennardJones, GminTip4p, Gupta, PairPotential, gmin_water_geometry, random_tip4p_cluster,
};
use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::fs::File;
use std::io::Read;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

mod padding {
    include!(concat!(env!("OUT_DIR"), "/gate_padding.rs"));
}

const HIT_TOL: f64 = 1.0e-3;
const STATUS_STRIDE_MASK: u64 = 1023;

#[derive(Clone, Copy)]
enum CellKind {
    Lj,
    Morse { rho: f64 },
    Au,
    Blj { n_a: usize, sigma_bb: f64 },
    Water,
}

#[derive(Clone, Copy)]
struct CellSpec {
    name: &'static str,
    family: &'static str,
    n: usize,
    budget: usize,
    reference: f64,
    kind: CellKind,
}

fn cell_spec(name: &str) -> Option<CellSpec> {
    let spec = match name {
        "lj38" => CellSpec {
            name: "lj38",
            family: "lj",
            n: 38,
            budget: 400_000,
            reference: -173.928427,
            kind: CellKind::Lj,
        },
        "lj55" => CellSpec {
            name: "lj55",
            family: "lj",
            n: 55,
            budget: 1_000_000,
            reference: -279.248470,
            kind: CellKind::Lj,
        },
        "lj75" => CellSpec {
            name: "lj75",
            family: "lj",
            n: 75,
            budget: 4_000_000,
            reference: -397.492331,
            kind: CellKind::Lj,
        },
        "lj98" => CellSpec {
            name: "lj98",
            family: "lj",
            n: 98,
            budget: 4_000_000,
            reference: -543.665361,
            kind: CellKind::Lj,
        },
        "lj102" => CellSpec {
            name: "lj102",
            family: "lj",
            n: 102,
            budget: 4_000_000,
            reference: -569.363652,
            kind: CellKind::Lj,
        },
        "lj103" => CellSpec {
            name: "lj103",
            family: "lj",
            n: 103,
            budget: 4_000_000,
            reference: -575.766131,
            kind: CellKind::Lj,
        },
        "lj104" => CellSpec {
            name: "lj104",
            family: "lj",
            n: 104,
            budget: 4_000_000,
            reference: -582.086642,
            kind: CellKind::Lj,
        },
        "lj150" => CellSpec {
            name: "lj150",
            family: "lj",
            n: 150,
            budget: 4_000_000,
            reference: -893.310258,
            kind: CellKind::Lj,
        },
        "blj105n38" => CellSpec {
            name: "blj105n38",
            family: "blj",
            n: 38,
            budget: 400_000,
            reference: -177.2090846748,
            kind: CellKind::Blj {
                n_a: 8,
                sigma_bb: 1.05,
            },
        },
        "blj105n55" => CellSpec {
            name: "blj105n55",
            family: "blj",
            n: 55,
            budget: 1_000_000,
            reference: -283.1545313810,
            kind: CellKind::Blj {
                n_a: 13,
                sigma_bb: 1.05,
            },
        },
        "blj115n38" => CellSpec {
            name: "blj115n38",
            family: "blj",
            n: 38,
            budget: 400_000,
            reference: -187.3676867453,
            kind: CellKind::Blj {
                n_a: 9,
                sigma_bb: 1.15,
            },
        },
        "blj115n55" => CellSpec {
            name: "blj115n55",
            family: "blj",
            n: 55,
            budget: 1_000_000,
            reference: -289.2931853058,
            kind: CellKind::Blj {
                n_a: 16,
                sigma_bb: 1.15,
            },
        },
        "au38" => CellSpec {
            name: "au38",
            family: "au",
            n: 38,
            budget: 400_000,
            reference: -130.54001708,
            kind: CellKind::Au,
        },
        "au55" => CellSpec {
            name: "au55",
            family: "au",
            n: 55,
            budget: 1_000_000,
            reference: -191.03085280,
            kind: CellKind::Au,
        },
        "m38r6" => CellSpec {
            name: "m38r6",
            family: "morse",
            n: 38,
            budget: 400_000,
            reference: -157.477108,
            kind: CellKind::Morse { rho: 6.0 },
        },
        "m38r14" => CellSpec {
            name: "m38r14",
            family: "morse",
            n: 38,
            budget: 400_000,
            reference: -144.321054,
            kind: CellKind::Morse { rho: 14.0 },
        },
        "m55r6" => CellSpec {
            name: "m55r6",
            family: "morse",
            n: 55,
            budget: 1_000_000,
            reference: -250.286609,
            kind: CellKind::Morse { rho: 6.0 },
        },
        "m55r14" => CellSpec {
            name: "m55r14",
            family: "morse",
            n: 55,
            budget: 1_000_000,
            reference: -220.646208,
            kind: CellKind::Morse { rho: 14.0 },
        },
        "m80r6" => CellSpec {
            name: "m80r6",
            family: "morse",
            n: 80,
            budget: 4_000_000,
            reference: -378.333471,
            kind: CellKind::Morse { rho: 6.0 },
        },
        "m80r14" => CellSpec {
            name: "m80r14",
            family: "morse",
            n: 80,
            budget: 4_000_000,
            reference: -340.811371,
            kind: CellKind::Morse { rho: 14.0 },
        },
        "tip4p8" => water_spec("tip4p8", 8, -305.5183221676),
        "tip4p9" => water_spec("tip4p9", 9, -344.4352092012),
        "tip4p10" => water_spec("tip4p10", 10, -391.0226842125),
        "tip4p11" => water_spec("tip4p11", 11, -431.4890723318),
        "tip4p12" => water_spec("tip4p12", 12, -492.9082357122),
        "tip4p13" => water_spec("tip4p13", 13, -532.9705363445),
        "tip4p14" => water_spec("tip4p14", 14, -582.9911036211),
        "tip4p15" => water_spec("tip4p15", 15, -628.3722525077),
        "tip4p16" => water_spec("tip4p16", 16, -681.1925945898),
        "tip4p17" => water_spec("tip4p17", 17, -723.8077531390),
        "tip4p18" => water_spec("tip4p18", 18, -773.2318693143),
        "tip4p19" => water_spec("tip4p19", 19, -821.0369191145),
        "tip4p20" => water_spec("tip4p20", 20, -872.9887544952),
        "tip4p21" => water_spec("tip4p21", 21, -916.7074302940),
        _ => return None,
    };
    Some(spec)
}

const fn water_spec(name: &'static str, n: usize, reference: f64) -> CellSpec {
    CellSpec {
        name,
        family: "water",
        n,
        budget: 2_000_000,
        reference,
        kind: CellKind::Water,
    }
}

struct StatusTracker {
    buf: Vec<u8>,
    peak_rss_anon_kb: u64,
    peak_vm_hwm_kb: u64,
    max_threads: u64,
    cpus_allowed: String,
    samples: u64,
    status_ns: u64,
}

impl StatusTracker {
    fn new() -> Self {
        Self {
            buf: Vec::with_capacity(4096),
            peak_rss_anon_kb: 0,
            peak_vm_hwm_kb: 0,
            max_threads: 0,
            cpus_allowed: String::with_capacity(32),
            samples: 0,
            status_ns: 0,
        }
    }

    fn sample(&mut self, record_ns: bool) {
        let mark = Instant::now();
        self.buf.clear();
        if let Ok(mut file) = File::open("/proc/self/status")
            && file.read_to_end(&mut self.buf).is_ok()
        {
            for line in self.buf.split(|&byte| byte == b'\n') {
                if let Some(rest) = line.strip_prefix(b"RssAnon:") {
                    if let Some(kb) = parse_leading_u64(rest) {
                        self.peak_rss_anon_kb = self.peak_rss_anon_kb.max(kb);
                    }
                } else if let Some(rest) = line.strip_prefix(b"VmHWM:") {
                    if let Some(kb) = parse_leading_u64(rest) {
                        self.peak_vm_hwm_kb = self.peak_vm_hwm_kb.max(kb);
                    }
                } else if let Some(rest) = line.strip_prefix(b"Threads:") {
                    if let Some(t) = parse_leading_u64(rest) {
                        self.max_threads = self.max_threads.max(t);
                    }
                } else if self.cpus_allowed.is_empty()
                    && let Some(rest) = line.strip_prefix(b"Cpus_allowed_list:")
                {
                    let trimmed = trim_ascii(rest);
                    for &byte in trimmed {
                        self.cpus_allowed.push(byte as char);
                    }
                }
            }
            self.samples += 1;
        }
        if record_ns {
            self.status_ns += mark.elapsed().as_nanos() as u64;
        }
    }
}

fn trim_ascii(mut slice: &[u8]) -> &[u8] {
    while let Some((&first, rest)) = slice.split_first() {
        if first.is_ascii_whitespace() {
            slice = rest;
        } else {
            break;
        }
    }
    while let Some((&last, rest)) = slice.split_last() {
        if last.is_ascii_whitespace() {
            slice = rest;
        } else {
            break;
        }
    }
    slice
}

fn parse_leading_u64(slice: &[u8]) -> Option<u64> {
    let trimmed = trim_ascii(slice);
    let mut val = 0u64;
    let mut any = false;
    for &byte in trimmed {
        if byte.is_ascii_digit() {
            val = val.saturating_mul(10).saturating_add((byte - b'0') as u64);
            any = true;
        } else {
            break;
        }
    }
    any.then_some(val)
}

#[cfg(feature = "cap-twist")]
fn apply_cap_twist(cfg: &mut Config) -> usize {
    if std::env::var("H2H_CAP_TWIST").is_ok_and(|value| value == "1") {
        cfg.cap_twist = true;
    }
    usize::from(
        cfg.resolved_json()
            .expect("resolved config json")
            .contains("\"cap_twist\":true"),
    )
}

#[cfg(not(feature = "cap-twist"))]
fn apply_cap_twist(_cfg: &mut Config) -> usize {
    assert!(
        !std::env::var("H2H_CAP_TWIST").is_ok_and(|value| value == "1"),
        "reference build does not include Config::cap_twist"
    );
    0
}

#[cfg(feature = "cap-twist")]
fn outcome_cap_escapes(outcome: &Outcome) -> usize {
    outcome.cap_escapes
}

#[cfg(not(feature = "cap-twist"))]
fn outcome_cap_escapes(_outcome: &Outcome) -> usize {
    0
}

struct RunMetrics {
    hit: bool,
    evals_to_hit: usize,
    charged: usize,
    energy_calls: u64,
    grad_calls: u64,
    best: f64,
    hops: usize,
    basins: usize,
    wall_s: f64,
    pot_s: f64,
    status_s: f64,
    outside_s: f64,
    rss_anon_kb: u64,
    vm_hwm_kb: u64,
    status_samples: u64,
    threads_max: u64,
    cpus_allowed: String,
    cap_twist_flag: usize,
    cap_escapes: usize,
    cap_arm_draws: usize,
    cap_arm_accepts: usize,
    arms: Vec<(String, usize, usize, f64)>,
}

fn cap_arm_counts(outcome: &Outcome) -> (usize, usize) {
    for (name, draws, accepts, _) in &outcome.arms {
        if name == "cap" {
            return (*draws, *accepts);
        }
    }
    (0, 0)
}

fn atomic_evals_to_hit(outcome: &Outcome, reference: f64, budget: usize) -> (bool, usize) {
    let hit = anneal_core::energy_hits(outcome.best, reference, HIT_TOL);
    if !hit {
        return (false, budget);
    }
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| anneal_core::energy_hits(*energy, reference, HIT_TOL))
        .map(|(_, spent, _, _)| *spent);
    let evals = match at_hit {
        Some(spent) if spent < budget => spent,
        _ => outcome.charged.min(budget),
    };
    (true, evals)
}

fn run_warm_lbfgs_cell<F>(
    cfg: &Config,
    start: &Array1<f64>,
    seed: u64,
    budget: usize,
    reference: f64,
    cap_twist_flag: usize,
    eval_vg: F,
) -> RunMetrics
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Sync,
{
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(reference + HIT_TOL);
    let mut optimiser = WarmLbfgs::default();
    let pot_ns = AtomicU64::new(0);
    let energy_calls = AtomicU64::new(0);
    let grad_calls = AtomicU64::new(0);
    let tracker = Mutex::new(StatusTracker::new());
    tracker.lock().expect("status lock").sample(false);

    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            if !led.charge() {
                return None;
            }
            let mark = Instant::now();
            let pair = eval_vg(v);
            pot_ns.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
            let call_idx = energy_calls.fetch_add(1, Ordering::Relaxed) + 1;
            if (call_idx & STATUS_STRIDE_MASK) == 0 {
                tracker.lock().expect("status lock").sample(true);
            }
            Some(pair)
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        let mark = Instant::now();
        let gradient = eval_vg(v).1;
        pot_ns.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let g_idx = grad_calls.fetch_add(1, Ordering::Relaxed) + 1;
        if (g_idx & STATUS_STRIDE_MASK) == 0 {
            tracker.lock().expect("status lock").sample(true);
        }
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
    let wall_s = started.elapsed().as_secs_f64();
    tracker.lock().expect("status lock").sample(false);
    std::hint::black_box(&outcome);
    std::hint::black_box(&ledger);

    let st = tracker.into_inner().expect("status tracker");
    let pot_s = pot_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
    let status_s = st.status_ns as f64 / 1.0e9;
    let outside_s = (wall_s - pot_s - status_s).max(0.0);
    let (hit, evals_to_hit) = atomic_evals_to_hit(&outcome, reference, budget);
    let (cap_arm_draws, cap_arm_accepts) = cap_arm_counts(&outcome);

    RunMetrics {
        hit,
        evals_to_hit,
        charged: outcome.charged,
        energy_calls: energy_calls.load(Ordering::Relaxed),
        grad_calls: grad_calls.load(Ordering::Relaxed),
        best: outcome.best,
        hops: outcome.hops,
        basins: outcome.basins,
        wall_s,
        pot_s,
        status_s,
        outside_s,
        rss_anon_kb: st.peak_rss_anon_kb,
        vm_hwm_kb: st.peak_vm_hwm_kb,
        status_samples: st.samples,
        threads_max: st.max_threads,
        cpus_allowed: st.cpus_allowed,
        cap_twist_flag,
        cap_escapes: outcome_cap_escapes(&outcome),
        cap_arm_draws,
        cap_arm_accepts,
        arms: outcome.arms.clone(),
    }
}

struct BljObjective {
    inner: BinaryLennardJones,
    pot_ns: AtomicU64,
    energy_calls: AtomicU64,
    grad_calls: AtomicU64,
    tracker: Mutex<StatusTracker>,
}

impl Objective<f64> for BljObjective {
    fn dim(&self) -> usize {
        Objective::dim(&self.inner)
    }

    fn bounds(&self) -> &Bounds<f64> {
        self.inner.bounds()
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let mark = Instant::now();
        let e = self.inner.eval(x);
        self.pot_ns
            .fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let idx = self.energy_calls.fetch_add(1, Ordering::Relaxed) + 1;
        if (idx & STATUS_STRIDE_MASK) == 0 {
            self.tracker.lock().expect("status lock").sample(true);
        }
        e
    }
}

impl Gradient<f64> for BljObjective {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let mark = Instant::now();
        let g = self.inner.grad(x);
        self.pot_ns
            .fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let idx = self.grad_calls.fetch_add(1, Ordering::Relaxed) + 1;
        if (idx & STATUS_STRIDE_MASK) == 0 {
            self.tracker.lock().expect("status lock").sample(true);
        }
        g
    }

    fn dim(&self) -> usize {
        Objective::dim(&self.inner)
    }
}

impl DifferentiableObjective<f64> for BljObjective {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let mark = Instant::now();
        let pair = self.inner.value_and_gradient(x);
        self.pot_ns
            .fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let idx = self.grad_calls.fetch_add(1, Ordering::Relaxed) + 1;
        if (idx & STATUS_STRIDE_MASK) == 0 {
            self.tracker.lock().expect("status lock").sample(true);
        }
        pair
    }
}

struct WaterSeen {
    calls: usize,
    lowest: f64,
}

struct WaterObjective {
    inner: GminTip4p,
    budget: usize,
    stop_at: f64,
    stop: Arc<AtomicBool>,
    seen: Mutex<WaterSeen>,
    pot_ns: AtomicU64,
    energy_calls: AtomicU64,
    grad_calls: AtomicU64,
    tracker: Mutex<StatusTracker>,
}

impl WaterObjective {
    fn counted(&self, x: ArrayView1<f64>, is_grad: bool) -> (f64, Array1<f64>) {
        let mark = Instant::now();
        let pair = self.inner.value_and_gradient(x);
        self.pot_ns
            .fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let total_idx = if is_grad {
            self.grad_calls.fetch_add(1, Ordering::Relaxed) + 1
        } else {
            self.energy_calls.fetch_add(1, Ordering::Relaxed) + 1
        };
        let mut seen = self.seen.lock().expect("water seen");
        seen.calls += 1;
        if seen.calls <= self.budget && pair.0 < seen.lowest {
            seen.lowest = pair.0;
        }
        if seen.calls >= self.budget || anneal_core::energy_hits(seen.lowest, self.stop_at, 0.0) {
            self.stop.store(true, Ordering::Relaxed);
        }
        let calls_now = seen.calls as u64;
        drop(seen);
        if (calls_now & STATUS_STRIDE_MASK) == 0 || (total_idx & STATUS_STRIDE_MASK) == 0 {
            self.tracker.lock().expect("status lock").sample(true);
        }
        pair
    }
}

impl Objective<f64> for WaterObjective {
    fn dim(&self) -> usize {
        Objective::dim(&self.inner)
    }

    fn bounds(&self) -> &Bounds<f64> {
        self.inner.bounds()
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.counted(x, false).0
    }
}

impl Gradient<f64> for WaterObjective {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.counted(x, true).1
    }

    fn dim(&self) -> usize {
        Objective::dim(&self.inner)
    }
}

impl DifferentiableObjective<f64> for WaterObjective {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.counted(x, true)
    }
}

fn run_cell(spec: &CellSpec, seed: u64, budget: usize) -> RunMetrics {
    match spec.kind {
        CellKind::Lj => {
            let mut cfg = Config::recommended(spec.n);
            let cap_twist_flag = apply_cap_twist(&mut cfg);
            let mut start_rng = StdRng::seed_from_u64(seed);
            let start = random_cluster_in_radius(
                spec.n,
                cfg.start_radius(),
                cfg.min_separation,
                &mut start_rng,
            );
            let potential = PairPotential::lennard_jones(spec.n);
            run_warm_lbfgs_cell(
                &cfg,
                &start,
                seed,
                budget,
                spec.reference,
                cap_twist_flag,
                |v| potential.value_and_gradient(v),
            )
        }
        CellKind::Morse { rho } => {
            let mut cfg = Config::recommended(spec.n);
            let cap_twist_flag = apply_cap_twist(&mut cfg);
            let mut start_rng = StdRng::seed_from_u64(seed);
            let start = random_cluster_in_radius(
                spec.n,
                cfg.start_radius(),
                cfg.min_separation,
                &mut start_rng,
            );
            let potential = PairPotential::morse(spec.n, rho);
            run_warm_lbfgs_cell(
                &cfg,
                &start,
                seed,
                budget,
                spec.reference,
                cap_twist_flag,
                |v| potential.value_and_gradient(v),
            )
        }
        CellKind::Au => {
            let species = Gupta::copper_gold_species(0, spec.n);
            let mut cfg = Config::recommended(spec.n).with_species(species, None);
            cfg.basin_stride = 8;
            let cap_twist_flag = apply_cap_twist(&mut cfg);
            let mut start_rng = StdRng::seed_from_u64(seed);
            let start = random_cluster_in_radius(
                spec.n,
                cfg.start_radius(),
                cfg.min_separation,
                &mut start_rng,
            );
            let potential = Gupta::copper_gold(0, spec.n);
            run_warm_lbfgs_cell(
                &cfg,
                &start,
                seed,
                budget,
                spec.reference,
                cap_twist_flag,
                |v| potential.value_and_gradient(v),
            )
        }
        CellKind::Blj { n_a, sigma_bb } => {
            let n_b = spec.n - n_a;
            let mut species = vec![1u32; n_a];
            species.extend(std::iter::repeat_n(2u32, n_b));
            let mut sigma = vec![1.0f64; n_a];
            sigma.extend(std::iter::repeat_n(sigma_bb, n_b));
            let mut cfg = Config::recommended(spec.n).with_species(species, Some(sigma));
            let cap_twist_flag = apply_cap_twist(&mut cfg);
            let mut start_rng = StdRng::seed_from_u64(seed);
            let start = random_cluster_in_radius(
                spec.n,
                cfg.start_radius(),
                cfg.min_separation,
                &mut start_rng,
            );
            let objective = BljObjective {
                inner: BinaryLennardJones::new(
                    n_a,
                    n_b,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.5 * (1.0 + sigma_bb),
                    sigma_bb,
                ),
                pot_ns: AtomicU64::new(0),
                energy_calls: AtomicU64::new(0),
                grad_calls: AtomicU64::new(0),
                tracker: Mutex::new(StatusTracker::new()),
            };
            objective
                .tracker
                .lock()
                .expect("status lock")
                .sample(false);
            let mut ledger = Ledger::new(budget);
            ledger.stop_below(spec.reference + HIT_TOL);
            let started = Instant::now();
            let (outcome, _) =
                search_from(&objective, &cfg, &mut ledger, start.view(), seed).expect("finite blj");
            let wall_s = started.elapsed().as_secs_f64();
            objective
                .tracker
                .lock()
                .expect("status lock")
                .sample(false);
            std::hint::black_box(&outcome);
            std::hint::black_box(&ledger);

            let best = ledger.best.min(outcome.best);
            let charged = ledger.spent();
            let hit = anneal_core::energy_hits(best, spec.reference, HIT_TOL);
            let at_hit = outcome
                .improvements
                .iter()
                .find(|(_, _, _, energy)| {
                    anneal_core::energy_hits(*energy, spec.reference, HIT_TOL)
                })
                .map(|(_, spent, _, _)| *spent);
            let evals_to_hit = if hit {
                match at_hit {
                    Some(spent) if spent < budget => spent,
                    _ => charged.min(budget),
                }
            } else {
                budget
            };
            let (cap_arm_draws, cap_arm_accepts) = cap_arm_counts(&outcome);
            let pot_s = objective.pot_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
            let energy_calls = objective.energy_calls.load(Ordering::Relaxed);
            let grad_calls = objective.grad_calls.load(Ordering::Relaxed);
            let st = objective.tracker.into_inner().expect("status tracker");
            let status_s = st.status_ns as f64 / 1.0e9;
            let outside_s = (wall_s - pot_s - status_s).max(0.0);

            RunMetrics {
                hit,
                evals_to_hit,
                charged,
                energy_calls,
                grad_calls,
                best,
                hops: outcome.hops,
                basins: outcome.basins,
                wall_s,
                pot_s,
                status_s,
                outside_s,
                rss_anon_kb: st.peak_rss_anon_kb,
                vm_hwm_kb: st.peak_vm_hwm_kb,
                status_samples: st.samples,
                threads_max: st.max_threads,
                cpus_allowed: st.cpus_allowed,
                cap_twist_flag,
                cap_escapes: outcome_cap_escapes(&outcome),
                cap_arm_draws,
                cap_arm_accepts,
                arms: outcome.arms.clone(),
            }
        }
        CellKind::Water => {
            let mut cfg = Config::recommended_tip4p(spec.n);
            cfg.rigid_body_geometry = Some(gmin_water_geometry());
            let cap_twist_flag = apply_cap_twist(&mut cfg);
            let mut start_rng =
                StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(3));
            let start = random_tip4p_cluster(spec.n, &mut start_rng);
            let stop = Arc::new(AtomicBool::new(false));
            let stop_at = spec.reference + HIT_TOL;
            let objective = WaterObjective {
                inner: GminTip4p::new(spec.n),
                budget,
                stop_at,
                stop: Arc::clone(&stop),
                seen: Mutex::new(WaterSeen {
                    calls: 0,
                    lowest: f64::INFINITY,
                }),
                pot_ns: AtomicU64::new(0),
                energy_calls: AtomicU64::new(0),
                grad_calls: AtomicU64::new(0),
                tracker: Mutex::new(StatusTracker::new()),
            };
            objective
                .tracker
                .lock()
                .expect("status lock")
                .sample(false);
            let mut ledger = Ledger::with_stop(budget, stop);
            let started = Instant::now();
            let (outcome, _) = search_from(&objective, &cfg, &mut ledger, start.view(), seed)
                .expect("finite tip4p");
            let wall_s = started.elapsed().as_secs_f64();
            objective
                .tracker
                .lock()
                .expect("status lock")
                .sample(false);
            std::hint::black_box(&outcome);
            std::hint::black_box(&ledger);

            let seen = objective.seen.into_inner().expect("water seen");
            let best = seen.lowest.min(outcome.best);
            let hit = anneal_core::energy_hits(best, stop_at, 0.0);
            let evals_to_hit = if hit {
                seen.calls.min(budget)
            } else {
                budget
            };
            let charged = ledger.spent();
            let (cap_arm_draws, cap_arm_accepts) = cap_arm_counts(&outcome);
            let pot_s = objective.pot_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
            let energy_calls = objective.energy_calls.load(Ordering::Relaxed);
            let grad_calls = objective.grad_calls.load(Ordering::Relaxed);
            let st = objective.tracker.into_inner().expect("status tracker");
            let status_s = st.status_ns as f64 / 1.0e9;
            let outside_s = (wall_s - pot_s - status_s).max(0.0);

            RunMetrics {
                hit,
                evals_to_hit,
                charged,
                energy_calls,
                grad_calls,
                best,
                hops: outcome.hops,
                basins: outcome.basins,
                wall_s,
                pot_s,
                status_s,
                outside_s,
                rss_anon_kb: st.peak_rss_anon_kb,
                vm_hwm_kb: st.peak_vm_hwm_kb,
                status_samples: st.samples,
                threads_max: st.max_threads,
                cpus_allowed: st.cpus_allowed,
                cap_twist_flag,
                cap_escapes: outcome_cap_escapes(&outcome),
                cap_arm_draws,
                cap_arm_accepts,
                arms: outcome.arms.clone(),
            }
        }
    }
}

fn main() {
    std::hint::black_box(padding::touch_padding());
    anneal_core::install_rgpot_kernels!();
    assert!(
        anneal_core::potentials::cluster_kernels().is_some(),
        "rgpot cluster kernels failed to install"
    );

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: cap_turn_driver <cell> <seed>");
        std::process::exit(2);
    }
    let Some(spec) = cell_spec(&args[0]) else {
        eprintln!("unknown cell {}", args[0]);
        std::process::exit(2);
    };
    let Ok(seed) = args[1].parse::<u64>() else {
        eprintln!("invalid seed {}", args[1]);
        std::process::exit(2);
    };
    if let Err(err) = Config::check_removed_env() {
        eprintln!("{err}");
        std::process::exit(2);
    }

    let budget = if std::env::var("GATE_ALLOW_BUDGET_OVERRIDE").is_ok_and(|v| v == "1") {
        std::env::var("H2H_BUDGET")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&b| b > 0)
            .unwrap_or(spec.budget)
    } else {
        spec.budget
    };

    let m = run_cell(&spec, seed, budget);
    let cpus = if m.cpus_allowed.is_empty() {
        "unknown"
    } else {
        m.cpus_allowed.as_str()
    };
    println!(
        "gate_row cell={} family={} n={} seed={} budget={} reference={:.10} hit={} evals_to_hit={} charged={} energy_calls={} grad_calls={} best={:.12} best_bits=0x{:016x} hops={} basins={} wall_s={:.6} pot_s={:.6} status_s={:.6} outside_s={:.6} rss_anon_kb={} vm_hwm_kb={} status_samples={} threads_max={} cpus_allowed={} pad_kb={} cap_twist_flag={} cap_escapes={} cap_arm_draws={} cap_arm_accepts={}",
        spec.name,
        spec.family,
        spec.n,
        seed,
        budget,
        spec.reference,
        u8::from(m.hit),
        m.evals_to_hit,
        m.charged,
        m.energy_calls,
        m.grad_calls,
        m.best,
        m.best.to_bits(),
        m.hops,
        m.basins,
        m.wall_s,
        m.pot_s,
        m.status_s,
        m.outside_s,
        m.rss_anon_kb,
        m.vm_hwm_kb,
        m.status_samples,
        m.threads_max,
        cpus,
        padding::GATE_PAD_KB_BUILT,
        m.cap_twist_flag,
        m.cap_escapes,
        m.cap_arm_draws,
        m.cap_arm_accepts,
    );
    for (name, draws, accepts, best) in &m.arms {
        println!("arm {name} draws {draws} accepts {accepts} best {best:.8}");
    }
}
