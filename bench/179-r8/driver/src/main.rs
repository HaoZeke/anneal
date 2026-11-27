// Copyright (c) anneal developers
//! Unified release gate driver for anneal #179 r8.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{BinaryLennardJones, PairPotential, SuttonChen, SuttonChenFamily};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

const HIT_TOL: f64 = anneal_core::DEFAULT_HIT_TOL;
const BLJ_LABEL_A: u32 = 200;
const BLJ_LABEL_B: u32 = 201;

#[cfg(feature = "placement-pad")]
#[unsafe(no_mangle)]
pub static PLACEMENT_PAD: [u8; 65536] = [0xA5; 65536];

#[cfg(feature = "placement-pad")]
#[inline(never)]
#[unsafe(no_mangle)]
pub extern "C" fn r8_placement_pad_stub(x: usize) -> u64 {
    let b = unsafe { std::ptr::read_volatile(&PLACEMENT_PAD[x % PLACEMENT_PAD.len()]) };
    (x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (b as u64)
}

#[derive(Clone, Copy)]
enum CellKind {
    Lj,
    Morse { rho: f64 },
    SuttonChen108,
    BinaryLj { n_a: usize, sigma_bb: f64 },
}

struct CellSpec {
    name: &'static str,
    n: usize,
    budget: usize,
    reference: f64,
    kind: CellKind,
}

fn cell_spec(name: &str) -> Option<CellSpec> {
    match name {
        "lj38" => Some(CellSpec {
            name: "lj38",
            n: 38,
            budget: 400_000,
            reference: -173.928427,
            kind: CellKind::Lj,
        }),
        "lj75" => Some(CellSpec {
            name: "lj75",
            n: 75,
            budget: 4_000_000,
            reference: -397.492331,
            kind: CellKind::Lj,
        }),
        "lj98" => Some(CellSpec {
            name: "lj98",
            n: 98,
            budget: 4_000_000,
            reference: -543.665361,
            kind: CellKind::Lj,
        }),
        "lj102" => Some(CellSpec {
            name: "lj102",
            n: 102,
            budget: 4_000_000,
            reference: -569.363652,
            kind: CellKind::Lj,
        }),
        "lj104" => Some(CellSpec {
            name: "lj104",
            n: 104,
            budget: 4_000_000,
            reference: -582.086642,
            kind: CellKind::Lj,
        }),
        "m80r14" => Some(CellSpec {
            name: "m80r14",
            n: 80,
            budget: 4_000_000,
            reference: -340.811371,
            kind: CellKind::Morse { rho: 14.0 },
        }),
        "sc108-80" => Some(CellSpec {
            name: "sc108-80",
            n: 80,
            budget: 1_000_000,
            reference: -21819.9952,
            kind: CellKind::SuttonChen108,
        }),
        "blj130n38" => Some(CellSpec {
            name: "blj130n38",
            n: 38,
            budget: 400_000,
            reference: -191.1840192602,
            kind: CellKind::BinaryLj {
                n_a: 13,
                sigma_bb: 1.30,
            },
        }),
        _ => None,
    }
}

fn read_rss_anon_kb() -> u64 {
    let Ok(text) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("RssAnon:") {
            if let Some(tok) = rest.split_whitespace().next() {
                if let Ok(kb) = tok.parse::<u64>() {
                    return kb;
                }
            }
        }
    }
    0
}

fn update_peak_rss(peak_kb: &AtomicU64) {
    let val = read_rss_anon_kb();
    let mut cur = peak_kb.load(Ordering::Relaxed);
    while val > cur {
        match peak_kb.compare_exchange_weak(cur, val, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => cur = actual,
        }
    }
}

enum Potential {
    Pair(PairPotential),
    Sc(SuttonChen),
    Blj(BinaryLennardJones),
}

impl Potential {
    fn from_spec(spec: &CellSpec) -> Self {
        match spec.kind {
            CellKind::Lj => Self::Pair(PairPotential::lennard_jones(spec.n)),
            CellKind::Morse { rho } => Self::Pair(PairPotential::morse(spec.n, rho)),
            CellKind::SuttonChen108 => Self::Sc(SuttonChen::new(spec.n, SuttonChenFamily::Sc10_8)),
            CellKind::BinaryLj { n_a, sigma_bb } => Self::Blj(BinaryLennardJones::new(
                n_a,
                spec.n - n_a,
                1.0,
                1.0,
                1.0,
                1.0,
                0.5 * (1.0 + sigma_bb),
                sigma_bb,
            )),
        }
    }

    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        match self {
            Self::Pair(p) => p.value_and_gradient(x),
            Self::Sc(p) => p.value_and_gradient(x),
            Self::Blj(p) => p.value_and_gradient(x),
        }
    }
}

fn build_config(spec: &CellSpec) -> Config {
    match spec.kind {
        CellKind::BinaryLj { n_a, .. } => {
            let mut labels = vec![BLJ_LABEL_A; spec.n];
            for label in labels.iter_mut().skip(n_a) {
                *label = BLJ_LABEL_B;
            }
            Config::recommended(spec.n).with_species(labels, None)
        }
        _ => Config::recommended(spec.n),
    }
}

fn cmd_run(args: &[String]) -> i32 {
    if args.len() != 2 {
        eprintln!("usage: driver run <cell> <seed>");
        return 2;
    }
    let cell = args[0].as_str();
    let seed: u64 = match args[1].parse() {
        Ok(v) => v,
        Err(_) => {
            eprintln!("invalid seed {}", args[1]);
            return 2;
        }
    };
    let Some(spec) = cell_spec(cell) else {
        eprintln!("unknown cell {cell}");
        return 2;
    };
    let cfg = build_config(&spec);
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(spec.n, cfg.start_radius(), cfg.min_separation, &mut rng);
    let potential = Potential::from_spec(&spec);
    let mut ledger = Ledger::new(spec.budget);
    ledger.stop_below(spec.reference + HIT_TOL);
    let mut optimiser = WarmLbfgs::default();
    let pot_ns = AtomicU64::new(0);
    let peak_rss_kb = AtomicU64::new(0);
    let calls_seen = AtomicU64::new(0);
    update_peak_rss(&peak_rss_kb);

    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            if !led.charge() {
                return None;
            }
            let c = calls_seen.fetch_add(1, Ordering::Relaxed);
            if (c & 0x3fff) == 0 {
                update_peak_rss(&peak_rss_kb);
            }
            let mark = Instant::now();
            let pair = potential.value_and_gradient(v);
            pot_ns.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
            Some(pair)
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        let c = calls_seen.fetch_add(1, Ordering::Relaxed);
        if (c & 0x3fff) == 0 {
            update_peak_rss(&peak_rss_kb);
        }
        let mark = Instant::now();
        let gradient = potential.value_and_gradient(v).1;
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
    update_peak_rss(&peak_rss_kb);

    let pot_s = pot_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
    let outside_s = (wall - pot_s).max(0.0);
    let rss_anon_kb = peak_rss_kb.load(Ordering::Relaxed);

    let hit = anneal_core::energy_hits(outcome.best, spec.reference, HIT_TOL);
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| anneal_core::energy_hits(*energy, spec.reference, HIT_TOL))
        .map(|(_, spent, _, _)| *spent);
    let evals = if hit {
        match at_hit {
            Some(spent) if spent < spec.budget => spent,
            _ => outcome.charged.min(spec.budget),
        }
    } else {
        spec.budget
    };

    println!(
        "gate_row cell={} seed={} budget={} hit={} evals_to_hit={} charged={} wall_s={:.6} best={:.10} pot_s={:.6} outside_s={:.6} rss_anon_kb={} symmetrised={} sym_gain={:.6}",
        spec.name,
        seed,
        spec.budget,
        u8::from(hit),
        evals,
        outcome.charged,
        wall,
        outcome.best,
        pot_s,
        outside_s,
        rss_anon_kb,
        outcome.symmetrised.0,
        outcome.symmetrised.1,
    );
    for count in &outcome.proposal_counts {
        eprintln!(
            "move {} proposed {} accepted {} improved {}",
            count.move_type, count.proposed, count.accepted, count.improved
        );
    }
    0
}

fn cmd_budget(args: &[String]) -> i32 {
    let Some(cell) = args.first().map(String::as_str) else {
        eprintln!("usage: driver budget <cell>");
        return 2;
    };
    let Some(spec) = cell_spec(cell) else {
        eprintln!("unknown cell {cell}");
        return 2;
    };
    println!("{} {:.10} {}", spec.budget, spec.reference, spec.n);
    0
}

fn cmd_probe() -> i32 {
    let n = 38usize;
    let cfg = Config::recommended(n);
    let mut rng = StdRng::seed_from_u64(2000);
    let start = random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng);
    let pot = PairPotential::lennard_jones(n);
    let mut opt = WarmLbfgs::default();
    let (_, relaxed, _) = opt.minimize(start.view(), 80, |v| Some(pot.value_and_gradient(v)));
    let classes = vec![0u32; n];
    let proj = anneal_core::csm::propose(relaxed.view(), &classes);
    let group_label = proj.as_ref().map(|p| p.group).unwrap_or("none");
    let measure = proj.as_ref().map(|p| p.measure).unwrap_or(f64::NAN);
    let sym_orders = vec![2usize, 3, 4, 5, 6];
    let sym_cands =
        anneal_core::symmetrise::detect_all(relaxed.view(), n, &sym_orders, cfg.symmetry_tolerance);
    // Slightly perturbed octahedron (6 atoms) to confirm B hits the detected CSM branch.
    let oct = Array1::from_vec(vec![
        1.01, 0.01, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        -1.0,
    ]);
    let oct_classes = vec![0u32; 6];
    let oct_proj = anneal_core::csm::propose(oct.view(), &oct_classes);
    let oct_group = oct_proj.as_ref().map(|p| p.group).unwrap_or("none");
    println!(
        "probe csm_group={} csm_measure={:.6} detect_low_orders={} oct_csm_group={}",
        group_label,
        measure,
        sym_cands.len(),
        oct_group,
    );
    0
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    #[cfg(feature = "placement-pad")]
    {
        let _ = std::hint::black_box(r8_placement_pad_stub(args.len()));
    }
    let cmd = args.first().map(String::as_str).unwrap_or("");
    let status = match cmd {
        "run" => cmd_run(&args[1..]),
        "budget" => cmd_budget(&args[1..]),
        "probe" => cmd_probe(),
        other => {
            eprintln!("usage: driver <run|budget|probe> ... (got {other:?})");
            2
        }
    };
    std::process::exit(status);
}
