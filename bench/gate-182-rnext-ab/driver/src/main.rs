use std::fs;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use anneal_core::DEFAULT_HIT_TOL;
use anneal_core::methods::cluster_hopping::{
    Config, Ledger, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::cluster_search;
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Short,
    Traj,
    Panel,
}

struct CellSpec {
    name: &'static str,
    mode: Mode,
    atoms: usize,
    budget: usize,
    deferral: usize,
    reference: f64,
}

const CELLS: &[CellSpec] = &[
    CellSpec {
        name: "lj98_short_d0",
        mode: Mode::Short,
        atoms: 98,
        budget: 2_500,
        deferral: 0,
        reference: -543.665361,
    },
    CellSpec {
        name: "lj98_short_d4",
        mode: Mode::Short,
        atoms: 98,
        budget: 2_500,
        deferral: 4,
        reference: -543.665361,
    },
    CellSpec {
        name: "lj98_traj_d0",
        mode: Mode::Traj,
        atoms: 98,
        budget: 5_000,
        deferral: 0,
        reference: -543.665361,
    },
    CellSpec {
        name: "lj98_traj_d4",
        mode: Mode::Traj,
        atoms: 98,
        budget: 5_000,
        deferral: 4,
        reference: -543.665361,
    },
    CellSpec {
        name: "lj38_short_d0",
        mode: Mode::Short,
        atoms: 38,
        budget: 2_500,
        deferral: 0,
        reference: -173.928427,
    },
    CellSpec {
        name: "lj38_short_d4",
        mode: Mode::Short,
        atoms: 38,
        budget: 2_500,
        deferral: 4,
        reference: -173.928427,
    },
    CellSpec {
        name: "lj38_d0",
        mode: Mode::Panel,
        atoms: 38,
        budget: 400_000,
        deferral: 0,
        reference: -173.928427,
    },
    CellSpec {
        name: "lj38_d4",
        mode: Mode::Panel,
        atoms: 38,
        budget: 400_000,
        deferral: 4,
        reference: -173.928427,
    },
    CellSpec {
        name: "lj55",
        mode: Mode::Panel,
        atoms: 55,
        budget: 1_000_000,
        deferral: 4,
        reference: -279.248470,
    },
    CellSpec {
        name: "lj75",
        mode: Mode::Panel,
        atoms: 75,
        budget: 4_000_000,
        deferral: 4,
        reference: -397.492331,
    },
    CellSpec {
        name: "lj98",
        mode: Mode::Panel,
        atoms: 98,
        budget: 4_000_000,
        deferral: 4,
        reference: -543.665361,
    },
    CellSpec {
        name: "lj102",
        mode: Mode::Panel,
        atoms: 102,
        budget: 4_000_000,
        deferral: 4,
        reference: -569.363652,
    },
    CellSpec {
        name: "lj103",
        mode: Mode::Panel,
        atoms: 103,
        budget: 4_000_000,
        deferral: 4,
        reference: -575.766131,
    },
    CellSpec {
        name: "lj104",
        mode: Mode::Panel,
        atoms: 104,
        budget: 4_000_000,
        deferral: 4,
        reference: -582.086642,
    },
    CellSpec {
        name: "lj150",
        mode: Mode::Panel,
        atoms: 150,
        budget: 4_000_000,
        deferral: 4,
        reference: -893.310258,
    },
];

fn read_rss_anon_kib() -> u64 {
    let Ok(text) = fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("RssAnon:") {
            if let Some(num) = rest.split_whitespace().next() {
                if let Ok(val) = num.parse::<u64>() {
                    return val;
                }
            }
        }
    }
    0
}

fn update_rss_peak(peak: &AtomicU64) {
    let current = read_rss_anon_kib();
    let mut prev = peak.load(Ordering::Relaxed);
    while current > prev {
        match peak.compare_exchange_weak(prev, current, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(observed) => prev = observed,
        }
    }
}

struct TimedPotential<'a> {
    inner: &'a PairPotential,
    pot_ns: &'a AtomicU64,
    calls: &'a AtomicUsize,
    rss_peak: &'a AtomicU64,
}

impl TimedPotential<'_> {
    #[inline]
    fn sample_rss_if_due(&self) {
        let count = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
        if count.is_multiple_of(4096) {
            update_rss_peak(self.rss_peak);
        }
    }
}

impl Objective<f64> for TimedPotential<'_> {
    fn dim(&self) -> usize {
        Objective::dim(self.inner)
    }

    fn bounds(&self) -> &Bounds<f64> {
        self.inner.bounds()
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let t0 = Instant::now();
        let val = self.inner.value_and_gradient(x).0;
        self.pot_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.sample_rss_if_due();
        val
    }
}

impl Gradient<f64> for TimedPotential<'_> {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let t0 = Instant::now();
        let g = self.inner.value_and_gradient(x).1;
        self.pot_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.sample_rss_if_due();
        g
    }

    fn dim(&self) -> usize {
        Gradient::dim(self.inner)
    }
}

impl DifferentiableObjective<f64> for TimedPotential<'_> {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let t0 = Instant::now();
        let vg = self.inner.value_and_gradient(x);
        self.pot_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.sample_rss_if_due();
        vg
    }
}

fn run_once_with_gradient(
    spec: &CellSpec,
    cfg: &Config,
    potential: &PairPotential,
    seed: u64,
    stop_at_hit: bool,
    pot_ns: &AtomicU64,
    calls: &AtomicUsize,
    rss_peak: &AtomicU64,
) -> (bool, usize, f64, usize) {
    let mut start_rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(
        spec.atoms,
        cfg.start_radius(),
        cfg.min_separation,
        &mut start_rng,
    );
    let mut ledger = Ledger::new(spec.budget);
    if stop_at_hit {
        ledger.stop_below(spec.reference + DEFAULT_HIT_TOL);
    }
    let mut optimiser = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            if !led.charge() {
                return None;
            }
            let mark = Instant::now();
            let evaluated = potential.value_and_gradient(v);
            pot_ns.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
            let n = calls.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_multiple_of(4096) {
                update_rss_peak(rss_peak);
            }
            Some(evaluated)
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        let mark = Instant::now();
        let gradient = potential.value_and_gradient(v).1;
        pot_ns.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let n = calls.fetch_add(1, Ordering::Relaxed) + 1;
        if n.is_multiple_of(4096) {
            update_rss_peak(rss_peak);
        }
        Some(gradient)
    };
    let mut walk_rng = StdRng::seed_from_u64(seed);
    let outcome = run_with_gradient(
        cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut walk_rng,
    )
    .expect("finite start");
    let hit = anneal_core::energy_hits(outcome.best, spec.reference, DEFAULT_HIT_TOL);
    let calls_to_hit = if hit {
        outcome
            .improvements
            .iter()
            .find(|(_, _, _, e)| anneal_core::energy_hits(*e, spec.reference, DEFAULT_HIT_TOL))
            .map(|(_, spent, _, _)| *spent)
            .unwrap_or(outcome.charged)
    } else {
        spec.budget + 1
    };
    (hit, calls_to_hit, outcome.best, outcome.hops)
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        eprintln!("usage: gate182-driver <cell> <seed> <inner_reps>");
        std::process::exit(2);
    }
    let cell_name = args[0].as_str();
    let seed: u64 = args[1].parse().expect("valid seed");
    let inner_reps: usize = args[2].parse().expect("valid inner_reps");
    assert!(inner_reps >= 1, "inner_reps must be >= 1");

    let spec = CELLS
        .iter()
        .find(|c| c.name == cell_name)
        .unwrap_or_else(|| panic!("unknown cell {cell_name}"));

    if spec.mode == Mode::Traj {
        let _ = rgmin::vecops::force_simd_path("scalar");
    }

    let mut cfg = Config::recommended(spec.atoms);
    if spec.deferral == 0 {
        cfg.packing_deferral = 0;
    } else {
        assert_eq!(
            cfg.packing_deferral, spec.deferral,
            "unexpected default packing_deferral"
        );
    }
    let potential = PairPotential::lennard_jones(spec.atoms);

    let pot_ns = AtomicU64::new(0);
    let eval_calls = AtomicUsize::new(0);
    let rss_peak = AtomicU64::new(0);
    update_rss_peak(&rss_peak);

    let mut last_result = (false, spec.budget + 1, f64::INFINITY, 0usize);
    let started = Instant::now();
    for _ in 0..inner_reps {
        last_result = match spec.mode {
            Mode::Short => {
                let timed = TimedPotential {
                    inner: &potential,
                    pot_ns: &pot_ns,
                    calls: &eval_calls,
                    rss_peak: &rss_peak,
                };
                let mut ledger = Ledger::new(spec.budget);
                let (outcome, _stats) =
                    cluster_search::search(&timed, &cfg, &mut ledger, seed).expect("finite");
                let hit = anneal_core::energy_hits(outcome.best, spec.reference, DEFAULT_HIT_TOL);
                let calls_to_hit = if hit {
                    outcome
                        .improvements
                        .iter()
                        .find(|(_, _, _, e)| {
                            anneal_core::energy_hits(*e, spec.reference, DEFAULT_HIT_TOL)
                        })
                        .map(|(_, spent, _, _)| *spent)
                        .unwrap_or(outcome.charged)
                } else {
                    spec.budget + 1
                };
                (hit, calls_to_hit, outcome.best, outcome.hops)
            }
            Mode::Traj => run_once_with_gradient(
                spec,
                &cfg,
                &potential,
                seed,
                false,
                &pot_ns,
                &eval_calls,
                &rss_peak,
            ),
            Mode::Panel => run_once_with_gradient(
                spec,
                &cfg,
                &potential,
                seed,
                true,
                &pot_ns,
                &eval_calls,
                &rss_peak,
            ),
        };
    }
    let wall_total = started.elapsed().as_secs_f64();
    update_rss_peak(&rss_peak);

    let pot_total = pot_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
    let outside_total = (wall_total - pot_total).max(0.0);
    let outside_s = outside_total / (inner_reps as f64);
    let (hit, calls_to_hit, best_energy, hops) = last_result;
    let rss_kib = rss_peak.load(Ordering::Relaxed);

    println!(
        "gate_row cell={} seed={} deferral={} hit={} calls_to_hit={} best_energy={} hops={} inner_reps={} wall_total_s={:.9} pot_total_s={:.9} outside_s={:.9} rss_anon_kib={}",
        spec.name,
        seed,
        spec.deferral,
        u8::from(hit),
        calls_to_hit,
        best_energy,
        hops,
        inner_reps,
        wall_total,
        pot_total,
        outside_s,
        rss_kib,
    );
}
