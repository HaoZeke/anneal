// Copyright (c) anneal developers
//! One default-path seed for the no-builders part-c gate.
//!
//! ```text
//! nb8_gate_c spec <system>
//! nb8_gate_c start <system> <seed> <path>
//! nb8_gate_c run <system> <seed> <path>
//! ```
//!
//! `spec` prints the paper budget and the reference. `start` writes the
//! seed's coordinates. `run` reads that file and does not draw another
//! start. The search is `cluster_search::search_from` on
//! `Config::recommended` (Sutton-Chen and binary Lennard-Jones) or
//! `Config::recommended_tip4p` with the GMIN rigid chart (TIP4P). The
//! budget is the paper budget compiled into this file. Nothing in the
//! environment changes it. A hit is a finite best at or below the
//! reference plus the shared one-sided tolerance.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anneal_core::methods::cluster_hopping::{Config, Ledger, random_cluster_in_radius};
use anneal_core::methods::cluster_search::search_from;
use anneal_core::potentials::{
    BinaryLennardJones, GminTip4p, SuttonChen, SuttonChenFamily, gmin_water_geometry,
    random_tip4p_cluster,
};
use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

const HIT_TOL: f64 = anneal_core::DEFAULT_HIT_TOL;
const TIP4P_BUDGET: usize = 2_000_000;

enum Family {
    SuttonChen(SuttonChenFamily),
    BinaryLennardJones { n_a: usize, sigma_bb: f64 },
    Tip4p,
}

struct Spec {
    name: &'static str,
    n: usize,
    budget: usize,
    reference: f64,
    family: Family,
}

fn specs() -> &'static [Spec] {
    &[
        Spec {
            name: "sc126-55",
            n: 55,
            budget: 1_000_000,
            reference: -52884.6806,
            family: Family::SuttonChen(SuttonChenFamily::Sc12_6),
        },
        Spec {
            name: "sc96-38",
            n: 38,
            budget: 400_000,
            reference: -8917.7056,
            family: Family::SuttonChen(SuttonChenFamily::Sc9_6),
        },
        Spec {
            name: "blj105n38",
            n: 38,
            budget: 400_000,
            reference: -177.2090846748,
            family: Family::BinaryLennardJones { n_a: 8, sigma_bb: 1.05 },
        },
        Spec {
            name: "blj105n55",
            n: 55,
            budget: 1_000_000,
            reference: -283.1545313810,
            family: Family::BinaryLennardJones { n_a: 13, sigma_bb: 1.05 },
        },
        Spec {
            name: "blj115n38",
            n: 38,
            budget: 400_000,
            reference: -187.3676867453,
            family: Family::BinaryLennardJones { n_a: 9, sigma_bb: 1.15 },
        },
        Spec {
            name: "blj115n55",
            n: 55,
            budget: 1_000_000,
            reference: -289.2931853058,
            family: Family::BinaryLennardJones { n_a: 16, sigma_bb: 1.15 },
        },
        Spec {
            name: "blj130n38",
            n: 38,
            budget: 400_000,
            reference: -191.1840192602,
            family: Family::BinaryLennardJones { n_a: 13, sigma_bb: 1.30 },
        },
        Spec {
            name: "blj130n55",
            n: 55,
            budget: 1_000_000,
            reference: -306.7337643721,
            family: Family::BinaryLennardJones { n_a: 19, sigma_bb: 1.30 },
        },
        Spec {
            name: "blj130n98",
            n: 98,
            budget: 4_000_000,
            reference: -590.7874132760,
            family: Family::BinaryLennardJones { n_a: 41, sigma_bb: 1.30 },
        },
        Spec {
            name: "tip4p8",
            n: 8,
            budget: TIP4P_BUDGET,
            reference: -305.5183221676,
            family: Family::Tip4p,
        },
        Spec {
            name: "tip4p9",
            n: 9,
            budget: TIP4P_BUDGET,
            reference: -344.4352092012,
            family: Family::Tip4p,
        },
        Spec {
            name: "tip4p10",
            n: 10,
            budget: TIP4P_BUDGET,
            reference: -391.0226842125,
            family: Family::Tip4p,
        },
        Spec {
            name: "tip4p11",
            n: 11,
            budget: TIP4P_BUDGET,
            reference: -431.4890723318,
            family: Family::Tip4p,
        },
        Spec {
            name: "tip4p12",
            n: 12,
            budget: TIP4P_BUDGET,
            reference: -492.9082357122,
            family: Family::Tip4p,
        },
        Spec {
            name: "tip4p13",
            n: 13,
            budget: TIP4P_BUDGET,
            reference: -532.9705363445,
            family: Family::Tip4p,
        },
        Spec {
            name: "tip4p14",
            n: 14,
            budget: TIP4P_BUDGET,
            reference: -582.9911036211,
            family: Family::Tip4p,
        },
        Spec {
            name: "tip4p15",
            n: 15,
            budget: TIP4P_BUDGET,
            reference: -628.3722525077,
            family: Family::Tip4p,
        },
        Spec {
            name: "tip4p16",
            n: 16,
            budget: TIP4P_BUDGET,
            reference: -681.1925945898,
            family: Family::Tip4p,
        },
        Spec {
            name: "tip4p17",
            n: 17,
            budget: TIP4P_BUDGET,
            reference: -723.8077531390,
            family: Family::Tip4p,
        },
        Spec {
            name: "tip4p18",
            n: 18,
            budget: TIP4P_BUDGET,
            reference: -773.2318693143,
            family: Family::Tip4p,
        },
        Spec {
            name: "tip4p19",
            n: 19,
            budget: TIP4P_BUDGET,
            reference: -821.0369191145,
            family: Family::Tip4p,
        },
        Spec {
            name: "tip4p20",
            n: 20,
            budget: TIP4P_BUDGET,
            reference: -872.9887544952,
            family: Family::Tip4p,
        },
        Spec {
            name: "tip4p21",
            n: 21,
            budget: TIP4P_BUDGET,
            reference: -916.7074302940,
            family: Family::Tip4p,
        },
    ]
}

fn spec_of(name: &str) -> &'static Spec {
    specs()
        .iter()
        .find(|spec| spec.name == name)
        .unwrap_or_else(|| panic!("unknown system {name}"))
}

enum Pot {
    Sc(SuttonChen),
    Blj(BinaryLennardJones),
    Tip(GminTip4p),
}

struct Timed {
    pot: Pot,
    pot_ns: AtomicU64,
    entries: AtomicU64,
}

impl Timed {
    fn pair(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let mark = Instant::now();
        let pair = match &self.pot {
            Pot::Sc(pot) => pot.value_and_gradient(x),
            Pot::Blj(pot) => pot.value_and_gradient(x),
            Pot::Tip(pot) => pot.value_and_gradient(x),
        };
        self.pot_ns
            .fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.entries.fetch_add(1, Ordering::Relaxed);
        pair
    }
}

impl Objective<f64> for Timed {
    fn dim(&self) -> usize {
        match &self.pot {
            Pot::Sc(pot) => Objective::dim(pot),
            Pot::Blj(pot) => Objective::dim(pot),
            Pot::Tip(pot) => Objective::dim(pot),
        }
    }

    fn bounds(&self) -> &Bounds<f64> {
        match &self.pot {
            Pot::Sc(pot) => Objective::bounds(pot),
            Pot::Blj(pot) => Objective::bounds(pot),
            Pot::Tip(pot) => Objective::bounds(pot),
        }
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.pair(x).0
    }
}

impl Gradient<f64> for Timed {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.pair(x).1
    }

    fn dim(&self) -> usize {
        Objective::dim(self)
    }
}

impl DifferentiableObjective<f64> for Timed {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.pair(x)
    }
}

fn binary_potential(n: usize, n_a: usize, sigma_bb: f64) -> (Config, BinaryLennardJones) {
    let n_b = n - n_a;
    let mut species = vec![1u32; n_a];
    species.extend(std::iter::repeat_n(2u32, n_b));
    let mut sigma = vec![1.0; n_a];
    sigma.extend(std::iter::repeat_n(sigma_bb, n_b));
    let cfg = Config::recommended(n).with_species(species, Some(sigma));
    let pot = BinaryLennardJones::new(
        n_a,
        n_b,
        1.0,
        1.0,
        1.0,
        1.0,
        0.5 * (1.0 + sigma_bb),
        sigma_bb,
    );
    (cfg, pot)
}

fn prepare(spec: &Spec) -> (Config, Pot, bool) {
    match spec.family {
        Family::SuttonChen(family) => (
            Config::recommended(spec.n),
            Pot::Sc(SuttonChen::new(spec.n, family)),
            false,
        ),
        Family::BinaryLennardJones { n_a, sigma_bb } => {
            let (cfg, pot) = binary_potential(spec.n, n_a, sigma_bb);
            (cfg, Pot::Blj(pot), false)
        }
        Family::Tip4p => {
            let mut cfg = Config::recommended_tip4p(spec.n);
            cfg.rigid_body_geometry = Some(gmin_water_geometry());
            (cfg, Pot::Tip(GminTip4p::new(spec.n)), true)
        }
    }
}

fn draw_start(spec: &Spec, cfg: &Config, seed: u64, rigid: bool) -> Array1<f64> {
    if rigid {
        let mut rng =
            StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(3));
        random_tip4p_cluster(spec.n, &mut rng)
    } else {
        let mut rng = StdRng::seed_from_u64(seed);
        random_cluster_in_radius(spec.n, cfg.start_radius(), cfg.min_separation, &mut rng)
    }
}

fn write_bits(path: &str, coords: &Array1<f64>) {
    let mut text = String::from("f64bits\n");
    for value in coords.iter() {
        text.push_str(&format!("{}\n", value.to_bits()));
    }
    std::fs::write(path, text).unwrap_or_else(|err| panic!("{path}: {err}"));
}

fn read_bits(path: &str) -> Array1<f64> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|err| panic!("{path}: {err}"));
    let mut lines = text.lines();
    let tag = lines.next().unwrap_or("");
    assert_eq!(tag, "f64bits", "{path}: missing header");
    let values: Vec<f64> = lines
        .filter(|line| !line.is_empty())
        .map(|line| {
            let bits: u64 = line.parse().unwrap_or_else(|_| panic!("{path}: {line}"));
            f64::from_bits(bits)
        })
        .collect();
    Array1::from_vec(values)
}

fn calls_to_hit(hit: bool, budget: usize, improvements: &[(usize, usize, usize, f64)], reference: f64) -> usize {
    if !hit {
        return budget;
    }
    improvements
        .iter()
        .find(|(_, _, _, energy)| anneal_core::energy_hits(*energy, reference, HIT_TOL))
        .map(|(_, spent, _, _)| *spent)
        .unwrap_or(budget)
}

fn cmd_spec(name: &str) -> i32 {
    let spec = spec_of(name);
    println!(
        "nb8 command=spec system={} n={} budget={} reference_bits={}",
        spec.name,
        spec.n,
        spec.budget,
        spec.reference.to_bits()
    );
    0
}

fn cmd_start(name: &str, seed: u64, path: &str) -> i32 {
    let spec = spec_of(name);
    let (cfg, _, rigid) = prepare(spec);
    let start = draw_start(spec, &cfg, seed, rigid);
    let expect = if rigid { 6 * spec.n } else { 3 * spec.n };
    assert_eq!(start.len(), expect, "start length");
    write_bits(path, &start);
    println!(
        "nb8 command=start system={} seed={} n_numbers={}",
        spec.name,
        seed,
        start.len()
    );
    0
}

fn cmd_run(name: &str, seed: u64, path: &str) -> i32 {
    let spec = spec_of(name);
    let (cfg, pot, rigid) = prepare(spec);
    let start = read_bits(path);
    let expect = if rigid { 6 * spec.n } else { 3 * spec.n };
    assert_eq!(start.len(), expect, "start length for {}", spec.name);
    let timed = Timed {
        pot,
        pot_ns: AtomicU64::new(0),
        entries: AtomicU64::new(0),
    };
    let mut ledger = Ledger::new(spec.budget);
    ledger.stop_below(spec.reference + HIT_TOL);
    let started = Instant::now();
    let (outcome, _) = search_from(&timed, &cfg, &mut ledger, start.view(), seed).expect("finite start");
    let wall = started.elapsed().as_secs_f64();
    let pot_s = timed.pot_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
    let outside = (wall - pot_s).max(0.0);
    let hit = anneal_core::energy_hits(outcome.best, spec.reference, HIT_TOL);
    let calls = calls_to_hit(hit, spec.budget, &outcome.improvements, spec.reference);
    println!(
        "nb8 command=run system={} seed={} budget={} hit={} calls={} charged={} wall_s={:.9} pot_s={:.9} outside_s={:.9} best_bits={} entries={}",
        spec.name,
        seed,
        spec.budget,
        u8::from(hit),
        calls,
        outcome.charged,
        wall,
        pot_s,
        outside,
        outcome.best.to_bits(),
        timed.entries.load(Ordering::Relaxed),
    );
    0
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("");
    let status = match command {
        "spec" => {
            let name = args.get(1).map(String::as_str).expect("system");
            cmd_spec(name)
        }
        "start" => {
            let name = args.get(1).map(String::as_str).expect("system");
            let seed: u64 = args.get(2).expect("seed").parse().expect("seed");
            let path = args.get(3).expect("path");
            cmd_start(name, seed, path)
        }
        "run" => {
            let name = args.get(1).map(String::as_str).expect("system");
            let seed: u64 = args.get(2).expect("seed").parse().expect("seed");
            let path = args.get(3).expect("path");
            cmd_run(name, seed, path)
        }
        _ => {
            eprintln!("usage: nb8_gate_c spec|start|run ...");
            2
        }
    };
    std::process::exit(status);
}
