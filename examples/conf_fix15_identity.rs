//! Bit-identity and timing row for one default cluster-search run.
//!
//! Usage:
//! `conf_fix15_identity <system> <seed> <coordinate-bits>`
//!
//! The supported systems are `lj38`, `lj75`, `m80r6`, `au38`, and
//! `tip4p20`. Each uses the repository's paper budget and default
//! recommended configuration. The final state's raw `f64` bits are written
//! in coordinate order so two runs can be compared without text rounding.

use anneal_core::methods::cluster_hopping::{random_cluster_in_radius, Config, Ledger, Outcome};
use anneal_core::methods::cluster_search::search_from;
use anneal_core::potentials::{
    ccd_tip4p_minimum, gmin_water_geometry, random_tip4p_cluster, GminTip4p, Gupta, PairPotential,
};
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use eindir_core::Objective;
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy)]
enum Kind {
    LennardJones,
    Morse { rho: f64 },
    Gold,
    Tip4p,
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
        "lj75" => Some(Spec {
            name: "lj75",
            n: 75,
            budget: 4_000_000,
            reference: -397.492331,
            kind: Kind::LennardJones,
        }),
        "m80r6" => Some(Spec {
            name: "m80r6",
            n: 80,
            budget: 4_000_000,
            reference: -378.333471,
            kind: Kind::Morse { rho: 6.0 },
        }),
        "au38" => Some(Spec {
            name: "au38",
            n: 38,
            budget: 400_000,
            reference: -130.54001708,
            kind: Kind::Gold,
        }),
        "tip4p20" => Some(Spec {
            name: "tip4p20",
            n: 20,
            budget: 2_000_000,
            reference: ccd_tip4p_minimum(20)?,
            kind: Kind::Tip4p,
        }),
        _ => None,
    }
}

enum Potential {
    Pair(PairPotential),
    Gold(Gupta),
    Tip4p(GminTip4p),
}

struct Counted {
    potential: Potential,
    budget: usize,
    reference: f64,
    stop: Arc<AtomicBool>,
    calls: AtomicUsize,
    first_hit: AtomicUsize,
    pot_ns: AtomicU64,
}

impl Counted {
    fn counted(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let mark = Instant::now();
        let evaluated = match &self.potential {
            Potential::Pair(potential) => potential.value_and_gradient(x),
            Potential::Gold(potential) => potential.value_and_gradient(x),
            Potential::Tip4p(potential) => potential.value_and_gradient(x),
        };
        self.pot_ns
            .fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let call = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
        if anneal_core::energy_hits(evaluated.0, self.reference, anneal_core::DEFAULT_HIT_TOL) {
            let _ = self
                .first_hit
                .compare_exchange(0, call, Ordering::Relaxed, Ordering::Relaxed);
            self.stop.store(true, Ordering::Relaxed);
        }
        if call >= self.budget {
            self.stop.store(true, Ordering::Relaxed);
        }
        evaluated
    }
}

impl Objective<f64> for Counted {
    fn dim(&self) -> usize {
        match &self.potential {
            Potential::Pair(potential) => Objective::dim(potential),
            Potential::Gold(potential) => Objective::dim(potential),
            Potential::Tip4p(potential) => Objective::dim(potential),
        }
    }

    fn bounds(&self) -> &Bounds<f64> {
        match &self.potential {
            Potential::Pair(potential) => potential.bounds(),
            Potential::Gold(potential) => potential.bounds(),
            Potential::Tip4p(potential) => potential.bounds(),
        }
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.counted(x).0
    }
}

impl Gradient<f64> for Counted {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.counted(x).1
    }

    fn dim(&self) -> usize {
        Objective::dim(self)
    }
}

impl DifferentiableObjective<f64> for Counted {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.counted(x)
    }
}

fn config_and_start(spec: Spec, seed: u64) -> (Config, Array1<f64>) {
    let cfg = match spec.kind {
        Kind::Gold => {
            let species = Gupta::copper_gold_species(0, spec.n);
            let mut cfg = Config::recommended(spec.n).with_species(species, None);
            cfg.basin_stride = 8;
            cfg
        }
        Kind::Tip4p => {
            let mut cfg = Config::recommended_tip4p(spec.n);
            cfg.rigid_body_geometry = Some(gmin_water_geometry());
            cfg
        }
        Kind::LennardJones | Kind::Morse { .. } => Config::recommended(spec.n),
    };
    let start = match spec.kind {
        Kind::Tip4p => {
            let mixed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(3);
            random_tip4p_cluster(spec.n, &mut StdRng::seed_from_u64(mixed))
        }
        _ => {
            let mut rng = StdRng::seed_from_u64(seed);
            random_cluster_in_radius(spec.n, cfg.start_radius(), cfg.min_separation, &mut rng)
        }
    };
    (cfg, start)
}

fn potential(spec: Spec) -> Potential {
    match spec.kind {
        Kind::LennardJones => Potential::Pair(PairPotential::lennard_jones(spec.n)),
        Kind::Morse { rho } => Potential::Pair(PairPotential::morse(spec.n, rho)),
        Kind::Gold => Potential::Gold(Gupta::copper_gold(0, spec.n)),
        Kind::Tip4p => Potential::Tip4p(GminTip4p::new(spec.n)),
    }
}

fn write_bits(path: &str, state: ArrayView1<f64>) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(state.len() * 8);
    for value in state {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    std::fs::write(path, bytes)
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert_eq!(args.len(), 3, "see the usage line");
    let spec = spec(&args[0]).unwrap_or_else(|| panic!("unknown system {}", args[0]));
    let seed: u64 = args[1].parse().expect("seed");
    let (cfg, start) = config_and_start(spec, seed);
    let stop = Arc::new(AtomicBool::new(false));
    let objective = Counted {
        potential: potential(spec),
        budget: spec.budget,
        reference: spec.reference,
        stop: Arc::clone(&stop),
        calls: AtomicUsize::new(0),
        first_hit: AtomicUsize::new(0),
        pot_ns: AtomicU64::new(0),
    };
    let mut ledger = Ledger::with_stop(spec.budget, stop);
    let started = Instant::now();
    let outcome: Outcome = search_from(&objective, &cfg, &mut ledger, start.view(), seed)
        .expect("finite start")
        .0;
    let wall = started.elapsed().as_secs_f64();
    let pot_s = objective.pot_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
    let outside_s = (wall - pot_s).max(0.0);
    let total_calls = objective.calls.load(Ordering::Relaxed).min(spec.budget);
    let first_hit = objective.first_hit.load(Ordering::Relaxed);
    let hit = first_hit > 0;
    let calls = if hit { first_hit } else { spec.budget };
    let best = outcome.best;
    let state = outcome
        .final_state
        .as_ref()
        .or(outcome.best_state.as_ref())
        .expect("final state");
    write_bits(&args[2], state.view()).expect("coordinate bits");
    println!(
        "system,seed,budget,reference,hit,calls,charged,wall_s,inside_s,outside_s,best,best_bits,state_len"
    );
    println!(
        "{},{},{},{:.12},{},{},{},{:.9},{:.9},{:.9},{:.17},{:016x},{}",
        spec.name,
        seed,
        spec.budget,
        spec.reference,
        u8::from(hit),
        calls,
        total_calls,
        wall,
        pot_s,
        outside_s,
        best,
        best.to_bits(),
        state.len(),
    );
}
