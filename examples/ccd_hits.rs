//! Hit counts against Cambridge Cluster Database minima at a fixed budget.
//!
//! Usage: `ccd_hits <problem> <budget> <first_seed> <last_seed> [arm]`
//!
//! `problem` is `morse55` (Morse, `rho = 14`), `morse55:RHO`, `tip4p15` or
//! `tip4p20`. Seeds are inclusive. `arm` is `gmin` for the structural-move
//! hopper in `methods::gmin_hop`, or names a `cluster_hopping` preset,
//! `recommended` by default. The registration in `paper/seeds.toml` fixes the problems, the
//! seeds, the budgets and the hit rule.
//!
//! Every call into the potential counts, whether or not the driver charges
//! it to its ledger: the objective is wrapped in a counter, and the counter
//! raises the ledger's stop flag when it reaches the budget. A seed is a hit
//! when some evaluated point has an energy at or below the reference plus
//! [`HIT_TOLERANCE`] at a call index within the budget. The reference is a
//! global minimum, so nothing lies below it; a point within the tolerance is
//! at the bottom of the global basin or of a minimum degenerate with it to
//! that tolerance.
//!
//! The Morse potential is Doye and Wales' reduced form, `epsilon = r_0 = 1`.
//! Rigid TIP4P is GMIN's parameter set, the one the database energies were
//! computed with, in kJ/mol.

use anneal_core::methods::cluster_hopping::{Config, Ledger, Outcome};
use anneal_core::methods::cluster_search::{search, search_from};
use anneal_core::methods::gmin_hop;
use anneal_core::potentials::{GminTip4p, PairPotential, ccd_tip4p_minimum, gmin_water_geometry};
use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Energy above the reference that still counts as the global minimum, in
/// the potential's own energy unit.
const HIT_TOLERANCE: f64 = 1.0e-3;

/// Cambridge Cluster Database global minimum for the problem.
fn reference(problem: &Problem) -> f64 {
    match *problem {
        Problem::Morse { n, rho } => PairPotential::ccd_morse_minimum(n, rho),
        Problem::Tip4p { n } => ccd_tip4p_minimum(n),
    }
    .unwrap_or_else(|| panic!("no database minimum for {problem:?}"))
}

#[derive(Debug, Clone, Copy)]
enum Problem {
    Morse { n: usize, rho: f64 },
    Tip4p { n: usize },
}

fn parse_problem(spec: &str) -> Problem {
    if let Some(rest) = spec.strip_prefix("morse") {
        let (n, rho) = match rest.split_once(':') {
            Some((n, rho)) => (n, rho.parse().expect("rho")),
            None => (rest, 14.0),
        };
        return Problem::Morse {
            n: n.parse().expect("Morse size"),
            rho,
        };
    }
    if let Some(n) = spec.strip_prefix("tip4p") {
        return Problem::Tip4p {
            n: n.parse().expect("TIP4P size"),
        };
    }
    panic!("unknown problem {spec}");
}

enum Model {
    Morse(PairPotential),
    Tip4p(GminTip4p),
}

struct FirstHit {
    calls: usize,
    hit_at: Option<usize>,
    lowest: f64,
}

/// The potential, with every call counted.
struct Counted {
    model: Model,
    budget: usize,
    target: f64,
    stop: Arc<AtomicBool>,
    state: Mutex<FirstHit>,
}

impl Counted {
    fn raw(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        match &self.model {
            Model::Morse(p) => p.value_and_gradient(x),
            Model::Tip4p(p) => p.value_and_gradient(x),
        }
    }

    fn counted(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let (e, g) = self.raw(x);
        let mut state = self.state.lock().unwrap();
        state.calls += 1;
        if state.calls <= self.budget {
            if e < state.lowest {
                state.lowest = e;
            }
            if state.hit_at.is_none() && e <= self.target {
                state.hit_at = Some(state.calls);
            }
        }
        if state.calls >= self.budget {
            self.stop.store(true, Ordering::Relaxed);
        }
        (e, g)
    }
}

impl Objective<f64> for Counted {
    fn dim(&self) -> usize {
        match &self.model {
            Model::Morse(p) => Objective::dim(p),
            Model::Tip4p(p) => Objective::dim(p),
        }
    }

    fn bounds(&self) -> &Bounds<f64> {
        match &self.model {
            Model::Morse(p) => p.bounds(),
            Model::Tip4p(p) => p.bounds(),
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

fn morse_config(arm: &str, n: usize) -> Config {
    match arm {
        "recommended" => Config::recommended(n),
        "base" => Config::for_cluster(n),
        other => panic!("no Morse arm {other}"),
    }
}

fn tip4p_config(arm: &str, n: usize) -> Config {
    let mut cfg = match arm {
        "recommended" => Config::recommended_tip4p(n),
        "communicating" => Config::communicating_tip4p(n),
        "base" => Config::for_tip4p(n),
        other => panic!("no TIP4P arm {other}"),
    };
    cfg.rigid_body_geometry = Some(gmin_water_geometry());
    cfg
}

/// Starting structures come off a stream separate from the search's own.
fn start_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5EED)
}

fn run_seed(problem: Problem, arm: &str, budget: usize, seed: u64) -> (Outcome, FirstHit, f64) {
    let target = reference(&problem) + HIT_TOLERANCE;
    let stop = Arc::new(AtomicBool::new(false));
    let fresh = FirstHit {
        calls: 0,
        hit_at: None,
        lowest: f64::INFINITY,
    };
    let mut ledger = Ledger::with_stop(budget, Arc::clone(&stop));
    let started = std::time::Instant::now();
    let (outcome, objective) = match problem {
        Problem::Morse { n, rho } => {
            let objective = Counted {
                model: Model::Morse(PairPotential::morse(n, rho)),
                budget,
                target,
                stop,
                state: Mutex::new(fresh),
            };
            let out = if arm == "gmin" {
                let start = gmin_hop::random_compact_cluster(n, &mut start_rng(seed));
                gmin_hop::hop(
                    &objective,
                    &gmin_hop::Config::morse(n),
                    &mut ledger,
                    start.view(),
                    seed,
                )
            } else {
                let cfg = morse_config(arm, n);
                search(&objective, &cfg, &mut ledger, seed)
                    .expect("finite start")
                    .0
            };
            (out, objective)
        }
        Problem::Tip4p { n } => {
            let objective = Counted {
                model: Model::Tip4p(GminTip4p::new(n)),
                budget,
                target,
                stop,
                state: Mutex::new(fresh),
            };
            let start = anneal_core::potentials::random_tip4p_cluster(n, &mut start_rng(seed));
            let out = if arm == "gmin" {
                gmin_hop::hop(
                    &objective,
                    &gmin_hop::Config::water(n),
                    &mut ledger,
                    start.view(),
                    seed,
                )
            } else {
                let cfg = tip4p_config(arm, n);
                search_from(&objective, &cfg, &mut ledger, start.view(), seed)
                    .expect("finite start")
                    .0
            };
            (out, objective)
        }
    };
    let secs = started.elapsed().as_secs_f64();
    let state = objective.state.into_inner().unwrap();
    (outcome, state, secs)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(args.len() >= 4, "see the usage line in the source");
    let problem = parse_problem(&args[0]);
    let budget: usize = args[1].parse().expect("budget");
    let first: u64 = args[2].parse().expect("first seed");
    let last: u64 = args[3].parse().expect("last seed");
    let arm = args.get(4).map(String::as_str).unwrap_or("recommended");
    let reference = reference(&problem);
    println!(
        "problem={} arm={arm} budget={budget} seeds={first}..={last} reference={reference:.10} tolerance={HIT_TOLERANCE}",
        args[0]
    );
    let seeds: Vec<u64> = (first..=last).collect();
    let queue = Mutex::new(seeds.clone().into_iter());
    let rows = Mutex::new(Vec::new());
    let workers = std::env::var("CCD_WORKERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let Some(seed) = queue.lock().unwrap().next() else {
                        break;
                    };
                    let (out, state, secs) = run_seed(problem, arm, budget, seed);
                    let hit = state.hit_at.is_some();
                    println!(
                        "seed={seed} hit={} first_hit={} lowest={:.10} best={:.10} gap={:+.6} calls={} hops={} basins={} secs={secs:.1}",
                        u8::from(hit),
                        state
                            .hit_at
                            .map(|c| c.to_string())
                            .unwrap_or_else(|| "-".into()),
                        state.lowest,
                        out.best,
                        state.lowest - reference,
                        state.calls,
                        out.hops,
                        out.basins,
                    );
                    rows.lock().unwrap().push((seed, state.hit_at));
                }
            });
        }
    });
    let mut rows = rows.into_inner().unwrap();
    rows.sort_unstable();
    let hits: Vec<usize> = rows.iter().filter_map(|(_, h)| *h).collect();
    let mut sorted = hits.clone();
    sorted.sort_unstable();
    let median = sorted.get(sorted.len() / 2).copied();
    println!(
        "summary problem={} arm={arm} budget={budget} hits={}/{} median_first_hit={}",
        args[0],
        hits.len(),
        rows.len(),
        median.map(|m| m.to_string()).unwrap_or_else(|| "-".into())
    );
}
