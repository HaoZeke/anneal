//! One head-to-head seed with its trace and its lowest geometry.
//!
//! Usage: `h2h_seed <lj|morse|tip4p> <n> <rho> <budget> <seed> <arm> <stop> <prefix>`
//!
//! `arm` is `recommended` or `base` for the `cluster_hopping` presets.
//! `stop` is an energy at which the run ends early, or `-` to spend the
//! whole budget.
//!
//! Each run starts as in the ABCluster head-to-head. Atoms come from
//! `random_cluster_in_radius` at the radius and separation of
//! `Config::for_cluster(n)`, drawn from `StdRng::seed_from_u64(seed)`. Waters
//! come from `random_tip4p_cluster`, drawn from the seed times the 64-bit
//! golden ratio plus 3. The search itself runs on `seed`.
//!
//! Every call into the potential is counted, whatever the driver charges,
//! and the count stops the ledger at the budget. Each new lowest evaluated
//! energy is written to `<prefix>.trace.csv` with its call index, so a hit
//! can be scored afterwards against any reference. The lowest point is
//! written to `<prefix>.xyz`: atoms as `X`, waters as O, H, H with the M
//! site left out. Water sites use the parameters of the global
//! minimisation program (GMIN).

use anneal_core::methods::cluster_hopping::{Config, Ledger, Outcome, random_cluster_in_radius};
use anneal_core::methods::cluster_search::search_from;
use anneal_core::potentials::{
    GminTip4p, PairPotential, energy_ohh, gmin_tip4p_sites, gmin_water_geometry,
    random_tip4p_cluster,
};
use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// A new lowest energy enters the trace only when it is this much lower.
const TRACE_STEP: f64 = 1.0e-7;

enum Model {
    Pair(PairPotential),
    Water(GminTip4p),
}

struct Seen {
    calls: usize,
    lowest: f64,
    at: Array1<f64>,
    trace: Vec<(usize, f64)>,
}

struct Counted {
    model: Model,
    budget: usize,
    stop_at: Option<f64>,
    stop: Arc<AtomicBool>,
    seen: Mutex<Seen>,
    pot_ns: AtomicU64,
}

impl Counted {
    fn counted(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let mark = Instant::now();
        let (e, g) = match &self.model {
            Model::Pair(p) => p.value_and_gradient(x),
            Model::Water(p) => p.value_and_gradient(x),
        };
        self.pot_ns
            .fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let mut seen = self.seen.lock().unwrap();
        seen.calls += 1;
        if seen.calls <= self.budget && e < seen.lowest {
            if e < seen.lowest - TRACE_STEP {
                let calls = seen.calls;
                seen.trace.push((calls, e));
            }
            seen.lowest = e;
            seen.at = x.to_owned();
        }
        if seen.calls >= self.budget || self.stop_at.is_some_and(|s| seen.lowest <= s) {
            self.stop.store(true, Ordering::Relaxed);
        }
        (e, g)
    }
}

impl Objective<f64> for Counted {
    fn dim(&self) -> usize {
        match &self.model {
            Model::Pair(p) => Objective::dim(p),
            Model::Water(p) => Objective::dim(p),
        }
    }

    fn bounds(&self) -> &Bounds<f64> {
        match &self.model {
            Model::Pair(p) => p.bounds(),
            Model::Water(p) => p.bounds(),
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

fn atomic_config(arm: &str, n: usize) -> Config {
    match arm {
        "recommended" => Config::recommended(n),
        "base" => Config::for_cluster(n),
        "temperature-1.5" => {
            let mut cfg = Config::recommended(n);
            cfg.temperature = 1.5 * cfg.energy_scale;
            cfg
        }
        "angular" => {
            let mut cfg = Config::recommended(n);
            cfg.angular_moves = true;
            cfg
        }
        other => panic!("no atomic arm {other}"),
    }
}

fn water_config(arm: &str, n: usize) -> Config {
    let mut cfg = match arm {
        "recommended" => Config::recommended_tip4p(n),
        "base" => Config::for_tip4p(n),
        other => panic!("no TIP4P arm {other}"),
    };
    cfg.rigid_body_geometry = Some(gmin_water_geometry());
    cfg
}

/// Rodrigues rotation of `v` by the rotation vector `w`.
fn rotate(w: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    let theta = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    if theta < 1e-12 {
        return v;
    }
    let k = [w[0] / theta, w[1] / theta, w[2] / theta];
    let kv = [
        k[1] * v[2] - k[2] * v[1],
        k[2] * v[0] - k[0] * v[2],
        k[0] * v[1] - k[1] * v[0],
    ];
    let kdv = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
    let (sin, cos) = theta.sin_cos();
    [0, 1, 2].map(|a| v[a] * cos + kv[a] * sin + k[a] * kdv * (1.0 - cos))
}

/// Lab-frame O, H, H of every molecule in the block chart, flattened.
fn ohh(n: usize, x: ArrayView1<f64>) -> Vec<f64> {
    let body = gmin_tip4p_sites();
    (0..n)
        .flat_map(|i| {
            let w = [x[3 * n + 3 * i], x[3 * n + 3 * i + 1], x[3 * n + 3 * i + 2]];
            (0..3).flat_map(move |s| {
                let p = rotate(w, body[s]);
                [0, 1, 2].map(|k| x[3 * i + k] + p[k])
            })
        })
        .collect()
}

fn xyz(symbols: &[&str], coords: &[f64], comment: &str) -> String {
    let mut s = format!("{}\n{comment}\n", symbols.len());
    for (sym, p) in symbols.iter().zip(coords.chunks(3)) {
        writeln!(s, "{sym} {:.10} {:.10} {:.10}", p[0], p[1], p[2]).unwrap();
    }
    s
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert_eq!(args.len(), 8, "see the usage line in the source");
    let kind = args[0].as_str();
    let n: usize = args[1].parse().expect("n");
    let rho: f64 = args[2].parse().expect("rho");
    let budget: usize = args[3].parse().expect("budget");
    let seed: u64 = args[4].parse().expect("seed");
    let arm = args[5].as_str();
    let stop_at: Option<f64> = (args[6] != "-").then(|| args[6].parse().expect("stop energy"));
    let prefix = &args[7];

    let water = kind == "tip4p";
    let (model, start) = if water {
        let mut rng =
            StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(3));
        (
            Model::Water(GminTip4p::new(n)),
            random_tip4p_cluster(n, &mut rng),
        )
    } else {
        let pair = match kind {
            "lj" => PairPotential::lennard_jones(n),
            "morse" => PairPotential::morse(n, rho),
            other => panic!("unknown system {other}"),
        };
        let sphere = Config::for_cluster(n);
        let mut rng = StdRng::seed_from_u64(seed);
        let start =
            random_cluster_in_radius(n, sphere.start_radius(), sphere.min_separation, &mut rng);
        (Model::Pair(pair), start)
    };
    if water {
        std::fs::write(format!("{prefix}.start.gmin"), gmin_block(n, start.view()))
            .expect("start");
    }
    let stop = Arc::new(AtomicBool::new(false));
    let objective = Counted {
        model,
        budget,
        stop_at,
        stop: Arc::clone(&stop),
        seen: Mutex::new(Seen {
            calls: 0,
            lowest: f64::INFINITY,
            at: start.clone(),
            trace: Vec::new(),
        }),
        pot_ns: AtomicU64::new(0),
    };
    let mut ledger = Ledger::with_stop(budget, stop);
    let started = Instant::now();
    let out: Outcome = match (water, arm) {
        (true, _) => {
            search_from(
                &objective,
                &water_config(arm, n),
                &mut ledger,
                start.view(),
                seed,
            )
            .expect("finite start")
            .0
        }
        (false, _) => {
            search_from(
                &objective,
                &atomic_config(arm, n),
                &mut ledger,
                start.view(),
                seed,
            )
            .expect("finite start")
            .0
        }
    };
    let wall = started.elapsed().as_secs_f64();
    let pot_s = objective.pot_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
    let outside_s = (wall - pot_s).max(0.0);
    let seen = objective.seen.into_inner().unwrap();

    let mut trace = String::from("calls,energy\n");
    for (calls, e) in &seen.trace {
        writeln!(trace, "{calls},{e:.10}").unwrap();
    }
    std::fs::write(format!("{prefix}.trace.csv"), trace).expect("trace");
    let comment = format!(
        "{kind} n={n} rho={rho} seed={seed} arm={arm} energy={:.10}",
        seen.lowest
    );
    let (geometry, check) = if water {
        let coords = ohh(n, seen.at.view());
        let symbols: Vec<&str> = (0..n).flat_map(|_| ["O", "H", "H"]).collect();
        let check = energy_ohh(&coords);
        (xyz(&symbols, &coords, &comment), check)
    } else {
        let coords = seen.at.to_vec();
        (xyz(&vec!["X"; n], &coords, &comment), seen.lowest)
    };
    std::fs::write(format!("{prefix}.xyz"), geometry).expect("geometry");

    println!("system: {kind}");
    println!("n: {n}");
    println!("rho: {rho}");
    println!("seed: {seed}");
    println!("arm: {arm}");
    println!("budget: {budget}");
    println!("calls: {}", seen.calls.min(budget));
    println!("charged: {}", ledger.spent());
    println!("lowest: {:.10}", seen.lowest);
    println!("geometry_energy: {check:.10}");
    println!("best: {:.10}", out.best);
    println!("hops: {}", out.hops);
    println!("basins: {}", out.basins);
    println!("wall_seconds: {wall:.6}");
    println!("pot_s: {pot_s:.6}");
    println!("outside_s: {outside_s:.6}");
    for (name, draws, accepts, best) in &out.arms {
        println!("arm {name} draws {draws} accepts {accepts} best {best:.6}");
    }
    let hit = stop_at.is_some_and(|level| seen.lowest <= level);
    println!(
        "h2h_row system={kind}{n} code=anneal seed={seed} budget={budget} hit={} evals_to_hit={} charged={} best={:.10} wall_s={wall:.6} pot_s={pot_s:.6} outside_s={outside_s:.6} basins={}",
        u8::from(hit),
        if hit { seen.calls.min(budget) } else { budget },
        ledger.spent(),
        seen.lowest,
        out.basins,
    );
    if water {
        std::fs::write(format!("{prefix}.gmin"), gmin_block(n, seen.at.view())).expect("gmin coords");
    }
}

/// Centres, then angle-axis with the opposite sense of the Rodrigues chart.
/// NEWTIP reads that block: `OFFSET = 3 NATOMS / 2`.
fn gmin_block(n: usize, x: ArrayView1<f64>) -> String {
    let mut text = String::new();
    for i in 0..n {
        writeln!(
            text,
            "{:.10} {:.10} {:.10}",
            x[3 * i],
            x[3 * i + 1],
            x[3 * i + 2]
        )
        .unwrap();
    }
    for i in 0..n {
        let base = 3 * n + 3 * i;
        writeln!(
            text,
            "{:.10} {:.10} {:.10}",
            -x[base],
            -x[base + 1],
            -x[base + 2]
        )
        .unwrap();
    }
    text
}
