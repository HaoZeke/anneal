//! Local stall-arm probe. Not part of the search.
//!
//! `stall_arm_probe <lj|tip4p> <n> <budget> <seed> <recommended|base>`
//!
//! Optional overrides, applied after the preset:
//! `STALL_PATIENCE`, `STALL_FACTOR`, `STALL_TRAIL`, `STALL_CLIMB`,
//! `STALL_RESTART`, `STALL_RETURN`, `STALL_CSYM` (`0` or `1`).

use anneal_core::methods::cluster_hopping::{
    Config, ContinuousSymmetry, Ledger, random_cluster_in_radius, run as run_gradfree,
};
use anneal_core::methods::cluster_search::search_from;
use anneal_core::potentials::{
    GminTip4p, PairPotential, SuttonChen, SuttonChenFamily, random_tip4p_cluster,
};
use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::env;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

struct TimedObjective {
    inner: Box<dyn DifferentiableObjective<f64>>,
    calls: AtomicUsize,
    pot_ns: AtomicU64,
}

impl Objective<f64> for TimedObjective {
    fn dim(&self) -> usize {
        Objective::dim(self.inner.as_ref())
    }

    fn bounds(&self) -> &Bounds<f64> {
        self.inner.bounds()
    }

    fn eval(&self, x: ndarray::ArrayView1<f64>) -> f64 {
        let t0 = Instant::now();
        let v = self.inner.eval(x);
        self.pot_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
        v
    }
}

impl Gradient<f64> for TimedObjective {
    fn grad(&self, x: ndarray::ArrayView1<f64>) -> ndarray::Array1<f64> {
        let t0 = Instant::now();
        let g = self.inner.grad(x);
        self.pot_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
        g
    }

    fn dim(&self) -> usize {
        Gradient::dim(self.inner.as_ref())
    }
}

impl DifferentiableObjective<f64> for TimedObjective {
    fn value_and_gradient(&self, x: ndarray::ArrayView1<f64>) -> (f64, ndarray::Array1<f64>) {
        let t0 = Instant::now();
        let vg = self.inner.value_and_gradient(x);
        self.pot_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
        vg
    }
}

fn flag(name: &str, cfg_flag: &mut bool) {
    if let Ok(text) = env::var(name) {
        *cfg_flag = text != "0";
    }
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = env::args().skip(1).collect();
    assert_eq!(
        args.len(),
        5,
        "lj|tip4p|lj-gradfree|sc-gradfree|tip4p-gradfree n budget seed recommended|base"
    );
    let kind = args[0].as_str();
    let n: usize = args[1].parse().expect("n");
    let budget: usize = args[2].parse().expect("budget");
    let seed: u64 = args[3].parse().expect("seed");
    let arm = args[4].as_str();
    let grad_free = kind.ends_with("-gradfree");
    let base_kind = kind.strip_suffix("-gradfree").unwrap_or(kind);
    let water = base_kind == "tip4p";
    let (objective, start, mut cfg): (Box<dyn eindir_core::DifferentiableObjective<f64>>, _, _) =
        if water {
            let mut rng =
                StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(3));
            let start = random_tip4p_cluster(n, &mut rng);
            let mut cfg = match arm {
                "recommended" => Config::recommended_tip4p(n),
                "base" => Config::for_tip4p(n),
                other => panic!("no tip4p arm {other}"),
            };
            cfg.rigid_body_geometry = Some(anneal_core::potentials::gmin_water_geometry());
            (Box::new(GminTip4p::new(n)), start, cfg)
        } else {
            let obj: Box<dyn eindir_core::DifferentiableObjective<f64>> = match base_kind {
                "lj" => Box::new(PairPotential::lennard_jones(n)),
                "sc" => Box::new(SuttonChen::new(n, SuttonChenFamily::Sc12_6)),
                other => panic!("unknown system {other}"),
            };
            let sphere = Config::for_cluster(n);
            let mut rng = StdRng::seed_from_u64(seed);
            let start =
                random_cluster_in_radius(n, sphere.start_radius(), sphere.min_separation, &mut rng);
            let cfg = match arm {
                "recommended" => Config::recommended(n),
                "base" => Config::for_cluster(n),
                other => panic!("no atomic arm {other}"),
            };
            (obj, start, cfg)
        };
    if let Ok(text) = env::var("STALL_PATIENCE") {
        cfg.escape_stall_patience = text.parse().expect("patience");
    }
    if let Ok(text) = env::var("STALL_FACTOR") {
        cfg.escape_stall_factor = text.parse().expect("factor");
    }
    flag("STALL_TRAIL", &mut cfg.trail_on_stall);
    flag("STALL_CLIMB", &mut cfg.escape_on_stall);
    flag("STALL_RESTART", &mut cfg.restart_on_stall);
    flag("STALL_RETURN", &mut cfg.return_screen);
    if let Ok(text) = env::var("STALL_CSYM") {
        cfg.continuous_symmetry = if text == "0" {
            ContinuousSymmetry::Off
        } else {
            ContinuousSymmetry::Measure { interval: 10 }
        };
    }
    let timed = TimedObjective {
        inner: objective,
        calls: AtomicUsize::new(0),
        pot_ns: AtomicU64::new(0),
    };
    let stop = Arc::new(AtomicBool::new(false));
    let mut ledger = Ledger::with_stop(budget, stop);
    let t0 = Instant::now();
    let out = if grad_free {
        let mut relax = |led: &mut Ledger, x: ndarray::ArrayView1<f64>, _steps: usize| {
            if !led.charge() {
                return (f64::INFINITY, x.to_owned());
            }
            (timed.eval(x), x.to_owned())
        };
        let mut run_rng = StdRng::seed_from_u64(seed);
        run_gradfree(&cfg, start.view(), &mut ledger, &mut relax, &mut run_rng)
            .expect("finite start")
    } else {
        search_from(&timed, &cfg, &mut ledger, start.view(), seed)
            .expect("finite start")
            .0
    };
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let pot_ms = timed.pot_ns.load(Ordering::Relaxed) as f64 / 1e6;
    let outside_ms = wall_ms - pot_ms;
    println!(
        "PROBE kind={kind} n={n} seed={seed} arm={arm} budget={budget} charged={} pot_calls={} hops={} best={:.6} final={:.6} best_bits={:016x} final_bits={:016x} trail_escapes={} stall_escapes={} restarts={} returned={} wall_ms={wall_ms:.3} pot_ms={pot_ms:.3} outside_ms={outside_ms:.3} stop={:?}",
        ledger.spent(),
        timed.calls.load(Ordering::Relaxed),
        out.hops,
        out.best,
        out.final_energy,
        out.best.to_bits(),
        out.final_energy.to_bits(),
        out.trail_escapes,
        out.stall_escapes,
        out.restarts,
        out.returned,
        out.stop,
    );
}
