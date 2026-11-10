//! One seed of the ABCluster head-to-head, on the default or the dual-core arm.
//!
//! ```text
//! h2h_arm3 <system> <default|arm3> <budget> <seed>
//! ```
//!
//! `system` is one of `lj38 lj55 lj75 lj98 m55r6 m55r14 m80r6 m80r14 w6 w10
//! w15 w20`. Atomic clusters run [`Config::for_cluster`], or
//! [`Config::for_cluster_dual_core`] on `arm3`, from `random_cluster` at
//! density 0.7. Morse then scales the minimum separation and the container
//! by `r_min / 2^(1/6)`, as `cluster_bench` does. Water runs
//! [`Config::for_tip4p`] on both arms: a rigid-body chain has no orbit
//! completion, so the two arms are one configuration there.
//!
//! The ledger stops at the first recorded minimum below the reference plus
//! the tolerance. `first_calls` is the charged count at that record, read on
//! the first closure entry or charge after it, or the final count when the
//! driver returns first. `first_relax` counts the relaxation calls before
//! it and `first_hops` is the hop counter on the improvement record, when the
//! trace holds one (an orbit completion records without it). `relaxations`
//! counts every call of the relaxation closure, screens included. One
//! `system=` line is printed per run.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{PairKind, PairPotential, Tip4pCluster, random_tip4p_cluster};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[derive(Clone, Copy)]
enum Kind {
    Lj,
    Morse(f64),
    Water,
}

/// Size, potential, reference and hit tolerance. LJ98 is the Leary Td
/// global minimum; the Deaven icosahedral minimum -543.642957 is above it.
fn system(name: &str) -> Option<(usize, Kind, f64, f64)> {
    Some(match name {
        "lj38" => (38, Kind::Lj, -173.928427, 0.001),
        "lj55" => (55, Kind::Lj, -279.248470, 0.001),
        "lj75" => (75, Kind::Lj, -397.492331, 0.001),
        "lj98" => (98, Kind::Lj, -543.665361, 0.001),
        "m55r6" => (55, Kind::Morse(6.0), -250.286609, 0.001),
        "m55r14" => (55, Kind::Morse(14.0), -220.646208, 0.001),
        "m80r6" => (80, Kind::Morse(6.0), -378.333471, 0.001),
        "m80r14" => (80, Kind::Morse(14.0), -340.811371, 0.001),
        "w6" => (6, Kind::Water, -197.78, 0.01),
        "w10" => (10, Kind::Water, -391.02, 0.01),
        "w15" => (15, Kind::Water, -628.37225, 0.01),
        "w20" => (20, Kind::Water, -872.99, 0.01),
        _ => return None,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let name = args.first().map(String::as_str).unwrap_or("lj38");
    let arm = args.get(1).map(String::as_str).unwrap_or("default");
    let budget: usize = args.get(2).and_then(|v| v.parse().ok()).expect("budget");
    let seed: u64 = args.get(3).and_then(|v| v.parse().ok()).expect("seed");
    let Some((n, kind, reference, tol)) = system(name) else {
        eprintln!("unknown system {name}");
        std::process::exit(2);
    };
    let dual = match arm {
        "default" => false,
        "arm3" => true,
        _ => {
            eprintln!("unknown arm {arm}; expected default or arm3");
            std::process::exit(2);
        }
    };
    let level = reference + tol;

    type Energy = Box<dyn Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync>;
    let (cfg, start, energy): (Config, Array1<f64>, Energy) = match kind {
        Kind::Water => {
            let pot = Tip4pCluster::new(n);
            let cfg = Config::for_tip4p(n);
            let mut rng = StdRng::seed_from_u64(seed);
            let start = random_tip4p_cluster(n, &mut rng);
            (cfg, start, Box::new(move |x| pot.value_and_gradient(x)))
        }
        Kind::Lj | Kind::Morse(_) => {
            let (pot, pair) = match kind {
                Kind::Morse(rho) => (PairPotential::morse(n, rho), PairKind::Morse { rho }),
                _ => (PairPotential::lennard_jones(n), PairKind::LennardJones),
            };
            let mut cfg = if dual {
                Config::for_cluster_dual_core(n)
            } else {
                Config::for_cluster(n)
            };
            let scale = pair.r_min() / 2.0_f64.powf(1.0 / 6.0);
            cfg.min_separation *= scale;
            cfg.container *= scale;
            let mut rng = StdRng::seed_from_u64(seed);
            let start = random_cluster(n, 0.7, cfg.min_separation, &mut rng);
            (cfg, start, Box::new(move |x| pot.value_and_gradient(x)))
        }
    };
    let water = matches!(kind, Kind::Water);
    let folder = water.then(|| Tip4pCluster::new(n));

    let mut ledger = Ledger::new(budget);
    ledger.stop_below(level);
    let mut optimiser = WarmLbfgs::default();
    let mut relaxations = 0usize;
    // Calls spent when the ledger first holds a record below the level,
    // read on the next closure entry or charge after that record.
    let first_seen = AtomicUsize::new(usize::MAX);
    let first_relax = AtomicUsize::new(usize::MAX);
    let mark = |led: &Ledger, relaxations: usize| {
        if led.target_reached() && first_seen.load(Ordering::Relaxed) == usize::MAX {
            first_seen.store(led.spent(), Ordering::Relaxed);
            first_relax.store(relaxations, Ordering::Relaxed);
        }
    };
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        mark(led, relaxations);
        relaxations += 1;
        optimiser.forget();
        let (value, mut relaxed, _) = optimiser.minimize(x, iterations, |v| {
            mark(led, relaxations - 1);
            if !led.charge() {
                return None;
            }
            Some(energy(v))
        });
        if let (Some(pot), Some(slice)) = (folder.as_ref(), relaxed.as_slice_mut()) {
            pot.fold_rotations(slice);
        }
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        mark(led, usize::MAX);
        if !led.charge() {
            return None;
        }
        Some(energy(v).1)
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let started = Instant::now();
    let outcome: Outcome = run_with_gradient(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start");
    let wall = started.elapsed().as_secs_f64();

    let best = outcome.best;
    let hit = best < level;
    let trace = outcome
        .improvements
        .iter()
        .find(|(_, _, _, e)| *e < level)
        .map(|(hops, _, _, _)| hops.to_string());
    let seen = first_seen.load(Ordering::Relaxed);
    let (first_calls, first_source) = match (hit, seen) {
        (false, _) => ("-".to_string(), "-"),
        (true, usize::MAX) => (outcome.charged.to_string(), "stop"),
        (true, calls) => (calls.to_string(), "ledger"),
    };
    let first_hops = if hit {
        trace.unwrap_or_else(|| "-".into())
    } else {
        "-".into()
    };
    let first_relax = match (hit, first_relax.load(Ordering::Relaxed)) {
        (true, usize::MAX) => relaxations.to_string(),
        (true, count) => count.to_string(),
        (false, _) => "-".into(),
    };
    println!(
        "system={name} code=anneal arm={arm} seed={seed} budget={budget} hit={} best={best:.8} \
         first_calls={first_calls} first_hops={first_hops} first_relax={first_relax} \
         first_source={first_source} charged={} hops={} relaxations={} reference={reference:.6} tol={tol} wall_s={wall:.3} \
         config_sha256={}",
        u8::from(hit),
        outcome.charged,
        outcome.hops,
        relaxations,
        cfg.resolved_sha256().unwrap_or_else(|_| "unset".into()),
    );
}
