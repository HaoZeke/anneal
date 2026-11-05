//! One arm-3 seed for the ABCluster head-to-head.
//!
//! `h2h_arm3 <system> <budget> <seed>`
//!
//! Systems: lj38 lj55 lj75 lj98 m55r6 m55r14 m80r6 m80r14 w6 w10 w15 w20.
//! The start is the driver's random cluster. The reference only scores a hit.
//! The atomic systems run [`Config::for_cluster`] with orbit completion on
//! every new basin from both cores; water runs [`Config::for_tip4p`].

use anneal_core::bias::BasinBias;
use anneal_core::methods::cluster_hopping::{
    ChainCheckpoint, CheckpointAction, ClusterFingerprint, Config, Ledger, OrbitCore,
    random_cluster, run_with_bias_at_checkpoints,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{
    PairPotential, Tip4pCluster, random_tip4p_cluster, wales_hodges_minimum,
};
use anneal_core::symmetrise::ShellCore;
use ndarray::ArrayView1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

struct Spec {
    name: &'static str,
    kind: Kind,
    n: usize,
    reference: f64,
    tol: f64,
}

enum Kind {
    Lj,
    Morse(f64),
    Tip4p,
}

fn spec(name: &str) -> Spec {
    match name {
        "lj38" => lj(38, -173.928427),
        "lj55" => lj(55, -279.248470),
        "lj75" => lj(75, -397.492331),
        // Cambridge/Doye table. The crate constant -543.665361 is below it.
        "lj98" => lj(98, -543.642957),
        "m55r6" => morse("m55r6", 55, 6.0, -250.286609),
        "m55r14" => morse("m55r14", 55, 14.0, -220.646208),
        "m80r6" => morse("m80r6", 80, 6.0, -378.333471),
        "m80r14" => morse("m80r14", 80, 14.0, -340.811371),
        "w6" => water(6),
        "w10" => water(10),
        "w15" => water(15),
        "w20" => water(20),
        other => panic!("unknown system {other}"),
    }
}

fn lj(n: usize, reference: f64) -> Spec {
    Spec {
        name: match n {
            38 => "lj38",
            55 => "lj55",
            75 => "lj75",
            98 => "lj98",
            _ => unreachable!(),
        },
        kind: Kind::Lj,
        n,
        reference,
        tol: 1e-3,
    }
}

fn morse(name: &'static str, n: usize, rho: f64, reference: f64) -> Spec {
    Spec {
        name,
        kind: Kind::Morse(rho),
        n,
        reference,
        tol: 1e-3,
    }
}

fn water(n: usize) -> Spec {
    let reference = match n {
        6 => wales_hodges_minimum(6).unwrap(),
        10 => wales_hodges_minimum(10).unwrap(),
        15 => -628.37225,
        20 => wales_hodges_minimum(20).unwrap(),
        _ => panic!("no water reference for {n}"),
    };
    Spec {
        name: match n {
            6 => "w6",
            10 => "w10",
            15 => "w15",
            20 => "w20",
            _ => unreachable!(),
        },
        kind: Kind::Tip4p,
        n,
        reference,
        tol: 0.01,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let system = args.get(1).map(String::as_str).unwrap_or("lj38");
    let budget: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(20_000);
    let seed: u64 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(0);
    let spec = spec(system);
    let started = Instant::now();
    match spec.kind {
        Kind::Lj => atomic(
            &spec,
            PairPotential::lennard_jones(spec.n),
            budget,
            seed,
            started,
        ),
        Kind::Morse(rho) => atomic(
            &spec,
            PairPotential::morse(spec.n, rho),
            budget,
            seed,
            started,
        ),
        Kind::Tip4p => tip4p(&spec, budget, seed, started),
    }
}

struct Hit {
    calls: usize,
    hops: usize,
}

fn atomic(spec: &Spec, pot: PairPotential, budget: usize, seed: u64, started: Instant) {
    let mut cfg = Config::for_cluster(spec.n);
    cfg.orbit_complete_on_new = true;
    cfg.orbit_core = OrbitCore::Both(ShellCore::scaled(cfg.length_scale));
    let scale = pot.kind().r_min() / 2.0_f64.powf(1.0 / 6.0);
    cfg.min_separation *= scale;
    cfg.container *= scale;
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster(spec.n, 0.7, cfg.min_separation, &mut rng);
    let mut ledger = Ledger::new(budget);
    let mut opt = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
        opt.forget();
        let (f, xr, _) = opt.minimize(x, iters, |v| {
            if !led.charge() {
                return None;
            }
            Some(pot.value_and_gradient(v))
        });
        (f, xr)
    };
    let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<ndarray::Array1<f64>> {
        if !led.charge() {
            return None;
        }
        Some(pot.value_and_gradient(x).1)
    };
    let mut bias = BasinBias::new(
        ClusterFingerprint::of_config(&cfg, &start),
        cfg.merge_radius,
        cfg.bias_height,
        cfg.bias_gamma,
    );
    let target = spec.reference + spec.tol;
    let mut seen: Option<Hit> = None;
    let mut checkpoint = |ck: ChainCheckpoint<'_>| {
        if ck.best_energy() < target {
            if seen.is_none() {
                seen = Some(Hit {
                    calls: ck.charged(),
                    hops: ck.hops(),
                });
            }
            CheckpointAction::Retire {
                reason: "hit".into(),
            }
        } else {
            CheckpointAction::Continue
        }
    };
    let out = run_with_bias_at_checkpoints(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut bias,
        &mut rng,
        1,
        &mut checkpoint,
    )
    .expect("finite start");
    let hit = earlier(seen, &out.improvements, target);
    emit(
        spec,
        &cfg,
        seed,
        budget,
        out.best,
        hit,
        ledger.spent(),
        out.hops,
        started,
    );
}

fn tip4p(spec: &Spec, budget: usize, seed: u64, started: Instant) {
    let n = spec.n;
    let pot = Tip4pCluster::new(n);
    let cfg = Config::for_tip4p(n);
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_tip4p_cluster(n, &mut rng);
    let mut ledger = Ledger::new(budget);
    let mut opt = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
        opt.forget();
        let (f, mut xr, _) = opt.minimize(x, iters, |v| {
            if !led.charge() {
                return None;
            }
            Some(pot.value_and_gradient(v))
        });
        if let Some(slice) = xr.as_slice_mut() {
            pot.fold_rotations(slice);
        }
        (f, xr)
    };
    let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<ndarray::Array1<f64>> {
        if !led.charge() {
            return None;
        }
        Some(pot.value_and_gradient(x).1)
    };
    let mut bias = BasinBias::new(
        ClusterFingerprint::of_config(&cfg, &start),
        cfg.merge_radius,
        cfg.bias_height,
        cfg.bias_gamma,
    );
    let target = spec.reference + spec.tol;
    let mut seen: Option<Hit> = None;
    let mut checkpoint = |ck: ChainCheckpoint<'_>| {
        if ck.best_energy() < target {
            if seen.is_none() {
                seen = Some(Hit {
                    calls: ck.charged(),
                    hops: ck.hops(),
                });
            }
            CheckpointAction::Retire {
                reason: "hit".into(),
            }
        } else {
            CheckpointAction::Continue
        }
    };
    let mut out = run_with_bias_at_checkpoints(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut bias,
        &mut rng,
        1,
        &mut checkpoint,
    )
    .expect("finite start");
    if let Some(state) = out.best_state.as_ref() {
        out.best = pot.value_and_gradient(state.view()).0;
    }
    let hit = earlier(seen, &out.improvements, target);
    emit(
        spec,
        &cfg,
        seed,
        budget,
        out.best,
        hit,
        ledger.spent(),
        out.hops,
        started,
    );
}

/// The improvement log is exact until it caps at 512 records. The checkpoint
/// still sees a later crossing. Keep the earlier of the two.
fn earlier(
    seen: Option<Hit>,
    improvements: &[(usize, usize, usize, f64)],
    target: f64,
) -> Option<Hit> {
    let logged = improvements
        .iter()
        .find(|(_, _, _, e)| *e < target)
        .map(|(hops, calls, _, _)| Hit {
            calls: *calls,
            hops: *hops,
        });
    match (seen, logged) {
        (Some(a), Some(b)) => Some(if a.calls <= b.calls { a } else { b }),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn emit(
    spec: &Spec,
    cfg: &Config,
    seed: u64,
    budget: usize,
    best: f64,
    hit: Option<Hit>,
    charged: usize,
    hops: usize,
    started: Instant,
) {
    println!(
        "system={system} seed={seed} budget={budget} reference={reference:.8} tol={tol} \
         best={best:.8} hit={hit} first_calls={fc} first_hops={fh} charged={charged} \
         hops={hops} wall_s={wall:.3} config_sha256={digest}",
        system = spec.name,
        reference = spec.reference,
        tol = spec.tol,
        best = best,
        hit = usize::from(hit.is_some()),
        fc = hit
            .as_ref()
            .map(|h| h.calls.to_string())
            .unwrap_or_else(|| "-".into()),
        fh = hit
            .as_ref()
            .map(|h| h.hops.to_string())
            .unwrap_or_else(|| "-".into()),
        wall = started.elapsed().as_secs_f64(),
        digest = cfg.resolved_sha256().expect("the configuration serialises"),
    );
}
