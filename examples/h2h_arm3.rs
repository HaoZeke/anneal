//! One arm-3 seed for the ABCluster head-to-head.
//!
//! `h2h_arm3 <system> <budget> <seed>`
//!
//! Systems: lj38 lj55 lj75 lj98 lj98-points m55r6 m55r14 m80r6 m80r14 w6 w10
//! w15 w20. `lj98-points` scores LJ98 against the energy of the Cambridge
//! points file, the Leary tetrahedron, where `lj98` reads the table.
//! The start is the driver's random cluster. The reference only scores a hit.
//! The atomic systems run [`Config::for_cluster`] with orbit completion on
//! every new basin from both cores, and a random restart after
//! [`stall_restart_calls`] charged calls without a new best. Water runs
//! [`Config::for_tip4p`].
//! `H2H_ARM` runs an atomic system as communicating chains instead, as
//! `common/h2h_chains.rs` describes.

#[path = "common/ensemble_report.rs"]
mod ensemble_report;
#[path = "common/h2h_chains.rs"]
mod h2h_chains;

use anneal_core::bias::BasinBias;
use anneal_core::methods::cluster_hopping::{
    ChainCheckpoint, CheckpointAction, ClusterFingerprint, Config, Ledger, random_cluster,
    run_with_bias_at_checkpoints,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{
    PairPotential, Tip4pCluster, random_tip4p_cluster, wales_hodges_minimum,
};
use h2h_chains::{Arm, run_chains};
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
        "lj98-points" => Spec {
            name: "lj98-points",
            ..lj(98, -543.665361)
        },
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
    hops: Option<usize>,
}

/// Charged calls without a new best before a random restart.
///
/// A quench costs `relax_steps` evaluations. The stall clock waits
/// `screen_steps` of those quenches. The product does not read the atom
/// count. It replaces the hand-set 4000 on the dual-core preset.
fn stall_restart_calls(cfg: &Config) -> usize {
    cfg.screen_steps.saturating_mul(cfg.relax_steps).max(1)
}

fn atomic(spec: &Spec, pot: PairPotential, budget: usize, seed: u64, started: Instant) {
    let scale = pot.kind().r_min() / 2.0_f64.powf(1.0 / 6.0);
    let mut base = Config::for_cluster(spec.n);
    base.min_separation *= scale;
    base.container *= scale;
    let mut cfg = base.clone().with_dual_orbit_core();
    cfg.restart_patience = stall_restart_calls(&cfg);
    let arm = Arm::from_env();
    if arm != Arm::Single {
        let chain = arm.chain_config(&cfg, base);
        let min_separation = chain.min_separation;
        let run = run_chains(
            &chain,
            budget,
            seed,
            spec.reference + spec.tol,
            &|x| pot.value_and_gradient(x),
            &|_, rng| random_cluster(spec.n, 0.7, min_separation, rng),
        );
        emit(Record {
            spec,
            cfg: &chain,
            arm,
            replicas: run.replicas,
            seed,
            budget,
            best: run.best,
            hit: run
                .first_target_calls
                .map(|calls| Hit { calls, hops: None }),
            charged: run.charged,
            hops: run.hops,
            started,
        });
        return;
    }
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
                    hops: Some(ck.hops()),
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
    emit(Record {
        spec,
        cfg: &cfg,
        arm: Arm::Single,
        replicas: 1,
        seed,
        budget,
        best: out.best,
        hit,
        charged: ledger.spent(),
        hops: out.hops,
        started,
    });
}

fn tip4p(spec: &Spec, budget: usize, seed: u64, started: Instant) {
    assert_eq!(
        Arm::from_env(),
        Arm::Single,
        "the communicating arms run the atomic systems"
    );
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
                    hops: Some(ck.hops()),
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
    emit(Record {
        spec,
        cfg: &cfg,
        arm: Arm::Single,
        replicas: 1,
        seed,
        budget,
        best: out.best,
        hit,
        charged: ledger.spent(),
        hops: out.hops,
        started,
    });
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
            hops: Some(*hops),
        });
    match (seen, logged) {
        (Some(a), Some(b)) => Some(if a.calls <= b.calls { a } else { b }),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// One output line.
struct Record<'a> {
    spec: &'a Spec,
    cfg: &'a Config,
    arm: Arm,
    replicas: usize,
    seed: u64,
    budget: usize,
    best: f64,
    hit: Option<Hit>,
    charged: usize,
    hops: usize,
    started: Instant,
}

fn emit(record: Record<'_>) {
    let Record {
        spec,
        cfg,
        arm,
        replicas,
        seed,
        budget,
        best,
        hit,
        charged,
        hops,
        started,
    } = record;
    println!(
        "system={system} seed={seed} budget={budget} reference={reference:.8} tol={tol} \
         best={best:.8} hit={hit} first_calls={fc} first_hops={fh} charged={charged} \
         hops={hops} wall_s={wall:.3} arm={arm} replicas={replicas} witness={witness} \
         config_sha256={digest}",
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
            .and_then(|h| h.hops)
            .map(|h| h.to_string())
            .unwrap_or_else(|| "-".into()),
        wall = started.elapsed().as_secs_f64(),
        arm = arm.name(),
        witness = if arm == Arm::Single {
            "-"
        } else {
            h2h_chains::witness_name()
        },
        digest = cfg.resolved_sha256().expect("the configuration serialises"),
    );
}
