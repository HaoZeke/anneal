//! A water search reads its stop target only to stop.
//!
//! A benchmark gives a global optimiser the reference energy so that the
//! search can end at the first hit. The energy must do nothing more, or the
//! known answer steers the trajectory. Two runs of one seed against two
//! references would then stop being the same search.

use anneal_core::bias::BasinBias;
use anneal_core::methods::cluster_hopping::{
    ChainCheckpoint, CheckpointAction, ClusterFingerprint, Config, Ledger, MoveLibrary,
    run_with_bias_at_checkpoints,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{Tip4pCluster, random_tip4p_cluster};
use ndarray::ArrayView1;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Debug, PartialEq)]
struct Trace {
    /// Ledger count and quenched energy bits at the end of every minimisation.
    quenches: Vec<(usize, u64)>,
    charged: usize,
    hops: usize,
    best: u64,
}

/// `Config::for_tip4p` with the cycle reversal off, so the chain moves by
/// translation and rotation alone.
fn translation_and_rotation(n: usize) -> Config {
    let mut cfg = Config::for_tip4p(n);
    let MoveLibrary::RigidBody { reverse_cycles, .. } = &mut cfg.move_library else {
        unreachable!("Config::for_tip4p moves rigid bodies");
    };
    *reverse_cycles = false;
    cfg
}

/// Returns the trace of one search under `cfg` with the seed and start of
/// the head-to-head runner. The search is stopped at `target` when one is
/// given, and otherwise after `budget` calls.
fn search(cfg: &Config, budget: usize, seed: u64, target: Option<f64>) -> Trace {
    let n = cfg.n_points;
    let pot = Tip4pCluster::new(n);
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(3));
    let start = random_tip4p_cluster(n, &mut rng);
    let mut ledger = Ledger::new(budget);
    if let Some(limit) = target {
        ledger.stop_below(limit);
    }
    let mut opt = WarmLbfgs::default();
    let mut quenches = Vec::new();
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
        quenches.push((led.spent(), f.to_bits()));
        (f, xr)
    };
    let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<ndarray::Array1<f64>> {
        if !led.charge() {
            return None;
        }
        Some(pot.value_and_gradient(x).1)
    };
    let mut bias = BasinBias::new(
        ClusterFingerprint::of_config(cfg, &start),
        cfg.merge_radius,
        cfg.bias_height,
        cfg.bias_gamma,
    );
    let mut checkpoint = |_: ChainCheckpoint<'_>| CheckpointAction::Continue;
    let out = run_with_bias_at_checkpoints(
        cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut bias,
        &mut rng,
        1_000,
        &mut checkpoint,
    )
    .expect("finite start");
    Trace {
        quenches,
        charged: ledger.spent(),
        hops: out.hops,
        best: out.best.to_bits(),
    }
}

/// The test runs seed 11030 of four-point transferable intermolecular
/// potential (TIP4P) water at n=14 against the two own-surface references
/// of the head-to-head, each with its 0.01 kJ/mol hit tolerance.
/// Neither reference is reached. Instead, the chain stalls at -574.978
/// kJ/mol after 1250 hops, and the two targets sit just inside and just
/// outside one hopping temperature, 8 kJ/mol, of that minimum. Without a
/// target, a run walks the same path. The chain moves by translation and
/// rotation alone: with the cycle reversal, it never stalls inside the
/// budget and it reaches the upper reference.
#[test]
fn two_unreached_references_give_one_water_trajectory() {
    let cfg = translation_and_rotation(14);
    let lower = search(&cfg, 200_000, 11030, Some(-582.99110 + 0.01));
    let upper = search(&cfg, 200_000, 11030, Some(-582.9744786324 + 0.01));
    assert!(
        lower.hops > 1_250,
        "no stall inside the budget: {} hops",
        lower.hops
    );
    assert_eq!(lower.charged, 200_000);
    assert_eq!(upper.charged, 200_000);
    let diverged = lower
        .quenches
        .iter()
        .zip(&upper.quenches)
        .position(|(a, b)| a != b);
    assert_eq!(diverged, None, "trajectories part at quench {diverged:?}");
    assert_eq!(lower, upper);
    assert_eq!(lower, search(&cfg, 200_000, 11030, None));
}

/// The search must stop at its first hit on a reference it reaches, and
/// that reference must do nothing else. Up to that hit, the trajectory
/// matches the one an unreached reference gives.
#[test]
fn a_reached_reference_only_ends_the_water_trajectory() {
    let cfg = Config::for_tip4p(8);
    let reached = search(&cfg, 200_000, 11000, Some(-305.5098595 + 0.01));
    let unreached = search(&cfg, 200_000, 11000, Some(-305.5098595 - 1.0));
    assert!(reached.charged < 200_000, "the reference was not reached");
    assert!(f64::from_bits(reached.best) <= -305.5098595 + 0.01);
    assert_eq!(unreached.charged, 200_000);
    assert!(reached.quenches.len() <= unreached.quenches.len());
    assert_eq!(
        reached.quenches[..],
        unreached.quenches[..reached.quenches.len()]
    );
    assert_eq!(unreached, search(&cfg, 200_000, 11000, None));
}
