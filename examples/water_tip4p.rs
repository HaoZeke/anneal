//! Rigid TIP4P water clusters against the Wales--Hodges putative minima.
//!
//! Usage: `water_tip4p <n> <budget> <seeds> [seed0]`
//!
//! A non-numeric fourth token is an options label and is ignored, so
//! `ttt.sh n budget seeds tip4p` can drive this binary the same way it
//! drives `lj_cluster_search`.

use anneal_core::bias::BasinBias;
use anneal_core::methods::cluster_hopping::{
    ChainCheckpoint, CheckpointAction, ClusterFingerprint, Config, Ledger,
    run_with_bias_at_checkpoints,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{Tip4pCluster, random_tip4p_cluster, wales_hodges_minimum};
use ndarray::ArrayView1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::io::{self, Write};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(6);
    let budget: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(20_000);
    let seeds: u64 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut seed0: u64 = std::env::var("SEED_OFFSET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if let Some(fourth) = args.get(4)
        && let Ok(parsed) = fourth.parse::<u64>()
    {
        seed0 = parsed;
    }

    let pot = Tip4pCluster::new(n);
    let cfg = Config::for_tip4p(n);
    let reference = wales_hodges_minimum(n);
    println!(
        "TIP4P (H2O){n}, budget {budget}, seeds {seed0}..{}",
        seed0 + seeds
    );
    if let Some(r) = reference {
        println!("  Wales--Hodges target {r:.2} kJ/mol");
    }

    let mut solved = 0usize;
    let mut deepest = f64::INFINITY;
    let mut total_hops = 0usize;
    let mut total_charged = 0usize;
    for seed in seed0..(seed0 + seeds) {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(3));
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
            Some(pot.grad(x))
        };
        let mut bias = BasinBias::new(
            ClusterFingerprint::of_config(&cfg, &start),
            cfg.merge_radius,
            cfg.bias_height,
            cfg.bias_gamma,
        );
        let mut checkpoint = |_: ChainCheckpoint<'_>| CheckpointAction::Continue;
        let mut out = run_with_bias_at_checkpoints(
            &cfg,
            start.view(),
            &mut ledger,
            &mut relax,
            Some(&mut grad),
            &mut bias,
            &mut rng,
            1_000,
            &mut checkpoint,
        );
        if let Some(state) = out.best_state.as_ref() {
            let e = pot.eval(state.view());
            out.best = e;
        }
        let hit = reference.map(|r| out.best < r + 0.01).unwrap_or(false);
        if hit {
            solved += 1;
        }
        let first_hit = reference.and_then(|r| {
            out.improvements
                .iter()
                .find(|(_, _, _, e)| *e < r + 0.01)
                .map(|(_, spent, _, _)| *spent)
        });
        deepest = deepest.min(out.best);
        total_hops += out.hops;
        total_charged += ledger.spent();
        println!(
            "  seed {seed}: best {:.6} kJ/mol  wales-hodges {}  first_hit {}  hops {}  charged {}{}",
            out.best,
            if hit { "yes" } else { "no" },
            first_hit
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".into()),
            out.hops,
            ledger.spent(),
            if hit { "  SOLVED" } else { "" }
        );
        println!(
            "{}/1 solved, deepest {:.6}   mean hops {:.0}, force per hop {:.0}",
            usize::from(hit),
            out.best,
            out.hops as f64,
            ledger.spent() as f64 / out.hops.max(1) as f64
        );
        let _ = io::stdout().flush();
    }
    println!(
        "{solved}/{seeds} solved, deepest {deepest:.6}   \
         mean hops {:.0}, force per hop {:.0}",
        total_hops as f64 / seeds.max(1) as f64,
        total_charged as f64 / total_hops.max(1) as f64
    );
    if let Some(r) = reference {
        println!("gap to reference {:+.6}", deepest - r);
    }
}
