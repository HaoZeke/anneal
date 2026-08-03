//! End-to-end cluster search on the crate's own machinery.
//!
//! Runs the Rust driver against a Lennard-Jones cluster under a charged ledger
//! and reports whether it reaches the published global minimum. The point is to
//! close the loop: everything else in this crate is checked against a unit test
//! or a synthetic spectrum, and none of that says the search finds the answer.
//!
//! Usage: `cargo run --release --example lj_cluster_search -- <n> <budget> <seeds>`

use anneal_core::methods::cluster_hopping::{optimize, Config, Ledger};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use ndarray::{Array1, ArrayView1};

/// Lennard-Jones value and gradient in reduced units, no cutoff.
fn lj(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut e = 0.0;
    let mut g = Array1::zeros(x.len());
    for i in 0..n {
        for j in (i + 1)..n {
            let d = [
                x[3 * i] - x[3 * j],
                x[3 * i + 1] - x[3 * j + 1],
                x[3 * i + 2] - x[3 * j + 2],
            ];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let inv2 = 1.0 / r2;
            let inv6 = inv2 * inv2 * inv2;
            let inv12 = inv6 * inv6;
            e += 4.0 * (inv12 - inv6);
            let coef = 24.0 * inv2 * (2.0 * inv12 - inv6);
            for k in 0..3 {
                g[3 * i + k] -= coef * d[k];
                g[3 * j + k] += coef * d[k];
            }
        }
    }
    (e, g)
}

/// Value and gradient, charged to the ledger, or `None` when it is spent.
fn charged(led: &mut Ledger, x: ArrayView1<f64>) -> Option<(f64, Array1<f64>)> {
    if !led.charge() {
        return None;
    }
    Some(lj(x))
}

/// Published global minima, for reporting only; nothing steers by these.
fn reference(n: usize) -> Option<f64> {
    Some(match n {
        13 => -44.326801,
        38 => -173.928427,
        55 => -279.248470,
        75 => -397.492331,
        98 => -543.665361,
        _ => return None,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(38);
    let budget: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(400_000);
    let seeds: u64 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(8);

    let reference = reference(n);
    println!(
        "LJ{n}, budget {budget} charged evaluations, {seeds} seeds{}",
        reference
            .map(|r| format!(", reference {r:.6}"))
            .unwrap_or_default()
    );

    // The temperature and step come from Wales and Doye's protocol for basin
    // hopping on the quenched surface, a reduced temperature of 0.8 and a step
    // between 0.36 and 0.40, rather than from tuning here.
    let mut cfg = Config::for_cluster(n);
    // Keying on shape rather than on the descriptor, so the merge threshold is
    // a length. Enabled by the fourth argument so both are measurable.
    // Mechanisms named on the command line, so each is measurable against the
    // others rather than all arriving at once.
    let opts: Vec<&str> = args.get(4).map(|v| v.split(',').collect()).unwrap_or_default();
    cfg.shape_keyed = opts.contains(&"shape");
    cfg.budget_window = opts.contains(&"bfwt");
    cfg.allocate_moves = opts.contains(&"thompson");
    cfg.adaptive_height = opts.contains(&"height");
    cfg.anneal_diversity = opts.contains(&"csa");
    cfg.path_on_stall = opts.contains(&"path");
    cfg.return_screen = opts.contains(&"rscreen");
    if opts.contains(&"pt") {
        // A ladder sharing one budget, not four budgets. The comparison is
        // against a single chain at the same total cost.
        cfg.replicas = 4;
        println!("  replica exchange: {} chains, swap every {} hops, top x{}",
                 cfg.replicas, cfg.swap_period, cfg.ladder_top);
    }
    // The deposit height matters only now that basins are revisited: at 33
    // revisits a height of 0.25 accumulates to about 8, against escape gaps
    // measured at 0.09 for the cheapest and 0.18 at the tenth percentile.
    if let Ok(h) = std::env::var("BIAS_HEIGHT") {
        if let Ok(v) = h.parse::<f64>() {
            cfg.bias_height = v;
            println!("  bias height {v}");
        }
    }
    if !opts.is_empty() {
        println!("  mechanisms: {}", opts.join(", "));
    }
    if cfg.shape_keyed {
        // A length now, not a number in descriptor space: two structures whose
        // atoms can be brought within this of each other by a permutation and
        // a rigid motion are the same basin.
        cfg.merge_radius = 0.2;
        println!("  keying on IRA shape distance, merge radius {} (a length)", cfg.merge_radius);
    }

    let mut solved = 0usize;
    let mut deepest = f64::INFINITY;
    let mut total_hops = 0usize;
    let mut total_charged = 0usize;
    for seed in 0..seeds {
        let mut ledger = Ledger::new(budget);
        // The driver owns the search; the numerics under it are the caller's.
        // A hand-rolled steepest descent with backtracking cost 830 charged
        // evaluations per hop here against about 79 for a quasi-Newton
        // relaxation, so a three million unit budget bought a few thousand
        // hops rather than tens of thousands, and the search failed for want
        // of relaxations rather than for want of a mechanism.
        // Convergence is counted, not assumed. A driver on the quenched
        // landscape is only on it if its relaxations reach minima; one that
        // stops at the iteration cap is hopping between arbitrary points and
        // every mechanism above it is acting on noise.
        let mut converged = 0usize;
        let mut capped = 0usize;
        let mut opt = WarmLbfgs::default();
        let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
            // Curvature is not carried between relaxations: measured on this
            // problem, retaining it across a structural change costs more than
            // it saves.
            opt.forget();
            let (f, xr, _) = opt.minimize(x, iters, |v| charged(led, v));
            let (_, g) = lj(xr.view());
            if g.iter().fold(0.0_f64, |a, v| a.max(v.abs())) < 1e-5 {
                converged += 1;
            } else {
                capped += 1;
            }
            (f, xr)
        };
        let out = optimize(&cfg, &mut ledger, &mut relax, seed);

        // The reported value is checked against a fresh evaluation of the
        // structure it claims to come from, off the ledger and outside the
        // driver. A search that reports a number its own answer does not have
        // is the failure worth catching, and nothing else here would catch it.
        let verified = match out.best_state.as_ref() {
            Some(x) => {
                assert_eq!(
                    x.len(),
                    3 * n,
                    "seed {seed} returned {} coordinates for {n} points",
                    x.len()
                );
                let (e, g) = lj(x.view());
                let gmax = g.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
                assert!(
                    (e - out.best).abs() < 1e-6,
                    "seed {seed} reported {:.6} but its structure is {:.6}",
                    out.best,
                    e
                );
                Some((e, gmax))
            }
            None => None,
        };
        let hit = reference.map(|r| out.best < r + 1e-4).unwrap_or(false);
        if hit {
            solved += 1;
        }
        deepest = deepest.min(out.best);
        total_hops += out.hops;
        total_charged += ledger.spent();
        println!(
            "  seed {seed}: best {:.6}  hops {}  screened {}  charged {}  \
             basins {} ({:.1} hops each)  returned {}  \
             swaps {}/{}  paths {} improved {} gain {:.3}  \
             relaxed {converged}/{} converged  verified {}{}",
            out.best,
            out.hops,
            out.screened_out,
            ledger.spent(),
            out.basins,
            out.hops as f64 / out.basins.max(1) as f64,
            out.returned,
            out.swaps_accepted,
            out.swaps_tried,
            out.paths,
            out.path_improvements,
            out.path_gain,
            converged + capped,
            verified
                .map(|(e, gmax)| format!("{e:.6} |g| {gmax:.1e}"))
                .unwrap_or_else(|| "NO STATE".into()),
            if hit { "  SOLVED" } else { "" }
        );
    }
    // Both counts, since a force budget and a hop budget are different
    // contests and the literature reports hops.
    println!(
        "{solved}/{seeds} solved, deepest {deepest:.6}   \
         mean hops {:.0}, force per hop {:.0}",
        total_hops as f64 / seeds as f64,
        total_charged as f64 / total_hops.max(1) as f64
    );
    if let Some(r) = reference {
        println!("gap to reference {:+.6}", deepest - r);
    }
}
