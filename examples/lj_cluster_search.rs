//! End-to-end cluster search on the crate's own machinery.
//!
//! Runs the Rust driver against a Lennard-Jones cluster under a charged ledger
//! and reports whether it reaches the published global minimum. The point is to
//! close the loop: everything else in this crate is checked against a unit test
//! or a synthetic spectrum, and none of that says the search finds the answer.
//!
//! Usage: `cargo run --release --example lj_cluster_search -- <n> <budget> <seeds>`

use anneal_core::methods::csa_cluster::{self, BankConfig};
use anneal_core::methods::cluster_hopping::{
    optimize_with_gradient, Config, Keying, Ledger, Outcome,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::terminate::Terminator;
use ndarray::{Array1, ArrayView1};

#[cfg(feature = "ira")]
use anneal_core::shape::IraMetric;

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
    // Where the seed numbering starts, so a campaign can put one seed on each
    // core instead of walking them in one process. Seeds are the same runs
    // either way: seed 5 of one process and seed 5 of another are identical.
    let seed0: u64 = std::env::var("SEED_OFFSET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

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
    let mut opts: Vec<&str> = args.get(4).map(|v| v.split(',').collect()).unwrap_or_default();
    // The working Python driver (askmc_hopping) is Thompson over moves, the
    // budget-window temperature, and the per-basin bias the crate always
    // carries. Naming that stack keeps the measurement comparable without
    // assembling the flags from memory each time.
    if opts.contains(&"askmc") {
        opts.extend_from_slice(&["thompson", "bfwt"]);
    }
    cfg.shape_keyed = opts.contains(&"shape");
    cfg.budget_window = opts.contains(&"bfwt");
    cfg.allocate_moves = opts.contains(&"thompson");
    cfg.adaptive_height = opts.contains(&"height");
    cfg.anneal_diversity = opts.contains(&"csa");
    cfg.path_on_stall = opts.contains(&"path");
    cfg.return_screen = opts.contains(&"rscreen");
    cfg.minima_hopping = opts.contains(&"mh");
    cfg.escape_on_stall = opts.contains(&"climb");
    // The radius read off the search's own step length rather than swept.
    cfg.calibrate_radius = opts.contains(&"calib");
    // The walker restarted, the landscape memory kept.
    cfg.restart_on_stall = opts.contains(&"restart");
    // Wales and Doye's angular move on the worst-bound point.
    cfg.angular_moves = opts.contains(&"angular");
    // The funnel forbidden rather than penalised.
    cfg.tabu_on_stall = opts.contains(&"tabu");
    // The relaxation decision taken under a posterior.
    cfg.bayes_screen = opts.contains(&"bayes");
    // Acceptance against the density of minima rather than against the energy.
    cfg.flat_histogram = opts.contains(&"flat");
    // The temperature taken from the entropy the run measures for itself.
    cfg.statistical_temperature = opts.contains(&"stemp");
    if let Ok(v) = std::env::var("FLAT_QUANTILE") {
        if let Ok(q) = v.parse::<f64>() {
            cfg.flat_quantile = q;
            println!("  flat below the {q} quantile of each sweep");
        }
    }
    // The move chosen from the structure the chain is standing on.
    cfg.contextual_moves = opts.contains(&"ctx");
    // Basins keyed on how well each point is bound.
    if opts.contains(&"sites") {
        cfg.keying = Keying::Sites;
        println!("  keying on sorted site energies");
    }
    if opts.contains(&"canon") {
        // A length in coordinate space now: two structures whose points can be
        // brought within this root-mean-square of each other by a permutation
        // and a rigid motion are one basin.
        cfg.keying = Keying::Canonical;
        cfg.merge_radius = 0.3;
        println!("  keying on a canonical order, merge radius {}", cfg.merge_radius);
    }
    if opts.contains(&"pt") {
        // A ladder sharing one budget, not four budgets. The comparison is
        // against a single chain at the same total cost.
        cfg.replicas = 4;
        cfg.bias_by_rung = opts.contains(&"rungbias");
        println!("  replica exchange: {} chains, swap every {} hops, top x{}",
                 cfg.replicas, cfg.swap_period, cfg.ladder_top);
    }
    // The deposit height matters only now that basins are revisited: at 33
    // revisits a height of 0.25 accumulates to about 8, against escape gaps
    // measured at 0.09 for the cheapest and 0.18 at the tenth percentile.
    // How coarse a basin is, which decides how deep the bias gets anywhere.
    //
    // Traced at 75 points, a hundred thousand hops register about three
    // thousand basins, so each one collects around thirty deposits and the
    // icosahedral funnel never fills: a run can spend ninety-eight thousand
    // hops inside it without a single improvement. A radius that merges the
    // variants of a funnel into one basin puts the same deposits in one place.
    if let Ok(v) = std::env::var("MERGE_RADIUS") {
        if let Ok(r) = v.parse::<f64>() {
            cfg.merge_radius = r;
            println!("  merge radius {r}");
        }
    }
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

    // The bank arm. Runs the same chains under the same total budget, with
    // where-to-start-next and what-to-keep decided by the diversity rule
    // rather than by the chain itself.
    let use_bank = opts.contains(&"bank");
    // The slice length is the shape of the method, not a tuning knob. A bank
    // whose slices are long is a handful of medium chains: at a sixteenth of
    // the budget each, a bank of eight saw every member twice and scored 0
    // seeds in 5. Conformational space annealing runs thousands of short
    // perturbations against a bank of tens, so each member is revisited on the
    // order of a hundred times.
    let env = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let capacity = env("BANK_CAPACITY", 30);
    let bank_cfg = BankConfig {
        capacity,
        acquisition: opts.contains(&"acq"),
        slice: env("BANK_SLICE", 3_000),
        seeding: capacity,
        dcut_floor: std::env::var("BANK_DCUT_FLOOR")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.2),
        mix_fraction: std::env::var("BANK_MIX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5),
        mix_images: 7,
    };
    if use_bank {
        #[cfg(not(feature = "ira"))]
        {
            eprintln!(
                "the bank arm needs the `ira` feature (shape distance for Dcut); rebuild with --features ira"
            );
            std::process::exit(2);
        }
        println!(
            "  bank of {} chains, {} charged per slice, Dcut floor {}",
            bank_cfg.capacity, bank_cfg.slice, bank_cfg.dcut_floor
        );
    }

    let mut solved = 0usize;
    let mut deepest = f64::INFINITY;
    let mut total_hops = 0usize;
    let mut total_charged = 0usize;
    for seed in seed0..(seed0 + seeds) {
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
        // The screening pass stopped as soon as its limit is decided.
        let early_stop = opts.contains(&"early");
        let mut early_stopped = 0usize;
        let mut early_saved = 0usize;
        let mut converged = 0usize;
        let mut capped = 0usize;
        let mut opt = WarmLbfgs::default();
        let screen_steps = cfg.screen_steps;
        let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
            // Curvature is not carried between relaxations: measured on this
            // problem, retaining it across a structural change costs more than
            // it saves.
            opt.forget();
            // Early termination applies to the screening pass only.
            //
            // The screen's output is allowed to be unconverged; the full
            // relaxation's is not, because the driver puts it into the chain
            // and every mechanism above assumes the chain stands on a minimum.
            // Stopping the full relaxation early is the same defect that broke
            // the escape controller, where 94 relaxations in 3148 reached a
            // minimum and the curvature it steered by came back negative at a
            // point being treated as one.
            if early_stop && iters <= screen_steps {
                let mut term = Terminator::default();
                let mut cur = x.to_owned();
                let mut f = f64::INFINITY;
                let mut done = 0usize;
                // Four at a time: enough for the ratio estimate to move, small
                // enough that the saving is not given back.
                while done < iters {
                    let take = 4.min(iters - done);
                    let (fi, xi, _) = opt.minimize(cur.view(), take, |v| charged(led, v));
                    f = fi;
                    cur = xi;
                    done += take;
                    term.observe(f);
                    if term.settled_above(led.best) {
                        early_stopped += 1;
                        early_saved += iters - done;
                        break;
                    }
                }
                capped += 1;
                return (f, cur);
            }
            let (f, xr, _) = opt.minimize(x, iters, |v| charged(led, v));
            let (_, g) = lj(xr.view());
            if g.iter().fold(0.0_f64, |a, v| a.max(v.abs())) < 1e-5 {
                converged += 1;
            } else {
                capped += 1;
            }
            (f, xr)
        };
        // The gradient the soft-mode escape needs, charged like everything
        // else: a Lanczos pass is two evaluations per step and the escape
        // must pay for them.
        let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<Array1<f64>> {
            if !led.charge() {
                return None;
            }
            Some(lj(x).1)
        };
        let out = if use_bank {
            #[cfg(feature = "ira")]
            {
                // A shape distance, so Dcut is a length: two structures whose
                // points can be brought within it of each other by a permutation
                // and a rigid motion are the same solution.
                let ira = IraMetric::default();
                let b = csa_cluster::run(
                    &cfg,
                    &bank_cfg,
                    &mut ledger,
                    &mut relax,
                    if cfg.minima_hopping || cfg.escape_on_stall {
                        Some(&mut grad)
                    } else {
                        None
                    },
                    |p, q| ira.distance(p, q),
                    seed,
                );
                println!(
                    "      bank: {} slices, Dcut {:.3} -> {:.3}, {} improved, {} novel, \
                     {} duplicate, {} mixes ({} admitted, {} below both ends), holding {:?}",
                    b.slices,
                    b.dcut.0,
                    b.dcut.1,
                    b.improved,
                    b.novel,
                    b.duplicates,
                    b.mixes,
                    b.mix_admitted,
                    b.mix_below_both,
                    b.bank.iter().map(|e| (e * 100.0).round() / 100.0).collect::<Vec<_>>()
                );
                Outcome {
                    best: b.best,
                    best_state: b.best_state,
                    hops: b.hops,
                    basins: b.basins,
                    ..Outcome::default()
                }
            }
            #[cfg(not(feature = "ira"))]
            {
                unreachable!("bank requires ira; checked at startup")
            }
        } else {
            optimize_with_gradient(
                &cfg,
                &mut ledger,
                &mut relax,
                if cfg.minima_hopping || cfg.escape_on_stall {
                    Some(&mut grad)
                } else {
                    None
                },
                seed,
            )
        };

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
                // The returned structure has to be a minimum, not merely carry
                // the energy that was reported for it. Checking only the energy
                // let an arm return a point with a gradient of 0.31, which the
                // campaign table caught rather than this assertion.
                assert!(
                    gmax < 1e-3,
                    "seed {seed} returned a structure with gradient {gmax:.2e}, \
                     which is not a minimum"
                );
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
        // Where the run got its answer. Printed for the last few improvements
        // only: the early ones are a descent from a random start and say
        // nothing.
        if let Some(r) = reference {
            if let Some((h, _, b, e)) = out
                .improvements
                .iter()
                .find(|(_, _, _, e)| *e < r + 1e-4)
            {
                println!(
                    "      crossed at hop {h} of {} ({:.1}% in), {b} basins, {e:.6}",
                    out.hops,
                    100.0 * *h as f64 / out.hops.max(1) as f64
                );
            } else if let Some((h, _, b, e)) = out.improvements.last() {
                println!(
                    "      last improvement at hop {h} of {} ({:.1}% in), {b} basins, {e:.6}",
                    out.hops,
                    100.0 * *h as f64 / out.hops.max(1) as f64
                );
            }
        }
        deepest = deepest.min(out.best);
        if seed == 0 && out.rungs.len() > 1 {
            for (t, b, en) in &out.rungs {
                println!("      rung T={t:.3}  basins {b:>5}  energy {en:>11.4}");
            }
        }
        total_hops += out.hops;
        total_charged += ledger.spent();
        println!(
            "  seed {seed}: best {:.6}  hops {}  screened {}  charged {}  \
             basins {} ({:.1} hops each)  returned {}  \
             swaps {}/{}  paths {} improved {} gain {:.3}  \
             escape {:.3} thr {:.4} same/known/new {}/{}/{} soft {}/{} lmin {:.4} climbs {} gain {:.2} radius {:.3} step {:.3} restarts {} angular {}/{} R {:.3} tabu {} vetoed {} screen {}/{} expl {} obs {} ctx {:?}  \
             relaxed {converged}/{} converged  early {early_stopped} saved {early_saved}  \
             verified {}{}",
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
            out.escape_scale,
            out.escape_threshold,
            out.visit_counts.0,
            out.visit_counts.1,
            out.visit_counts.2,
            out.soft_crossed,
            out.soft_escapes,
            out.soft_lambda,
            out.stall_escapes,
            out.stall_escape_gain,
            out.merge_radius,
            out.mean_step,
            out.restarts,
            out.angular.1,
            out.angular.0,
            out.angular.2,
            out.tabu.0,
            out.tabu.1,
            out.screen.1,
            out.screen.0,
            out.screen.2,
            out.screen.3,
            out.contextual.0,
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
