//! End-to-end cluster search on the crate's own machinery.
//!
//! Runs the Rust driver against a Lennard-Jones cluster under a charged ledger
//! and reports whether it reaches the published global minimum. The point is to
//! close the loop: everything else in this crate is checked against a unit test
//! or a synthetic spectrum, and none of that says the search finds the answer.
//!
//! Usage: `cargo run --release --example lj_cluster_search -- <n> <budget> <seeds>`

use anneal_core::methods::cluster_hopping::{optimize, Config, Ledger};
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
    let cfg = Config::for_cluster(n);

    let mut solved = 0usize;
    let mut deepest = f64::INFINITY;
    for seed in 0..seeds {
        let mut ledger = Ledger::new(budget);
        // The driver owns the search; the numerics under it are the caller's,
        // so the relaxation is supplied here. Steepest descent with a
        // backtracking line search, every evaluation charged, stopping the
        // moment the ledger refuses.
        let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
            let mut cur = x.to_owned();
            let (mut f, mut g) = match charged(led, cur.view()) {
                Some(v) => v,
                None => return (f64::INFINITY, cur),
            };
            let mut step = 1e-3;
            for _ in 0..iters {
                let gnorm = g.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
                if gnorm < 1e-8 {
                    break;
                }
                let mut moved = false;
                for _ in 0..20 {
                    let mut trial = cur.clone();
                    for i in 0..trial.len() {
                        trial[i] -= step * g[i];
                    }
                    match charged(led, trial.view()) {
                        None => return (f, cur),
                        Some((ft, gt)) => {
                            if ft < f {
                                cur = trial;
                                f = ft;
                                g = gt;
                                step *= 1.6;
                                moved = true;
                                break;
                            }
                            step *= 0.5;
                        }
                    }
                }
                if !moved {
                    break;
                }
            }
            (f, cur)
        };
        let out = optimize(&cfg, &mut ledger, &mut relax, seed);
        let hit = reference.map(|r| out.best < r + 1e-4).unwrap_or(false);
        if hit {
            solved += 1;
        }
        deepest = deepest.min(out.best);
        println!(
            "  seed {seed}: best {:.6}  hops {}  screened {}  charged {}{}",
            out.best,
            out.hops,
            out.screened_out,
            ledger.spent(),
            if hit { "  SOLVED" } else { "" }
        );
    }
    println!("{solved}/{seeds} solved, deepest {deepest:.6}");
    if let Some(r) = reference {
        println!("gap to reference {:+.6}", deepest - r);
    }
}
