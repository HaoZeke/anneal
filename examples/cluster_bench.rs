//! Cluster global optimisation across more than one potential.
//!
//! One benchmark family is not a result. This runs the same driver, the same
//! charged ledger and the same mechanisms against Lennard-Jones and Morse
//! clusters, so a mechanism that only helps on one of them is visible as such.
//!
//! Morse is the useful second family because its range parameter is a dial on
//! how hard the landscape is, with the same code path either side. Doye and
//! Wales, *Structural consequences of the range of the interatomic potential*,
//! survey 20 to 80 points across that dial; the short-range cases are the ones
//! the literature reports as out of reach for unbiased methods.
//!
//! Usage:
//! `cargo run --release --example cluster_bench -- <potential> <n> <budget> <seeds> [mechanisms]`
//! where `<potential>` is `lj`, or `morse:RHO` such as `morse:6`.

use anneal_core::methods::cluster_hopping::{
    optimize_with_gradient, Config, Keying, Ledger, Outcome,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use ndarray::{Array1, ArrayView1};

/// A pair potential in reduced units, as value and gradient of the whole
/// configuration.
#[derive(Debug, Clone, Copy)]
enum Potential {
    /// `4 (r^-12 - r^-6)`, well depth 1 at `r = 2^(1/6)`.
    LennardJones,
    /// `e^{rho (1 - r)} (e^{rho (1 - r)} - 2)`, well depth 1 at `r = 1`.
    ///
    /// Doye and Wales' form with `epsilon = 1` and `r_0 = 1`, so `rho` is the
    /// only parameter and sets the range of the force.
    Morse { rho: f64 },
}

impl Potential {
    fn parse(s: &str) -> Option<Self> {
        if s == "lj" {
            return Some(Potential::LennardJones);
        }
        let rho = s.strip_prefix("morse:")?.parse().ok()?;
        Some(Potential::Morse { rho })
    }

    fn name(&self) -> String {
        match self {
            Potential::LennardJones => "LJ".into(),
            Potential::Morse { rho } => format!("Morse rho={rho}"),
        }
    }

    /// Equilibrium pair separation, which sets the container and the closest
    /// two points may be placed when a structure is seeded.
    fn r_min(&self) -> f64 {
        match self {
            Potential::LennardJones => 2.0_f64.powf(1.0 / 6.0),
            Potential::Morse { .. } => 1.0,
        }
    }

    /// Value and gradient of the configuration.
    fn eval(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
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
                // `coef` is `(dV/dr) / r`, so the force follows by multiplying
                // the separation vector.
                let (v, coef) = match self {
                    Potential::LennardJones => {
                        let inv2 = 1.0 / r2;
                        let inv6 = inv2 * inv2 * inv2;
                        let inv12 = inv6 * inv6;
                        (4.0 * (inv12 - inv6), -24.0 * inv2 * (2.0 * inv12 - inv6))
                    }
                    Potential::Morse { rho } => {
                        let r = r2.sqrt();
                        let a = (rho * (1.0 - r)).exp();
                        // V = a^2 - 2a, dV/dr = 2 rho (a - a^2).
                        (a * (a - 2.0), 2.0 * rho * (a - a * a) / r)
                    }
                };
                e += v;
                for k in 0..3 {
                    g[3 * i + k] += coef * d[k];
                    g[3 * j + k] -= coef * d[k];
                }
            }
        }
        (e, g)
    }
}

/// Published global minima, for reporting only; nothing steers by these.
///
/// Lennard-Jones from the Cambridge Cluster Database, Morse from Doye and
/// Wales' table for the same database, in units of the pair well depth.
fn reference(p: Potential, n: usize) -> Option<f64> {
    match p {
        Potential::LennardJones => Some(match n {
            13 => -44.326801,
            38 => -173.928427,
            55 => -279.248470,
            75 => -397.492331,
            98 => -543.665361,
            _ => return None,
        }),
        Potential::Morse { rho } => {
            let r = (rho * 2.0).round() as i64;
            // Taken from the bold entries of Doye and Wales' table, which are
            // the global minima; the table also lists competing minima for the
            // same size and range, and the global one changes structure along
            // a row. Two values here were first taken from non-bold rows, so
            // the success bar was set below the global minimum: at rho = 6 and
            // N = 38 the minimum is 38E at -157.477108, not 38D at -157.406902,
            // and at rho = 14 and N = 55 it is 55C at -220.646208, not 55B at
            // -213.523774. The runs found the true minima and were scored
            // against the wrong ones.
            Some(match (r, n) {
                (12, 38) => -157.477108,
                (12, 55) => -250.286609,
                (12, 75) => -351.472365,
                (20, 38) => -145.849817,
                (20, 55) => -225.814286,
                (20, 75) => -322.643558,
                (28, 38) => -144.321054,
                (28, 55) => -220.646208,
                (28, 75) => -318.407330,
                _ => return None,
            })
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pot = args
        .get(1)
        .and_then(|v| Potential::parse(v))
        .unwrap_or(Potential::LennardJones);
    let n: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(38);
    let budget: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(400_000);
    let seeds: u64 = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(8);
    let seed0: u64 = std::env::var("SEED_OFFSET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let reference = reference(pot, n);
    println!(
        "{} N={n}, budget {budget} charged evaluations, {seeds} seeds{}",
        pot.name(),
        reference
            .map(|r| format!(", reference {r:.6}"))
            .unwrap_or_else(|| ", no published reference".into())
    );

    let mut cfg = Config::for_cluster(n);
    // The move library and the container are set from the potential's own
    // length scale rather than from Lennard-Jones numbers. A Morse cluster at
    // rho = 14 sits at r_0 = 1 against 1.12 for Lennard-Jones, and a container
    // or a minimum separation carried over from the other potential is a
    // different problem.
    let scale = pot.r_min() / 2.0_f64.powf(1.0 / 6.0);
    cfg.min_separation *= scale;
    cfg.container *= scale;

    let opts: Vec<&str> = args.get(5).map(|v| v.split(',').collect()).unwrap_or_default();
    cfg.allocate_moves = opts.contains(&"thompson");
    cfg.return_screen = opts.contains(&"rscreen");
    cfg.contextual_moves = opts.contains(&"ctx");
    cfg.bayes_screen = opts.contains(&"bayes");
    cfg.angular_moves = opts.contains(&"angular");
    cfg.budget_window = opts.contains(&"bfwt");
    if opts.contains(&"sites") {
        cfg.keying = Keying::Sites;
    }
    if opts.contains(&"canon") {
        cfg.keying = Keying::Canonical;
        cfg.merge_radius = 0.3;
    }
    if let Ok(v) = std::env::var("MERGE_RADIUS") {
        if let Ok(r) = v.parse::<f64>() {
            cfg.merge_radius = r;
        }
    }
    if !opts.is_empty() {
        println!("  mechanisms: {}", opts.join(", "));
    }

    let mut solved = 0usize;
    let mut deepest = f64::INFINITY;
    let mut total_charged = 0usize;
    let mut total_hops = 0usize;
    for seed in seed0..(seed0 + seeds) {
        let mut ledger = Ledger::new(budget);
        let mut converged = 0usize;
        let mut capped = 0usize;
        let mut opt = WarmLbfgs::default();
        let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
            opt.forget();
            let (f, xr, _) = opt.minimize(x, iters, |v| {
                if !led.charge() {
                    return None;
                }
                Some(pot.eval(v))
            });
            let (_, g) = pot.eval(xr.view());
            if g.iter().fold(0.0_f64, |a, v| a.max(v.abs())) < 1e-5 {
                converged += 1;
            } else {
                capped += 1;
            }
            (f, xr)
        };
        let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<Array1<f64>> {
            if !led.charge() {
                return None;
            }
            Some(pot.eval(x).1)
        };
        let out: Outcome = optimize_with_gradient(
            &cfg,
            &mut ledger,
            &mut relax,
            // Only the soft-mode escape needs it, and this benchmark does not
            // enable that arm; passing it unconditionally would still be
            // correct but would suggest it is used.
            None,
            seed,
        );
        // The reported value is checked against a fresh evaluation of the
        // structure it claims to come from, off the ledger and outside the
        // driver, exactly as in the Lennard-Jones example.
        let verified = out.best_state.as_ref().map(|x| {
            assert_eq!(x.len(), 3 * n, "seed {seed} returned {} coordinates", x.len());
            let (e, g) = pot.eval(x.view());
            let gmax = g.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            // A minimum, not merely a structure carrying the reported energy.
            assert!(
                gmax < 1e-3,
                "seed {seed} returned a structure with gradient {gmax:.2e}, \
                 which is not a minimum"
            );
            assert!(
                (e - out.best).abs() < 1e-6,
                "seed {seed} reported {:.6} but its structure is {e:.6}",
                out.best
            );
            (e, gmax)
        });
        let hit = reference.map(|r| out.best < r + 1e-4).unwrap_or(false);
        if hit {
            solved += 1;
        }
        deepest = deepest.min(out.best);
        total_charged += ledger.spent();
        total_hops += out.hops;
        println!(
            "  seed {seed}: best {:.6}  hops {}  basins {}  relaxed {converged}/{}  verified {}{}",
            out.best,
            out.hops,
            out.basins,
            converged + capped,
            verified
                .map(|(e, gmax)| format!("{e:.6} |g| {gmax:.1e}"))
                .unwrap_or_else(|| "NO STATE".into()),
            if hit { "  SOLVED" } else { "" }
        );
    }
    println!(
        "{solved}/{seeds} solved, deepest {deepest:.6}   mean hops {}, force per hop {}",
        total_hops / seeds.max(1) as usize,
        total_charged / total_hops.max(1)
    );
    if let Some(r) = reference {
        println!("gap to reference {:+.6}", deepest - r);
    }
}
