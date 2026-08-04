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

use anneal_core::methods::cluster_hopping::{Config, Keying, Ledger};
use anneal_core::methods::cluster_search::{search, verify};
use anneal_core::potentials::{PairKind, PairPotential};

/// The potential named on the command line.
fn parse_potential(spec: &str, n: usize) -> Option<PairPotential> {
    if spec == "lj" {
        return Some(PairPotential::lennard_jones(n));
    }
    let rho: f64 = spec.strip_prefix("morse:")?.parse().ok()?;
    Some(PairPotential::morse(n, rho))
}

/// A name for the report, from what the caller asked for.
fn potential_name(spec: &str) -> String {
    if spec == "lj" {
        "LJ".into()
    } else {
        spec.replace("morse:", "Morse rho=")
    }
}

/// Published global minima, for reporting only; nothing steers by these.
///
/// Lennard-Jones from the Cambridge Cluster Database, Morse from Doye and
/// Wales' table for the same database, in units of the pair well depth.
fn reference(p: PairKind, n: usize) -> Option<f64> {
    match p {
        PairKind::LennardJones => Some(match n {
            13 => -44.326801,
            38 => -173.928427,
            55 => -279.248470,
            75 => -397.492331,
            98 => -543.665361,
            _ => return None,
        }),
        PairKind::Morse { rho } => {
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
    let spec = args.get(1).cloned().unwrap_or_else(|| "lj".into());
    let n: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(38);
    let pot = match parse_potential(&spec, n) {
        Some(p) => p,
        None => {
            eprintln!("unknown potential {spec}; expected lj or morse:RHO");
            std::process::exit(2);
        }
    };
    let budget: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(400_000);
    let seeds: u64 = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(8);
    let seed0: u64 = std::env::var("SEED_OFFSET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let reference = reference(pot.kind(), n);
    println!(
        "{} N={n}, budget {budget} charged evaluations, {seeds} seeds{}",
        potential_name(&spec),
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
    let scale = pot.kind().r_min() / 2.0_f64.powf(1.0 / 6.0);
    cfg.min_separation *= scale;
    cfg.container *= scale;

    let opts: Vec<&str> = args.get(5).map(|v| v.split(',').collect()).unwrap_or_default();
    cfg.allocate_moves = opts.contains(&"thompson");
    cfg.return_screen = opts.contains(&"rscreen");
    cfg.contextual_moves = opts.contains(&"ctx");
    cfg.bayes_screen = opts.contains(&"bayes");
    cfg.angular_moves = opts.contains(&"angular");
    // Symmetrise onto the structure's own approximate symmetry when stuck.
    cfg.symmetrise_on_stall = opts.contains(&"sym");
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
        // The whole of the plumbing, in one call: the relaxation, the charged
        // gradient and the convergence count all come from the crate now, so
        // this example cannot quietly run a different potential or a different
        // relaxation from any other caller.
        let (out, stats) = search(&pot, &cfg, &mut ledger, seed);

        // The reported value is checked against a fresh evaluation of the
        // structure it claims to come from, off the ledger and outside the
        // driver, and the structure has to be a minimum rather than merely
        // carry the right energy.
        let verified = verify(&pot, &out);
        if let Some((e, gmax)) = verified {
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
        }
        let hit = reference.map(|r| out.best < r + 1e-4).unwrap_or(false);
        if hit {
            solved += 1;
        }
        deepest = deepest.min(out.best);
        total_charged += ledger.spent();
        total_hops += out.hops;
        println!(
            "  seed {seed}: best {:.6}  hops {}  basins {}  relaxed {}/{}  sym {}/{:.2}  verified {}{}",
            out.best,
            out.hops,
            out.basins,
            stats.converged,
            stats.total(),
            out.symmetrised.0,
            out.symmetrised.1,
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
