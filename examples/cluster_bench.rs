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

use anneal_core::methods::cluster_hopping::{Config, Keying, Ledger, MoveLibrary};
use anneal_core::methods::cluster_search::{
    Encounter, first_encounter, median_encounter, search, verify,
};
use anneal_core::methods::csa_cluster::{self, BankConfig};
use anneal_core::potentials::{PairKind, PairPotential};
use anneal_core::structure::{cna, ptm, ptm_fractions};

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

    let opts: Vec<&str> = args
        .get(5)
        .map(|v| v.split(',').collect())
        .unwrap_or_default();
    cfg.allocate_moves = opts.contains(&"thompson");
    cfg.return_screen = opts.contains(&"rscreen");
    cfg.adaptive_screen = opts.contains(&"aq");
    cfg.probe_screen = opts.contains(&"probe");
    cfg.track_funnels = opts.contains(&"funnel");
    cfg.delayed_acceptance = opts.contains(&"da");
    let selected: Vec<MoveLibrary> = [
        ("reseed", MoveLibrary::Reseed),
        ("twin", MoveLibrary::Twin),
        ("learn", MoveLibrary::LearnedReseed),
    ]
    .into_iter()
    .filter_map(|(name, library)| opts.contains(&name).then_some(library))
    .collect();
    assert!(selected.len() <= 1, "select at most one move library");
    if let Some(library) = selected.into_iter().next() {
        cfg.move_library = library;
    }
    if let Ok(v) = std::env::var("SCREEN_STEPS") {
        cfg.screen_steps = v.parse().unwrap_or(25);
    }
    if let Ok(v) = std::env::var("TEMP") {
        cfg.temperature = v.parse().unwrap_or(0.8);
    }
    if let Ok(v) = std::env::var("QUENCH_WARMUP") {
        cfg.quench_warmup = v.parse().unwrap_or(4);
    }
    if let Ok(v) = std::env::var("QUENCH_CONF") {
        cfg.quench_confidence = v.parse().unwrap_or(2.0);
    }
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

    // The bank arm, with the next start chosen by expected improvement over
    // morphology rather than by a round robin.
    let use_bank = opts.contains(&"bank");
    let bank_cfg = BankConfig {
        capacity: 30,
        slice: budget / 400,
        seeding: 30,
        dcut_floor: 0.4,
        mix_fraction: 0.5,
        mix_images: 7,
        random_images: 0,
        deadlock_iters: 3,
        deadlock_inject: 0,
        acquisition: opts.contains(&"ei"),
    };
    if use_bank {
        println!(
            "  bank of {} , slice {}, acquisition {}",
            bank_cfg.capacity, bank_cfg.slice, bank_cfg.acquisition
        );
    }

    let mut solved = 0usize;
    let mut encounters: Vec<Encounter> = Vec::new();
    let mut deepest = f64::INFINITY;
    let mut total_charged = 0usize;
    let mut total_hops = 0usize;
    for seed in seed0..(seed0 + seeds) {
        let mut ledger = Ledger::new(budget);
        // The whole of the plumbing, in one call: the relaxation, the charged
        // gradient and the convergence count all come from the crate now, so
        // this example cannot quietly run a different potential or a different
        // relaxation from any other caller.
        let (out, stats) = if use_bank {
            let mut opt2 = anneal_core::methods::warm_lbfgs::WarmLbfgs::default();
            let mut relax = |led: &mut Ledger, x: ndarray::ArrayView1<f64>, iters: usize| {
                opt2.forget();
                let (f, xr, _) = opt2.minimize(x, iters, |v| {
                    if !led.charge() {
                        return None;
                    }
                    Some(pot.value_and_gradient(v))
                });
                (f, xr)
            };
            let b = csa_cluster::run(
                &cfg,
                &bank_cfg,
                &mut ledger,
                &mut relax,
                None,
                anneal_core::methods::csa_cluster::spectrum_distance(n),
                seed,
            );
            println!(
                "      bank: {} slices, {} morphologies, Dcut {:.3} -> {:.3}, {} mixes",
                b.slices, b.morphologies, b.dcut.0, b.dcut.1, b.mixes
            );
            (
                anneal_core::methods::cluster_hopping::Outcome {
                    best: b.best,
                    best_state: b.best_state,
                    hops: b.hops,
                    basins: b.basins,
                    // Carried, or the encounter time censors every run of this
                    // arm including the ones that found the answer.
                    improvements: b.improvements.clone(),
                    ..Default::default()
                },
                anneal_core::methods::cluster_search::RelaxStats::default(),
            )
        } else {
            search(&pot, &cfg, &mut ledger, seed)
        };

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
        // What morphology the run actually ended in, which a success count
        // does not say. Cheap enough to do once per seed on the answer.
        let morphology = out.best_state.as_ref().map(|st| {
            let c = cna(
                st.view(),
                n,
                1.39 * pot.kind().r_min() / 2.0_f64.powf(1.0 / 6.0),
            );
            let f = ptm_fractions(st.view(), n, 0.12);
            // The residual distribution, because a classification that reports
            // nothing is either a structure with no order or a cutoff set in
            // the wrong place, and the fractions alone cannot say which.
            let mut r: Vec<f64> = ptm(st.view(), n, f64::INFINITY)
                .iter()
                .map(|m| m.rmsd)
                .collect();
            r.sort_by(|a, b| a.partial_cmp(b).unwrap());
            format!(
                "cna 555 {:.3} 421 {:.3} 422 {:.3} | ptm fcc {:.2} hcp {:.2} ico {:.2} other {:.2}",
                c.fraction((5, 5, 5)),
                c.fraction((4, 2, 1)),
                c.fraction((4, 2, 2)),
                f[0],
                f[1],
                f[2],
                f[4]
            ) + &format!(
                " | rmsd min {:.3} median {:.3}",
                r.first().copied().unwrap_or(f64::NAN),
                r.get(r.len() / 2).copied().unwrap_or(f64::NAN)
            )
        });
        let hit = reference.map(|r| out.best < r + 1e-4).unwrap_or(false);
        if hit {
            solved += 1;
        }
        // The work to first reach the published minimum, which is the
        // statistic worth comparing. A run that never reached it contributes a
        // lower bound rather than being dropped.
        if let Some(r) = reference {
            encounters.push(first_encounter(&out, r, 1e-4, ledger.spent()));
        }
        deepest = deepest.min(out.best);
        total_charged += ledger.spent();
        total_hops += out.hops;
        if let Some((s1, s1r, s2, s2r)) = out.delayed {
            println!(
                "    delayed: stage1 {s1} rejected {s1r} ({:.3}), stage2 {s2} rejected {s2r} ({:.3})",
                s1r as f64 / s1.max(1) as f64,
                s2r as f64 / s2.max(1) as f64
            );
        }
        for (name, draws, accepts, best) in &out.arms {
            if *draws > 0 {
                println!(
                    "    arm {name:<14} draws {draws:>7}  accepts {accepts:>7} ({:.3})  best {:.4}",
                    *accepts as f64 / *draws as f64,
                    best
                );
            }
        }
        println!(
            "  seed {seed}: best {:.6}  hops {}  basins {}  relaxed {}/{}  sym {}/{:.2}  \
             charged screen {} full {} check {} ({:.0}% screen)  accept {:.3}  qsteps {:.1}  probe {} at {:.1} err {:.4}  verified {}{}",
            out.best,
            out.hops,
            out.basins,
            stats.converged,
            stats.total(),
            out.symmetrised.0,
            out.symmetrised.1,
            stats.screen_charged,
            stats.full_charged,
            stats.check_charged,
            100.0 * stats.screen_share(),
            out.accepted as f64 / out.hops.max(1) as f64,
            stats.screen_steps_taken as f64 / stats.screens.max(1) as f64,
            stats.probe_stops,
            stats.probe_steps as f64 / stats.probe_stops.max(1) as f64,
            stats.probe_error / stats.probe_stops.max(1) as f64,
            verified
                .map(|(e, gmax)| format!("{e:.6} |g| {gmax:.1e}"))
                .unwrap_or_else(|| "NO STATE".into()),
            if hit { "  SOLVED" } else { "" }
        );
        if let Some(m) = morphology {
            println!("      {m}");
        }
    }
    println!(
        "{solved}/{seeds} solved, deepest {deepest:.6}   mean hops {}, force per hop {}",
        total_hops / seeds.max(1) as usize,
        total_charged / total_hops.max(1)
    );
    if let Some(r) = reference {
        println!("gap to reference {:+.6}", deepest - r);
    }
    if !encounters.is_empty() {
        let found: Vec<usize> = encounters
            .iter()
            .filter(|e| e.found())
            .map(|e| e.charged())
            .collect();
        let censored = encounters.len() - found.len();
        match median_encounter(&encounters) {
            Some(m) => println!(
                "first encounter: median {m} charged evaluations \
                 ({} reached, {censored} censored)",
                found.len()
            ),
            None => println!(
                "first encounter: no median, {} of {} runs censored; \
                 the median has not been observed",
                censored,
                encounters.len()
            ),
        }
        if !found.is_empty() {
            let lo = found.iter().copied().min().unwrap_or(0);
            let hi = found.iter().copied().max().unwrap_or(0);
            println!("  observed encounters span {lo} to {hi}");
        }
    }
}
