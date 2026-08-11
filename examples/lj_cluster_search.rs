//! End-to-end cluster search on the crate's own machinery.
//!
//! Runs the Rust driver against a Lennard-Jones cluster under a charged ledger
//! and reports whether it reaches the published global minimum. The point is to
//! close the loop: everything else in this crate is checked against a unit test
//! or a synthetic spectrum, and none of that says the search finds the answer.
//!
//! Usage: `cargo run --release --example lj_cluster_search -- <n> <budget> <seeds>`

use anneal_core::bias::BasinBias;
use anneal_core::methods::cluster_hopping::{
    ClusterFingerprint, Config, Keying, Ledger, MoveLibrary, Outcome, random_cluster,
    run_with_bias, optimize_with_gradient,
};
use anneal_core::methods::csa_cluster::{self, BankConfig};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::terminate::Terminator;
use ndarray::{Array1, ArrayView1};
use std::io::{self, Write};

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

/// The objective with isotropic noise on the gradient, for the screening pass.
///
/// The basin of attraction of a starting point is a property of the minimiser,
/// not of the landscape alone, so perturbing the descent sends the same
/// starting point to a different minimum. That is the one factor in
/// "perturbation then quench" that no acceptance rule, sampling weight,
/// temperature or bias reaches, and measurement puts the funnel crossing
/// squarely inside it: every crossing observed arrives in a single quench.
///
/// The noise is isotropic and scaled to the gradient's own magnitude, so it
/// carries no information about any structure and cannot encode an answer the
/// way a template library does. It is also dimensionless: `eta` is a fraction
/// of the local gradient, so nothing here is a length or an energy belonging to
/// a particular system.
///
/// Applied to the screening pass only. The full relaxation stays clean, because
/// the driver puts its output into the chain and every mechanism above assumes
/// the chain stands on a minimum.
fn charged_noisy<R: rand::Rng + ?Sized>(
    led: &mut Ledger,
    x: ArrayView1<f64>,
    eta: f64,
    rng: &mut R,
) -> Option<(f64, Array1<f64>)> {
    if !led.charge() {
        return None;
    }
    let (e, mut g) = lj(x);
    let norm = g.iter().fold(0.0_f64, |a, v| a + v * v).sqrt();
    if norm > 0.0 && eta > 0.0 {
        let scale = eta * norm / (g.len() as f64).sqrt();
        for v in g.iter_mut() {
            let u1: f64 = rng.random::<f64>().max(1e-12);
            let u2: f64 = rng.random::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            *v += scale * z;
        }
    }
    Some((e, g))
}

/// Gradient of the pair energy with respect to the listed atoms only.
///
/// Computes the k rows of the interaction that involve a moved atom: k*n pair
/// terms against n(n-1)/2 for the full system, which is the fraction charged.
/// Frozen atoms contribute forces to the moved ones; their own entries stay
/// zero, which is the frozen-environment constraint.
fn lj_partial_grad(x: ndarray::ArrayView1<f64>, moved: &[usize]) -> Array1<f64> {
    let n = x.len() / 3;
    let mut g = Array1::zeros(x.len());
    for &i in moved {
        for j in 0..n {
            if j == i {
                continue;
            }
            let d = [
                x[3 * i] - x[3 * j],
                x[3 * i + 1] - x[3 * j + 1],
                x[3 * i + 2] - x[3 * j + 2],
            ];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let inv2 = 1.0 / r2;
            let inv6 = inv2 * inv2 * inv2;
            let inv12 = inv6 * inv6;
            let coef = 24.0 * inv2 * (2.0 * inv12 - inv6);
            for k in 0..3 {
                g[3 * i + k] -= coef * d[k];
            }
        }
    }
    g
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
    // Random structure search: the evaluation-matched baseline. Random starts,
    // full quenches, nothing else, so every stack above it is measured against
    // what pure sampling buys at the same number of charged evaluations.
    if std::env::args()
        .nth(4)
        .map(|v| v.contains("rss"))
        .unwrap_or(false)
    {
        let mut solved = 0usize;
        for seed in seed0..(seed0 + seeds) {
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(
                seed.wrapping_mul(0x9E3779B9).wrapping_add(7),
            );
            let mut ledger = Ledger::new(budget);
            let mut opt = WarmLbfgs::default();
            let mut best = f64::INFINITY;
            let mut relaxes = 0usize;
            while ledger.remaining() > 0 {
                let x0 =
                    anneal_core::methods::cluster_hopping::random_cluster(n, 0.7, 0.5, &mut rng);
                opt.forget();
                let (e, _xr, _) = opt.minimize(x0.view(), 500, |v| {
                    if !ledger.charge() {
                        return None;
                    }
                    Some(lj(v))
                });
                relaxes += 1;
                if let Ok(prefix) = std::env::var("ANNEAL_MIN_DUMP") {
                    use std::io::Write as _;
                    if let Ok(mut fh) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&prefix)
                    {
                        let mut line = format!("{e:.8}");
                        for v in _xr.iter() {
                            line.push_str(&format!(" {v:.6}"));
                        }
                        line.push('\n');
                        let _ = fh.write_all(line.as_bytes());
                    }
                }
                if e < best {
                    best = e;
                }
            }
            let hit = reference.map(|r| best < r + 1e-4).unwrap_or(false);
            if hit {
                solved += 1;
            }
            println!(
                "  seed {seed}: best {best:.6}  relaxes {relaxes}{}",
                if hit { "  SOLVED" } else { "" }
            );
        }
        println!("{solved}/{seeds} solved (rss)");
        return;
    }
    // Archive-ratchet mode: the minima network explored from a permanent
    // keyed archive, launches by discovery posterior.
    #[cfg(feature = "graphkey")]
    if std::env::args()
        .nth(4)
        .map(|v| v.contains("archive"))
        .unwrap_or(false)
    {
        use anneal_core::methods::ffs::{FfsConfig, ffs_descent};
        let fcfg = FfsConfig::for_cluster(n);
        let mut solved = 0usize;
        for seed in seed0..(seed0 + seeds) {
            let mut ledger = Ledger::new(budget);
            let mut opt = WarmLbfgs::default();
            let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
                opt.forget();
                let (f, xr, _) = opt.minimize(x, iters, |v| charged(led, v));
                (f, xr)
            };
            let out = ffs_descent(&fcfg, &mut ledger, &mut relax, seed);
            let hit = reference.map(|r| out.best < r + 1e-4).unwrap_or(false);
            if hit {
                solved += 1;
            }
            println!(
                "  seed {seed}: best {:.6}  archive {} inserts {} launches {} barren {}{}",
                out.best,
                out.descents,
                out.stored,
                out.continuations,
                out.returns,
                if hit { "  SOLVED" } else { "" }
            );
        }
        println!("{solved}/{seeds} solved (archive)");
        return;
    }
    // Committor-population mode: short chains of the configured stack,
    // resampled by improvement posterior.
    if std::env::args()
        .nth(4)
        .map(|v| v.contains("committor"))
        .unwrap_or(false)
    {
        use anneal_core::methods::committor_pop::committor_population;
        let mut ccfg = Config::for_cluster(n);
        ccfg.move_library = MoveLibrary::LeanBurst;
        ccfg.allocate_moves = true;
        ccfg.depth_reward = true;
        let walkers = 6usize;
        let seg = (budget / (walkers * 6)).max(20_000);
        let mut solved = 0usize;
        for seed in seed0..(seed0 + seeds) {
            let mut ledger = Ledger::new(budget);
            let mut opt = WarmLbfgs::default();
            let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
                opt.forget();
                let (f, xr, _) = opt.minimize(x, iters, |v| charged(led, v));
                (f, xr)
            };
            let out = committor_population(&ccfg, walkers, seg, &mut ledger, &mut relax, seed);
            let hit = reference.map(|r| out.best < r + 1e-4).unwrap_or(false);
            if hit {
                solved += 1;
            }
            println!(
                "  seed {seed}: best {:.6}  segments {} improvements {} resamples {}{}",
                out.best,
                out.segments,
                out.improvements,
                out.resamples,
                if hit { "  SOLVED" } else { "" }
            );
        }
        println!("{solved}/{seeds} solved (committor)");
        return;
    }
    // Nested mode replaces the chain entirely: population under a descending
    // ceiling, stopping by the run's own volume curve.
    if std::env::args()
        .nth(4)
        .map(|v| v.contains("nested"))
        .unwrap_or(false)
    {
        use anneal_core::methods::nested::{NestedConfig, nested_search};
        let ncfg = NestedConfig::for_cluster(n);
        let mut solved = 0usize;
        for seed in seed0..(seed0 + seeds) {
            let mut ledger = Ledger::new(budget);
            let mut opt = WarmLbfgs::default();
            let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
                opt.forget();
                let (f, xr, _) = opt.minimize(x, iters, |v| charged(led, v));
                (f, xr)
            };
            let out = nested_search(&ncfg, &mut ledger, &mut relax, seed);
            let hit = reference.map(|r| out.best < r + 1e-4).unwrap_or(false);
            if hit {
                solved += 1;
            }
            println!(
                "  seed {seed}: best {:.6}  replacements {}  steps {}  taken {}  ceiling {:.4}  repop {}{}",
                out.best,
                out.replacements,
                out.steps,
                out.taken,
                out.final_ceiling,
                out.repopulations,
                if hit { "  SOLVED" } else { "" }
            );
        }
        println!("{solved}/{seeds} solved (nested)");
        return;
    }
    // Residual archive search. Token is `ras`, not `archive` (that is FFS).
    #[cfg(feature = "graphkey")]
    if std::env::args()
        .nth(4)
        .map(|v| v.split(',').any(|t| t == "ras" || t == "pair"))
        .unwrap_or(false)
    {
        use anneal_core::methods::archive_search::{Archive, archive_search};
        use anneal_core::methods::cluster_hopping::{random_cluster_in_radius, run_with_gradient};
        use rand::SeedableRng;
        let pair = std::env::args()
            .nth(4)
            .map(|v| v.split(',').any(|t| t == "pair"))
            .unwrap_or(false);
        let cfg = Config::recommended(n);
        println!(
            "LJ{n}, budget {budget} charged evaluations, {seeds} seeds{}  arm {}",
            reference
                .map(|r| format!(", reference {r:.6}"))
                .unwrap_or_default(),
            if pair { "pair rec+ras" } else { "ras" }
        );
        let mut rec_solved = 0usize;
        let mut ras_solved = 0usize;
        let mut rec_hit_at = Vec::new();
        let mut ras_hit_at = Vec::new();
        for seed in seed0..(seed0 + seeds) {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let start =
                random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng);
            let mut rng_ras = rng.clone();
            if pair {
                let mut ledger = Ledger::new(budget);
                let mut opt = WarmLbfgs::default();
                let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
                    opt.forget();
                    let (f, xr, _) = opt.minimize(x, iters, |v| charged(led, v));
                    (f, xr)
                };
                let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<Array1<f64>> {
                    if !led.charge() {
                        return None;
                    }
                    Some(lj(x).1)
                };
                let mut rng_rec = rng_ras.clone();
                let out = run_with_gradient(
                    &cfg,
                    start.view(),
                    &mut ledger,
                    &mut relax,
                    Some(&mut grad),
                    &mut rng_rec,
                );
                let hit = reference.map(|r| out.best < r + 1e-4).unwrap_or(false);
                if hit {
                    rec_solved += 1;
                }
                let hat = out
                    .improvements
                    .iter()
                    .find(|(_, _, _, e)| reference.map(|r| *e < r + 1e-4).unwrap_or(false))
                    .map(|(_, sp, _, _)| *sp);
                if let Some(sp) = hat {
                    rec_hit_at.push(sp);
                }
                println!(
                    "  seed {seed} rec: best {:.6}  charged {}  hit_at {}{}",
                    out.best,
                    ledger.spent(),
                    hat.map(|v| v.to_string()).unwrap_or_else(|| "-".into()),
                    if hit { "  SOLVED" } else { "" }
                );
                let _ = io::stdout().flush();
            }
            let mut ledger = Ledger::new(budget);
            let mut opt = WarmLbfgs::default();
            let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
                opt.forget();
                let (f, xr, _) = opt.minimize(x, iters, |v| charged(led, v));
                (f, xr)
            };
            let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<Array1<f64>> {
                if !led.charge() {
                    return None;
                }
                Some(lj(x).1)
            };
            let mut archive = Archive::new();
            let out = archive_search(
                &cfg,
                start.view(),
                &mut ledger,
                &mut relax,
                Some(&mut grad),
                &mut archive,
                &mut rng_ras,
            );
            let hit = reference.map(|r| out.best < r + 1e-4).unwrap_or(false);
            if hit {
                ras_solved += 1;
            }
            if hit {
                ras_hit_at.push(out.best_at);
            }
            println!(
                "  seed {seed} ras: best {:.6}  charged {}  hit_at {}  screens {} full {} returned {} same_floor {} floors {} events {} artn {}{}",
                out.best,
                out.charged,
                if hit {
                    out.best_at.to_string()
                } else {
                    "-".into()
                },
                out.screens,
                out.full,
                out.returned,
                out.same_floor,
                out.floors,
                out.events,
                out.artn,
                if hit { "  SOLVED" } else { "" }
            );
            let _ = io::stdout().flush();
        }
        if pair {
            let rec_mean = if rec_hit_at.is_empty() {
                0
            } else {
                rec_hit_at.iter().sum::<usize>() / rec_hit_at.len()
            };
            let ras_mean = if ras_hit_at.is_empty() {
                0
            } else {
                ras_hit_at.iter().sum::<usize>() / ras_hit_at.len()
            };
            println!(
                "{rec_solved}/{seeds} solved (rec)  {ras_solved}/{seeds} solved (ras)  mean_hit_at rec {rec_mean} ras {ras_mean}"
            );
        } else {
            println!("{ras_solved}/{seeds} solved (ras)");
        }
        return;
    }
    println!(
        "LJ{n}, budget {budget} charged evaluations, {seeds} seeds{}",
        reference
            .map(|r| format!(", reference {r:.6}"))
            .unwrap_or_default()
    );

    // The temperature and step come from Wales and Doye's protocol for basin
    // hopping on the quenched surface, a reduced temperature of 0.8 and a step
    // between 0.36 and 0.40, rather than from tuning here.
    let mut cfg = if args.get(4).map(|v| v.contains("rec")).unwrap_or(false) {
        println!("  recommended configuration");
        Config::recommended(n)
    } else {
        Config::for_cluster(n)
    };
    // Keying on shape rather than on the descriptor, so the merge threshold is
    // a length. Enabled by the fourth argument so both are measurable.
    // Mechanisms named on the command line, so each is measurable against the
    // others rather than all arriving at once.
    let mut opts: Vec<&str> = args
        .get(4)
        .map(|v| v.split(',').collect())
        .unwrap_or_default();
    // The working Python driver (askmc_hopping) is Thompson over moves, the
    // budget-window temperature, and the per-basin bias the crate always
    // carries. Naming that stack keeps the measurement comparable without
    // assembling the flags from memory each time.
    if opts.contains(&"askmc") {
        opts.extend_from_slice(&["thompson", "bfwt"]);
    }
    cfg.shape_keyed = opts.contains(&"shape");
    cfg.budget_window = opts.contains(&"bfwt");
    cfg.allocate_moves = cfg.allocate_moves || opts.contains(&"thompson");
    cfg.adaptive_height = opts.contains(&"height");
    cfg.anneal_diversity = opts.contains(&"csa");
    cfg.path_on_stall = opts.contains(&"path");
    // Do not clobber Config::recommended: that hop already turns the
    // return screen on. The flag only adds it to for_cluster.
    if opts.contains(&"rscreen") {
        cfg.return_screen = true;
    }
    if opts.contains(&"soapclass") {
        cfg.soap_class_residual = true;
        println!("  SOAP residual: class 555->421 (oracle)");
    }
    if opts.contains(&"soapmean") {
        cfg.soap_class_residual = false;
        println!("  SOAP residual: mean (2p-mu)");
    }
    if cfg.soap_hop {
        #[cfg(feature = "featomic")]
        println!(
            "  SOAP hop: SOFI C5 residual while fivefold, else featomic packing-mean kick / leftover, l>=5, no 421/fcc"
        );
        #[cfg(feature = "ira")]
        println!("  IRA: libira_match Hausdorff on the shared bank, SOFI libira_try_mat on the hop");
        if cfg.keying == Keying::SoapPacking {
            println!(
                "  SOAP superbasin: mean-SOAP merge {}, adaptive height N_f={}",
                cfg.merge_radius, cfg.height_revisits
            );
        }
        #[cfg(not(feature = "featomic"))]
        println!("  SOAP hop: in-crate leftover (rebuild with --features featomic)");
    }
    cfg.minima_hopping = opts.contains(&"mh");
    cfg.escape_on_stall = opts.contains(&"climb");
    // The radius read off the search's own step length rather than swept.
    cfg.calibrate_radius = opts.contains(&"calib");
    // The walker restarted, the landscape memory kept.
    cfg.restart_on_stall = opts.contains(&"restart");
    // Wales and Doye's angular move on the worst-bound point.
    cfg.angular_moves = opts.contains(&"angular");
    // The funnel forbidden rather than penalised.
    cfg.tabu_on_stall = cfg.tabu_on_stall || opts.contains(&"tabu");
    // The relaxation decision taken under a posterior.
    cfg.bayes_screen = opts.contains(&"bayes");
    // Acceptance against the density of minima rather than against the energy.
    cfg.flat_histogram = opts.contains(&"flat");
    // The temperature taken from the entropy the run measures for itself.
    cfg.statistical_temperature = opts.contains(&"stemp");
    // A well-tempered bias in quenched energy, scales from the run itself.
    cfg.energy_bias = opts.contains(&"ebias");
    let requested_libraries = [
        ("visit", MoveLibrary::Visit),
        ("reseed", MoveLibrary::Reseed),
        ("selfseed", MoveLibrary::SelfReseed),
        ("learncon", MoveLibrary::LearnedReseed),
        ("lean", MoveLibrary::Lean),
        ("burst", MoveLibrary::LeanBurst),
        ("twin", MoveLibrary::Twin),
        ("gtwin", MoveLibrary::GrowthAndTwin),
    ];
    let selected: Vec<MoveLibrary> = requested_libraries
        .into_iter()
        .filter_map(|(name, library)| opts.contains(&name).then_some(library))
        .collect();
    assert!(
        selected.len() <= 1,
        "select at most one move library: visit,reseed,selfseed,learncon,lean,burst,twin,gtwin"
    );
    if let Some(library) = selected.into_iter().next() {
        cfg.move_library = library;
    }
    // Local order and global twinning together use one typed library.
    // Arms rewarded by depth reached rather than by acceptance.
    cfg.depth_reward = cfg.depth_reward || opts.contains(&"depth");
    // Perturbation drawn in the soft subspace of the incumbent's curvature.
    cfg.soft_perturb = opts.contains(&"softsub");
    // Proposal covariance learned from the run's accepted displacements.
    cfg.cov_perturb = opts.contains(&"covper");
    // Settle moved atoms at fractional price before the full-system screen.
    cfg.staged_quench = opts.contains(&"staged");
    // Arm selection has to be under an allocator at all before the reward rule
    // matters: without this the arm is drawn uniformly and both allocators are
    // inert.
    if cfg.depth_reward {
        cfg.allocate_moves = true;
    }
    // The screening pass is the quench, so its length is the one number that
    // decides whether the chain moves on the transformed landscape at all.
    if let Ok(v) = std::env::var("SCREEN_STEPS") {
        if let Ok(k) = v.parse::<usize>() {
            cfg.screen_steps = k;
            println!("  screen steps {k}");
        }
    }
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
        println!(
            "  keying on a canonical order, merge radius {}",
            cfg.merge_radius
        );
    }
    if opts.contains(&"pt") {
        // A ladder sharing one budget, not four budgets. The comparison is
        // against a single chain at the same total cost.
        cfg.replicas = 4;
        cfg.bias_by_rung = opts.contains(&"rungbias");
        println!(
            "  replica exchange: {} chains, swap every {} hops, top x{}",
            cfg.replicas, cfg.swap_period, cfg.ladder_top
        );
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
        println!(
            "  keying on IRA shape distance, merge radius {} (a length)",
            cfg.merge_radius
        );
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
            .unwrap_or(0.4),
        mix_fraction: std::env::var("BANK_MIX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5),
        mix_images: env("BANK_MIX_IMAGES", 20),
        random_images: env("BANK_RANDOM", 10),
        deadlock_iters: env("BANK_DEADLOCK_ITERS", 3),
        deadlock_inject: env("BANK_DEADLOCK", 50),
    };
    if use_bank {
        println!(
            "  bank of {} chains, {} charged per slice, Dcut floor {}, mix {} ({} splice + {} random), deadlock {}x{}, acq {}",
            bank_cfg.capacity,
            bank_cfg.slice,
            bank_cfg.dcut_floor,
            bank_cfg.mix_fraction,
            bank_cfg.mix_images,
            bank_cfg.random_images,
            bank_cfg.deadlock_iters,
            bank_cfg.deadlock_inject,
            bank_cfg.acquisition
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
        // Noise on the screening descent, as a fraction of the local gradient.
        // Zero reproduces the clean quench exactly.
        let noise_eta: f64 = std::env::var("QUENCH_NOISE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0);
        let mut qrng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(
            seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(17),
        );
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
            let (f, xr, _) = if noise_eta > 0.0 && iters <= screen_steps {
                opt.minimize(x, iters, |v| charged_noisy(led, v, noise_eta, &mut qrng))
            } else {
                opt.minimize(x, iters, |v| charged(led, v))
            };
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
        let mut out = if let Ok(sock) = std::env::var("BANK_RPC") {
            #[cfg(feature = "bank-rpc")]
            {
                println!("  capnp bank {sock}");
                #[cfg(feature = "featomic")]
                println!(
                    "  bank SOAP: soap_bank_distance / packing wells merge {}, Dcut fallback {}",
                    anneal_core::featomic_hop::SOAP_PACK_MERGE,
                    anneal_core::featomic_hop::SOAP_DCUT_FALLBACK
                );
                #[cfg(feature = "ira")]
                println!("  bank IRA: Hausdorff same-state, then SOAP Lee Dcut");
                println!(
                    "  bank explore: well-UCB start (QD), archive-null SOAP hop, Lee splice + Dcut"
                );
                run_capnp_bank(&cfg, &mut ledger, &mut relax, &mut grad, seed as u64, &sock)
            }
            #[cfg(not(feature = "bank-rpc"))]
            {
                let _ = sock;
                panic!("BANK_RPC set; rebuild with --features bank-rpc");
            }
        } else if use_bank {
            {
                // Shape distance when IRA is linked; otherwise the pairwise
                // spectrum. The bank rule is Lee's Dcut replacement, not the
                // metric: two members closer than Dcut are one solution.
                #[cfg(feature = "featomic")]
                let mut dist = {
                    let rcut = 3.5 * cfg.length_scale;
                    let z = cfg.species.clone();
                    println!(
                        "  bank Dcut: featomic soap_bank_distance, fallback {}",
                        anneal_core::featomic_hop::SOAP_DCUT_FALLBACK
                    );
                    move |p: ArrayView1<f64>, q: ArrayView1<f64>| {
                        anneal_core::featomic_hop::soap_bank_distance(
                            p,
                            q,
                            rcut,
                            z.as_deref(),
                            None,
                        )
                    }
                };
                #[cfg(all(feature = "ira", not(feature = "featomic")))]
                let mut dist = {
                    let ira = IraMetric::default();
                    move |p: ArrayView1<f64>, q: ArrayView1<f64>| ira.distance(p, q)
                };
                #[cfg(not(any(feature = "ira", feature = "featomic")))]
                let mut dist = csa_cluster::spectrum_distance(n);
                let b = csa_cluster::run(
                    &cfg,
                    &bank_cfg,
                    &mut ledger,
                    &mut relax,
                    if cfg.minima_hopping || cfg.escape_on_stall || cfg.soft_perturb {
                        Some(&mut grad)
                    } else {
                        None
                    },
                    &mut dist,
                    seed,
                );
                println!(
                    "      bank: {} slices, Dcut {:.3} -> {:.3}, {} improved, {} novel, \
                     {} duplicate, {} mixes ({} admitted, {} below both ends), \
                     {} deadlocks ({} injected), holding {:?}",
                    b.slices,
                    b.dcut.0,
                    b.dcut.1,
                    b.improved,
                    b.novel,
                    b.duplicates,
                    b.mixes,
                    b.mix_admitted,
                    b.mix_below_both,
                    b.deadlocks,
                    b.injected,
                    b.bank
                        .iter()
                        .map(|e| (e * 100.0).round() / 100.0)
                        .collect::<Vec<_>>()
                );
                Outcome {
                    best: b.best,
                    best_state: b.best_state,
                    hops: b.hops,
                    basins: b.basins,
                    screened_out: b.screened_out,
                    returned: b.returned,
                    ..Outcome::default()
                }
            }
        } else {
            {
                // The settle stage: steepest descent of the moved atoms in the
                // frozen field, charged at the exact fraction of a full
                // evaluation the partial rows represent. Audited on the first
                // call against the full gradient when AUDIT_SETTLE is set.
                let mut audited = false;
                let mut settle = |led: &mut Ledger,
                                  x: ArrayView1<f64>,
                                  moved: &[usize],
                                  iters: usize|
                 -> Array1<f64> {
                    let np = x.len() / 3;
                    let frac = (2.0 * moved.len() as f64) / ((np.max(2) - 1) as f64);
                    if std::env::var("AUDIT_SETTLE").is_ok() && !audited {
                        audited = true;
                        let (_, full) = lj(x);
                        let part = lj_partial_grad(x, moved);
                        for &m in moved {
                            for k in 0..3 {
                                assert!(
                                    (full[3 * m + k] - part[3 * m + k]).abs() < 1e-9,
                                    "partial gradient diverges from full at atom {m}"
                                );
                            }
                        }
                        println!("  settle audit passed: partial rows match the full gradient");
                    }
                    let mut cur = x.to_owned();
                    for _ in 0..iters {
                        if !led.charge_frac(frac) {
                            break;
                        }
                        let g = lj_partial_grad(cur.view(), moved);
                        let mut gmax = 0.0_f64;
                        for &m in moved {
                            for k in 0..3 {
                                gmax = gmax.max(g[3 * m + k].abs());
                            }
                        }
                        if gmax < 1e-4 {
                            break;
                        }
                        // A conservative step against the stiffest component,
                        // enough to drain the worst of the overlap the move
                        // created; the full screen finishes the job.
                        let step = 0.05 / gmax.max(1.0);
                        for &m in moved {
                            for k in 0..3 {
                                cur[3 * m + k] -= step * g[3 * m + k];
                            }
                        }
                    }
                    cur
                };
                anneal_core::methods::cluster_hopping::optimize_with_settle(
                    &cfg,
                    &mut ledger,
                    &mut relax,
                    if cfg.minima_hopping || cfg.escape_on_stall || cfg.soft_perturb {
                        Some(&mut grad)
                    } else {
                        None
                    },
                    if cfg.staged_quench {
                        Some(&mut settle)
                    } else {
                        None
                    },
                    seed,
                )
            }
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
                let (mut e, g) = lj(x.view());
                let mut gmax = g.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
                // A hop quench can stop short of a minimum and still be
                // recorded when the driver did not pass a gradient to the
                // recordable guard. Finish the relaxation off the ledger
                // and report the minimum that structure actually is.
                let (e, gmax) = if gmax >= 1e-3 {
                    let mut opt = WarmLbfgs::default();
                    let (er, xr, _) = opt.minimize(x.view(), 2000, |v| Some(lj(v)));
                    let (_, gr) = lj(xr.view());
                    let gm = gr.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
                    (er, gm)
                } else {
                    (e, gmax)
                };
                assert!(
                    gmax < 1e-3,
                    "seed {seed} returned a structure with gradient {gmax:.2e}, \
                     which is not a minimum"
                );
                Some((e, gmax))
            }
            None => None,
        };
        if let Some((e, _)) = verified {
            out.best = e;
        }
        let hit = reference.map(|r| out.best < r + 1e-4).unwrap_or(false);
        if hit {
            solved += 1;
        }
        // Where the run got its answer. Printed for the last few improvements
        // only: the early ones are a descent from a random start and say
        // nothing.
        if std::env::var("DUMP_IMPROVEMENTS").is_ok() {
            for (h, sp, b, en) in out.improvements.iter() {
                println!("IMP hop {h} spend {sp} basins {b} energy {en:.6}");
            }
        }
        if let Some(r) = reference {
            if let Some((h, _, b, e)) = out.improvements.iter().find(|(_, _, _, e)| *e < r + 1e-4) {
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
             escape {:.3} thr {:.4} same/known/new {}/{}/{} soft {}/{} sub {}/{} lmin {:.4} climbs {} gain {:.2} radius {:.3} step {:.3} restarts {} angular {}/{} R {:.3} tabu {} vetoed {} screen {}/{} expl {} obs {} ctx {:?}  \
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
            out.soft_perturbs,
            out.soft_subspaces,
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

#[cfg(feature = "bank-rpc")]
fn packing_of(x: ArrayView1<f64>, cfg: &Config) -> Array1<f64> {
    #[cfg(feature = "featomic")]
    {
        anneal_core::featomic_hop::soap_cloud_mean(
            x,
            3.5 * cfg.length_scale,
            cfg.species.as_deref(),
            None,
        )
    }
    #[cfg(not(feature = "featomic"))]
    {
        let _ = (x, cfg);
        Array1::zeros(0)
    }
}

/// One HQ chain against the Cap'n Proto bank: slice, offer, deposit, repeat.
#[cfg(feature = "bank-rpc")]
fn run_capnp_bank(
    cfg: &Config,
    ledger: &mut Ledger,
    relax: &mut dyn FnMut(&mut Ledger, ArrayView1<f64>, usize) -> (f64, Array1<f64>),
    grad: &mut dyn FnMut(&mut Ledger, ArrayView1<f64>) -> Option<Array1<f64>>,
    seed: u64,
    sock: &str,
) -> Outcome {
    use anneal_core::bank_rpc::BankClient;
    use anneal_core::diversity::DiversityAnnealer;
    use anneal_core::methods::splice::cut_and_splice;
    use rand::Rng;
    let mut client = BankClient::connect(sock).unwrap_or_else(|e| panic!("BANK_RPC {sock}: {e}"));
    let mut bias = BasinBias::new(
        ClusterFingerprint::of_config(cfg, &Array1::zeros(0)),
        cfg.merge_radius,
        cfg.bias_height,
        cfg.bias_gamma,
    );
    let slice = std::env::var("BANK_SLICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3_000);
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
    let mut best = f64::INFINITY;
    let mut best_state = None;
    let mut hops = 0usize;
    let mut basins = 0usize;
    let mut screened_out = 0usize;
    let mut returned = 0usize;
    let mut slices = 0usize;
    let mut mixes = 0usize;
    let mut random_starts = 0usize;
    let total = ledger.remaining();
    let mut schedule: Option<DiversityAnnealer> = None;
    let well_cap = cfg.bias_height * cfg.height_revisits.max(1.0);
    while ledger.remaining() > 0 {
        let snap = client.snapshot().ok();
        if let Some(s) = snap.as_ref() {
            for (soap, h) in &s.wells {
                bias.import_well(soap.clone(), *h);
            }
            #[cfg(feature = "featomic")]
            anneal_core::featomic_hop::set_packing_archive(
                s.wells.iter().map(|(soap, _)| soap.clone()).collect(),
            );
            if s.size >= 2 {
                let sched = schedule.get_or_insert_with(|| {
                    DiversityAnnealer::from_initial(s.dcut.max(0.05)).with_final_fraction(0.4)
                });
                let progress = 1.0 - ledger.remaining() as f64 / total.max(1) as f64;
                let _ = client.set_dcut(sched.threshold(progress));
            }
        }
        // Mix two members (working + first bank) instead of searching one
        // ico copy again. Lee's operator; the hop alone stays in-funnel.
        if snap.as_ref().map(|s| s.size >= 2).unwrap_or(false) && rng.random::<f64>() < 0.5 {
            if let (Ok(Some((_, a))), Ok(Some((_, b)))) = (
                client.sample(rng.random::<u64>() & !1),
                client.sample(rng.random::<u64>() | 1),
            ) {
                if a.len() == 3 * cfg.n_points && b.len() == 3 * cfg.n_points {
                    let trial = cut_and_splice(
                        a.view(),
                        b.view(),
                        cfg.species.as_deref(),
                        cfg.min_separation,
                        &mut rng,
                    );
                    let mut mix_led = Ledger::new(cfg.relax_steps.max(32).min(ledger.remaining()));
                    let (e, x) = relax(&mut mix_led, trial.view(), cfg.relax_steps);
                    ledger.charge_many(mix_led.spent());
                    ledger.record(e, x.view());
                    let soap = packing_of(x.view(), cfg);
                    let _ = client.offer(e, x.view(), soap.view());
                    let _ = client.deposit(soap.view(), cfg.bias_height);
                    mixes += 1;
                    if e < best {
                        best = e;
                        best_state = Some(x);
                    }
                    slices += 1;
                    continue;
                }
            }
        }
        let start = match client.sample(rng.random()) {
            Ok(Some((_, x))) if x.len() == 3 * cfg.n_points => {
                let soap = packing_of(x.view(), cfg);
                let h = client.bias_of(soap.view()).unwrap_or(0.0);
                // Known packing already filled: do not start there again.
                if h >= well_cap {
                    random_starts += 1;
                    random_cluster(cfg.n_points, 0.7, cfg.min_separation, &mut rng)
                } else {
                    x
                }
            }
            _ => random_cluster(cfg.n_points, 0.7, cfg.min_separation, &mut rng),
        };
        let mut slice_led = Ledger::new(slice.min(ledger.remaining()));
        let out = run_with_bias(
            cfg,
            start.view(),
            &mut slice_led,
            relax,
            Some(grad),
            &mut bias,
            &mut rng,
        );
        ledger.charge_many(slice_led.spent());
        if let Some(st) = slice_led.best_state.as_ref() {
            ledger.record(slice_led.best, st.view());
        }
        hops += out.hops;
        basins += out.basins;
        screened_out += out.screened_out;
        returned += out.returned;
        slices += 1;
        if out.best < best {
            best = out.best;
            best_state = out.best_state.clone();
        }
        if let Some(st) = out.best_state.as_ref() {
            let soap = packing_of(st.view(), cfg);
            let _ = client.offer(out.best, st.view(), soap.view());
            let _ = client.deposit(soap.view(), cfg.bias_height);
        }
    }
    println!(
        "      capnp bank: {slices} slices, {mixes} splices, {random_starts} novel starts, best {best:.6}"
    );
    Outcome {
        best,
        best_state,
        hops,
        basins,
        screened_out,
        returned,
        ..Outcome::default()
    }
}
