// Copyright (c) anneal developers
//! One seed of the general search on a GFN2-xTB silicon or carbon cluster.
//!
//! ```text
//! gfn2_covalent_h2h <si10|c10> <seed> <budget> [target_eV] [best.xyz]
//! ```
//!
//! The energy and gradient are the rgpot GFN2 handle in eV and angstrom,
//! `RGPOT_XTB_ENGINE` naming `libxtb_engine.so`. Silicon clusters have a
//! vanishing gap and the SCC does not converge at 300 K, so the bench sets
//! `RGPOT_XTB_ETEMP` and passes the same electronic temperature to CREST.
//!
//! The search is the covalent panel's `general` arm: the recommended flags,
//! priors off, on scales from this potential's own dimer, the bond over
//! 2^(1/6) and the well depth against two atoms at 20 angstrom. The dimer
//! probe runs on a two-atom handle and is charged to the same ledger. There
//! is no pair-range cut, because dispersion has no finite reach.
//!
//! The start is a random cluster in the scaled start radius from
//! `StdRng::seed_from_u64(seed)`. `H2H_START_ONLY=1` writes it to
//! `best.xyz`'s path and exits, so CREST is given the same geometry, and
//! charges nothing but the dimer probe it needs for the radius.
//!
//! A hit is a recorded energy at or below `target_eV` plus 1e-3 eV.

#[path = "common/rgpot_eindir.rs"]
mod rgpot_eindir;

use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, MoveLibrary, Outcome, SoapProposalMode, random_cluster_in_radius,
    run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use eindir_core::gradient::DifferentiableObjective;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rgpot_eindir::RgpotObjective;

const HIT_TOL_EV: f64 = 1e-3;
const FAR_A: f64 = 20.0;

static POTENTIAL_NS: AtomicU64 = AtomicU64::new(0);

struct System {
    n: usize,
    z: i32,
    symbol: &'static str,
}

fn system(name: &str) -> Option<System> {
    match name {
        "si10" => Some(System {
            n: 10,
            z: 14,
            symbol: "Si",
        }),
        "c10" => Some(System {
            n: 10,
            z: 6,
            symbol: "C",
        }),
        _ => None,
    }
}

fn evaluate(pot: &RgpotObjective, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let mark = Instant::now();
    let pair = pot.wrapper().value_and_gradient(x);
    POTENTIAL_NS.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
    pair
}

/// Bond length over 2^(1/6) and well depth of the homonuclear dimer.
///
/// A geometric scan finds the sign change of the bond force and false
/// position closes it, as in `h2h_covalent`; the depth is taken against the
/// pair at `FAR_A`.
fn dimer_scales(z: i32, mut charge: impl FnMut() -> bool) -> Option<(f64, f64)> {
    let box_ = [60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 60.0];
    let mut pot = RgpotObjective::xtb(&[z, z], box_);
    pot.restart();
    let at = |r: f64| {
        let x = Array1::from(vec![0.0, 0.0, 0.0, r, 0.0, 0.0]);
        let (energy, gradient) = evaluate(&pot, x.view());
        (energy, gradient[3])
    };
    let mut probe = |r: f64| charge().then(|| at(r));
    let far = probe(FAR_A)?.0;
    let mut r = 1.0_f64;
    let mut before = probe(r)?;
    let (mut lo, mut hi, mut slope_lo, mut slope_hi) = loop {
        let next = 1.05 * r;
        if next >= FAR_A {
            return None;
        }
        let now = probe(next)?;
        if before.1 < 0.0 && now.1 > 0.0 {
            break (r, next, before.1, now.1);
        }
        (r, before) = (next, now);
    };
    for _ in 0..40 {
        if hi - lo <= 1.0e-8 * hi {
            break;
        }
        let mid = (lo * slope_hi - hi * slope_lo) / (slope_hi - slope_lo);
        let (_, slope) = probe(mid)?;
        if slope < 0.0 {
            lo = mid;
            slope_lo = slope;
        } else {
            hi = mid;
            slope_hi = slope;
        }
    }
    let bond = 0.5 * (lo + hi);
    let (energy, _) = probe(bond)?;
    let depth = far - energy;
    (depth > 0.0).then(|| (bond / 2.0_f64.powf(1.0 / 6.0), depth))
}

/// The covalent panel's `general` arm.
fn general(n: usize, length: f64, energy: f64) -> Config {
    let mut cfg = Config::with_scales(n, length, energy);
    cfg.bayes_screen = true;
    cfg.bayes_threshold =
        anneal_core::screen::cost_asymmetric_threshold(cfg.screen_steps, cfg.relax_steps);
    cfg.move_library = MoveLibrary::LeanBurst;
    cfg.allocate_moves = true;
    cfg.depth_reward = true;
    cfg.tabu_on_stall = true;
    cfg.return_screen = true;
    cfg.soap_class_residual = false;
    cfg.soap_mode = SoapProposalMode::Flexible;
    cfg.trail_on_stall = true;
    cfg.escape_on_stall = true;
    cfg.orbit_complete_on_new = true;
    cfg.orbit_arm = true;
    cfg.prior_soap = false;
    cfg.prior_reseed = false;
    cfg.prior_growth = false;
    cfg.prior_core = false;
    cfg.prior_motif = false;
    cfg.symmetrise_on_stall = false;
    cfg.continuous_symmetry = anneal_core::methods::cluster_hopping::ContinuousSymmetry::Off;
    cfg.energy_lattice = false;
    cfg.surfaces.clear();
    cfg.budget_window = true;
    cfg
}

fn write_xyz(path: &str, symbol: &str, x: &[f64], comment: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    writeln!(file, "{}", x.len() / 3)?;
    writeln!(file, "{comment}")?;
    for atom in x.chunks(3) {
        writeln!(
            file,
            "{symbol} {:.10} {:.10} {:.10}",
            atom[0], atom[1], atom[2]
        )?;
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(sys) = args.first().and_then(|name| system(name)) else {
        eprintln!("usage: gfn2_covalent_h2h <si10|c10> <seed> <budget> [target_eV] [best.xyz]");
        std::process::exit(2);
    };
    let seed: u64 = args.get(1).and_then(|v| v.parse().ok()).expect("seed");
    let budget: usize = args.get(2).and_then(|v| v.parse().ok()).expect("budget");
    let target: Option<f64> = args
        .get(3)
        .filter(|v| v.as_str() != "-")
        .and_then(|v| v.parse().ok());
    let out = args.get(4).cloned();

    let mut ledger = Ledger::new(budget);
    if let Some(target) = target {
        ledger.stop_below(target + HIT_TOL_EV);
    }
    let started = Instant::now();
    let (length, energy) =
        dimer_scales(sys.z, || ledger.charge()).expect("a bound dimer inside the budget");
    let probe_calls = ledger.spent();
    let cfg = general(sys.n, length, energy);
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(sys.n, cfg.start_radius(), cfg.min_separation, &mut rng);
    if std::env::var("H2H_START_ONLY").ok().as_deref() == Some("1") {
        let path = out.expect("start-only needs the xyz path");
        write_xyz(
            &path,
            sys.symbol,
            start.as_slice().expect("contiguous"),
            "start",
        )
        .expect("start xyz");
        println!("start {path} length {length:.6} energy {energy:.6}");
        return;
    }

    let box_ = [60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 60.0];
    let mut pot = RgpotObjective::xtb(&vec![sys.z; sys.n], box_);
    pot.restart();
    let pot = Mutex::new(pot);
    let mut optimiser = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            if !led.charge() {
                return None;
            }
            Some(evaluate(&pot.lock().expect("pot"), v))
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        Some(evaluate(&pot.lock().expect("pot"), v).1)
    };
    let outcome: Outcome = run_with_gradient(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start");
    let wall = started.elapsed().as_secs_f64();
    let pot_s = POTENTIAL_NS.load(Ordering::Relaxed) as f64 / 1e9;
    let level = target.map(|t| t + HIT_TOL_EV);
    let hit = level.is_some_and(|level| outcome.best <= level);
    let evals = if hit {
        outcome
            .improvements
            .iter()
            .find(|(_, _, _, e)| level.is_some_and(|level| *e <= level))
            .map_or(outcome.charged, |(_, spent, _, _)| {
                (*spent).min(outcome.charged)
            })
    } else {
        budget
    };
    let calls = outcome.charged.max(1) as f64;
    let outside = (wall - pot_s).max(0.0);
    println!(
        "system,code,seed,budget,hit,evals_to_hit,charged,probe_calls,wall_s,best_eV,pot_s,outside_s,pot_per_eval_s,outside_per_eval_s,scc_rejected,geometry_rejected"
    );
    println!(
        "{},anneal-general,{seed},{budget},{hit},{evals},{},{probe_calls},{wall:.6},{:.8},{pot_s:.6},{outside:.6},{:.9},{:.9},{},{}",
        args[0],
        outcome.charged,
        outcome.best,
        pot_s / calls,
        outside / calls,
        rgpot_eindir::scc_rejected_count(),
        rgpot_eindir::geometry_rejected_count(),
    );
    if let (Some(path), Some(best)) = (out, outcome.best_state.as_ref()) {
        write_xyz(
            &path,
            sys.symbol,
            best.as_slice().expect("contiguous"),
            &format!("anneal {:.10} eV", outcome.best),
        )
        .expect("best xyz");
    }
}
