// Copyright (c) anneal developers
//! One seed of the general search on a GFN2-xTB silicon or carbon cluster.
//!
//! ```text
//! gfn2_covalent_h2h <si10|si13|si20|c10|c20> <seed> <budget> [target_eV] [best.xyz]
//! ```
//!
//! The energy and gradient are the rgpot GFN2 handle in eV and angstrom,
//! `RGPOT_XTB_ENGINE` naming `libxtb_engine.so`. Silicon clusters have a
//! vanishing gap and the SCC does not converge at 300 K, so the bench sets
//! `RGPOT_XTB_ETEMP` and passes the same electronic temperature to CREST.
//! Each call's SCC starts fresh, as in each of CREST's xtb processes: on
//! Si13 and Si20 the previous call's wavefunction as the guess converges
//! to lower SCC solutions that a fresh SCC does not reproduce.
//!
//! The search is the covalent panel's `general` arm: the recommended flags,
//! priors off, on scales from this potential's own dimer, the bond over
//! 2^(1/6) and the well depth against two atoms at 20 angstrom. The dimer
//! probe runs on a two-atom handle and is charged to the same ledger. There
//! is no pair-range cut, because dispersion has no finite reach.
//! The call floor is the handle's 0.90 Å, raised pair by pair to
//! `element_call_floor` of the atomic numbers: 1.55 Å for Si–Si and 1.06 Å
//! for C–C.
//!
//! The start is a random cluster in the scaled start radius from
//! `StdRng::seed_from_u64(seed)`. `H2H_START_ONLY=1` writes it to
//! `best.xyz`'s path and exits, so CREST is given the same geometry, and
//! charges nothing but the dimer probe it needs for the radius.
//!
//! A hit is a recorded energy at or below `target_eV` plus 1e-3 eV.
//!
//! `H2H_WALL_S` closes the ledger that many seconds after the start, and
//! the search returns its best so far. `charged` is then the calls made
//! before it closed.

#[path = "common/rgpot_eindir.rs"]
mod rgpot_eindir;

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
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

const HIT_TOL_EV: f64 = anneal_core::DEFAULT_HIT_TOL;
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
        "si13" => Some(System {
            n: 13,
            z: 14,
            symbol: "Si",
        }),
        "si20" => Some(System {
            n: 20,
            z: 14,
            symbol: "Si",
        }),
        "c10" => Some(System {
            n: 10,
            z: 6,
            symbol: "C",
        }),
        "c20" => Some(System {
            n: 20,
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

/// One call of the search's handle.
enum Call {
    /// The ledger is closed.
    Spent,
    /// The handle has no energy here: it refused the geometry before xtb,
    /// uncharged, or the SCC did not converge. The eindir wrapper leaves its
    /// gradient buffer at zero then, which is not a gradient.
    Refused,
    Answer(f64, Array1<f64>),
}

fn call(pot: &Mutex<RgpotObjective>, led: &mut Ledger, x: ArrayView1<f64>) -> Call {
    if rgpot_eindir::refuses_geometry(&x.to_vec()) {
        return Call::Refused;
    }
    if !led.charge() {
        return Call::Spent;
    }
    let (energy, gradient) = evaluate(&pot.lock().expect("pot"), x);
    classify(energy, gradient)
}

/// What the handle returned: an answer only when the energy and every
/// gradient component are finite.
fn classify(energy: f64, gradient: Array1<f64>) -> Call {
    if energy.is_finite() && gradient.iter().all(|value| value.is_finite()) {
        Call::Answer(energy, gradient)
    } else {
        Call::Refused
    }
}

/// A line search's evaluation: a refusal is a non-finite value, which it
/// reads as an uphill step and shortens.
fn line_search_call(
    pot: &Mutex<RgpotObjective>,
    led: &mut Ledger,
    x: ArrayView1<f64>,
) -> Option<(f64, Array1<f64>)> {
    match call(pot, led, x) {
        Call::Spent => None,
        Call::Refused => Some((f64::NAN, Array1::from_elem(x.len(), f64::NAN))),
        Call::Answer(energy, gradient) => Some((energy, gradient)),
    }
}

/// The search's gradient: none where the handle has no energy, so a refused
/// geometry is neither certified as a minimum nor climbed from.
fn gradient_call(
    pot: &Mutex<RgpotObjective>,
    led: &mut Ledger,
    x: ArrayView1<f64>,
) -> Option<Array1<f64>> {
    match call(pot, led, x) {
        Call::Answer(_, gradient) => Some(gradient),
        Call::Spent | Call::Refused => None,
    }
}

/// Bond length over 2^(1/6) and well depth of the homonuclear dimer.
///
/// A geometric scan finds the sign change of the bond force and false
/// position closes it, as in `h2h_covalent`; the depth is taken against the
/// pair at `FAR_A`.
fn dimer_scales(z: i32, mut charge: impl FnMut() -> bool) -> Option<(f64, f64)> {
    let box_ = [60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 60.0];
    let mut pot = RgpotObjective::xtb_fresh(&[z, z], box_);
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
    cfg.symmetrise_on_stall = false;
    cfg.continuous_symmetry = anneal_core::methods::cluster_hopping::ContinuousSymmetry::Off;
    cfg.energy_lattice = false;
    cfg.surfaces.clear();
    cfg.budget_window = true;
    cfg.call_floor = Some(rgpot_eindir::OVERLAP_CALL_LIMIT_A);
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
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(sys) = args.first().and_then(|name| system(name)) else {
        eprintln!(
            "usage: gfn2_covalent_h2h <si10|si13|si20|c10|c20> <seed> <budget> [target_eV] [best.xyz]"
        );
        std::process::exit(2);
    };
    let seed: u64 = args.get(1).and_then(|v| v.parse().ok()).expect("seed");
    let budget: usize = args.get(2).and_then(|v| v.parse().ok()).expect("budget");
    let target: Option<f64> = args
        .get(3)
        .filter(|v| v.as_str() != "-")
        .and_then(|v| v.parse().ok());
    let out = args.get(4).cloned();

    let mut ledger = match std::env::var("H2H_WALL_S").ok() {
        Some(text) => {
            let seconds: f64 = text.parse().expect("H2H_WALL_S in seconds");
            let stop = Arc::new(AtomicBool::new(false));
            let raise = Arc::clone(&stop);
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_secs_f64(seconds));
                raise.store(true, Ordering::Relaxed);
            });
            Ledger::with_stop(budget, stop)
        }
        None => Ledger::new(budget),
    };
    if let Some(target) = target {
        ledger.stop_below(target + HIT_TOL_EV);
    }
    let started = Instant::now();
    let (length, energy) =
        dimer_scales(sys.z, || ledger.charge()).expect("a bound dimer inside the budget");
    let probe_calls = ledger.spent();
    let mut cfg = general(sys.n, length, energy);
    cfg.call_floor_elements = Some(vec![sys.z as u32; sys.n]);
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
    let mut pot = RgpotObjective::xtb_fresh(&vec![sys.z; sys.n], box_);
    pot.restart();
    let pot = Mutex::new(pot);
    let mut optimiser = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) =
            optimiser.minimize(x, iterations, |v| line_search_call(&pot, led, v));
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| gradient_call(&pot, led, v);
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
    let hit = target.is_some_and(|t| anneal_core::energy_hits(outcome.best, t, HIT_TOL_EV));
    let evals = if hit {
        outcome
            .improvements
            .iter()
            .find(|(_, _, _, e)| {
                target.is_some_and(|t| anneal_core::energy_hits(*e, t, HIT_TOL_EV))
            })
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

/// Against real xtb: `RGPOT_XTB_ENGINE` names `libxtb_engine.so` and
/// `LD_LIBRARY_PATH` its libxtb. They fail without it rather than skip.
///
/// ```text
/// cargo test --release --features rgpot-ex --example gfn2_covalent_h2h
/// ```
#[cfg(test)]
mod tests {
    use super::*;

    const BOX: [f64; 9] = [60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 60.0];

    /// A compact Si13 whose fresh SCC does not converge at 300 K or at the
    /// bench's 3000 K, with no pair inside the handle's 0.90 Å.
    const SI13_SCC_FAILS: [f64; 39] = [
        -0.5957572740562841,
        -1.2217069293964795,
        0.9865349689621185,
        0.9268266042665285,
        -0.41172432716729984,
        1.9085195608484748,
        1.4304528902425964,
        -0.8898970109587083,
        -1.4687854013128288,
        -2.061061354262374,
        0.10173279503140154,
        2.031372579410227,
        2.182068219042771,
        -1.1714298473684326,
        -0.09782090194581013,
        -2.0431916393136964,
        -1.5406026489486877,
        0.5675528708204733,
        -0.12620548020935812,
        2.4894767753651057,
        0.4037140818031978,
        1.692028451833346,
        0.07660959382815108,
        -1.5992042413638194,
        0.17221464522341975,
        -0.7660468031333149,
        0.03260001162428287,
        0.4126836999357364,
        0.7854169146804258,
        0.5561469335403733,
        0.9008056093258453,
        1.529545514981933,
        -1.716409892035046,
        1.7029756306265302,
        -1.8943662564254033,
        0.6001958426248252,
        -0.7392478901300448,
        0.8037416796483207,
        -1.0011518294499242,
    ];

    fn handle(z: i32, n: usize) -> Mutex<RgpotObjective> {
        let mut pot = RgpotObjective::xtb_fresh(&vec![z; n], BOX);
        pot.restart();
        Mutex::new(pot)
    }

    fn silicon_pair(r: f64) -> Array1<f64> {
        Array1::from(vec![0.0, 0.0, 0.0, r, 0.0, 0.0])
    }

    #[test]
    fn the_wrappers_zero_buffer_after_a_failed_call_is_refused_not_answered() {
        let zeros = Array1::<f64>::zeros(6);
        assert!(matches!(classify(f64::NAN, zeros.clone()), Call::Refused));
        assert!(matches!(classify(f64::INFINITY, zeros), Call::Refused));
    }

    #[test]
    fn a_nonfinite_gradient_component_is_refused() {
        let gradient = Array1::from(vec![0.1, f64::NAN, -0.1]);
        assert!(matches!(classify(-3.0, gradient), Call::Refused));
        let gradient = Array1::from(vec![0.1, f64::NEG_INFINITY, -0.1]);
        assert!(matches!(classify(-3.0, gradient), Call::Refused));
    }

    #[test]
    fn a_finite_answer_keeps_its_gradient_and_a_zero_one_is_a_real_answer() {
        let gradient = Array1::from(vec![0.25, -0.5, 0.0]);
        match classify(-3.0, gradient.clone()) {
            Call::Answer(energy, returned) => {
                assert_eq!(energy, -3.0);
                assert_eq!(returned, gradient);
            }
            _ => panic!("a finite energy and gradient is an answer"),
        }
        assert!(matches!(
            classify(-3.0, Array1::zeros(6)),
            Call::Answer(_, _)
        ));
    }

    #[test]
    fn a_geometry_refused_before_xtb_has_no_gradient_and_costs_nothing() {
        let pot = handle(14, 2);
        let x = silicon_pair(0.5);
        let mut ledger = Ledger::new(10);
        assert!(gradient_call(&pot, &mut ledger, x.view()).is_none());
        let (energy, _) = line_search_call(&pot, &mut ledger, x.view()).expect("a refusal");
        assert!(!energy.is_finite());
        assert_eq!(ledger.spent(), 0);
    }

    #[test]
    fn an_scc_that_does_not_converge_has_no_gradient() {
        let pot = handle(14, 13);
        let x = Array1::from(SI13_SCC_FAILS.to_vec());
        assert!(!rgpot_eindir::refuses_geometry(&SI13_SCC_FAILS));
        let (energy, gradient) = evaluate(&pot.lock().expect("pot"), x.view());
        assert!(energy.is_nan());
        assert!(gradient.iter().all(|value| *value == 0.0));

        let mut ledger = Ledger::new(10);
        assert!(gradient_call(&pot, &mut ledger, x.view()).is_none());
        assert_eq!(ledger.spent(), 1);
        let (energy, _) = line_search_call(&pot, &mut ledger, x.view()).expect("a refusal");
        assert!(!energy.is_finite());
        assert_eq!(ledger.spent(), 2);
    }

    #[test]
    fn a_bound_geometry_has_its_gradient() {
        let pot = handle(14, 2);
        let x = silicon_pair(2.3);
        let mut ledger = Ledger::new(10);
        let gradient = gradient_call(&pot, &mut ledger, x.view()).expect("an answer");
        let (energy, along) = line_search_call(&pot, &mut ledger, x.view()).expect("an answer");
        assert!(energy.is_finite());
        assert_eq!(gradient, along);
        assert!(gradient.iter().all(|value| value.is_finite()));
        assert!(gradient[3].abs() > 1e-3);
        assert_eq!(ledger.spent(), 2);
    }

    #[test]
    fn a_closed_ledger_ends_the_search_rather_than_refuse() {
        let pot = handle(14, 2);
        let x = silicon_pair(2.3);
        let mut ledger = Ledger::new(0);
        assert!(gradient_call(&pot, &mut ledger, x.view()).is_none());
        assert!(line_search_call(&pot, &mut ledger, x.view()).is_none());
    }
}
