//! One fresh run of [`Config::recommended`].
//!
//! `lj38_honest N SEED` spends [`OAKLEY_LJ98_MEAN_EVALUATIONS`].
//! That count is the one call ceiling the article uses at every size.
//! A further argument replaces it.
//! `lj38_honest plain SEED BUDGET` turns the stall exits and the Bayes screen off.
//!
//! The hit is the first improvement whose energy lies strictly below the
//! published reference plus [`HIT_TOLERANCE`].
//! A crossing recorded when the ledger has spent exactly the ceiling is the
//! last charge [`Ledger::charge`] accepts.
//! `charge` returns false once `spent >= budget`, so the count does not
//! step past the ceiling.

use std::env;
use std::time::Instant;

use anneal_core::methods::cluster_hopping::{Config, Ledger, MoveLibrary};
use anneal_core::methods::cluster_search::{first_encounter, search};
use anneal_core::potentials::PairPotential;

/// First converged energy strictly below the published reference plus this window.
///
/// The Cambridge Cluster Database quotes these minima to six decimal places.
/// On its LJ38 row the next listed minimum, the icosahedron at -173.252378,
/// lies 0.676 above the octahedron, so this window is below that gap and
/// above the last quoted digit.
/// Table: <https://www-wales.ch.cam.ac.uk/~wales/CCD/jon/structures/LJ/tables.150.html>
const HIT_TOLERANCE: f64 = 0.001;

/// Oakley, Johnston and Wales, Phys. Chem. Chem. Phys. 15, 3965 (2013), Table 1.
///
/// Mean first encounter of core-orbit basin hopping on LJ98, over 100
/// successful random starts, in evaluations.
/// The article uses this one count as the call ceiling at every size.
const OAKLEY_LJ98_MEAN_EVALUATIONS: usize = 109_753;

/// Cambridge Cluster Database, Lennard-Jones global minima, the digits on the
/// table cited above.
///
/// LJ38, Oh, -173.928427, first reported by Gomez and Romero, Pillardy and
/// Piela, and Doye, Wales and Berry.
/// LJ55, Ih, -279.248470, Hoare and Pal, Adv. Phys. 20, 161 (1971).
/// LJ75, D5h, -397.492331, Doye, Wales and Berry, J. Chem. Phys. 103, 4234 (1995).
/// LJ98, Td, -543.665361, Leary and Doye, Phys. Rev. E 60, R6320 (1999).
/// LJ104, C2v, -582.086642, Doye and Wales, Chem. Phys. Lett. 247, 339 (1995).
const LJ38: f64 = -173.928427;
const LJ55: f64 = -279.248470;
const LJ75: f64 = -397.492331;
const LJ98: f64 = -543.665361;
const LJ104: f64 = -582.086642;

fn library_name(library: &MoveLibrary) -> &'static str {
    match library {
        MoveLibrary::LeanBurst => "LeanBurst",
        MoveLibrary::Reseed => "Reseed",
        MoveLibrary::WalesDoye => "WalesDoye",
        _ => "other",
    }
}

fn assert_preset() {
    let sizes = [13usize, 38, 55, 75, 97, 98, 104];
    let reference = Config::recommended(sizes[0]);
    for &n in &sizes {
        let cfg = Config::recommended(n);
        assert!(
            matches!(cfg.move_library, MoveLibrary::LeanBurst),
            "recommended({n}) keeps the lean burst library"
        );
        assert!(
            cfg.surfaces.is_empty(),
            "recommended({n}) leaves surfaces empty"
        );
        assert_eq!(cfg.relax_steps, Config::RECOMMENDED_RELAX_STEPS);
        assert!((cfg.record_gradient - 1.0e-3).abs() < 1.0e-15);
        let mut left = cfg;
        let mut right = reference.clone();
        left.n_points = 0;
        right.n_points = 0;
        left.container = 0.0;
        right.container = 0.0;
        assert_eq!(format!("{left:?}"), format!("{right:?}"), "n={n}");
    }
}

fn reference(n: usize) -> f64 {
    match n {
        38 => LJ38,
        55 => LJ55,
        75 => LJ75,
        98 => LJ98,
        104 => LJ104,
        _ => panic!("no published reference for {n}"),
    }
}

fn report(
    n: usize,
    seed: u64,
    ceiling: usize,
    cfg: &Config,
    out_best: f64,
    encounter_found: bool,
    encounter_calls: usize,
    spent: usize,
    hops: usize,
    wall_s: f64,
    plain: bool,
) {
    let target = if plain { LJ38 } else { reference(n) };
    let best_below = out_best < target + HIT_TOLERANCE;
    let at_ceiling = u8::from(encounter_found && encounter_calls == ceiling);
    let past_ceiling = u8::from(encounter_calls > ceiling || spent > ceiling);
    println!(
        "arm={}",
        if plain {
            "lj38-plain".to_string()
        } else {
            format!("lj{n}")
        }
    );
    println!("seed={seed}");
    println!("n={n}");
    println!("ceiling={ceiling}");
    println!("library={}", library_name(&cfg.move_library));
    println!("surfaces={}", cfg.surfaces.len());
    println!("relax_steps={}", cfg.relax_steps);
    println!("record_gradient={:.6e}", cfg.record_gradient);
    println!("screen_steps={}", cfg.screen_steps);
    println!("hit={}", u8::from(encounter_found));
    println!("calls={encounter_calls}");
    println!("trace_found={}", u8::from(encounter_found));
    println!("trace_calls={encounter_calls}");
    println!("best_below={}", u8::from(best_below));
    println!("at_ceiling={at_ceiling}");
    println!("past_ceiling={past_ceiling}");
    println!("energy={out_best:.6}");
    println!("spent={spent}");
    println!("wall_s={wall_s:.6}");
    println!("hops={hops}");
}

fn run_recommended(n: usize, seed: u64, ceiling: usize) {
    assert_preset();
    let cfg = Config::recommended(n);
    let pot = PairPotential::lennard_jones(n);
    let mut ledger = Ledger::new(ceiling);
    let target = reference(n);
    let started = Instant::now();
    let (out, _) = search(&pot, &cfg, &mut ledger, seed).expect("finite start");
    let wall_s = started.elapsed().as_secs_f64();
    let encounter = first_encounter(&out, target, HIT_TOLERANCE, ledger.spent());
    report(
        n,
        seed,
        ceiling,
        &cfg,
        ledger.best,
        encounter.found(),
        encounter.charged(),
        ledger.spent(),
        out.hops,
        wall_s,
        false,
    );
}

fn run_plain(seed: u64, budget: usize) {
    assert_preset();
    let mut cfg = Config::recommended(38);
    cfg.trail_on_stall = false;
    cfg.escape_on_stall = false;
    cfg.restart_on_stall = false;
    cfg.bayes_screen = false;
    cfg.surfaces.clear();
    let pot = PairPotential::lennard_jones(38);
    let mut ledger = Ledger::new(budget);
    let started = Instant::now();
    let (out, _) = search(&pot, &cfg, &mut ledger, seed).expect("finite start");
    let wall_s = started.elapsed().as_secs_f64();
    let encounter = first_encounter(&out, LJ38, HIT_TOLERANCE, ledger.spent());
    report(
        38,
        seed,
        budget,
        &cfg,
        ledger.best,
        encounter.found(),
        encounter.charged(),
        ledger.spent(),
        out.hops,
        wall_s,
        true,
    );
    println!("stall_exits=off");
}

fn main() {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        None => assert_preset(),
        Some("plain") => {
            let seed: u64 = args.next().expect("seed").parse().expect("seed");
            let budget: usize = args.next().expect("budget").parse().expect("budget");
            run_plain(seed, budget);
        }
        Some(n) => {
            let n: usize = n.parse().expect("n");
            let seed: u64 = args.next().expect("seed").parse().expect("seed");
            let ceiling = match args.next() {
                Some(text) => text.parse().expect("ceiling"),
                None => OAKLEY_LJ98_MEAN_EVALUATIONS,
            };
            run_recommended(n, seed, ceiling);
        }
    }
}
