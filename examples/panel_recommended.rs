//! One recommended search.
//!
//! `panel_recommended N SEED BUDGET`
//!
//! The budget is the call ceiling for that run. This file does not change
//! the library. The printed `calls` value is the first recorded energy at
//! the published reference. A miss prints `calls=none`. The budget is not
//! that charge.

use std::env;
use std::time::Instant;

use anneal_core::methods::cluster_hopping::{Config, Ledger, MoveLibrary};
use anneal_core::methods::cluster_search::{first_encounter, search};
use anneal_core::potentials::PairPotential;

const HIT_TOLERANCE: f64 = 0.001;

fn reference(n: usize) -> f64 {
    match n {
        38 => -173.928427,
        55 => -279.248470,
        75 => -397.492331,
        98 => -543.665361,
        104 => -582.086642,
        _ => panic!("no published reference for {n}"),
    }
}

fn library_name(library: &MoveLibrary) -> &'static str {
    match library {
        MoveLibrary::LeanBurst => "LeanBurst",
        MoveLibrary::Reseed => "Reseed",
        MoveLibrary::WalesDoye => "WalesDoye",
        _ => "other",
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let n: usize = args.next().expect("n").parse().expect("n");
    let seed: u64 = args.next().expect("seed").parse().expect("seed");
    let budget: usize = args.next().expect("budget").parse().expect("budget");
    let cfg = Config::recommended(n);
    let pot = PairPotential::lennard_jones(n);
    let mut ledger = Ledger::new(budget);
    let target = reference(n);
    let started = Instant::now();
    let (out, _) = search(&pot, &cfg, &mut ledger, seed).expect("finite start");
    let wall_s = started.elapsed().as_secs_f64();
    let encounter = first_encounter(&out, target, HIT_TOLERANCE, ledger.spent());
    println!("arm=lj{n}");
    println!("seed={seed}");
    println!("n={n}");
    println!("ceiling={budget}");
    println!("library={}", library_name(&cfg.move_library));
    println!("surfaces={}", cfg.surfaces.len());
    println!("relax_steps={}", cfg.relax_steps);
    println!("record_gradient={:.6e}", cfg.record_gradient);
    println!("screen_steps={}", cfg.screen_steps);
    println!("hit={}", u8::from(encounter.found()));
    if encounter.found() {
        println!("calls={}", encounter.charged());
    } else {
        println!("calls=none");
    }
    println!("energy={:.6}", ledger.best);
    println!("spent={}", ledger.spent());
    println!("wall_s={wall_s:.6}");
    println!("hops={}", out.hops);
}
