// Copyright (c) anneal developers
//! One seed of `Config::recommended` on a Lennard-Jones cluster, with the
//! potential calls counted at the potential boundary.
//!
//! ```text
//! paired_check <arm> <n> <seed> <budget>
//! ```
//!
//! The row is `arm,n,seed,budget,hit,evals_to_hit,charged,raw,wall_s,best,head,hops`.
//! `raw` counts each `value_and_gradient` call the relaxation and the
//! gradient closures make, so it is independent of the ledger. `charged` is
//! `Outcome::charged`. The search stops at the published minimum plus
//! `DEFAULT_HIT_TOL`, or at the budget. A miss reports the budget as its
//! evaluations. Run the same file against two trees to pair them.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::ArrayView1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Cambridge Cluster Database Lennard-Jones global minima.
fn reference(n: usize) -> Option<f64> {
    match n {
        26 => Some(-108.315616),
        38 => Some(-173.928427),
        55 => Some(-279.248470),
        75 => Some(-397.492331),
        98 => Some(-543.665361),
        _ => None,
    }
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(arm), Some(n), Some(seed), Some(budget)) = (
        args.first(),
        args.get(1).and_then(|v| v.parse::<usize>().ok()),
        args.get(2).and_then(|v| v.parse::<u64>().ok()),
        args.get(3).and_then(|v| v.parse::<usize>().ok()),
    ) else {
        eprintln!("usage: paired_check <arm> <n> <seed> <budget>");
        std::process::exit(2);
    };
    let Some(reference) = reference(n) else {
        eprintln!("no reference minimum for n = {n}");
        std::process::exit(2);
    };
    let cfg = Config::recommended(n);
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng);
    let potential = PairPotential::lennard_jones(n);
    let raw = AtomicU64::new(0);
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(reference + anneal_core::DEFAULT_HIT_TOL);
    let mut optimiser = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            if !led.charge() {
                return None;
            }
            raw.fetch_add(1, Ordering::Relaxed);
            Some(potential.value_and_gradient(v))
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        raw.fetch_add(1, Ordering::Relaxed);
        Some(potential.value_and_gradient(v).1)
    };
    let mut search_rng = StdRng::seed_from_u64(seed);
    let started = Instant::now();
    let outcome = run_with_gradient(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut search_rng,
    )
    .expect("finite start");
    let wall = started.elapsed().as_secs_f64();
    let hit = anneal_core::energy_hits(outcome.best, reference, anneal_core::DEFAULT_HIT_TOL);
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| {
            anneal_core::energy_hits(*energy, reference, anneal_core::DEFAULT_HIT_TOL)
        })
        .map(|(_, spent, _, _)| *spent);
    let evals = if hit {
        at_hit.unwrap_or(outcome.charged)
    } else {
        budget
    };
    let head = std::process::Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|text| text.trim().to_string())
        .unwrap_or_default();
    println!(
        "{arm},{n},{seed},{budget},{hit},{evals},{},{},{wall:.6},{:.8},{head},{}",
        outcome.charged,
        raw.load(Ordering::Relaxed),
        outcome.best,
        outcome.hops,
    );
}
