//! One Lennard-Jones seed on the library stream.
//!
//! `optimize_with_gradient` draws the start and continues the search on one
//! `StdRng`. A second stream reseeded at the same integer is a different
//! search. The hit is one-sided: a returned improvement at or below the
//! reference plus the tolerance.

use anneal_core::methods::cluster_hopping::{Config, Ledger, optimize_with_gradient};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::ArrayView1;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().expect("n").parse().expect("n");
    let seed: u64 = args.next().expect("seed").parse().expect("seed");
    let budget: usize = args.next().expect("budget").parse().expect("budget");
    let reference: f64 = args.next().expect("reference").parse().expect("reference");
    let tol: f64 = args.next().expect("tol").parse().expect("tol");
    let preset = args.next().unwrap_or_else(|| "recommended".to_owned());
    let cfg = match preset.as_str() {
        "recommended" => Config::recommended(n),
        "derived" => Config::derived(n),
        "for_cluster" => Config::for_cluster(n),
        "communicating" => Config::communicating(n),
        other => panic!("unknown preset {other}"),
    };
    let digest = cfg.resolved_sha256().expect("digest");
    let potential = PairPotential::lennard_jones(n);
    let mut ledger = Ledger::new(budget);
    let mut optimiser = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            if !led.charge() {
                return None;
            }
            Some(potential.value_and_gradient(v))
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        Some(potential.value_and_gradient(v).1)
    };
    let started = Instant::now();
    let outcome = optimize_with_gradient(&cfg, &mut ledger, &mut relax, Some(&mut grad), seed)
        .expect("finite start");
    let wall = started.elapsed().as_secs_f64();
    let force_to_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| *energy <= reference + tol)
        .map(|(_, spent, _, _)| *spent);
    let hit = force_to_hit.is_some();
    println!(
        "{{\"n\":{n},\"seed\":{seed},\"budget\":{budget},\"reference\":{reference},\"tol\":{tol},\
\"config\":\"{preset}\",\"config_digest\":\"{digest}\",\
\"charged\":{},\"best\":{:.12},\"hit\":{hit},\"force_to_hit\":{},\"wall_seconds\":{wall:.6},\
\"hops\":{}}}",
        outcome.charged,
        outcome.best,
        force_to_hit
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".into()),
        outcome.hops,
    );
}
