//! One held-out LJ38 seed of the recommended anneal arm.
//!
//! Usage: lj38_board_anneal SEED
//! The budget and the reference are the Lennard-Jones board values.
//! A hit is an energy below the reference plus 1e-4, a maximum absolute
//! force at or below 0.0514221, and a charged count inside the budget.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::ArrayView1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::io::Write;
use std::time::Instant;

const N: usize = 38;
const BUDGET: usize = 400_000;
const REFERENCE: f64 = -173.928427;
const HIT_TOL: f64 = 1.0e-4;
const FORCE_BAR: f64 = 0.0514221;

fn main() {
    let seed: u64 = std::env::args()
        .nth(1)
        .expect("usage: lj38_board_anneal SEED")
        .parse()
        .expect("seed");
    if !(6400..=6405).contains(&seed) {
        eprintln!("seed {seed} is outside 6400 through 6405");
        std::process::exit(2);
    }
    let cfg = Config::recommended(N);
    let potential = PairPotential::lennard_jones(N);
    let mut start_rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(
        N,
        cfg.start_radius(),
        cfg.min_separation,
        &mut start_rng,
    );
    let mut ledger = Ledger::new(BUDGET);
    ledger.stop_below(REFERENCE + HIT_TOL);
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
    let mut rng = StdRng::seed_from_u64(seed);
    println!(
        "start seed={seed} budget={BUDGET} reference={REFERENCE:.6} arm=recommended"
    );
    let _ = std::io::stdout().flush();
    let started = Instant::now();
    let outcome = run_with_gradient(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start");
    let wall = started.elapsed().as_secs_f64();
    let (energy, force) = match outcome.best_state.as_ref() {
        Some(state) => {
            let (value, gradient) = potential.value_and_gradient(state.view());
            let force = gradient.iter().fold(0.0_f64, |acc, term| acc.max(term.abs()));
            (value, force)
        }
        None => (outcome.best, f64::INFINITY),
    };
    let hit = energy < REFERENCE + HIT_TOL && force <= FORCE_BAR && outcome.charged <= BUDGET;
    let stop = outcome
        .stop
        .as_ref()
        .map(|reason| reason.code())
        .unwrap_or("unset");
    println!(
        "code=anneal n={N} seed={seed} budget={BUDGET} reference={REFERENCE:.6} \
         hit_tol={HIT_TOL:.6e} force_bar={FORCE_BAR:.7} arm=recommended \
         best={energy:.8} recorded_best={:.8} hit_force={force:.6e} hit={} \
         calls={} charged={} hops={} wall_s={wall:.6} stop={stop} \
         config_sha256={}",
        outcome.best,
        u8::from(hit),
        outcome.charged,
        outcome.charged,
        outcome.hops,
        cfg.resolved_sha256().unwrap_or_else(|_| "unset".into()),
    );
}
