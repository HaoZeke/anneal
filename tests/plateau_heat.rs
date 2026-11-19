//! `plateau_heat` rides the call-plateau clock of the climb.
//!
//! A chain that never runs out that clock hops at the preset temperature
//! and draws the same trajectory as the preset. Once a plateau passes, the
//! walk is hotter and the trajectory leaves the preset one.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use rand::SeedableRng;
use rand::rngs::StdRng;

const SEED: u64 = 2000;
const ATOMS: usize = 26;
const BUDGET: usize = 30_000;

fn walk(cfg: &Config) -> Outcome {
    let potential = PairPotential::lennard_jones(ATOMS);
    let mut rng = StdRng::seed_from_u64(SEED);
    let start = random_cluster_in_radius(ATOMS, cfg.start_radius(), cfg.min_separation, &mut rng);
    let mut ledger = Ledger::new(BUDGET);
    let mut optimiser = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ndarray::ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |point| {
            if !led.charge() {
                return None;
            }
            Some(potential.value_and_gradient(point))
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, point: ndarray::ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        Some(potential.value_and_gradient(point).1)
    };
    let mut rng = StdRng::seed_from_u64(SEED);
    run_with_gradient(
        cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start")
}

fn transitions(outcome: &Outcome) -> Vec<(usize, String, u64, u64)> {
    outcome
        .accepted_transitions
        .iter()
        .map(|edge| {
            (
                edge.hop,
                edge.action.clone(),
                edge.from_energy.to_bits(),
                edge.to_energy.to_bits(),
            )
        })
        .collect()
}

#[test]
fn a_chain_without_a_plateau_walks_the_preset_trajectory() {
    let mut preset = Config::recommended(ATOMS);
    preset.restart_patience = 10 * BUDGET;
    let mut heated = preset.clone();
    heated.plateau_heat = true;
    let (cold, hot) = (walk(&preset), walk(&heated));
    assert_eq!(cold.best.to_bits(), hot.best.to_bits());
    assert_eq!(cold.charged, hot.charged);
    assert_eq!(cold.hops, hot.hops);
    assert_eq!(transitions(&cold), transitions(&hot));
}

#[test]
fn a_plateau_heats_the_walk() {
    let mut preset = Config::recommended(ATOMS);
    preset.restart_patience = 2_000;
    let mut heated = preset.clone();
    heated.plateau_heat = true;
    let (cold, hot) = (walk(&preset), walk(&heated));
    assert_ne!(
        transitions(&cold),
        transitions(&hot),
        "the walk after a plateau is the preset walk"
    );
}
