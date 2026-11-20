// Copyright (c) anneal developers
//! `Knockouts` with nothing named must leave the preset bit-identical, and
//! each named knockout must set only the keys it documents.

use anneal_core::methods::cluster_hopping::{
    ArmSelection, Config, Knockouts, Ledger, Outcome, SoapProposalMode, random_cluster_in_radius,
    run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use rand::SeedableRng;
use rand::rngs::StdRng;

const ATOMS: usize = 26;

fn walk(cfg: &Config, budget: usize, seed: u64) -> Outcome {
    let potential = PairPotential::lennard_jones(ATOMS);
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(ATOMS, cfg.start_radius(), cfg.min_separation, &mut rng);
    let mut ledger = Ledger::new(budget);
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
    let mut rng = StdRng::seed_from_u64(seed);
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

/// FNV-1a over the accepted edges, the best energy, and the counters.
fn fingerprint(outcome: &Outcome) -> u64 {
    let mut state = 0xcbf29ce484222325u64;
    let mut mix = |bytes: &[u8]| {
        for byte in bytes {
            state ^= *byte as u64;
            state = state.wrapping_mul(0x100000001b3);
        }
    };
    for edge in &outcome.accepted_transitions {
        mix(&edge.hop.to_le_bytes());
        mix(edge.action.as_bytes());
        mix(&edge.from_energy.to_bits().to_le_bytes());
        mix(&edge.to_energy.to_bits().to_le_bytes());
    }
    mix(&outcome.best.to_bits().to_le_bytes());
    for value in outcome.best_state.iter().flatten() {
        mix(&value.to_bits().to_le_bytes());
    }
    mix(&(outcome.hops as u64).to_le_bytes());
    mix(&(outcome.charged as u64).to_le_bytes());
    state
}

#[test]
fn no_knockout_leaves_the_preset_bit_identical() {
    let stock = Config::recommended(ATOMS);
    let mut switched = stock.clone();
    switched.apply_knockouts(Knockouts::parse("").expect("empty list"));
    assert_eq!(
        stock.resolved_json().expect("stock json"),
        switched.resolved_json().expect("switched json")
    );
    for seed in [0, 1, 2] {
        let a = walk(&stock, 20_000, seed);
        let b = walk(&switched, 20_000, seed);
        assert_eq!(fingerprint(&a), fingerprint(&b), "seed {seed}");
        assert_eq!(a.best.to_bits(), b.best.to_bits(), "seed {seed}");
    }
}

#[test]
fn each_knockout_sets_only_its_own_keys() {
    let stock = Config::recommended(ATOMS);
    assert!(stock.prior_soap);
    assert_eq!(stock.arm_selection, ArmSelection::Posterior);

    let mut soap = stock.clone();
    soap.apply_knockouts(Knockouts::parse("prior_soap").expect("prior_soap"));
    assert!(!soap.prior_soap);
    assert_eq!(soap.soap_mode, SoapProposalMode::Off);
    assert_eq!(soap.arm_selection, stock.arm_selection);

    let mut uniform = stock.clone();
    uniform.apply_knockouts(Knockouts::parse("move_choice").expect("move_choice"));
    assert_eq!(uniform.arm_selection, ArmSelection::Uniform);
    assert_eq!(uniform.prior_soap, stock.prior_soap);
    assert_eq!(uniform.soap_mode, stock.soap_mode);

    for knocked in [&soap, &uniform] {
        assert_ne!(
            stock.resolved_json().expect("stock json"),
            knocked.resolved_json().expect("knocked json")
        );
    }
}

#[test]
fn an_unknown_knockout_is_refused() {
    for spec in ["templates", "prior_soap,restart", "soap"] {
        let err = Knockouts::parse(spec).expect_err(spec);
        assert!(
            err.message.contains("unknown knockout"),
            "{spec}: {}",
            err.message
        );
    }
}
