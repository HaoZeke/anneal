// Copyright (c) anneal developers
//! `plateau_heat` is an acceptance temperature. Each test pins a walk
//! that moves when one heated site is reverted.

use anneal_core::methods::cluster_hopping::{
    Config, ContinuousSymmetry, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
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

/// FNV-1a over the accepted edges. Stable for this toolchain, unlike
/// `std` hashing.
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
    mix(&(outcome.hops as u64).to_le_bytes());
    mix(&(outcome.charged as u64).to_le_bytes());
    state
}

fn pair(cfg: &Config, budget: usize, seed: u64) -> (Outcome, Outcome) {
    let mut hot = cfg.clone();
    hot.plateau_heat = true;
    (walk(cfg, budget, seed), walk(&hot, budget, seed))
}

#[test]
fn a_chain_without_a_plateau_walks_the_preset_trajectory() {
    let budget = 30_000;
    let mut cfg = Config::recommended(ATOMS);
    cfg.restart_patience = budget * 10;
    let (cold, hot) = pair(&cfg, budget, 2000);
    assert_eq!(hot.best, cold.best);
    assert_eq!(hot.hops, cold.hops);
    assert_eq!(hot.charged, cold.charged);
    assert_eq!(hot.accepted_transitions, cold.accepted_transitions);
}

/// `energy_lattice` is off, which also gates the generic packing probe
/// in `run_full`, so these are the walks develop pins. They are not the
/// walks the branch pinned before it merged develop (cold
/// `0xa4cf0197d21e3961`, hot `0xca208bc0360518d7`). What moved them is
/// develop's observed-basis stall probe, which withholds its side calls
/// from the walk's budget: with that probe switched off in a scratch
/// build, this test returns the older pair.
#[test]
fn metropolis_heat_follows_the_plateau() {
    let mut cfg = Config::recommended(ATOMS);
    cfg.restart_patience = 2_000;
    cfg.energy_lattice = false;
    let (cold, hot) = pair(&cfg, 30_000, 2000);
    let cold_fp = fingerprint(&cold);
    let hot_fp = fingerprint(&hot);
    assert_eq!(cold_fp, 0xc2e2c1ef481b1a93, "preset walk");
    assert_ne!(hot_fp, cold_fp, "temperature *= heat");
    assert_eq!(hot_fp, 0xcc74d55e4c74ed2f, "PLATEAU_HEAT_CAP");
}

/// Two code paths meet in this walk, and each pin has a different cause.
///
/// Cold `0x100885ac118293e1` is the branch's walk. The generic packing
/// probe changes it from develop's `0xfb5afaab4e40a443`; the observed-basis
/// stall probe fires three times here and does not change it.
///
/// Hot `0x1a623a8e37ad1d43` is in neither parent of the merge that first
/// pinned it. The branch before that merge pinned `0x44038697bde16fb7`
/// and develop pinned `0xb2078435e4bed155`. It is the composition: the
/// packing probe from this branch, and develop's observed-basis probe,
/// whose side calls come out of the heated walk's budget and cost it one
/// hop. With the observed-basis probe switched off in a scratch build the
/// hot walk is `0x44038697bde16fb7` again; with the packing probe
/// suppressed it is develop's value, asserted below.
#[test]
fn lattice_exchange_follows_the_plateau() {
    let mut cfg = Config::recommended(ATOMS);
    cfg.restart_patience = 2_000;
    let (cold, hot) = pair(&cfg, 30_000, 2000);
    assert_eq!(fingerprint(&cold), 0x100885ac118293e1, "preset walk");
    assert_eq!(
        fingerprint(&hot),
        0x1a623a8e37ad1d43,
        "heat * cfg.temperature"
    );

    // Measured: moving the offer from hop 36 to before the first hop leaves
    // both fingerprints where they are in this walk.
    cfg.packing_deferral = 0;
    let (cold, hot) = pair(&cfg, 30_000, 2000);
    assert_eq!(fingerprint(&cold), 0x100885ac118293e1, "probe at hop 0");
    assert_eq!(fingerprint(&hot), 0x1a623a8e37ad1d43, "probe at hop 0");

    cfg.suppress_generic_packings = true;
    let (cold, hot) = pair(&cfg, 30_000, 2000);
    assert_eq!(
        fingerprint(&cold),
        0xfb5afaab4e40a443,
        "without the packing probe the preset walk is develop's"
    );
    assert_eq!(
        fingerprint(&hot),
        0xb2078435e4bed155,
        "without the packing probe the heated walk is develop's"
    );
}

/// `energy_lattice` is off and the observed-basis probe never fires in
/// this 8_000-call walk, so these pins are the same on develop, on the
/// branch before and after its merges, and on the merged tree.
#[test]
fn a_new_lowest_returns_the_walk_to_the_preset() {
    let mut cfg = Config::recommended(ATOMS);
    cfg.restart_patience = 80;
    cfg.relax_steps = 20;
    cfg.energy_lattice = false;
    let (cold, hot) = pair(&cfg, 8_000, 2000);
    assert_eq!(fingerprint(&cold), 0x79a1ba0ae27370ad, "preset walk");
    assert_eq!(fingerprint(&hot), 0x87b7ee9f3579cc7e, "heat = 1.0");
}

/// The walk makes fewer than 36 hops in 6_000 calls, so at the default the
/// generic packing probe is never offered and both pins are develop's (cold
/// `0x3f6e77c9980269b9`, hot `0xc711fc71269b412d`), the same as with the
/// probe suppressed. With `packing_deferral = 0` the probe runs before the
/// first hop and the walk is the one the branch pinned until the offer was
/// deferred (cold `0xdd5b3f896df84eb7`, hot `0x4f443a96ed586f29`). The
/// observed-basis probe does not fire in this walk, so develop's changes to
/// it do not touch these.
#[test]
fn a_same_iteration_record_returns_to_the_preset() {
    let mut cfg = Config::recommended(ATOMS);
    cfg.restart_patience = 60;
    cfg.relax_steps = 40;
    cfg.general_collective = true;
    cfg.continuous_symmetry = ContinuousSymmetry::Measure { interval: 1 };

    let (cold, hot) = pair(&cfg, 6_000, 10);
    assert!(cold.hops < 36 && hot.hops < 36, "the probe is not offered");
    assert_eq!(
        fingerprint(&cold),
        0x3f6e77c9980269b9,
        "preset walk, probe not offered"
    );
    assert_eq!(
        fingerprint(&hot),
        0xc711fc71269b412d,
        "same-iteration heat = 1.0, probe not offered"
    );

    cfg.packing_deferral = 0;
    let (cold, hot) = pair(&cfg, 6_000, 10);
    assert_eq!(
        fingerprint(&cold),
        0xdd5b3f896df84eb7,
        "preset walk, probe at hop 0"
    );
    assert_eq!(
        fingerprint(&hot),
        0x4f443a96ed586f29,
        "same-iteration heat = 1.0, probe at hop 0"
    );

    cfg.suppress_generic_packings = true;
    let (cold, hot) = pair(&cfg, 6_000, 10);
    assert_eq!(
        fingerprint(&cold),
        0x3f6e77c9980269b9,
        "without the packing probe the preset walk is develop's"
    );
    assert_eq!(
        fingerprint(&hot),
        0xc711fc71269b412d,
        "without the packing probe the heated walk is develop's"
    );
}
