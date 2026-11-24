// Copyright (c) anneal developers
//! Every potential call a search makes is on its ledger.
//!
//! The counters sit at the potential boundary: one on the relaxation
//! closure's `value_and_gradient` call and one on the gradient closure's.
//! They do not read the ledger. After `run_with_gradient` returns, the
//! ledger must hold at least as many calls as the potential saw
//! (`raw <= spent + settled_beyond_cap`), and `outcome.charged` must equal
//! `ledger.spent()`, which never passes the budget.
//!
//! `spent - raw` is not always zero. The energy-lattice arm sums pair
//! energies over vacant sites with its own arithmetic, makes no call into
//! the potential, and charges the work as a fraction of one evaluation
//! (`Ledger::charge_frac`). `Ledger::priced` reports the whole units that
//! settled that way, so the invariant is exact:
//! `raw + priced == spent + settled_beyond_cap`. The last term holds settled
//! calls that did not fit under the budget, and it is zero in these runs.
//! With the lattice arm off, `priced` is zero and `raw == spent`.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::ArrayView1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Cambridge Cluster Database Lennard-Jones global minima.
fn reference(n: usize) -> f64 {
    match n {
        26 => -108.315616,
        38 => -173.928427,
        75 => -397.492331,
        98 => -543.665361,
        _ => unreachable!("no reference for {n}"),
    }
}

struct Tally {
    outcome: Outcome,
    spent: usize,
    priced: usize,
    beyond_cap: usize,
    budget: usize,
    raw: usize,
}

fn walk(cfg: &Config, n: usize, seed: u64, budget: usize, stop: bool) -> Tally {
    anneal_core::install_rgpot_kernels!();
    let potential = PairPotential::lennard_jones(n);
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng);
    let relaxed_calls = AtomicUsize::new(0);
    let gradient_calls = AtomicUsize::new(0);
    let mut ledger = Ledger::new(budget);
    if stop {
        ledger.stop_below(reference(n) + anneal_core::DEFAULT_HIT_TOL);
    }
    let mut optimiser = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |point| {
            if !led.charge() {
                return None;
            }
            relaxed_calls.fetch_add(1, Ordering::Relaxed);
            Some(potential.value_and_gradient(point))
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, point: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        gradient_calls.fetch_add(1, Ordering::Relaxed);
        Some(potential.value_and_gradient(point).1)
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let outcome = run_with_gradient(
        cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start");
    Tally {
        outcome,
        spent: ledger.spent(),
        priced: ledger.priced(),
        beyond_cap: ledger.settled_beyond_cap(),
        budget,
        raw: relaxed_calls.load(Ordering::Relaxed) + gradient_calls.load(Ordering::Relaxed),
    }
}

fn assert_parity(label: &str, tally: &Tally) {
    println!(
        "{label}: raw {} priced {} spent {} beyond cap {} hops {}",
        tally.raw, tally.priced, tally.spent, tally.beyond_cap, tally.outcome.hops
    );
    assert_eq!(
        tally.outcome.charged, tally.spent,
        "{label}: outcome.charged and ledger.spent() differ"
    );
    assert!(
        tally.spent <= tally.budget,
        "{label}: spent {} passed the budget {}",
        tally.spent,
        tally.budget
    );
    let settled = tally.spent + tally.beyond_cap;
    assert!(
        tally.raw <= settled,
        "{label}: the potential saw {} calls and the ledger holds {settled}",
        tally.raw,
    );
    assert_eq!(
        tally.raw + tally.priced,
        settled,
        "{label}: raw {} + priced {} != spent {} + beyond cap {}",
        tally.raw,
        tally.priced,
        tally.spent,
        tally.beyond_cap
    );
}

/// The default offers the packing probe after 36 hops. Zero offers it before
/// the first hop. Both settle the probe's side ledger onto the main one.
const DEFERRALS: [Option<usize>; 2] = [None, Some(0)];

fn recommended(n: usize, deferral: Option<usize>) -> Config {
    let mut cfg = Config::recommended(n);
    if let Some(hops) = deferral {
        cfg.packing_deferral = hops;
    }
    cfg
}

#[test]
fn every_potential_call_is_on_the_ledger_for_lj26_and_lj38() {
    for n in [26usize, 38] {
        for seed in [2000u64, 2001, 2002] {
            for deferral in DEFERRALS {
                let tally = walk(&recommended(n, deferral), n, seed, 60_000, true);
                assert_parity(&format!("LJ{n} seed {seed} deferral {deferral:?}"), &tally);
            }
        }
    }
}

#[test]
fn every_potential_call_is_on_the_ledger_for_lj75_and_lj98() {
    for n in [75usize, 98] {
        for seed in [2000u64, 2001, 2002] {
            for deferral in DEFERRALS {
                let tally = walk(&recommended(n, deferral), n, seed, 40_000, true);
                assert_parity(&format!("LJ{n} seed {seed} deferral {deferral:?}"), &tally);
            }
        }
    }
}

#[test]
fn without_the_lattice_arm_the_potential_count_equals_the_ledger() {
    for n in [26usize, 38, 75, 98] {
        for seed in [2000u64, 2001, 2002] {
            let mut cfg = Config::recommended(n);
            cfg.displacement_only = true;
            let tally = walk(&cfg, n, seed, 20_000, true);
            assert_eq!(tally.priced, 0, "LJ{n} seed {seed}: nothing is priced");
            assert_eq!(
                tally.raw, tally.spent,
                "LJ{n} seed {seed}: the potential saw {} calls and the ledger holds {}",
                tally.raw, tally.spent
            );
            assert_parity(&format!("LJ{n} seed {seed} displacement"), &tally);
        }
    }
}
