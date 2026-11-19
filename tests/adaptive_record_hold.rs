//! The adaptive arm holds the incumbent on the single-chain proposal stream,
//! then spends whatever budget remains on a fork of the stream it did not use.
//!
//! Under the opening allowance the adaptive arm and [`Config::recommended`]
//! record the same energy at the same call. Drawing the return move on that
//! stream, or quenching a second replica from it, leaves the single chain.
//! Past the allowance, removing the fork leaves the search on the held chain.
//! This test is both decisions.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{PairKind, PairPotential};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicUsize, Ordering};

const TOL: f64 = 0.001;

fn search(
    adaptive: bool,
    n: usize,
    rho: Option<f64>,
    reference: f64,
    seed: u64,
    budget: usize,
) -> (f64, usize) {
    let length = match rho {
        Some(rho) => PairKind::Morse { rho }.length_scale(),
        None => PairKind::LennardJones.length_scale(),
    };
    let cfg = if adaptive {
        Config::recommended_adaptive_with_scales(n, length, 1.0)
    } else {
        Config::recommended_with_scales(n, length, 1.0)
    };
    let potential = match rho {
        Some(rho) => PairPotential::morse(n, rho),
        None => PairPotential::lennard_jones(n),
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng);
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(reference + TOL);
    let mut optimiser = WarmLbfgs::default();
    let first_seen = AtomicUsize::new(usize::MAX);
    let mut relax = |led: &mut Ledger, x: ndarray::ArrayView1<f64>, iterations: usize| {
        if led.target_reached() && first_seen.load(Ordering::Relaxed) == usize::MAX {
            first_seen.store(led.spent(), Ordering::Relaxed);
        }
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |point| {
            if led.target_reached() && first_seen.load(Ordering::Relaxed) == usize::MAX {
                first_seen.store(led.spent(), Ordering::Relaxed);
            }
            if !led.charge() {
                return None;
            }
            Some(potential.value_and_gradient(point))
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, point: ndarray::ArrayView1<f64>| {
        if led.target_reached() && first_seen.load(Ordering::Relaxed) == usize::MAX {
            first_seen.store(led.spent(), Ordering::Relaxed);
        }
        if !led.charge() {
            return None;
        }
        Some(potential.value_and_gradient(point).1)
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let outcome = run_with_gradient(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start");
    let seen = first_seen.load(Ordering::Relaxed);
    let first_calls = if seen == usize::MAX {
        ledger.spent()
    } else {
        seen
    };
    (outcome.best, first_calls)
}

#[test]
fn adaptive_search_holds_the_incumbent_then_forks_the_unused_stream() {
    // Inside the opening allowance the fork never runs, so the adaptive arm
    // is the single chain. These two seeds hit on that chain.
    for seed in [2000_u64, 2001] {
        let (base, base_calls) = search(false, 38, None, -173.928427, seed, 200_000);
        let (held, held_calls) = search(true, 38, None, -173.928427, seed, 200_000);
        assert!(
            base <= -173.928427 + TOL,
            "recommended LJ38 seed {seed} stopped at {base:.6}"
        );
        assert_eq!(
            held.to_bits(),
            base.to_bits(),
            "LJ38 seed {seed} left the held stream: {held:.6} against {base:.6}"
        );
        assert_eq!(
            held_calls, base_calls,
            "LJ38 seed {seed} first hit at {held_calls}, the single chain at {base_calls}"
        );
    }

    // Past the allowance the held chain has stopped. This seed does not hit
    // inside the opening run, so a search that never forks reports the
    // opening spend. The fork spends the calls that were held back.
    let budget = (1usize << 19) + 80_000;
    let (forked, forked_calls) = search(true, 75, None, -397.492331, 2000, budget);
    assert!(
        forked_calls > 1usize << 19,
        "LJ75 seed 2000 stopped at {forked_calls} calls ({forked:.6}) and never spent the forked tail"
    );
    assert!(
        forked_calls == budget || forked <= -397.492331 + TOL,
        "LJ75 seed 2000 stopped early at {forked_calls} with {forked:.6}"
    );
}
