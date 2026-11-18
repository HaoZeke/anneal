//! The adaptive arm holds the incumbent on the single-chain proposal stream,
//! then spends whatever budget remains on a fork of the stream it did not use.
//!
//! Lennard-Jones 38 and Morse 38 at rho = 10 are solved by that single chain.
//! Drawing the untried escape arm in its place, or spending the incumbent's
//! random stream on a second replica, leaves those basins and the chain sits
//! on one shelf. A later fork cannot recover a hit the opening run already
//! had the calls to find. The larger Morse gain is the other way round: the
//! held chain does not find it, and removing the fork leaves the search on
//! that chain. This test is both decisions. It fails if either is removed.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{PairKind, PairPotential};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicUsize, Ordering};

const TOL: f64 = 0.001;

fn search(n: usize, rho: Option<f64>, reference: f64, seed: u64, budget: usize) -> (f64, usize) {
    let length = match rho {
        Some(rho) => PairKind::Morse { rho }.length_scale(),
        None => PairKind::LennardJones.length_scale(),
    };
    let cfg = Config::recommended_adaptive_with_scales(n, length, 1.0);
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
    let (lj, lj_calls) = search(38, None, -173.928427, 2004, 40_000);
    assert!(
        lj <= -173.928427 + TOL,
        "LJ38 seed 2004 left the held stream and stopped at {lj:.6}"
    );
    assert_eq!(lj_calls, 6_483, "LJ38 seed 2004 first hit at {lj_calls}");

    let (lj_shifted, _) = search(38, None, -173.928427, 2000, 40_000);
    assert!(
        lj_shifted <= -173.928427 + TOL,
        "LJ38 seed 2000 spent the incumbent stream before the first hop and stopped at {lj_shifted:.6}"
    );

    let (morse, _) = search(38, Some(10.0), -145.849817, 2001, 80_000);
    assert!(
        morse <= -145.849817 + TOL,
        "Morse 38 rho=10 seed 2001 left the held stream and stopped at {morse:.6}"
    );

    let (late, late_calls) = search(38, Some(10.0), -145.849817, 2003, 400_000);
    assert!(
        late <= -145.849817 + TOL,
        "Morse 38 rho=10 seed 2003 was cut off before its single-chain hit and stopped at {late:.6}"
    );
    assert_eq!(
        late_calls, 348_489,
        "Morse 38 rho=10 seed 2003 first hit at {late_calls}"
    );

    let (gain, gain_calls) = search(75, Some(14.0), -318.40732976, 2007, 700_000);
    assert!(
        gain <= -318.40732976 + TOL,
        "Morse 75 rho=14 seed 2007 never ran the forked stream and stopped at {gain:.6}"
    );
    assert_eq!(
        gain_calls, 614_874,
        "Morse 75 rho=14 seed 2007 first hit at {gain_calls}"
    );
}
