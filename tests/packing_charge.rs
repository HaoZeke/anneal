// Copyright (c) anneal developers
//! The generic packing probe's potential calls are all on the ledger, and a
//! walk that hits before the probe is offered does not pay for it.
//!
//! develop `983f6f69ea1555e51a71874870410966ac726616` has no probe. It hits
//! Lennard-Jones 26, seed 2000, at 502 charged calls. The probe is offered
//! at the top of the hop loop after `Config::packing_deferral` times nine
//! hops, 36 hops by default, and the walk's hit comes after two hops. The
//! default walk is therefore develop's walk and charges 502.
//!
//! With `packing_deferral = 0` the probe runs before the first hop and
//! spends 362 potential calls, 931 before each candidate was screened
//! against the walk's best and 325 before refused candidates took the
//! walk's audit escape. The 37 the escape adds are the full relaxation
//! of audited refusals. The same hit is then charged at 502 + 362 = 864,
//! and the ledger, the improvement row and `outcome.charged` all say so.
//!
//! Lennard-Jones 38, seed 2000, is a walk that runs past the offer. The
//! probe runs once, reaches the target and its calls are on the ledger.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicUsize, Ordering};

const SEED: u64 = 2000;
const BUDGET: usize = 30_000;

/// The energy-lattice arm charges its site scans straight to the ledger,
/// with no potential call. The Lennard-Jones 26 walk is charged four calls
/// that way. develop shows the same four: 498 potential calls against 502
/// charged.
const SITE_SCAN_CHARGE: usize = 4;

struct Walk {
    outcome: Outcome,
    spent: usize,
    priced: usize,
    evaluations: usize,
    at_hit: Option<usize>,
}

fn walk(cfg: &Config, atoms: usize, reference: f64) -> Walk {
    anneal_core::install_rgpot_kernels!();
    let potential = PairPotential::lennard_jones(atoms);
    let mut rng = StdRng::seed_from_u64(SEED);
    let start = random_cluster_in_radius(atoms, cfg.start_radius(), cfg.min_separation, &mut rng);
    let evaluations = AtomicUsize::new(0);
    let mut ledger = Ledger::new(BUDGET);
    ledger.stop_below(reference + anneal_core::DEFAULT_HIT_TOL);
    let mut optimiser = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ndarray::ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |point| {
            if !led.charge() {
                return None;
            }
            evaluations.fetch_add(1, Ordering::Relaxed);
            Some(potential.value_and_gradient(point))
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, point: ndarray::ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        evaluations.fetch_add(1, Ordering::Relaxed);
        Some(potential.value_and_gradient(point).1)
    };
    let mut rng = StdRng::seed_from_u64(SEED);
    let outcome = run_with_gradient(
        cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start");
    let at_hit = outcome
        .improvements
        .iter()
        .find(|row| anneal_core::energy_hits(row.3, reference, anneal_core::DEFAULT_HIT_TOL))
        .map(|row| row.1);
    assert!(
        anneal_core::energy_hits(outcome.best, reference, anneal_core::DEFAULT_HIT_TOL),
        "best {:.6}",
        outcome.best
    );
    Walk {
        spent: ledger.spent(),
        priced: ledger.priced(),
        evaluations: evaluations.load(Ordering::Relaxed),
        at_hit,
        outcome,
    }
}

#[test]
fn a_walk_hit_is_charged_for_every_potential_call_including_the_packing_probe() {
    let mut cfg = Config::recommended(26);
    cfg.packing_deferral = 0;
    let walk = walk(&cfg, 26, -108.315616);
    assert_eq!(walk.at_hit, Some(864), "hit spend");
    assert_eq!(walk.outcome.charged, 864, "charged calls");
    assert_eq!(walk.outcome.charged, walk.spent, "outcome and ledger agree");
    assert_eq!(
        walk.evaluations + SITE_SCAN_CHARGE,
        walk.spent,
        "every potential call is charged"
    );
    assert_eq!(walk.evaluations + walk.priced, walk.spent);
}

#[test]
fn a_walk_that_hits_before_the_offer_is_develops_walk_and_pays_nothing_for_the_probe() {
    let cfg = Config::recommended(26);
    assert_eq!(cfg.packing_deferral, 4);
    let walk = walk(&cfg, 26, -108.315616);
    assert_eq!(walk.at_hit, Some(502), "develop's hit spend");
    assert_eq!(walk.outcome.charged, 502, "develop's charged calls");
    assert!(walk.outcome.hops < 36, "the hit came before the offer");
    assert_eq!(walk.evaluations + SITE_SCAN_CHARGE, walk.spent);
}

#[test]
fn a_walk_past_the_offer_runs_the_probe_once_and_charges_it() {
    let cfg = Config::recommended(38);
    let walk = walk(&cfg, 38, -173.928427);
    assert_eq!(
        walk.outcome.hops, 36,
        "offered at hop 36, hit inside the probe"
    );
    assert_eq!(
        walk.at_hit,
        Some(3777),
        "hit spend, develop's is 5527 and the unscreened probe's 3864"
    );
    assert_eq!(walk.outcome.charged, 3778, "charged calls");
    assert_eq!(walk.outcome.charged, walk.spent, "outcome and ledger agree");
    assert_eq!(walk.evaluations, 3702, "potential calls");
    assert_eq!(
        walk.priced, 76,
        "lattice site scans priced, no potential call"
    );
    assert_eq!(walk.evaluations + walk.priced, walk.spent);
}
