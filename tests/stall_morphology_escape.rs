// Copyright (c) anneal developers
//! A stalled monoatomic cluster gates template-free morphology escapes
//! (`twin` and `observed`) via the search's existing stall counters and
//! bandit reward bookkeeping (`FlooredThompson`), leaving short walks,
//! disabled configurations, and frozen, labelled or slab systems untouched.

use anneal_core::methods::cluster_hopping::{
    Config, ContinuousSymmetry, Ledger, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::ArrayView1;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn run_pair(
    cfg: &Config,
    potential: &PairPotential,
    budget: usize,
    seed: u64,
    stop_below: Option<f64>,
) -> anneal_core::methods::cluster_hopping::Outcome {
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(
        cfg.n_points,
        cfg.start_radius(),
        cfg.min_separation,
        &mut rng,
    );
    let mut ledger = Ledger::new(budget);
    if let Some(limit) = stop_below {
        ledger.stop_below(limit);
    }
    let mut optimiser = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |point| {
            if !led.charge() {
                return None;
            }
            Some(potential.value_and_gradient(point))
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, point: ArrayView1<f64>| {
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

fn morph_draws(outcome: &anneal_core::methods::cluster_hopping::Outcome) -> usize {
    outcome
        .proposal_counts
        .iter()
        .filter(|c| c.move_type == "twin" || c.move_type == "observed")
        .map(|c| c.proposed)
        .sum()
}

#[test]
fn stalled_monoatomic_search_escapes_short_range_funnel_without_templates() {
    anneal_core::install_rgpot_kernels!();
    let cfg = Config::recommended(38);
    let potential = PairPotential::morse(38, 14.0);
    let reference = -144.321054;
    let outcome = run_pair(
        &cfg,
        &potential,
        300_000,
        2000,
        Some(reference + anneal_core::DEFAULT_HIT_TOL),
    );
    assert!(
        anneal_core::energy_hits(outcome.best, reference, anneal_core::DEFAULT_HIT_TOL),
        "expected global minimum near {reference}, got {}",
        outcome.best
    );
    assert!(
        morph_draws(&outcome) > 0,
        "expected twin/observed stall escapes after patience, proposal_counts={:?}",
        outcome.proposal_counts
    );
}

#[test]
fn morphology_escape_respects_disabled_stall_or_displacement_only() {
    anneal_core::install_rgpot_kernels!();
    let potential = PairPotential::lennard_jones(13);

    let stock = run_pair(&Config::recommended(13), &potential, 40_000, 2000, None);
    assert!(
        morph_draws(&stock) > 0,
        "the stock walk must reach the escape for the cases below to test it: {:?}",
        stock.proposal_counts
    );

    let mut no_stall = Config::recommended(13);
    no_stall.escape_on_stall = false;
    no_stall.trail_on_stall = false;
    let out_no_stall = run_pair(&no_stall, &potential, 40_000, 2000, None);
    assert!(
        morph_draws(&out_no_stall) == 0,
        "disabled stall exits must not run morphology escapes: {:?}",
        out_no_stall.proposal_counts
    );

    let mut disp_only = Config::recommended(13);
    disp_only.displacement_only = true;
    let out_disp = run_pair(&disp_only, &potential, 40_000, 2000, None);
    assert!(
        morph_draws(&out_disp) == 0,
        "displacement_only must not run morphology escapes: {:?}",
        out_disp.proposal_counts
    );

    let mut zero_restart = Config::recommended(13);
    zero_restart.restart_patience = 0;
    let out_zero_restart = run_pair(&zero_restart, &potential, 40_000, 2000, None);
    assert!(
        morph_draws(&out_zero_restart) == 0,
        "restart_patience=0 must not run morphology escapes: {:?}",
        out_zero_restart.proposal_counts
    );

    let mut zero_stall = Config::recommended(13);
    zero_stall.stall_patience = 0;
    let out_zero_stall = run_pair(&zero_stall, &potential, 40_000, 2000, None);
    assert!(
        morph_draws(&out_zero_stall) == 0,
        "stall_patience=0 must not run morphology escapes: {:?}",
        out_zero_stall.proposal_counts
    );
}

#[test]
fn morphology_escape_excludes_frozen_labelled_and_slab_systems() {
    anneal_core::install_rgpot_kernels!();
    let n = 13;
    let potential = PairPotential::lennard_jones(n);
    let free = || {
        let mut cfg = Config::recommended(n);
        cfg.continuous_symmetry = ContinuousSymmetry::Off;
        cfg
    };

    let control = run_pair(&free(), &potential, 40_000, 2000, None);
    assert!(
        morph_draws(&control) > 0,
        "the unconstrained walk must reach the escape for the cases below to test it: {:?}",
        control.proposal_counts
    );

    let mut frozen = free();
    frozen.frozen = Some((0..n).map(|i| i == 0).collect());
    let out_frozen = run_pair(&frozen, &potential, 40_000, 2000, None);
    assert_eq!(
        morph_draws(&out_frozen),
        0,
        "a frozen frame must not run morphology escapes: {:?}",
        out_frozen.proposal_counts
    );

    let mut labelled = free();
    labelled.species = Some((0..n).map(|i| u32::from(2 * i >= n)).collect());
    let out_labelled = run_pair(&labelled, &potential, 40_000, 2000, None);
    assert_eq!(
        morph_draws(&out_labelled),
        0,
        "two labels must not run morphology escapes: {:?}",
        out_labelled.proposal_counts
    );

    let mut slab = free();
    slab.species = Some(vec![0; n]);
    slab.active_region = Some((vec![0], 1));
    let out_slab = run_pair(&slab, &potential, 40_000, 2000, None);
    assert_eq!(
        morph_draws(&out_slab),
        0,
        "an active region must not run morphology escapes: {:?}",
        out_slab.proposal_counts
    );
}
