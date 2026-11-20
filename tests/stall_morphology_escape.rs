// Copyright (c) anneal developers
//! A stalled monoatomic cluster gates template-free morphology escapes
//! (`twin` and `observed`) via the search's existing stall counters and
//! bandit reward bookkeeping (`FlooredThompson`), leaving short walks and
//! disabled configurations untouched.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, random_cluster_in_radius, run_with_gradient,
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
    let morph_draws: usize = outcome
        .proposal_counts
        .iter()
        .filter(|c| c.move_type == "twin" || c.move_type == "observed")
        .map(|c| c.proposed)
        .sum();
    assert!(
        morph_draws > 0,
        "expected twin/observed stall escapes after patience, proposal_counts={:?}",
        outcome.proposal_counts
    );
}

#[test]
fn morphology_escape_respects_disabled_stall_or_displacement_only() {
    anneal_core::install_rgpot_kernels!();
    let potential = PairPotential::lennard_jones(13);

    let mut no_stall = Config::recommended(13);
    no_stall.escape_on_stall = false;
    no_stall.trail_on_stall = false;
    let out_no_stall = run_pair(&no_stall, &potential, 40_000, 2000, None);
    assert!(
        !out_no_stall
            .proposal_counts
            .iter()
            .any(|c| c.move_type == "twin" || c.move_type == "observed"),
        "disabled stall exits must not run morphology escapes: {:?}",
        out_no_stall.proposal_counts
    );

    let mut disp_only = Config::recommended(13);
    disp_only.displacement_only = true;
    let out_disp = run_pair(&disp_only, &potential, 40_000, 2000, None);
    assert!(
        !out_disp
            .proposal_counts
            .iter()
            .any(|c| c.move_type == "twin" || c.move_type == "observed"),
        "displacement_only must not run morphology escapes: {:?}",
        out_disp.proposal_counts
    );

    let mut zero_restart = Config::recommended(13);
    zero_restart.restart_patience = 0;
    let out_zero_restart = run_pair(&zero_restart, &potential, 40_000, 2000, None);
    assert!(
        !out_zero_restart
            .proposal_counts
            .iter()
            .any(|c| c.move_type == "twin" || c.move_type == "observed"),
        "restart_patience=0 must not run morphology escapes: {:?}",
        out_zero_restart.proposal_counts
    );

    let mut zero_stall = Config::recommended(13);
    zero_stall.stall_patience = 0;
    let out_zero_stall = run_pair(&zero_stall, &potential, 40_000, 2000, None);
    assert!(
        !out_zero_stall
            .proposal_counts
            .iter()
            .any(|c| c.move_type == "twin" || c.move_type == "observed"),
        "stall_patience=0 must not run morphology escapes: {:?}",
        out_zero_stall.proposal_counts
    );
}
