// Copyright (c) anneal developers
//! Under [`Ledger::keep_evidence_until_shown`] a run trims only the quench
//! boundaries it recorded itself, and leaves the ledger keeping evidence as
//! it did before the run, whether the run returns or unwinds.

use anneal_core::bias::BasinBias;
use anneal_core::methods::cluster_hopping::{
    ChainCheckpoint, CheckpointAction, ClusterFingerprint, Config, Ledger, random_cluster,
    run_with_bias_at_checkpoints, run_with_gradient,
};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::panic::{AssertUnwindSafe, catch_unwind};

const OUTSIDE: f64 = -7.25;

fn recording_relax(
    ledger: &mut Ledger,
    state: ArrayView1<f64>,
    steps: usize,
) -> (f64, Array1<f64>) {
    let before = ledger.spent();
    let mut relaxed = state.to_owned();
    for _ in 0..steps {
        if !ledger.charge() {
            break;
        }
        relaxed.mapv_inplace(|value| value * 0.85);
    }
    let energy = relaxed.iter().map(|value| value * value).sum();
    let gradient = relaxed.mapv(|value| 2.0 * value);
    ledger.record_quench_boundary(before, energy, relaxed.clone(), Some(gradient));
    (energy, relaxed)
}

fn record_outside_a_run(ledger: &mut Ledger, n: usize) {
    let before = ledger.spent();
    assert!(ledger.charge());
    assert!(ledger.record_quench_boundary(
        before,
        OUTSIDE,
        Array1::from_elem(3 * n, 0.5),
        Some(Array1::zeros(3 * n)),
    ));
}

fn setup() -> (Config, Array1<f64>, StdRng, Ledger) {
    let mut cfg = Config::recommended(6);
    cfg.max_hops = Some(6);
    cfg.relax_steps = 1;
    let mut rng = StdRng::seed_from_u64(0x5c0e);
    let start = random_cluster(cfg.n_points, 0.7, cfg.min_separation, &mut rng);
    let mut ledger = Ledger::new(10_000);
    ledger.keep_evidence_until_shown();
    (cfg, start, rng, ledger)
}

#[test]
fn checkpoints_trim_only_the_boundaries_their_run_recorded() {
    let (cfg, start, mut rng, mut ledger) = setup();
    record_outside_a_run(&mut ledger, cfg.n_points);
    let mut bias = BasinBias::new(
        ClusterFingerprint::for_keying(cfg.n_points, cfg.shape_keyed),
        cfg.merge_radius,
        cfg.bias_height,
        cfg.bias_gamma,
    );
    let mut shown = Vec::new();
    let mut checkpoint = |snapshot: ChainCheckpoint<'_>| {
        shown.extend(
            snapshot
                .quench_boundaries()
                .iter()
                .map(|boundary| boundary.energy().to_bits()),
        );
        CheckpointAction::Continue
    };
    run_with_bias_at_checkpoints(
        &cfg,
        start.view(),
        &mut ledger,
        &mut recording_relax,
        None,
        &mut bias,
        &mut rng,
        1,
        &mut checkpoint,
    )
    .expect("finite start");
    assert!(shown.len() > 1, "no checkpoint saw a run boundary");
    assert_eq!(
        shown[0],
        OUTSIDE.to_bits(),
        "the first checkpoint did not see the boundary before the run"
    );
    assert_eq!(ledger.dropped_quench_boundaries(), shown.len() - 1);
    let first = ledger
        .quench_boundaries()
        .first()
        .expect("the boundary recorded before the run was dropped");
    assert_eq!(first.energy().to_bits(), OUTSIDE.to_bits());
    assert_eq!(first.charged_calls(), 1);
}

#[test]
fn a_run_that_panics_restores_how_the_ledger_keeps_evidence() {
    let (cfg, start, mut rng, mut ledger) = setup();
    record_outside_a_run(&mut ledger, cfg.n_points);
    let mut relaxations = 0;
    let mut relax = |ledger: &mut Ledger, state: ArrayView1<f64>, steps: usize| {
        relaxations += 1;
        assert!(relaxations < 4, "relaxation failed mid-run");
        recording_relax(ledger, state, steps)
    };
    let unwound = catch_unwind(AssertUnwindSafe(|| {
        run_with_gradient(&cfg, start.view(), &mut ledger, &mut relax, None, &mut rng)
    }));
    assert!(unwound.is_err(), "the run did not panic");
    assert_eq!(ledger.dropped_quench_boundaries(), 3);
    record_outside_a_run(&mut ledger, cfg.n_points);
    let energies: Vec<u64> = ledger
        .quench_boundaries()
        .iter()
        .map(|boundary| boundary.energy().to_bits())
        .collect();
    assert_eq!(
        energies,
        vec![OUTSIDE.to_bits(); 2],
        "a boundary recorded after the unwound run was not kept"
    );
}
