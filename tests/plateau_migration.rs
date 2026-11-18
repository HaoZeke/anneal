// Copyright (c) anneal developers
//! A two-species recommended plateau moves onto a stored lower minimum.
//!
//! The diverse-seat helper never calls the search. This drives `run` with the
//! recommended visit policy and the recommended climb, and checks that the
//! live chain is back on the lower seat after it has accepted the worse one.

use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;

use anneal_core::methods::cluster_hopping::{
    CheckpointAction, Config, ContinuousSymmetry, Ledger, SharedVisitPolicy, SoapProposalMode,
    run_with_energy_gradient_at_checkpoints,
};

fn octahedron() -> Array1<f64> {
    let s = 0.85;
    Array1::from_vec(vec![
        s, 0.0, 0.0, -s, 0.0, 0.0, 0.0, s, 0.0, 0.0, -s, 0.0, 0.0, 0.0, s, 0.0, 0.0, -s,
    ])
}

fn grid() -> Array1<f64> {
    let mut coords = Vec::with_capacity(18);
    for row in 0..2 {
        for col in 0..3 {
            coords.push(row as f64 * 1.7);
            coords.push(col as f64 * 1.7);
            coords.push(0.0);
        }
    }
    Array1::from_vec(coords)
}

fn pair_spectrum(x: &Array1<f64>) -> Vec<f64> {
    let n = x.len() / 3;
    let mut distances = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let mut squared = 0.0;
            for axis in 0..3 {
                let delta = x[3 * i + axis] - x[3 * j + axis];
                squared += delta * delta;
            }
            distances.push(squared.sqrt());
        }
    }
    distances.sort_by(f64::total_cmp);
    distances
}

fn spectrum_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn a_recommended_binary_plateau_migrates_to_the_stored_lower_seat() {
    let worse = octahedron();
    let lower = grid();
    let apart = spectrum_distance(&pair_spectrum(&worse), &pair_spectrum(&lower));
    let labels = vec![200_u32, 200, 200, 201, 201, 201];
    let pairs = 3 * 3;
    let mut cfg = Config::recommended(labels.len()).with_species(labels, None);
    assert_eq!(
        cfg.shared_visit_policy,
        SharedVisitPolicy::Tabu,
        "the recommended policy leaves the recognition bank empty"
    );
    assert!(cfg.escape_on_stall);
    assert!(!cfg.packing_cna_applies());
    assert!(apart > cfg.merge_radius, "the two wells must not merge");
    // The patience is the only clock change. The visit policy and the climb
    // stay the recommended ones; a short patience reaches that clock inside
    // the test budget.
    cfg.restart_patience = 1;
    cfg.relax_steps = 1;
    cfg.screen_steps = 1;
    cfg.quench_once = true;
    cfg.temperature = 1.0e6;
    cfg.prior_soap = false;
    cfg.prior_reseed = false;
    cfg.prior_growth = false;
    cfg.prior_core = false;
    cfg.prior_handover = false;
    cfg.prior_motif = false;
    cfg.continuous_symmetry = ContinuousSymmetry::Off;
    cfg.energy_lattice = false;
    cfg.surfaces.clear();
    cfg.return_screen = false;
    cfg.calibrated_screen = false;
    cfg.bayes_screen = false;
    cfg.screen_margin = 1.0e9;
    cfg.soap_mode = SoapProposalMode::Off;
    // This fixture pins the call ordering around the plateau branch itself,
    // independent of the budget-window temperature law.
    cfg.budget_window = false;

    let worse_energy = -1.0;
    let lower_energy = -3.0;
    // The opening quench, then one round of unlike-pair polishes that do not
    // improve it. The next full quench is the hop, and it is the lower well.
    let opening = 1 + pairs;
    let mut full = 0_usize;
    let mut relax = |ledger: &mut Ledger, x: ndarray::ArrayView1<f64>, steps: usize| {
        if steps == 0 {
            return (0.0, x.to_owned());
        }
        if !ledger.charge() {
            return (f64::NAN, Array1::zeros(0));
        }
        let index = full;
        full += 1;
        if index == opening {
            (lower_energy, lower.clone())
        } else {
            (worse_energy, worse.clone())
        }
    };
    let mut ledger = Ledger::new(4_000);
    let mut rng = StdRng::seed_from_u64(11);
    let mut checks = 0_usize;
    let mut checkpoint = |_snapshot: anneal_core::methods::cluster_hopping::ChainCheckpoint<'_>| {
        let seen = checks;
        checks += 1;
        if seen >= 1 {
            CheckpointAction::Retire {
                reason: "seated".to_owned(),
            }
        } else {
            CheckpointAction::Continue
        }
    };
    let outcome = run_with_energy_gradient_at_checkpoints(
        &cfg,
        worse.view(),
        &mut ledger,
        &mut relax,
        None,
        None,
        &mut rng,
        1,
        &mut checkpoint,
    )
    .expect("the binary start quenches");

    let adopted: Vec<f64> = outcome
        .accepted_transitions
        .iter()
        .map(|edge| edge.to_energy)
        .collect();
    let last = adopted
        .last()
        .copied()
        .expect("the chain accepts the worse well after the lower one");
    assert!(
        adopted
            .iter()
            .any(|energy| (*energy - lower_energy).abs() < 1e-9),
        "the lower well is accepted before the plateau, got {adopted:?}"
    );
    assert!(
        (last - worse_energy).abs() < 1e-9,
        "the last acceptance is the worse well, so a later proposal did not restore the seat, got {adopted:?}"
    );
    assert!(
        (outcome.final_energy - lower_energy).abs() < 1e-9,
        "the plateau leaves the worse well for the stored seat, final {:.6} after {adopted:?}",
        outcome.final_energy
    );
    let seated = outcome
        .final_state
        .expect("the migrated chain keeps the stored coordinates");
    assert_eq!(seated.len(), lower.len());
    for (got, want) in seated.iter().zip(lower.iter()) {
        assert!((got - want).abs() < 1e-12);
    }
}
