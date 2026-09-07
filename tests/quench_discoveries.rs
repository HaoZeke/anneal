use anneal_core::methods::cluster_hopping::{Config, Ledger, Outcome, run_with_gradient};
use ndarray::{Array1, ArrayView1};
use rand::{SeedableRng, rngs::StdRng};

fn polishing_run(valid: bool) -> Outcome {
    let mut cfg = Config::for_cluster(2);
    cfg.max_hops = Some(1);
    cfg.relax_steps = 2;
    cfg.polish_records = 7;
    cfg.return_screen = false;
    let start = Array1::from(vec![-0.6, 0.0, 0.0, 0.6, 0.0, 0.0]);
    let reached = Array1::from(vec![-0.7, 0.0, 0.0, 0.7, 0.0, 0.0]);
    let polished = Array1::from(vec![-0.8, 0.0, 0.0, 0.8, 0.0, 0.0]);
    let mut ledger = Ledger::new(100);
    let mut rng = StdRng::seed_from_u64(0x50115);
    let mut first = true;
    let mut relax = |ledger: &mut Ledger, _: ArrayView1<f64>, steps: usize| {
        assert!(ledger.charge());
        if first {
            first = false;
            (0.0, start.clone())
        } else if steps == cfg.polish_records {
            (-0.5, polished.clone())
        } else {
            (-0.25, reached.clone())
        }
    };
    let mut gradient = |ledger: &mut Ledger, state: ArrayView1<f64>| {
        assert!(ledger.charge());
        Some(Array1::from_elem(
            state.len(),
            if state == polished.view() && !valid { 1.0 } else { 0.0 },
        ))
    };
    run_with_gradient(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut gradient),
        &mut rng,
    )
}

#[test]
fn unconverged_polishing_cannot_replace_a_validated_minimum() {
    let outcome = polishing_run(false);
    assert_eq!(outcome.best, -0.25);
    assert_eq!(outcome.best_state.unwrap()[0], -0.7);
}

#[test]
fn validated_polishing_discoveries_enter_the_charged_improvement_curve() {
    let outcome = polishing_run(true);
    assert_eq!(outcome.best, -0.5);
    assert_eq!(outcome.best_state.as_ref().unwrap()[0], -0.8);
    let improvement = outcome.improvements.last().unwrap();
    assert_eq!(improvement.0, outcome.hops);
    assert_eq!(improvement.1, outcome.charged);
    assert_eq!(improvement.3, outcome.best);
}

#[test]
fn result_verification_cannot_optimize_or_replace_the_search_answer() {
    let source = include_str!("../examples/lj_cluster_search.rs");
    let verification = source
        .split_once("let verified =")
        .unwrap()
        .1
        .split_once("let hit =")
        .unwrap()
        .0;
    assert!(!verification.contains(".minimize("), "reporting performs an uncharged search");
    assert!(!verification.contains("out.best ="), "reporting replaces the scored objective");
}
