use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, QuenchStatus, run_with_gradient,
};
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
            if state == polished.view() && !valid {
                1.0
            } else {
                0.0
            },
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
fn malformed_gradients_cannot_certify_the_initial_minimum() {
    let mut cfg = Config::for_cluster(2);
    cfg.max_hops = Some(0);
    let start = Array1::from(vec![-0.6, 0.0, 0.0, 0.6, 0.0, 0.0]);
    for bad_gradient in [
        Array1::from_elem(start.len(), f64::NAN),
        Array1::zeros(0),
        Array1::zeros(start.len() - 1),
    ] {
        let mut ledger = Ledger::new(100);
        let mut rng = StdRng::seed_from_u64(0xbad);
        let mut relax = |ledger: &mut Ledger, _: ArrayView1<f64>, _: usize| {
            assert!(ledger.charge());
            (-0.25, start.clone())
        };
        let mut gradient = |ledger: &mut Ledger, _: ArrayView1<f64>| {
            assert!(ledger.charge());
            Some(bad_gradient.clone())
        };
        let outcome = run_with_gradient(
            &cfg,
            start.view(),
            &mut ledger,
            &mut relax,
            Some(&mut gradient),
            &mut rng,
        );
        assert!(
            outcome.best_state.is_none(),
            "accepted gradient {bad_gradient:?}"
        );
        assert_eq!(outcome.best, f64::INFINITY);
        assert!(outcome.improvements.is_empty());
    }
}

#[test]
fn malformed_quench_boundaries_remain_paid_unresolved_observations() {
    let state = Array1::from(vec![-0.6, 0.0, 0.0, 0.6, 0.0, 0.0]);
    for (energy, coordinates, gradient) in [
        (-0.25, state.clone(), Array1::from_elem(state.len(), f64::NAN)),
        (-0.25, state.clone(), Array1::zeros(state.len() - 1)),
        (f64::NAN, state.clone(), Array1::zeros(state.len())),
        (-0.25, Array1::from_elem(state.len(), f64::NAN), Array1::zeros(state.len())),
        (-0.25, Array1::zeros(0), Array1::zeros(0)),
    ] {
        let mut ledger = Ledger::new(1);
        assert!(ledger.charge());
        assert!(ledger.record_quench_boundary(0, energy, coordinates, Some(gradient)));
        let [boundary] = ledger.quench_boundaries() else {
            panic!("a failed quench remains in the observation denominator");
        };
        assert_eq!(boundary.status(), QuenchStatus::Rejected);
        assert!(boundary.gradient().is_none());
        assert_eq!(boundary.charged_calls(), 1);
    }
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
    assert!(
        !verification.contains(".minimize("),
        "reporting performs an uncharged search"
    );
    assert!(
        !verification.contains("out.best ="),
        "reporting replaces the scored objective"
    );
}
