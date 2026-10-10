use anneal_core::methods::cluster_hopping::{Config, Ledger, Outcome, run_with_gradient};
use ndarray::{Array1, ArrayView1, array};
use rand::{SeedableRng, rngs::StdRng};

fn symmetrised_run(endpoint_gradient: f64) -> Outcome {
    let start = array![
        1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        -1.0,
    ];
    let mut candidate = start.clone();
    for (i, value) in candidate.iter_mut().enumerate() {
        *value += 0.03 * (((i * 37 + 11) % 17) as f64 / 8.0 - 1.0);
    }
    let endpoint = &candidate * 0.9;
    let mut cfg = Config::for_cluster(6);
    cfg.max_hops = Some(1);
    cfg.screen_steps = 1;
    cfg.relax_steps = 2;
    cfg.return_screen = false;
    cfg.screen_margin = f64::INFINITY;
    cfg.angular_moves = false;
    cfg.bias_height = 0.0;
    cfg.min_separation = 0.0;
    cfg.point_symmetrise_on_new = true;
    cfg.point_symmetrise_every_accept = true;
    cfg.symmetrise_core_fraction = 1.0;
    cfg.symmetry_tolerance = 0.5;
    assert!(
        anneal_core::symmetrise::symmetrise_core(
            candidate.view(),
            6,
            cfg.symmetry_tolerance,
            cfg.symmetry_merge_radius,
            cfg.symmetrise_core_fraction,
        )
        .is_some()
    );

    let mut full_quenches = 0;
    let mut relax = |ledger: &mut Ledger, _: ArrayView1<f64>, steps: usize| {
        let before = ledger.spent();
        assert!(ledger.charge());
        let (energy, point) = if steps == cfg.screen_steps {
            (-0.5, candidate.clone())
        } else {
            full_quenches += 1;
            match full_quenches {
                1 => (0.0, start.clone()),
                2 => (-1.0, candidate.clone()),
                3 => (-2.0, endpoint.clone()),
                _ => panic!("unexpected relaxation {full_quenches}"),
            }
        };
        let share_gradient = (point != endpoint).then(|| Array1::zeros(point.len()));
        assert!(ledger.record_quench_boundary(before, energy, point.clone(), share_gradient));
        (energy, point)
    };
    let mut gradient = |ledger: &mut Ledger, point: ArrayView1<f64>| {
        assert!(ledger.charge());
        Some(Array1::from_elem(
            point.len(),
            if point == endpoint.view() {
                endpoint_gradient
            } else {
                0.0
            },
        ))
    };
    let mut ledger = Ledger::new(100);
    let mut rng = StdRng::seed_from_u64(37);
    let outcome = run_with_gradient(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut gradient),
        &mut rng,
    );
    assert_eq!(full_quenches, 3);
    assert_eq!(outcome.symmetrised.0, 1);
    assert_eq!(outcome.final_state.as_ref(), Some(&endpoint));
    assert_eq!(
        outcome.final_energy, -2.0,
        "search adoption is not answer certification"
    );
    outcome
}

#[test]
fn unconverged_symmetrisation_cannot_replace_the_validated_answer() {
    let outcome = symmetrised_run(2.16e-3);
    assert_eq!(outcome.best, -1.0);
    assert_eq!(outcome.unconverged_records, 1);
}

#[test]
fn answer_eligible_symmetrisation_does_not_require_share_eligibility() {
    let outcome = symmetrised_run(5e-4);
    assert_eq!(outcome.best, -2.0);
    assert_eq!(outcome.unconverged_records, 0);
}

#[test]
fn nonfinite_symmetrisation_gradient_cannot_certify_an_answer() {
    let outcome = symmetrised_run(f64::NAN);
    assert_eq!(outcome.best, -1.0);
    assert_eq!(outcome.unconverged_records, 1);
}

#[test]
fn ordinary_lj_driver_supplies_charged_answer_validation_without_escape_flags() {
    let source = include_str!("../examples/lj_cluster_search.rs");
    let invocation = source
        .split_once("anneal_core::methods::cluster_hopping::optimize_with_settle(")
        .unwrap()
        .1
        .split_once("if cfg.staged_quench")
        .unwrap()
        .0;
    assert!(invocation.contains("Some(&mut grad)"));
    assert!(!invocation.contains("cfg.minima_hopping"));
    assert!(!invocation.contains("None"));
}
