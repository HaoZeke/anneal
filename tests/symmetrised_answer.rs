use anneal_core::methods::cluster_hopping::{Config, GradFn, Ledger, Outcome, run_with_gradient};
use ndarray::{Array1, ArrayView1, array};
use rand::{SeedableRng, rngs::StdRng};

struct Run {
    outcome: Outcome,
    relax_calls: usize,
    gradient_calls: usize,
    spent: usize,
    full_quenches: usize,
    endpoint: Array1<f64>,
}

fn completed_run(endpoint_gradient: f64) -> Outcome {
    scripted_checked(Some(endpoint_gradient)).outcome
}

fn scripted_checked(endpoint_gradient: Option<f64>) -> Run {
    let run = scripted_run(endpoint_gradient, |_| {});
    assert_eq!(run.full_quenches, 3);
    assert_eq!(run.outcome.symmetry_completions.0, 1);
    assert_eq!(run.outcome.final_state.as_ref(), Some(&run.endpoint));
    assert_eq!(
        run.outcome.final_energy, -2.0,
        "search adoption is not answer certification"
    );
    run
}

fn scripted_run(endpoint_gradient: Option<f64>, adjust: impl FnOnce(&mut Config)) -> Run {
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
    // The quench and its symmetry completion are a hop each.
    cfg.max_hops = Some(2);
    cfg.screen_steps = 1;
    cfg.relax_steps = 2;
    cfg.return_screen = false;
    cfg.screen_margin = f64::INFINITY;
    cfg.angular_moves = false;
    cfg.bias_height = 0.0;
    cfg.min_separation = 0.0;
    cfg.symmetry_completion_on_new = true;
    cfg.symmetry_core_fraction = 1.0;
    cfg.symmetry_tolerance = 0.5;
    cfg.finish_improving = false;
    adjust(&mut cfg);
    assert!(
        anneal_core::detected_symmetry::complete_detected(
            candidate.view(),
            6,
            cfg.symmetry_tolerance,
            cfg.symmetry_merge_radius,
            cfg.symmetry_core_fraction,
            cfg.min_separation,
        )
        .is_some()
    );

    let mut full_quenches = 0;
    let mut relax_calls = 0;
    let mut gradient_calls = 0;
    let mut relax = |ledger: &mut Ledger, _: ArrayView1<f64>, steps: usize| {
        relax_calls += 1;
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
        gradient_calls += 1;
        assert!(ledger.charge());
        Some(Array1::from_elem(
            point.len(),
            if point == endpoint.view() {
                endpoint_gradient.unwrap_or(0.0)
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
        endpoint_gradient.map(|_| &mut gradient as &mut GradFn<'_>),
        &mut rng,
    )
    .expect("finite start");
    Run {
        outcome,
        relax_calls,
        gradient_calls,
        spent: ledger.spent(),
        full_quenches,
        endpoint,
    }
}

#[test]
fn unconverged_completion_cannot_replace_the_validated_answer() {
    let outcome = completed_run(2.16e-3);
    assert_eq!(outcome.best, -1.0);
    assert_eq!(outcome.unconverged_records, 1);
}

#[test]
fn answer_eligible_completion_does_not_require_share_eligibility() {
    let outcome = completed_run(5e-4);
    assert_eq!(outcome.best, -2.0);
    assert_eq!(outcome.unconverged_records, 0);
}

#[test]
fn nonfinite_completion_gradient_cannot_certify_an_answer() {
    let outcome = completed_run(f64::NAN);
    assert_eq!(outcome.best, -1.0);
    assert_eq!(outcome.unconverged_records, 1);
}

#[test]
fn a_supplied_gradient_is_called_and_charged_to_the_ledger() {
    let run = scripted_checked(Some(5e-4));
    assert!(
        run.gradient_calls > 0,
        "the search never asked for the gradient"
    );
    assert_eq!(
        run.spent,
        run.relax_calls + run.gradient_calls,
        "every relaxation and every gradient call is one charged evaluation"
    );
}

#[test]
fn only_a_supplied_gradient_validates_the_answer() {
    let validated = scripted_checked(Some(2.16e-3));
    assert_eq!(validated.outcome.best, -1.0);
    assert_eq!(validated.outcome.unconverged_records, 1);

    let assumed = scripted_checked(None);
    assert_eq!(assumed.gradient_calls, 0);
    assert_eq!(
        assumed.outcome.best, -2.0,
        "without a gradient the quench is taken as the answer"
    );
    assert_eq!(assumed.outcome.unconverged_records, 0);
}

#[test]
fn the_stock_configuration_takes_no_minima_hopping_path() {
    for cfg in [Config::for_cluster(6), Config::recommended(6)] {
        assert!(!cfg.minima_hopping);
        assert!(!cfg.md_escape);
    }
    let ordinary = scripted_run(Some(5e-4), |cfg| {
        assert!(!cfg.minima_hopping);
    });
    assert_eq!(ordinary.outcome.md_escape, (0, 0, 0));
    assert!(!ordinary.outcome.nve_unavailable);

    let hopping = scripted_run(Some(5e-4), |cfg| cfg.minima_hopping = true);
    assert_ne!(
        (
            hopping.outcome.escape_scale,
            hopping.outcome.escape_threshold
        ),
        (
            ordinary.outcome.escape_scale,
            ordinary.outcome.escape_threshold
        ),
        "the controller moves only under the minima-hopping switch"
    );
}
