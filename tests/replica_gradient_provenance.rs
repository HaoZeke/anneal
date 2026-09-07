use anneal_core::methods::cluster_hopping::{Config, LadderMode, Ledger, run_with_gradient};
use ndarray::{Array1, ArrayView1, array};
use rand::{SeedableRng, rngs::StdRng};

fn harmonic_evaluation(state: ArrayView1<f64>) -> (f64, Array1<f64>) {
    const STIFFNESS: f64 = 1e-8;
    let energy = 0.5 * STIFFNESS * state.iter().map(|value| value * value).sum::<f64>();
    let gradient = state.mapv(|value| STIFFNESS * value);
    (energy, gradient)
}

#[test]
fn independently_initialized_rung_retains_its_own_gradient_on_activation() {
    let mut cfg = Config::for_cluster(2);
    cfg.replicas = 2;
    cfg.ladder_mode = LadderMode::Independent;
    cfg.swap_period = 1;
    cfg.ladder_pilot = 1;
    cfg.max_hops = Some(2);
    cfg.screen_steps = 1;
    cfg.relax_steps = 1;
    cfg.screen_margin = f64::INFINITY;
    cfg.return_screen = false;
    cfg.bias_height = 0.0;
    cfg.displacement_only = true;
    cfg.temperature = 1e12;

    let start = array![-0.6, 0.0, 0.0, 0.6, 0.0, 0.0];
    let mut initialized_states = Vec::new();
    let mut relax = |ledger: &mut Ledger, state: ArrayView1<f64>, _steps: usize| {
        assert!(ledger.charge());
        if initialized_states.len() < 2 {
            initialized_states.push(state.to_owned());
        }
        // The analytic force lies within the stationarity tolerance, so
        // this fixture's quench can retain its supplied coordinates.
        let (energy, gradient) = harmonic_evaluation(state);
        assert!(
            gradient
                .iter()
                .all(|value| value.abs() < cfg.record_gradient)
        );
        (energy, state.to_owned())
    };
    let mut gradient = |ledger: &mut Ledger, state: ArrayView1<f64>| {
        assert!(ledger.charge());
        Some(harmonic_evaluation(state).1)
    };
    let mut ledger = Ledger::new(100);
    let mut rng = StdRng::seed_from_u64(0x6a_72_61_64);
    let outcome = run_with_gradient(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut gradient),
        &mut rng,
    );

    assert_eq!(outcome.hops, 2);
    assert_eq!(outcome.rungs.len(), 2);
    assert_eq!(outcome.swaps_tried, 0);
    assert_eq!(outcome.swaps_accepted, 0);
    assert_eq!(initialized_states.len(), 2);
    assert_eq!(initialized_states[0], start);
    let initialized_rung = &initialized_states[1];
    assert_ne!(initialized_rung, &start);

    let transitions = &outcome.accepted_transitions;
    assert_eq!(transitions.len(), 2, "both rung slices must accept a hop");
    assert_eq!(transitions[0].from_state, start);
    assert_ne!(
        &transitions[0].to_state, initialized_rung,
        "the initialized rung must differ from the outgoing rung"
    );
    let activated = transitions
        .iter()
        .find(|transition| &transition.from_state == initialized_rung)
        .expect("a transition must execute from the separately initialized rung");
    let (expected_energy, expected_gradient) = harmonic_evaluation(activated.from_state.view());
    assert_eq!(activated.from_energy, expected_energy);
    let actual_gradient = activated
        .from_gradient
        .as_ref()
        .expect("the valid initialized rung has paid gradient evidence to retain");
    assert_eq!(
        actual_gradient, &expected_gradient,
        "rung activation must carry the gradient belonging to its coordinates"
    );
}
