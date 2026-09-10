use anneal_core::methods::cluster_hopping::{Config, Ledger, run_with_gradient};
use ndarray::{Array1, ArrayView1, array};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn mobile_harmonic(state: ArrayView1<f64>, minimum: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let mut gradient = &state - &minimum;
    for coordinate in gradient.iter_mut().take(3) {
        *coordinate = 0.0;
    }
    (0.5 * gradient.dot(&gradient), gradient)
}

#[test]
fn hopping_nve_keeps_frozen_coordinates_fixed_at_every_force_evaluation() {
    let minimum = array![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let mut cfg = Config::for_cluster(3);
    cfg.frozen = Some(vec![true, false, false]);
    cfg.minima_hopping = true;
    cfg.md_escape = true;
    cfg.md_escape_dt = 0.05;
    cfg.md_escape_kinetic = 0.05;
    cfg.md_escape_minima = 1;
    cfg.md_escape_max_steps = 100;
    cfg.md_escape_soften = 0;
    cfg.max_hops = Some(1);
    cfg.replicas = 1;
    cfg.displacement_only = true;
    cfg.angular_moves = false;
    cfg.soft_perturb = false;
    cfg.cov_perturb = false;
    cfg.hmc = None;
    cfg.two_phase = None;
    cfg.surfaces.clear();
    cfg.return_screen = false;
    cfg.bayes_screen = false;
    cfg.screen_margin = 10.0;
    cfg.screen_steps = 1;
    cfg.relax_steps = 1;
    cfg.polish_records = 0;
    cfg.merge_radius = 0.1;
    cfg.bias_height = 0.0;
    cfg.shared_deposits = 0;
    cfg.restart_on_stall = false;
    cfg.escape_on_stall = false;
    cfg.trail_on_stall = false;
    cfg.jump_on_stall = false;
    cfg.path_on_stall = false;
    cfg.symmetrise_on_stall = false;
    cfg.point_symmetrise_on_new = false;
    cfg.orbit_complete_on_new = false;
    cfg.superbasin_escape = false;

    let mut raw_states = Vec::new();
    let mut relaxed_states = Vec::new();
    let mut relax_calls = 0usize;
    let mut force_calls = Vec::new();
    let mut relax = |ledger: &mut Ledger, state: ArrayView1<f64>, steps: usize| {
        assert!(ledger.charge());
        relax_calls += 1;
        let mut result = state.to_owned();
        if steps == 0 {
            raw_states.push(result.clone());
        } else {
            assert_eq!(steps, 1);
            // The exact constrained quench changes only mobile coordinates.
            for coordinate in 3..result.len() {
                result[coordinate] = minimum[coordinate];
            }
        }
        let (energy, gradient) = mobile_harmonic(result.view(), minimum.view());
        if steps > 0 {
            assert_eq!(gradient, Array1::zeros(result.len()));
            relaxed_states.push((energy, result.clone()));
        }
        (energy, result)
    };
    let mut gradient = |ledger: &mut Ledger, state: ArrayView1<f64>| {
        assert!(ledger.charge());
        let (_, force) = mobile_harmonic(state, minimum.view());
        force_calls.push((state.to_owned(), force.clone()));
        Some(force)
    };
    let mut ledger = Ledger::new(256);
    let mut rng = StdRng::seed_from_u64(17);
    let out = run_with_gradient(
        &cfg,
        minimum.view(),
        &mut ledger,
        &mut relax,
        Some(&mut gradient),
        &mut rng,
    );

    assert_eq!(out.hops, 1);
    assert_eq!(out.md_escape.0, 1, "the hop must execute an NVE escape");
    assert!(out.md_escape.1 > 0);
    assert_eq!(
        out.md_escape.2, 0,
        "the harmonic escape must find a minimum"
    );
    assert!(raw_states.len() > 1, "NVE must evaluate a post-start frame");
    assert!(raw_states.iter().any(|state| {
        state
            .iter()
            .zip(minimum.iter())
            .skip(3)
            .any(|(actual, initial)| actual.to_bits() != initial.to_bits())
    }));
    assert_eq!(ledger.spent(), relax_calls + force_calls.len());
    assert_eq!(out.charged, ledger.spent());
    assert!(ledger.spent() < ledger.budget());
    for (energy, state) in &relaxed_states {
        assert_eq!(*energy, mobile_harmonic(state.view(), minimum.view()).0);
    }
    for state in &raw_states {
        assert!(force_calls.iter().any(|(evaluated, _)| evaluated == state));
    }
    for (state, gradient) in &force_calls {
        assert_eq!(*gradient, mobile_harmonic(state.view(), minimum.view()).1);
        for coordinate in 0..3 {
            assert_eq!(
                state[coordinate].to_bits(),
                minimum[coordinate].to_bits(),
                "frozen coordinate {coordinate} moved at a charged force evaluation: {state:?}"
            );
        }
    }
}
