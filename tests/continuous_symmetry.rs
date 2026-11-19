use anneal_core::assignment::minimum_cost_assignment;
use anneal_core::continuous_symmetry::{inversion_rms, project_inversion};
use anneal_core::methods::cluster_hopping::{Config, ContinuousSymmetry, Ledger, run};
use ndarray::{Array1, ArrayView1};
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn assignment_is_bijective_when_rowwise_nearest_neighbours_collide() {
    let costs = [
        1.0, 2.0, 100.0, // row 0
        1.0, 100.0, 100.0, // row 1
        100.0, 1.0, 1.0, // row 2
    ];

    let assignment = minimum_cost_assignment(&costs, 3).expect("finite square assignment");

    assert_eq!(assignment, vec![1, 0, 2]);
}

#[test]
fn assignment_rejects_malformed_or_nonfinite_costs() {
    assert!(minimum_cost_assignment(&[1.0, 2.0, 3.0], 2).is_none());
    assert!(minimum_cost_assignment(&[0.0, f64::NAN, 1.0, 0.0], 2).is_none());
}

#[test]
fn ci_projection_leaves_an_exact_centrosymmetric_cluster_unchanged() {
    let x = Array1::from(vec![
        3.0, -1.0, 0.5, // translated +x partner
        1.0, -1.0, 0.5, // translated -x partner
        2.0, 1.0, 0.5, // translated +y partner
        2.0, -3.0, 0.5, // translated -y partner
    ]);
    let classes = [0_u32; 4];

    let projection = project_inversion(x.view(), &classes).expect("valid point set");

    for (actual, expected) in projection.coordinates.iter().zip(x.iter()) {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }
    assert!(projection.residual_rms < 1e-12);
}

#[test]
fn ci_projection_uses_a_species_preserving_permutation_and_reduces_csm() {
    let x = Array1::from(vec![
        1.20, 0.10, 0.00, // class 0
        0.05, 2.10, 0.20, // class 1
        -1.00, 0.00, 0.00, // class 0
        0.00, -2.00, 0.00, // class 1
    ]);
    let classes = [0_u32, 1, 0, 1];
    let before = inversion_rms(x.view(), &classes).expect("valid point set");

    let projection = project_inversion(x.view(), &classes).expect("valid point set");
    let after = inversion_rms(projection.coordinates.view(), &classes).expect("valid projection");

    for (row, &column) in projection.assignment.iter().enumerate() {
        assert_eq!(classes[row], classes[column]);
    }
    assert!(before > 1e-3);
    assert!(after < before * 1e-8, "CSM residual {before} -> {after}");
}

#[test]
fn ci_projection_commutes_with_translation() {
    let x = Array1::from(vec![
        1.2, 0.1, 0.3, -0.9, -0.2, -0.1, 0.2, 1.7, -0.4, -0.1, -1.9, 0.2,
    ]);
    let classes = [0_u32; 4];
    let shift = [4.0, -3.0, 1.5];
    let mut shifted = x.clone();
    for atom in 0..classes.len() {
        for (axis, &offset) in shift.iter().enumerate() {
            shifted[3 * atom + axis] += offset;
        }
    }

    let base = project_inversion(x.view(), &classes).expect("base projection");
    let translated = project_inversion(shifted.view(), &classes).expect("translated projection");

    assert_eq!(base.assignment, translated.assignment);
    for atom in 0..classes.len() {
        for (axis, &offset) in shift.iter().enumerate() {
            let expected = base.coordinates[3 * atom + axis] + offset;
            assert!((translated.coordinates[3 * atom + axis] - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn scheduled_ci_move_is_quenched_charged_and_adopted_only_downhill() {
    let start = Array1::from(vec![1.4, 0.1, 0.0, -0.4, 1.2, 0.2, -0.2, -0.3, 0.0]);
    let initial_energy = start.iter().map(|value| value * value).sum::<f64>();
    let mut config = Config::for_cluster(3);
    config.relax_steps = 1;
    config.return_screen = false;
    config.max_hops = Some(1);
    config.continuous_symmetry = ContinuousSymmetry::Inversion { interval: 1 };
    let mut ledger = Ledger::new(2);
    let mut relax =
        |ledger: &mut Ledger, state: ArrayView1<'_, f64>, _steps: usize| -> (f64, Array1<f64>) {
            if !ledger.charge() {
                return (f64::INFINITY, state.to_owned());
            }
            let energy = state.iter().map(|value| value * value).sum();
            (energy, state.to_owned())
        };
    let mut rng = StdRng::seed_from_u64(7);

    let outcome =
        run(&config, start.view(), &mut ledger, &mut relax, &mut rng).expect("finite start");

    assert_eq!(outcome.continuous_symmetry.0, 1);
    assert!(outcome.continuous_symmetry.1 > 0.0);
    assert!(outcome.best < initial_energy);
    assert_eq!(outcome.charged, 2);
}

#[test]
fn ci_interval_counts_the_initial_quench_as_quench_one() {
    let start = Array1::from(vec![1.4, 0.1, 0.0, -0.4, 1.2, 0.2, -0.2, -0.3, 0.0]);
    let mut config = Config::for_cluster(3);
    config.relax_steps = 1;
    config.screen_steps = 1;
    config.return_screen = false;
    config.max_hops = Some(1);
    config.continuous_symmetry = ContinuousSymmetry::Inversion { interval: 2 };
    let mut ledger = Ledger::new(8);
    let mut relax =
        |ledger: &mut Ledger, state: ArrayView1<'_, f64>, _steps: usize| -> (f64, Array1<f64>) {
            if !ledger.charge() {
                return (f64::INFINITY, state.to_owned());
            }
            let energy = state.iter().map(|value| value * value).sum();
            (energy, state.to_owned())
        };
    let mut rng = StdRng::seed_from_u64(7);

    let outcome =
        run(&config, start.view(), &mut ledger, &mut relax, &mut rng).expect("finite start");

    assert_eq!(outcome.hops, 1);
    assert_eq!(outcome.continuous_symmetry.0, 0);
}

#[test]
fn ci_quench_itself_advances_the_csm_quench_counter() {
    let start = Array1::from(vec![1.4, 0.1, 0.0, -0.4, 1.2, 0.2, -0.2, -0.3, 0.0]);
    let mut config = Config::for_cluster(3);
    config.relax_steps = 1;
    config.screen_steps = 1;
    config.return_screen = false;
    config.max_hops = Some(3);
    config.continuous_symmetry = ContinuousSymmetry::Inversion { interval: 2 };
    let mut ledger = Ledger::new(12);
    let mut relax =
        |ledger: &mut Ledger, state: ArrayView1<'_, f64>, _steps: usize| -> (f64, Array1<f64>) {
            if !ledger.charge() {
                return (f64::INFINITY, state.to_owned());
            }
            let energy = state.iter().map(|value| value * value).sum();
            (energy, state.to_owned())
        };
    let mut rng = StdRng::seed_from_u64(7);

    let outcome =
        run(&config, start.view(), &mut ledger, &mut relax, &mut rng).expect("finite start");

    assert_eq!(outcome.hops, 3);
    assert_eq!(outcome.continuous_symmetry.0, 2);
}

#[test]
fn general_symmetry_flag_is_off_by_default_and_selects_threefold_measure_when_enabled() {
    assert!(!Config::for_cluster(3).general_symmetry);
    assert!(!Config::recommended(3).general_symmetry);
    let rigid_geom =
        anneal_core::rigid_body::RigidBodyGeometry::new(vec![[0.1, 0.0, 0.0]], vec![8]).unwrap();
    let mut preset = Config::for_cluster(2);
    preset.general_symmetry = true;
    assert!(!preset.with_rigid_bodies(rigid_geom).general_symmetry);

    let start = Array1::from(vec![
        1.15,
        0.0,
        0.0,
        -0.5,
        0.866_025_403_784_438_6,
        0.0,
        -0.5,
        -0.866_025_403_784_438_6,
        0.0,
    ]);
    let side_variance = |state: ArrayView1<'_, f64>| -> f64 {
        let d = |i: usize, j: usize| -> f64 {
            let dx = state[3 * i] - state[3 * j];
            let dy = state[3 * i + 1] - state[3 * j + 1];
            let dz = state[3 * i + 2] - state[3 * j + 2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        };
        let (d01, d12, d20) = (d(0, 1), d(1, 2), d(2, 0));
        let scale = (d01 * d01 + d12 * d12 + d20 * d20).max(1e-12);
        ((d01 - d12).powi(2) + (d12 - d20).powi(2) + (d20 - d01).powi(2)) / scale
    };
    let initial_energy = side_variance(start.view());
    assert!(initial_energy > 3e-3, "{initial_energy}");

    let run_with_flag = |general_symmetry: bool| {
        let mut config = Config::for_cluster(3);
        config.relax_steps = 1;
        config.screen_steps = 1;
        config.return_screen = false;
        config.max_hops = Some(1);
        config.continuous_symmetry = ContinuousSymmetry::Measure { interval: 1 };
        config.general_symmetry = general_symmetry;
        let mut ledger = Ledger::new(4);
        let mut relax = |ledger: &mut Ledger,
                         state: ArrayView1<'_, f64>,
                         _steps: usize|
         -> (f64, Array1<f64>) {
            if !ledger.charge() {
                return (f64::INFINITY, state.to_owned());
            }
            (side_variance(state), state.to_owned())
        };
        let mut rng = StdRng::seed_from_u64(11);
        run(&config, start.view(), &mut ledger, &mut relax, &mut rng).expect("finite start")
    };

    let off = run_with_flag(false);
    assert_eq!(off.continuous_symmetry.0, 1);
    assert!(off.best > 2e-4, "{}", off.best);

    let on = run_with_flag(true);
    assert_eq!(on.continuous_symmetry.0, 1);
    assert!(
        on.continuous_symmetry.1 > off.continuous_symmetry.1 + 2e-4,
        "on gain {} vs off gain {}",
        on.continuous_symmetry.1,
        off.continuous_symmetry.1
    );
    assert!(on.best < 1e-24, "{}", on.best);
}
