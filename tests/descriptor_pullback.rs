use anneal_core::descriptor_space::pullback::{
    PullbackConfig, PullbackConstraints, PullbackError, regularized_pullback,
};
use ndarray::{Array1, Array2, array};

fn unconstrained(coordinate_dim: usize) -> PullbackConstraints {
    PullbackConstraints {
        frozen_coordinates: vec![false; coordinate_dim],
        rigid_group_labels: Vec::new(),
        remove_translation: false,
    }
}

fn config(damping: f64, trust_radius: f64) -> PullbackConfig {
    PullbackConfig {
        damping,
        trust_radius,
        length_scale: 1.0,
    }
}

#[test]
fn diagonal_problem_matches_the_weighted_tikhonov_solution() {
    let jacobian = array![[2.0, 0.0], [0.0, 1.0]];
    let desired = array![1.0, 1.0];
    let weights = array![1.0, 4.0];

    let result = regularized_pullback(
        jacobian.view(),
        desired.view(),
        weights.view(),
        None,
        &unconstrained(2),
        config(0.5, 10.0),
    )
    .unwrap();

    assert!((result.step()[0] - 2.0 / 4.25).abs() < 1e-12);
    assert!((result.step()[1] - 4.0 / 4.25).abs() < 1e-12);
    assert!(result.realized_weighted_residual() < result.requested_weighted_norm());
    assert!(!result.clipped());
}

#[test]
fn rank_deficiency_produces_a_finite_residual_reducing_step() {
    let jacobian = array![[1.0, 1.0], [2.0, 2.0]];
    let desired = array![1.0, 2.0];
    let weights = Array1::ones(2);

    let result = regularized_pullback(
        jacobian.view(),
        desired.view(),
        weights.view(),
        None,
        &unconstrained(2),
        config(1e-3, 10.0),
    )
    .unwrap();

    assert!(result.step().iter().all(|value| value.is_finite()));
    assert!(result.realized_weighted_residual() < 1e-5);
}

#[test]
fn frozen_coordinates_are_exactly_zero() {
    let jacobian = Array2::eye(3);
    let desired = array![1.0, 1.0, 1.0];
    let weights = Array1::ones(3);
    let constraints = PullbackConstraints {
        frozen_coordinates: vec![false, true, false],
        rigid_group_labels: Vec::new(),
        remove_translation: false,
    };

    let result = regularized_pullback(
        jacobian.view(),
        desired.view(),
        weights.view(),
        None,
        &constraints,
        config(1e-3, 10.0),
    )
    .unwrap();

    assert_eq!(result.step()[1], 0.0);
    assert!(result.step()[0] > 0.9);
    assert!(result.step()[2] > 0.9);
}

#[test]
fn rigid_translation_is_projected_out() {
    let jacobian = Array2::eye(6);
    let desired = array![2.0, -1.0, 0.5, 4.0, 3.0, -0.5];
    let weights = Array1::ones(6);
    let constraints = PullbackConstraints {
        frozen_coordinates: vec![false; 6],
        rigid_group_labels: vec![0, 1],
        remove_translation: true,
    };

    let result = regularized_pullback(
        jacobian.view(),
        desired.view(),
        weights.view(),
        Some(array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0].view()),
        &constraints,
        config(1e-3, 10.0),
    )
    .unwrap();

    for axis in 0..3 {
        assert!((result.step()[axis] + result.step()[3 + axis]).abs() < 1e-12);
    }
}

#[test]
fn grouped_atoms_preserve_internal_distance_to_first_order() {
    let jacobian = Array2::eye(6);
    let desired = array![-1.0, 0.4, 0.0, 1.0, -0.2, 0.3];
    let weights = Array1::ones(6);
    let coordinates = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let constraints = PullbackConstraints {
        frozen_coordinates: vec![false; 6],
        rigid_group_labels: vec![4, 4],
        remove_translation: false,
    };

    let result = regularized_pullback(
        jacobian.view(),
        desired.view(),
        weights.view(),
        Some(coordinates.view()),
        &constraints,
        config(1e-3, 10.0),
    )
    .unwrap();

    let bond_derivative = result.step()[3] - result.step()[0];
    assert!(bond_derivative.abs() < 1e-12);
}

#[test]
fn trust_radius_clips_the_dimensionless_cartesian_norm() {
    let jacobian = Array2::eye(3);
    let desired = array![4.0, 0.0, 0.0];
    let weights = Array1::ones(3);
    let mut cfg = config(1e-6, 0.25);
    cfg.length_scale = 2.0;

    let result = regularized_pullback(
        jacobian.view(),
        desired.view(),
        weights.view(),
        None,
        &unconstrained(3),
        cfg,
    )
    .unwrap();

    let norm = result
        .step()
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    assert!((norm / 2.0 - 0.25).abs() < 1e-12);
    assert!(result.clipped());
}

#[test]
fn malformed_dimensions_and_parameters_are_classified() {
    let jacobian = Array2::eye(2);
    let desired = array![1.0];
    let weights = Array1::ones(2);
    assert_eq!(
        regularized_pullback(
            jacobian.view(),
            desired.view(),
            weights.view(),
            None,
            &unconstrained(2),
            config(1e-3, 1.0),
        )
        .unwrap_err(),
        PullbackError::DescriptorDimension {
            jacobian_rows: 2,
            desired: 1,
            weights: 2,
        }
    );

    assert_eq!(
        regularized_pullback(
            jacobian.view(),
            array![1.0, 1.0].view(),
            weights.view(),
            None,
            &unconstrained(2),
            config(0.0, 1.0),
        )
        .unwrap_err(),
        PullbackError::InvalidDamping
    );
}
