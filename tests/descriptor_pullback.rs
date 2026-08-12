use anneal_core::descriptor_space::pullback::{
    PullbackConfig, PullbackConstraints, PullbackError, regularized_pullback,
};
use anneal_core::descriptor_space::{
    DescriptorBlockKind, DescriptorBlockSpec, DescriptorError, DescriptorSchema, DescriptorSpace,
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

fn nonlinear_descriptor_space() -> DescriptorSpace {
    DescriptorSpace::new(
        DescriptorSchema::new(
            "pullback-soap",
            1,
            vec![DescriptorBlockSpec::new(DescriptorBlockKind::SoapMean, 2, 2, 3.5).unwrap()],
        )
        .unwrap(),
    )
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

#[test]
fn finite_difference_jacobian_matches_an_explicit_central_column() {
    let descriptor_space = nonlinear_descriptor_space();
    let coordinates = array![
        0.0, 0.0, 0.0, 1.1, 0.2, -0.1, -0.3, 1.3, 0.4, 0.4, -0.5, 1.5
    ];
    let step = 1e-6;
    let jacobian = descriptor_space
        .jacobian_fd(coordinates.view(), None, step)
        .unwrap();
    let column = 4;
    let mut plus = coordinates.clone();
    let mut minus = coordinates.clone();
    plus[column] += step;
    minus[column] -= step;
    let plus_descriptor = descriptor_space.describe(plus.view(), None).unwrap();
    let minus_descriptor = descriptor_space.describe(minus.view(), None).unwrap();

    assert_eq!(jacobian.nrows(), plus_descriptor.values().len());
    assert_eq!(jacobian.ncols(), coordinates.len());
    for row in 0..jacobian.nrows() {
        let expected =
            (plus_descriptor.values()[row] - minus_descriptor.values()[row]) / (2.0 * step);
        assert!((jacobian[[row, column]] - expected).abs() < 1e-12);
    }
}

#[test]
fn finite_difference_jacobian_rejects_an_invalid_step() {
    let descriptor_space = nonlinear_descriptor_space();
    let coordinates = array![0.0, 0.0, 0.0, 1.1, 0.2, -0.1];

    assert_eq!(
        descriptor_space
            .jacobian_fd(coordinates.view(), None, 0.0)
            .unwrap_err(),
        DescriptorError::InvalidFiniteDifferenceStep
    );
}

#[test]
fn descriptor_pullback_contracts_the_actual_nonlinear_residual() {
    let descriptor_space = nonlinear_descriptor_space();
    let coordinates = array![
        0.0, 0.0, 0.0, 1.1, 0.2, -0.1, -0.3, 1.3, 0.4, 0.4, -0.5, 1.5
    ];
    let mut target_coordinates = coordinates.clone();
    target_coordinates[3] += 2e-3;
    target_coordinates[7] -= 1e-3;
    let current = descriptor_space.describe(coordinates.view(), None).unwrap();
    let target = descriptor_space
        .describe(target_coordinates.view(), None)
        .unwrap();
    let desired = Array1::from_iter(
        target
            .values()
            .iter()
            .zip(current.values())
            .map(|(target, current)| target - current),
    );
    let jacobian = descriptor_space
        .jacobian_fd(coordinates.view(), None, 1e-6)
        .unwrap();
    let result = regularized_pullback(
        jacobian.view(),
        desired.view(),
        Array1::ones(desired.len()).view(),
        None,
        &unconstrained(coordinates.len()),
        config(1e-5, 0.01),
    )
    .unwrap();
    let moved_coordinates = &coordinates + result.step();
    let moved = descriptor_space
        .describe(moved_coordinates.view(), None)
        .unwrap();
    let initial_residual = desired.iter().map(|value| value * value).sum::<f64>().sqrt();
    let final_residual = moved
        .values()
        .iter()
        .zip(target.values())
        .map(|(moved, target)| (moved - target).powi(2))
        .sum::<f64>()
        .sqrt();

    assert!(
        final_residual < initial_residual,
        "nonlinear residual did not contract: {initial_residual} -> {final_residual}"
    );
}
