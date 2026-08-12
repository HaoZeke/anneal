use anneal_core::catalog_policy::proposal::{
    ProposalError, attraction_differential, catalog_differential, farthest_hole, pullback_increment,
};
use anneal_core::descriptor_space::pullback::{PullbackConfig, PullbackConstraints, PullbackError};
use ndarray::{Array1, Array2, array};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn mean(samples: &[Array1<f64>]) -> Array1<f64> {
    let mut value = Array1::zeros(samples[0].len());
    for sample in samples {
        value += sample;
    }
    value / samples.len() as f64
}

#[test]
fn exhaustive_independent_differentials_match_d27_mean_and_covariance() {
    let catalog = vec![array![0.0, 1.0], array![2.0, -1.0], array![4.0, 3.0]];
    let scale = 0.4;
    let mut differentials = Vec::new();
    for left in &catalog {
        for right in &catalog {
            differentials.push(scale * (left - right));
        }
    }

    let differential_mean = mean(&differentials);
    assert!(differential_mean.iter().all(|value| value.abs() < 1e-12));

    let catalog_mean = mean(&catalog);
    for row in 0..2 {
        for column in 0..2 {
            let catalog_covariance = catalog
                .iter()
                .map(|point| {
                    (point[row] - catalog_mean[row]) * (point[column] - catalog_mean[column])
                })
                .sum::<f64>()
                / catalog.len() as f64;
            let differential_covariance = differentials
                .iter()
                .map(|step| step[row] * step[column])
                .sum::<f64>()
                / differentials.len() as f64;
            assert!(
                (differential_covariance - 2.0 * scale * scale * catalog_covariance).abs() < 1e-12
            );
        }
    }
}

#[test]
fn seeded_differential_selection_is_deterministic_and_allows_equal_draws() {
    let catalog = vec![array![0.0], array![2.0], array![5.0]];
    let mut first_rng = StdRng::seed_from_u64(41);
    let mut second_rng = StdRng::seed_from_u64(41);

    let first = catalog_differential(&catalog, 0.5, &mut first_rng).unwrap();
    let second = catalog_differential(&catalog, 0.5, &mut second_rng).unwrap();

    assert_eq!(first, second);
    assert!(first.left_index() < catalog.len());
    assert!(first.right_index() < catalog.len());
    let expected = 0.5 * (&catalog[first.left_index()] - &catalog[first.right_index()]);
    assert_eq!(first.increment(), &expected);

    let singleton = vec![array![3.0, -1.0]];
    let zero = catalog_differential(&singleton, 0.8, &mut first_rng).unwrap();
    assert_eq!(zero.left_index(), zero.right_index());
    assert_eq!(zero.increment(), &array![0.0, 0.0]);
}

#[test]
fn farthest_hole_is_seeded_unit_length_and_no_closer_than_the_current_point() {
    let current = array![1.0, 0.0, 0.0];
    let catalog = vec![
        array![1.0, 0.0, 0.0],
        array![0.0, 1.0, 0.0],
        array![0.0, 0.0, 1.0],
    ];
    let mut first_rng = StdRng::seed_from_u64(73);
    let mut second_rng = StdRng::seed_from_u64(73);

    let first = farthest_hole(&current, &catalog, 128, &mut first_rng).unwrap();
    let second = farthest_hole(&current, &catalog, 128, &mut second_rng).unwrap();

    assert_eq!(first, second);
    let target_norm = first
        .target()
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    assert!((target_norm - 1.0).abs() < 1e-12);
    assert!(first.nearest_catalog_distance() >= 0.0);
    assert_eq!(first.increment(), &(first.target() - &current));
}

#[test]
fn attraction_and_differential_terms_are_composed_then_bounded() {
    let current = array![0.0, 0.0];
    let anchor = array![2.0, 0.0];
    let left = array![1.0, 3.0];
    let right = array![-1.0, 1.0];

    let proposal =
        attraction_differential(&current, &anchor, &left, &right, 0.25, 0.5, 10.0).unwrap();
    assert_eq!(proposal.increment(), &array![1.5, 0.5]);
    assert!(!proposal.clipped());

    let clipped =
        attraction_differential(&current, &anchor, &left, &right, 0.25, 0.5, 0.5).unwrap();
    let norm = clipped
        .increment()
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    assert!((norm - 0.5).abs() < 1e-12);
    assert!(clipped.clipped());
}

#[test]
fn pullback_failures_remain_classified_at_the_proposal_boundary() {
    let jacobian = Array2::eye(2);
    let desired = array![1.0, 1.0];
    let weights = Array1::ones(2);
    let constraints = PullbackConstraints {
        frozen_coordinates: vec![false; 2],
        rigid_group_labels: Vec::new(),
        remove_translation: false,
    };

    assert_eq!(
        pullback_increment(
            jacobian.view(),
            desired.view(),
            weights.view(),
            None,
            &constraints,
            PullbackConfig {
                damping: 0.0,
                trust_radius: 1.0,
                length_scale: 1.0,
            },
        )
        .unwrap_err(),
        ProposalError::Pullback(PullbackError::InvalidDamping)
    );
}
