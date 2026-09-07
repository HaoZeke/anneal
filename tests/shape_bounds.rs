use anneal_core::bias::SortedPairs;
use ndarray::{Array1, array};

#[test]
fn sorted_pair_bound_cannot_exceed_a_known_rigid_permutation_match() {
    for n in [4, 13, 38, 75] {
        let epsilon = 0.03;
        let left = Array1::from_iter((0..3 * n).map(|i| (i as f64 * 1.37).sin()));
        let mut right = Array1::zeros(3 * n);
        for atom in 0..n {
            let source = n - 1 - atom;
            let noise = epsilon * (atom as f64 * 0.83).sin();
            right[3 * atom] =
                0.6 * left[3 * source] - 0.8 * left[3 * source + 1] + 3.0 + 0.6 * noise;
            right[3 * atom + 1] =
                0.8 * left[3 * source] + 0.6 * left[3 * source + 1] - 2.0 + 0.8 * noise;
            right[3 * atom + 2] = left[3 * source + 2] + 1.0;
        }
        let lower = SortedPairs { n_points: n }
            .bottleneck_lower_bound(left.view(), right.view())
            .unwrap();
        assert!(lower >= 0.0);
        assert!(
            lower <= epsilon,
            "a witnessed {epsilon} match must survive: {lower}"
        );
    }
}

#[test]
fn a_separated_pair_spectrum_proves_a_rigid_match_is_outside_the_radius() {
    let left = array![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0];
    let right = &left * 3.0;
    let lower = SortedPairs { n_points: 4 }
        .bottleneck_lower_bound(left.view(), right.view())
        .unwrap();
    assert!(lower > 2.8);
    assert!(lower <= 2.0 * 2.0_f64.sqrt());
}

#[test]
fn a_zero_pair_bound_does_not_claim_structural_identity() {
    let line = |points: &[f64]| Array1::from_iter(points.iter().flat_map(|&x| [x, 0.0, 0.0]));
    let left = line(&[0.0, 1.0, 4.0, 10.0, 12.0, 17.0]);
    let right = line(&[0.0, 1.0, 8.0, 11.0, 13.0, 17.0]);
    let lower = SortedPairs { n_points: 6 }
        .bottleneck_lower_bound(left.view(), right.view())
        .unwrap();
    assert_eq!(lower, 0.0);
    assert_ne!(left, right);
}

#[test]
fn invalid_pair_bound_inputs_return_no_certificate() {
    let pairs = SortedPairs { n_points: 2 };
    let valid = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    assert!(
        pairs
            .bottleneck_lower_bound(valid.view(), array![0.0].view())
            .is_none()
    );
    let invalid = array![0.0, 0.0, 0.0, f64::NAN, 0.0, 0.0];
    assert!(
        pairs
            .bottleneck_lower_bound(valid.view(), invalid.view())
            .is_none()
    );
    assert!(
        SortedPairs { n_points: 0 }
            .bottleneck_lower_bound(array![].view(), array![].view())
            .is_none()
    );
}

#[cfg(feature = "ira")]
#[test]
fn bounded_ira_matching_returns_the_rejection_certificate() {
    use anneal_core::bias::BasinMetric;
    use anneal_core::shape::IraMetric;

    let left = array![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0];
    let right = &left * 3.0;
    let bounded = IraMetric::default().distance_bounded(left.view(), right.view(), 0.1);
    assert!(bounded > 2.8);
    assert!(bounded <= 2.0 * 2.0_f64.sqrt());
}

#[test]
fn pair_bound_preserves_radius_boundary_and_extreme_scales() {
    let radius = 0.125;
    let left = array![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let right = array![-1.125, 0.0, 0.0, 1.125, 0.0, 0.0];
    let pairs = SortedPairs { n_points: 2 };
    let lower = pairs.bottleneck_lower_bound(left.view(), right.view()).unwrap();
    assert!(lower > 0.99 * radius);
    assert!(lower <= radius);
    for scale in [1e-100, 1e-10, 1.0, 1e10, 1e100] {
        let scaled_left = &left * scale;
        let scaled_right = &right * scale;
        let lower = pairs
            .bottleneck_lower_bound(scaled_left.view(), scaled_right.view())
            .unwrap();
        assert!(lower >= 0.0);
        assert!(lower <= radius * scale);
    }
}

#[test]
fn pair_bound_overflow_returns_no_certificate() {
    let left = array![-1e308, 0.0, 0.0, 1e308, 0.0, 0.0];
    assert!(
        SortedPairs { n_points: 2 }
            .bottleneck_lower_bound(left.view(), left.view())
            .is_none()
    );
}

#[cfg(feature = "ira")]
#[test]
fn bounded_ira_preserves_a_valid_rigid_permutation_match() {
    use anneal_core::bias::BasinMetric;
    use anneal_core::shape::IraMetric;

    let left = array![0.0, 0.0, 0.0, 1.3, 0.1, 0.0, 0.2, 1.7, 0.3, 0.1, 0.3, 2.1];
    let mut right = Array1::zeros(12);
    for atom in 0..4 {
        let source = 3 - atom;
        right[3 * atom] = -left[3 * source + 1] + 3.0;
        right[3 * atom + 1] = left[3 * source] - 2.0;
        right[3 * atom + 2] = left[3 * source + 2] + 1.0;
    }
    let metric = IraMetric::default();
    let full = metric.distance(left.view(), right.view());
    let bounded = metric.distance_bounded(left.view(), right.view(), 0.1);
    assert!(full < 1e-10, "the rigid permutation is witnessed: {full}");
    assert!((bounded - full).abs() < 1e-10);
}
