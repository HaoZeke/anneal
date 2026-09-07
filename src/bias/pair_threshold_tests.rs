use super::{SortedPairs, pair_spectrum_preparation_count};
use ndarray::{Array1, ArrayView1, array, s};

fn radii() -> [f64; 10] {
    [
        f64::NEG_INFINITY,
        -1.0,
        -f64::from_bits(1),
        -0.0,
        0.0,
        f64::from_bits(1),
        0.125,
        f64::MAX,
        f64::INFINITY,
        f64::NAN,
    ]
}

fn assert_matches_full_bound(
    pairs: &SortedPairs,
    left: ArrayView1<f64>,
    right: ArrayView1<f64>,
    radius: f64,
) {
    let expected = pairs
        .bottleneck_lower_bound(left, right)
        .map(|bound| bound > radius);
    assert_eq!(
        pairs.bottleneck_exceeds(left, right, radius),
        expected,
        "raw coordinates, radius {radius:?}"
    );
    if let (Some(left), Some(right)) = (pairs.prepare(left), pairs.prepare(right)) {
        assert_eq!(
            left.bottleneck_exceeds(&right, radius),
            expected,
            "prepared coordinates, radius {radius:?}"
        );
    }
}

#[test]
fn threshold_matches_full_bound_for_deterministic_clusters() {
    for n in [1, 2, 4, 13, 75] {
        let pairs = SortedPairs { n_points: n };
        let left = Array1::from_iter((0..3 * n).map(|i| (i as f64 * 1.37).sin()));
        for dilation in [1.0, 1.01, 3.0] {
            let mut right = Array1::zeros(left.len());
            for atom in 0..n {
                let source = n - 1 - atom;
                right[3 * atom] = -dilation * left[3 * source + 1] + 3.0;
                right[3 * atom + 1] = dilation * left[3 * source] - 2.0;
                right[3 * atom + 2] = dilation * left[3 * source + 2] + 1.0;
            }
            for radius in radii() {
                assert_matches_full_bound(&pairs, left.view(), right.view(), radius);
            }
        }
    }
}

#[test]
fn zero_bound_preserves_clamping_and_ieee_radius_comparisons() {
    for coordinates in [array![0.0, 0.0, 0.0], array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]] {
        let pairs = SortedPairs {
            n_points: coordinates.len() / 3,
        };
        let prepared = pairs.prepare(coordinates.view()).unwrap();
        assert_eq!(prepared.bottleneck_lower_bound(&prepared), Some(0.0));
        for radius in radii() {
            assert_eq!(
                prepared.bottleneck_exceeds(&prepared, radius),
                Some(0.0 > radius),
                "radius {radius:?}"
            );
            assert_matches_full_bound(&pairs, coordinates.view(), coordinates.view(), radius);
        }
    }
}

#[test]
fn threshold_preserves_roundoff_and_adjacent_radius_boundaries() {
    let pairs = SortedPairs { n_points: 2 };
    let base_left = array![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let base_right = array![-1.125, 0.0, 0.0, 1.125, 0.0, 0.0];
    for scale in [1e-100, 1e-10, 1.0, 1e10, 1e100] {
        let left = &base_left * scale;
        let right = &base_right * scale;
        let bound = pairs
            .bottleneck_lower_bound(left.view(), right.view())
            .unwrap();
        assert!(bound >= 0.0 && bound <= 0.125 * scale);
        let lower_radius = if bound == 0.0 {
            -f64::from_bits(1)
        } else {
            f64::from_bits(bound.to_bits() - 1)
        };
        let upper_radius = f64::from_bits(bound.to_bits() + 1);
        assert_eq!(
            pairs.bottleneck_exceeds(left.view(), right.view(), lower_radius),
            Some(true)
        );
        assert_eq!(
            pairs.bottleneck_exceeds(left.view(), right.view(), bound),
            Some(false)
        );
        assert_eq!(
            pairs.bottleneck_exceeds(left.view(), right.view(), upper_radius),
            Some(false)
        );
        for radius in [lower_radius, bound, upper_radius, 0.125 * scale] {
            assert_matches_full_bound(&pairs, left.view(), right.view(), radius);
        }
    }
}

#[test]
fn invalid_shapes_never_produce_a_threshold_certificate() {
    let valid = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let pairs = SortedPairs { n_points: 2 };
    for length in [0, 1, 3, 5, 7] {
        let invalid = Array1::zeros(length);
        for radius in radii() {
            assert_eq!(
                pairs.bottleneck_exceeds(valid.view(), invalid.view(), radius),
                None
            );
            assert_eq!(
                pairs.bottleneck_exceeds(invalid.view(), valid.view(), radius),
                None
            );
        }
    }
    let empty = Array1::zeros(0);
    for n_points in [0, usize::MAX] {
        for radius in radii() {
            assert_eq!(
                SortedPairs { n_points }.bottleneck_exceeds(empty.view(), empty.view(), radius),
                None
            );
        }
    }
}

#[test]
fn nonfinite_coordinates_and_distance_overflow_never_certify() {
    let valid = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let pairs = SortedPairs { n_points: 2 };
    let mut invalid = vec![array![-1e308, 0.0, 0.0, 1e308, 0.0, 0.0]];
    for value in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
        invalid.push(array![0.0, 0.0, 0.0, value, 0.0, 0.0]);
    }
    for coordinates in invalid {
        for radius in radii() {
            assert_eq!(
                pairs.bottleneck_exceeds(valid.view(), coordinates.view(), radius),
                None
            );
            assert_eq!(
                pairs.bottleneck_exceeds(coordinates.view(), valid.view(), radius),
                None
            );
            assert_matches_full_bound(&pairs, valid.view(), coordinates.view(), radius);
        }
    }
}

#[test]
fn prepared_spectra_with_unequal_atom_counts_never_certify() {
    let left = SortedPairs { n_points: 1 }
        .prepare(array![0.0, 0.0, 0.0].view())
        .unwrap();
    let right = SortedPairs { n_points: 2 }
        .prepare(array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0].view())
        .unwrap();
    for radius in radii() {
        assert_eq!(left.bottleneck_exceeds(&right, radius), None);
        assert_eq!(right.bottleneck_exceeds(&left, radius), None);
    }
}

#[test]
fn threshold_preserves_strided_coordinate_views() {
    let coordinates = array![0.0, 0.0, 0.0, 1.3, 0.1, 0.0, 0.2, 1.7, 0.3, 0.1, 0.3, 2.1];
    let mut interleaved = Array1::zeros(2 * coordinates.len());
    for (index, value) in coordinates.iter().enumerate() {
        interleaved[2 * index] = 1.2 * value;
        interleaved[2 * index + 1] = f64::NAN;
    }
    let right = interleaved.slice(s![..;2]);
    for radius in radii() {
        assert_matches_full_bound(
            &SortedPairs { n_points: 4 },
            coordinates.view(),
            right,
            radius,
        );
    }
}

#[test]
fn preparation_count_is_thread_local_and_includes_invalid_attempts() {
    let before = pair_spectrum_preparation_count();
    let pairs = SortedPairs { n_points: 1 };
    assert!(pairs.prepare(array![0.0, 0.0, 0.0].view()).is_some());
    assert!(pairs.prepare(array![].view()).is_none());
    assert_eq!(pair_spectrum_preparation_count(), before + 2);
    let child_count = std::thread::spawn(|| {
        assert_eq!(pair_spectrum_preparation_count(), 0);
        assert!(
            SortedPairs { n_points: 1 }
                .prepare(array![0.0, 0.0, 0.0].view())
                .is_some()
        );
        pair_spectrum_preparation_count()
    })
    .join()
    .unwrap();
    assert_eq!(child_count, 1);
    assert_eq!(pair_spectrum_preparation_count(), before + 2);
}
