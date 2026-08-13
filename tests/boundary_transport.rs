use anneal_core::boundary_transport::{
    BoundaryTransportConfig, ObservedCrossing, boundary_transport,
};
use ndarray::Array1;

#[test]
fn observed_crossing_displacement_aligns_to_the_current_frame() {
    let from = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let to = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
    let current = Array1::from(vec![3.0, 4.0, 0.0, 3.0, 5.0, 0.0, 2.0, 4.0, 0.0]);
    let crossing = ObservedCrossing::new(from, to).unwrap();
    let proposal = boundary_transport(
        current.view(),
        &crossing,
        Array1::zeros(current.len()).view(),
        &BoundaryTransportConfig::unconstrained(10.0),
    )
    .unwrap();

    let expected = Array1::from(vec![3.0, 4.0, 0.0, 3.0, 5.0, 0.0, 1.0, 4.0, 0.0]);
    for (actual, expected) in proposal.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() < 1e-8,
            "aligned coordinate {actual} differs from {expected}; proposal={proposal:?}"
        );
    }
}

#[test]
fn perturbation_is_zero_mean_and_frozen_coordinates_do_not_move() {
    let current = Array1::from(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
    let crossing = ObservedCrossing::new(current.clone(), current.clone()).unwrap();
    let noise = Array1::from(vec![1.0, 0.0, 0.0, 3.0, 0.0, 0.0]);
    let proposal = boundary_transport(
        current.view(),
        &crossing,
        noise.view(),
        &BoundaryTransportConfig {
            noise_scale: 1.0,
            trust_radius: 10.0,
            frozen_coordinates: vec![true, true, true, false, false, false],
            rigid_groups: Vec::new(),
        },
    )
    .unwrap();

    assert_eq!(&proposal.as_slice().unwrap()[..3], &[0.0, 0.0, 0.0]);
    assert!((proposal[3] - 2.0).abs() < 1e-12);
}

#[test]
fn rigid_group_transport_preserves_internal_distances() {
    let current = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let crossing = ObservedCrossing::new(current.clone(), current.clone()).unwrap();
    let noise = Array1::from(vec![0.2, 0.1, 0.0, -0.4, 0.3, 0.0, 0.1, -0.5, 0.0]);
    let proposal = boundary_transport(
        current.view(),
        &crossing,
        noise.view(),
        &BoundaryTransportConfig {
            noise_scale: 1.0,
            trust_radius: 10.0,
            frozen_coordinates: vec![false; current.len()],
            rigid_groups: vec![vec![0, 1, 2]],
        },
    )
    .unwrap();

    for (first, second) in [(0, 1), (0, 2), (1, 2)] {
        let distance = |coordinates: &Array1<f64>| {
            (0..3)
                .map(|axis| {
                    let delta = coordinates[3 * first + axis] - coordinates[3 * second + axis];
                    delta * delta
                })
                .sum::<f64>()
                .sqrt()
        };
        assert!((distance(&proposal) - distance(&current)).abs() < 1e-8);
    }
}
