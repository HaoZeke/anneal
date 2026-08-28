use anneal_core::curvature::rigid_basis;
use anneal_core::descriptor_space::DescriptorGeometry;
use anneal_core::pes_exploration::{RideModeDirection, gaussian_nd_mode, localized_cartesian_mode};
use ndarray::{ArrayView1, array};

fn norm(values: ArrayView1<'_, f64>) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

#[test]
fn arbitrary_dimensional_gaussian_modes_are_reproducible_ranked_and_signed() {
    let positive = gaussian_nd_mode(5, 0xdecaf, 2, RideModeDirection::Positive).unwrap();
    let replay = gaussian_nd_mode(5, 0xdecaf, 2, RideModeDirection::Positive).unwrap();
    let negative = gaussian_nd_mode(5, 0xdecaf, 2, RideModeDirection::Negative).unwrap();
    let other_rank = gaussian_nd_mode(5, 0xdecaf, 3, RideModeDirection::Positive).unwrap();

    assert_eq!(positive, replay);
    assert!((norm(positive.view()) - 1.0).abs() < 1e-12);
    assert!(
        positive
            .iter()
            .zip(&negative)
            .all(|(forward, reverse)| (forward + reverse).abs() < 1e-14)
    );
    let overlap = positive.dot(&other_rank).abs();
    assert!(overlap < 0.999, "ranked modes must not replay one vector");
}

#[test]
fn finite_cluster_mode_is_localized_and_contains_no_rigid_motion() {
    let coordinates = array![
        0.0, 0.0, 0.0, // selected atom
        1.4, 0.1, 0.0, // distant atoms
        -0.2, 1.5, 0.2, 0.1, -0.3, 1.6,
    ];
    let geometry = DescriptorGeometry::finite(1.0).unwrap();
    let mode = localized_cartesian_mode(
        coordinates.view(),
        0,
        &[false; 4],
        geometry,
        0.45,
        71,
        0,
        RideModeDirection::Positive,
    )
    .unwrap();

    assert!((norm(mode.view()) - 1.0).abs() < 1e-12);
    for rigid in rigid_basis(coordinates.view()) {
        assert!(mode.dot(&rigid).abs() < 1e-10);
    }
    let selected_norm = norm(mode.slice(ndarray::s![0..3]));
    let largest_other = (1..4)
        .map(|atom| norm(mode.slice(ndarray::s![3 * atom..3 * atom + 3])))
        .fold(0.0, f64::max);
    assert!(selected_norm > largest_other);
}

#[test]
fn frozen_atoms_receive_no_localized_displacement() {
    let coordinates = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.5, 0.0];
    let mode = localized_cartesian_mode(
        coordinates.view(),
        1,
        &[true, false, false],
        DescriptorGeometry::finite(1.0).unwrap(),
        0.8,
        91,
        1,
        RideModeDirection::Positive,
    )
    .unwrap();

    assert_eq!(&mode.as_slice().unwrap()[0..3], &[0.0; 3]);
    assert!((norm(mode.view()) - 1.0).abs() < 1e-12);
}
