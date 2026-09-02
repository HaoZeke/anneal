use anneal_core::assignment::minimum_cost_assignment;
use anneal_core::continuous_symmetry::{inversion_rms, project_inversion};
use ndarray::Array1;

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
        for axis in 0..3 {
            shifted[3 * atom + axis] += shift[axis];
        }
    }

    let base = project_inversion(x.view(), &classes).expect("base projection");
    let translated = project_inversion(shifted.view(), &classes).expect("translated projection");

    assert_eq!(base.assignment, translated.assignment);
    for atom in 0..classes.len() {
        for axis in 0..3 {
            let expected = base.coordinates[3 * atom + axis] + shift[axis];
            assert!((translated.coordinates[3 * atom + axis] - expected).abs() < 1e-12);
        }
    }
}
