use anneal_core::catalog::lj::{
    accepts_calibration_minimum, descriptor_space, fresh_evaluation, parse_reference_coordinates,
    perturb_reference, system_signature, validator_config,
};
use ndarray::ArrayView1;

#[test]
fn larger_lj_signatures_scale_without_changing_the_descriptor_contract() {
    let lj75 = system_signature(75).unwrap();
    let lj104 = system_signature(104).unwrap();

    assert_eq!(lj75.coordinate_dim, 225);
    assert_eq!(lj104.coordinate_dim, 312);
    assert_eq!(lj75.descriptor, lj104.descriptor);
    assert_ne!(lj75.digest(), lj104.digest());
    assert_eq!(lj75.engine.kind, "lennard-jones-reduced-v1");
}

#[test]
fn lj_preset_descriptor_validator_and_engine_agree() {
    let r = 2.0_f64.powf(1.0 / 6.0);
    let coordinates = vec![0.0, 0.0, 0.0, r, 0.0, 0.0];
    let signature = system_signature(2).unwrap();
    let descriptor = descriptor_space()
        .describe(
            ArrayView1::from(&coordinates),
            Some(&signature.atomic_numbers),
        )
        .unwrap();
    let validator = validator_config(&coordinates, descriptor.values().len()).unwrap();
    let fresh = fresh_evaluation(2, &coordinates).unwrap();

    assert!(descriptor.values().iter().all(|value| value.is_finite()));
    assert_eq!(validator.descriptor_dim, descriptor.values().len());
    assert!((fresh.energy + 1.0).abs() < 1e-12);
    assert!(fresh.forces.iter().all(|force| force.abs() < 1e-10));
}

#[test]
fn invalid_lj_sizes_and_reference_dimensions_are_rejected() {
    assert!(system_signature(0).is_err());
    assert!(fresh_evaluation(3, &[0.0; 6]).is_err());
    assert!(validator_config(&[0.0; 5], 8).is_err());
}

#[test]
fn development_reference_parser_requires_one_finite_xyz_row_per_site() {
    let coordinates = parse_reference_coordinates("1.0 2.0 3.0\n-1.0 -2.0 -3.0\n", 2).unwrap();
    assert_eq!(coordinates, vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0]);

    assert!(parse_reference_coordinates("1 2 3 4\n0 0 0\n", 2).is_err());
    assert!(parse_reference_coordinates("1 2 inf\n0 0 0\n", 2).is_err());
    assert!(parse_reference_coordinates("1 2 3\n", 2).is_err());
}

#[test]
fn calibration_identity_requires_energy_gradient_and_ira_evidence() {
    assert!(accepts_calibration_minimum(
        -397.492331,
        -397.492331 + 5e-8,
        8e-6,
        2e-5,
    ));
    assert!(!accepts_calibration_minimum(
        -397.492331,
        -397.492331 + 2e-6,
        8e-6,
        2e-5,
    ));
    assert!(!accepts_calibration_minimum(
        -397.492331,
        -397.492331,
        2e-5,
        2e-5,
    ));
    assert!(!accepts_calibration_minimum(
        -397.492331,
        -397.492331,
        8e-6,
        2e-4,
    ));
    assert!(!accepts_calibration_minimum(
        -397.492331,
        f64::NAN,
        8e-6,
        2e-5,
    ));
}

#[test]
fn calibration_perturbations_are_seeded_independent_and_centred() {
    let reference = vec![-1.0, 0.5, 0.25, 0.0, -0.5, 0.5, 1.0, 0.0, -0.75];
    let left = perturb_reference(&reference, 3, 17, 0.01).unwrap();
    let replay = perturb_reference(&reference, 3, 17, 0.01).unwrap();
    let right = perturb_reference(&reference, 3, 18, 0.01).unwrap();

    assert_eq!(left, replay);
    assert_ne!(left, right);
    for axis in 0..3 {
        let reference_centroid = (0..3).map(|atom| reference[3 * atom + axis]).sum::<f64>() / 3.0;
        let perturbed_centroid = (0..3).map(|atom| left[3 * atom + axis]).sum::<f64>() / 3.0;
        assert!((perturbed_centroid - reference_centroid).abs() < 1e-14);
    }
    assert!(perturb_reference(&reference, 3, 19, 0.0).is_err());
    assert!(perturb_reference(&reference[..6], 3, 19, 0.01).is_err());
}
