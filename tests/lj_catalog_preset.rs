use anneal_core::catalog::lj::{
    descriptor_space, fresh_evaluation, system_signature, validator_config,
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
