#![cfg(feature = "featomic")]

use anneal_core::descriptor_space::{
    DescriptorBlockKind, DescriptorError, DescriptorGeometry, DescriptorSpace,
    FeatomicSoapProvider, FeatomicSoapScale,
};
use ndarray::Array1;

fn provider(cutoff: f64) -> FeatomicSoapProvider {
    FeatomicSoapProvider::new(
        vec![1, 8],
        vec![FeatomicSoapScale::new(cutoff, 2, 2).unwrap()],
    )
    .unwrap()
}

fn water() -> Array1<f64> {
    Array1::from_vec(vec![
        0.7572, 0.5865, 0.0, 0.0, 0.0, 0.0, -0.7572, 0.5865, 0.0,
    ])
}

fn rigid_transform(coordinates: &Array1<f64>) -> Array1<f64> {
    let angle = 0.63_f64;
    let (sine, cosine) = angle.sin_cos();
    let mut transformed = coordinates.clone();
    for atom in 0..coordinates.len() / 3 {
        let x = coordinates[3 * atom];
        let y = coordinates[3 * atom + 1];
        transformed[3 * atom] = cosine * x - sine * y + 1.7;
        transformed[3 * atom + 1] = sine * x + cosine * y - 0.9;
        transformed[3 * atom + 2] = coordinates[3 * atom + 2] + 2.4;
    }
    transformed
}

#[test]
fn featomic_provider_exposes_fixed_invariant_system_and_atomic_features() {
    let coordinates = water();
    let moved = rigid_transform(&coordinates);
    let permuted = Array1::from_vec(vec![
        coordinates[6],
        coordinates[7],
        coordinates[8],
        coordinates[3],
        coordinates[4],
        coordinates[5],
        coordinates[0],
        coordinates[1],
        coordinates[2],
    ]);
    let geometry = DescriptorGeometry::finite(1.0).unwrap();
    let space = DescriptorSpace::from_provider(geometry, provider(4.0)).unwrap();

    let reference = space
        .describe(coordinates.view(), Some(&[1, 8, 1]))
        .unwrap();
    let transformed = space.describe(moved.view(), Some(&[1, 8, 1])).unwrap();
    let reordered = space.describe(permuted.view(), Some(&[1, 8, 1])).unwrap();
    let local = space
        .describe_local(coordinates.view(), Some(&[1, 8, 1]))
        .unwrap();
    let local_reordered = space
        .describe_local(permuted.view(), Some(&[1, 8, 1]))
        .unwrap();

    let expected_dimension = 2 * 3 * 3 * 3 * 3;
    assert_eq!(reference.values().len(), expected_dimension);
    assert_eq!(local.dim(), (3, expected_dimension));
    assert_eq!(
        reference.blocks()[0].kind(),
        DescriptorBlockKind::ProviderFeature
    );
    assert!(reference.distance(&transformed).unwrap() < 1e-10);
    assert!(reference.distance(&reordered).unwrap() < 1e-10);
    for (new_atom, old_atom) in [2, 1, 0].into_iter().enumerate() {
        assert!(
            local_reordered
                .row(new_atom)
                .iter()
                .zip(local.row(old_atom))
                .all(|(left, right)| (left - right).abs() < 1e-10)
        );
    }
}

#[test]
fn featomic_configuration_digest_separates_descriptor_metrics() {
    let coordinates = water();
    let geometry = DescriptorGeometry::finite(1.0).unwrap();
    let short = DescriptorSpace::from_provider(geometry, provider(4.0)).unwrap();
    let long = DescriptorSpace::from_provider(geometry, provider(5.0)).unwrap();
    let short_descriptor = short
        .describe(coordinates.view(), Some(&[1, 8, 1]))
        .unwrap();
    let long_descriptor = long.describe(coordinates.view(), Some(&[1, 8, 1])).unwrap();

    assert_ne!(
        short.provider_contract().unwrap().model_digest(),
        long.provider_contract().unwrap().model_digest()
    );
    assert_eq!(
        short_descriptor.distance(&long_descriptor),
        Err(DescriptorError::IncompatibleDescriptorVectors)
    );
}
