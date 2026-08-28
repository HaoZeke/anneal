use anneal_core::descriptor_space::{
    DescriptorBlockKind, DescriptorBlockSpec, DescriptorGeometry, DescriptorSchema,
    DescriptorSpace, DescriptorVector, universal_descriptor_space,
};
use anneal_core::funnel_bo::FunnelModel;
use anneal_core::universal_coverage::{CoverageConfig, StableDeepKernel, UniversalCoverage};
use ndarray::Array1;

fn descriptor(coordinates: Vec<f64>, species: &[u32]) -> DescriptorVector {
    universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap())
        .describe(Array1::from_vec(coordinates).view(), Some(species))
        .unwrap()
}

fn tetrahedron() -> DescriptorVector {
    descriptor(tetrahedron_coordinates(), &[6, 8, 6, 8])
}

fn tetrahedron_coordinates() -> Vec<f64> {
    vec![0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.2, 1.0, 0.0, 0.3, 0.4, 0.9]
}

fn proper_transform(coordinates: &[f64]) -> Vec<f64> {
    let axis = [
        1.0_f64 / 6.0_f64.sqrt(),
        2.0 / 6.0_f64.sqrt(),
        -1.0 / 6.0_f64.sqrt(),
    ];
    let angle = 0.643_f64;
    let (sine, cosine) = angle.sin_cos();
    let one_minus_cosine = 1.0 - cosine;
    let rotation = [
        cosine + axis[0] * axis[0] * one_minus_cosine,
        axis[0] * axis[1] * one_minus_cosine - axis[2] * sine,
        axis[0] * axis[2] * one_minus_cosine + axis[1] * sine,
        axis[1] * axis[0] * one_minus_cosine + axis[2] * sine,
        cosine + axis[1] * axis[1] * one_minus_cosine,
        axis[1] * axis[2] * one_minus_cosine - axis[0] * sine,
        axis[2] * axis[0] * one_minus_cosine - axis[1] * sine,
        axis[2] * axis[1] * one_minus_cosine + axis[0] * sine,
        cosine + axis[2] * axis[2] * one_minus_cosine,
    ];
    let mut transformed = coordinates.to_vec();
    for atom in 0..coordinates.len() / 3 {
        let point = [
            coordinates[3 * atom],
            coordinates[3 * atom + 1],
            coordinates[3 * atom + 2],
        ];
        transformed[3 * atom] =
            rotation[0] * point[0] + rotation[1] * point[1] + rotation[2] * point[2] + 0.7;
        transformed[3 * atom + 1] =
            rotation[3] * point[0] + rotation[4] * point[1] + rotation[5] * point[2] - 1.4;
        transformed[3 * atom + 2] =
            rotation[6] * point[0] + rotation[7] * point[1] + rotation[8] * point[2] + 2.1;
    }
    transformed
}

fn chain() -> DescriptorVector {
    descriptor(
        vec![0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 2.3, 0.0, 0.0, 3.8, 0.0, 0.0],
        &[6, 8, 6, 8],
    )
}

#[test]
fn universal_gp_uses_euclidean_geometry_for_nonnegative_vectors() {
    let left = Array1::from_vec(vec![1.0, 1.0]);
    let scaled = Array1::from_vec(vec![2.0, 2.0]);
    let automatic = FunnelModel::new(1.0, 1.0, 1e-6);
    let euclidean = FunnelModel::new_euclidean(1.0, 1.0, 1e-6);

    assert!((automatic.similarity(left.view(), scaled.view()) - 1.0).abs() < 1e-12);
    assert!(euclidean.similarity(left.view(), scaled.view()) < 0.5);
}

#[test]
fn unseen_block_structure_outranks_an_observed_minimum_without_a_radius() {
    let observed = tetrahedron();
    let unseen = chain();
    let mut coverage = UniversalCoverage::new(&observed, CoverageConfig::default()).unwrap();
    coverage.observe(0, &observed, -12.0).unwrap();

    let known = coverage.evidence(&observed, Some(0)).unwrap();
    let novel = coverage.evidence(&unseen, None).unwrap();

    assert!(
        known
            .nearest_block_distances
            .iter()
            .all(|value| *value == Some(0.0))
    );
    assert!(novel.block_novelty > known.block_novelty);
    assert!(novel.energy_standard_deviation > known.energy_standard_deviation);
    assert!(novel.residual_variance > known.residual_variance);
    assert!(novel.acquisition > known.acquisition);
}

#[test]
fn exact_classes_survive_a_zero_descriptor_distance() {
    let first = descriptor(vec![0.0, 0.0, 0.0], &[2]);
    let translated = descriptor(vec![3.0, -2.0, 1.0], &[2]);
    assert_eq!(first.distance(&translated).unwrap(), 0.0);

    let mut coverage = UniversalCoverage::new(&first, CoverageConfig::default()).unwrap();
    coverage.observe(0, &first, -1.0).unwrap();
    coverage.observe(1, &translated, -0.8).unwrap();
    coverage.connect(0, 1).unwrap();

    assert_eq!(coverage.exact_class_count(), 2);
    assert_eq!(coverage.observation_count(), 2);
    let evidence = coverage.evidence(&translated, Some(1)).unwrap();
    assert!(
        evidence
            .nearest_block_distances
            .iter()
            .all(|value| *value == Some(0.0))
    );
}

#[test]
fn stable_deep_kernel_preserves_invariance_and_raw_separation() {
    let original = tetrahedron();
    let rotated = descriptor(proper_transform(&tetrahedron_coordinates()), &[6, 8, 6, 8]);
    let distinct = chain();
    let kernel = StableDeepKernel::seeded(original.values().len(), &[24, 8], 73).unwrap();

    assert!(
        kernel
            .distance(original.values(), rotated.values())
            .unwrap()
            < 1e-3
    );
    assert!(
        kernel
            .distance(original.values(), distinct.values())
            .unwrap()
            >= original.distance(&distinct).unwrap()
    );

    let mut coverage =
        UniversalCoverage::with_deep_kernel(&original, CoverageConfig::default(), kernel).unwrap();
    coverage.observe(0, &original, -12.0).unwrap();
    let evidence = coverage.evidence(&distinct, None).unwrap();
    assert!(evidence.deep_kernel_mean.is_some());
    assert!(evidence.deep_kernel_standard_deviation.is_some());
    assert!(evidence.model_disagreement.is_finite());
}

#[test]
fn coverage_rejects_an_incompatible_descriptor_schema() {
    let reference = tetrahedron();
    let other_space = DescriptorSpace::new(
        DescriptorSchema::new(
            "other-descriptor",
            1,
            vec![DescriptorBlockSpec::new(DescriptorBlockKind::SoapMean, 2, 2, 3.0).unwrap()],
        )
        .unwrap(),
    );
    let other = other_space
        .describe(
            Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).view(),
            Some(&[6, 6]),
        )
        .unwrap();
    let mut coverage = UniversalCoverage::new(&reference, CoverageConfig::default()).unwrap();

    assert!(coverage.observe(0, &other, -1.0).is_err());
    assert!(coverage.evidence(&other, None).is_err());
}
