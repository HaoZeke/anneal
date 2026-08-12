use anneal_core::descriptor_space::{
    DescriptorBlockKind, DescriptorBlockSpec, DescriptorSchema, DescriptorSpace,
};
use ndarray::Array1;

fn schema() -> DescriptorSchema {
    DescriptorSchema::new(
        "jcc-multiscale-soap",
        1,
        vec![
            DescriptorBlockSpec::new(DescriptorBlockKind::SoapMean, 3, 6, 3.5).unwrap(),
            DescriptorBlockSpec::new(DescriptorBlockKind::SoapVariance, 3, 6, 3.5).unwrap(),
            DescriptorBlockSpec::new(DescriptorBlockKind::AceNu3Mean, 2, 3, 2.5).unwrap(),
        ],
    )
    .unwrap()
}

fn asymmetric_cluster() -> (Array1<f64>, Vec<u32>) {
    (
        Array1::from_vec(vec![
            0.0, 0.0, 0.0, 1.1, 0.2, -0.1, -0.3, 1.3, 0.4, 0.4, -0.5, 1.5, -1.0, -0.7, 0.2,
        ]),
        vec![6, 8, 6, 8, 6],
    )
}

fn rigid_transform(coordinates: &Array1<f64>) -> Array1<f64> {
    let angle = 0.731_f64;
    let (sine, cosine) = angle.sin_cos();
    let mut transformed = coordinates.clone();
    for atom in 0..coordinates.len() / 3 {
        let x = coordinates[3 * atom];
        let y = coordinates[3 * atom + 1];
        let z = coordinates[3 * atom + 2];
        transformed[3 * atom] = cosine * x - sine * y + 2.5;
        transformed[3 * atom + 1] = sine * x + cosine * y - 1.25;
        transformed[3 * atom + 2] = z + 0.75;
    }
    transformed
}

fn permute(coordinates: &Array1<f64>, species: &[u32], order: &[usize]) -> (Array1<f64>, Vec<u32>) {
    let mut permuted = Vec::with_capacity(coordinates.len());
    let mut permuted_species = Vec::with_capacity(species.len());
    for &atom in order {
        permuted.extend_from_slice(&coordinates.as_slice().unwrap()[3 * atom..3 * atom + 3]);
        permuted_species.push(species[atom]);
    }
    (Array1::from_vec(permuted), permuted_species)
}

fn distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let delta = left - right;
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
}

fn load_xyz(text: &str) -> Array1<f64> {
    let coordinates = text
        .lines()
        .skip(2)
        .filter(|line| !line.trim().is_empty())
        .flat_map(|line| {
            line.split_whitespace()
                .skip(1)
                .take(3)
                .map(str::parse::<f64>)
                .collect::<Vec<_>>()
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    Array1::from_vec(coordinates)
}

#[test]
fn multiscale_descriptor_is_rigid_motion_invariant() {
    let (coordinates, species) = asymmetric_cluster();
    let descriptor_space = DescriptorSpace::new(schema());
    let original = descriptor_space
        .describe(coordinates.view(), Some(&species))
        .unwrap();
    let transformed = descriptor_space
        .describe(rigid_transform(&coordinates).view(), Some(&species))
        .unwrap();

    assert!(
        distance(original.values(), transformed.values()) < 1e-9,
        "multiscale descriptor moved under translation and rotation"
    );
}

#[test]
fn multiscale_descriptor_is_like_species_permutation_invariant() {
    let (coordinates, species) = asymmetric_cluster();
    let (permuted, permuted_species) = permute(&coordinates, &species, &[4, 1, 2, 3, 0]);
    let descriptor_space = DescriptorSpace::new(schema());
    let original = descriptor_space
        .describe(coordinates.view(), Some(&species))
        .unwrap();
    let reordered = descriptor_space
        .describe(permuted.view(), Some(&permuted_species))
        .unwrap();

    assert!(distance(original.values(), reordered.values()) < 1e-9);
}

#[test]
fn every_block_is_normalized_and_has_stable_metadata() {
    let (coordinates, species) = asymmetric_cluster();
    let descriptor = DescriptorSpace::new(schema())
        .describe(coordinates.view(), Some(&species))
        .unwrap();

    assert_eq!(descriptor.blocks().len(), 3);
    assert_eq!(descriptor.blocks()[0].kind(), DescriptorBlockKind::SoapMean);
    assert_eq!(
        descriptor.blocks()[1].kind(),
        DescriptorBlockKind::SoapVariance
    );
    assert_eq!(
        descriptor.blocks()[2].kind(),
        DescriptorBlockKind::AceNu3Mean
    );
    for block in descriptor.blocks() {
        assert_eq!(block.normalization(), "l2-v1");
        assert!(block.raw_norm().is_finite());
        let values = &descriptor.values()[block.offset()..block.offset() + block.len()];
        let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-12 || norm == 0.0);
    }
}

#[test]
fn lj75_mackay_and_marks_shelves_are_separated_without_runtime_labels() {
    let mackay = load_xyz(include_str!("fixtures/lj75_ico.xyz"));
    let marks = load_xyz(include_str!("fixtures/lj75_marks.xyz"));
    let descriptor_space = DescriptorSpace::new(schema());
    let mackay_descriptor = descriptor_space.describe(mackay.view(), None).unwrap();
    let marks_descriptor = descriptor_space.describe(marks.view(), None).unwrap();
    let separation = distance(mackay_descriptor.values(), marks_descriptor.values());

    assert!(
        separation > 0.05,
        "LJ75 competing shelves have descriptor distance {separation}"
    );
}
