use anneal_core::descriptor_space::{
    DescriptorBlockKind, DescriptorGeometry, UNIVERSAL_DESCRIPTOR_SCHEMA,
    UNIVERSAL_DESCRIPTOR_VERSION, universal_descriptor_space,
};
use ndarray::Array1;

fn finite(length_scale: f64) -> anneal_core::descriptor_space::DescriptorSpace {
    universal_descriptor_space(DescriptorGeometry::finite(length_scale).unwrap())
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

fn permute(
    coordinates: &Array1<f64>,
    species: &[u32],
    order: &[usize],
) -> (Array1<f64>, Vec<u32>) {
    let mut permuted = Vec::with_capacity(coordinates.len());
    let mut permuted_species = Vec::with_capacity(species.len());
    for &atom in order {
        permuted.extend_from_slice(&coordinates.as_slice().unwrap()[3 * atom..3 * atom + 3]);
        permuted_species.push(species[atom]);
    }
    (Array1::from_vec(permuted), permuted_species)
}

fn load_xyz(text: &str) -> Array1<f64> {
    Array1::from_vec(
        text.lines()
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
            .unwrap(),
    )
}

#[test]
fn one_schema_has_one_dimension_for_lj_molecules_and_surfaces() {
    let lj = finite(1.0)
        .describe(
            Array1::from_vec(vec![0.0, 0.0, 0.0, 1.12, 0.0, 0.0, 0.2, 1.1, 0.0]).view(),
            Some(&[18, 18, 18]),
        )
        .unwrap();
    let water = finite(1.32)
        .describe(
            Array1::from_vec(vec![
                0.0, 0.0, 0.0, 0.7572, 0.5865, 0.0, -0.7572, 0.5865, 0.0, 2.8, 0.1,
                0.0, 3.5572, 0.6865, 0.0, 2.0428, 0.6865, 0.0,
            ])
            .view(),
            Some(&[8, 1, 1, 8, 1, 1]),
        )
        .unwrap();
    let surface_geometry = DescriptorGeometry::new(
        1.0,
        Some([8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 20.0]),
        [true, true, false],
    )
    .unwrap();
    let surface = universal_descriptor_space(surface_geometry)
        .describe(
            Array1::from_vec(vec![
                0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.8,
                0.8, 1.7, 1.6, 0.8, 1.7,
            ])
            .view(),
            Some(&[29, 29, 29, 29, 1, 1]),
        )
        .unwrap();

    assert_eq!(lj.schema_name(), UNIVERSAL_DESCRIPTOR_SCHEMA);
    assert_eq!(lj.schema_version(), UNIVERSAL_DESCRIPTOR_VERSION);
    assert_eq!(lj.values().len(), water.values().len());
    assert_eq!(lj.values().len(), surface.values().len());
    assert!(lj.distance(&water).is_ok());
    assert!(water.distance(&surface).is_ok());

    let kinds = lj
        .blocks()
        .iter()
        .map(|block| block.kind())
        .collect::<Vec<_>>();
    assert!(kinds.contains(&DescriptorBlockKind::PairRadial));
    assert!(kinds.contains(&DescriptorBlockKind::ThreeBodyAngular));
    assert!(kinds.contains(&DescriptorBlockKind::GraphTopology));
    assert!(kinds.contains(&DescriptorBlockKind::InvariantSoapMean));
    assert!(kinds.contains(&DescriptorBlockKind::InvariantAceNu3Mean));
}

#[test]
fn universal_descriptor_is_rigid_motion_and_like_species_permutation_invariant() {
    let coordinates = Array1::from_vec(vec![
        0.0, 0.0, 0.0, 1.1, 0.2, -0.1, -0.3, 1.3, 0.4, 0.4, -0.5, 1.5, -1.0, -0.7,
        0.2,
    ]);
    let species = [6, 8, 6, 8, 6];
    let descriptor_space = finite(1.0);
    let original = descriptor_space
        .describe(coordinates.view(), Some(&species))
        .unwrap();
    let moved = descriptor_space
        .describe(rigid_transform(&coordinates).view(), Some(&species))
        .unwrap();
    let (permuted, permuted_species) = permute(&coordinates, &species, &[4, 1, 2, 3, 0]);
    let reordered = descriptor_space
        .describe(permuted.view(), Some(&permuted_species))
        .unwrap();

    assert!(distance(original.values(), moved.values()) < 1e-9);
    assert!(distance(original.values(), reordered.values()) < 1e-9);
}

#[test]
fn periodic_images_have_the_same_descriptor() {
    let geometry = DescriptorGeometry::new(
        1.0,
        Some([10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]),
        [true, true, true],
    )
    .unwrap();
    let descriptor_space = universal_descriptor_space(geometry);
    let wrapped = Array1::from_vec(vec![0.1, 0.2, 0.3, 9.8, 0.2, 0.3, 5.0, 5.0, 5.0]);
    let unwrapped = Array1::from_vec(vec![0.1, 0.2, 0.3, -0.2, 0.2, 0.3, 5.0, 5.0, 5.0]);
    let species = [29, 1, 29];
    let left = descriptor_space
        .describe(wrapped.view(), Some(&species))
        .unwrap();
    let right = descriptor_space
        .describe(unwrapped.view(), Some(&species))
        .unwrap();

    assert!(distance(left.values(), right.values()) < 1e-9);
}

#[test]
fn unlike_species_assignment_changes_the_geometry_embedding() {
    let coordinates = Array1::from_vec(vec![
        0.0, 0.0, 0.0, 0.9, 0.1, 0.0, -0.2, 1.4, 0.3, 1.7, 1.2, -0.4,
    ]);
    let descriptor_space = finite(1.0);
    let first = descriptor_space
        .describe(coordinates.view(), Some(&[8, 1, 1, 6]))
        .unwrap();
    let reassigned = descriptor_space
        .describe(coordinates.view(), Some(&[1, 8, 1, 6]))
        .unwrap();

    assert!(distance(first.values(), reassigned.values()) > 1e-4);
}

#[test]
fn universal_descriptor_separates_the_lj75_competing_shelves() {
    let mackay = load_xyz(include_str!("fixtures/lj75_ico.xyz"));
    let marks = load_xyz(include_str!("fixtures/lj75_marks.xyz"));
    let descriptor_space = finite(1.0);
    let mackay_descriptor = descriptor_space.describe(mackay.view(), None).unwrap();
    let marks_descriptor = descriptor_space.describe(marks.view(), None).unwrap();
    let separation = distance(mackay_descriptor.values(), marks_descriptor.values());

    assert!(
        separation > 0.05,
        "universal descriptor merged LJ75 Mackay and Marks at {separation}"
    );
}

#[test]
fn invalid_geometry_is_rejected() {
    assert!(DescriptorGeometry::finite(0.0).is_err());
    assert!(DescriptorGeometry::finite(f64::NAN).is_err());
    assert!(DescriptorGeometry::new(1.0, None, [true, false, false]).is_err());
    assert!(
        DescriptorGeometry::new(
            1.0,
            Some([1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            [true; 3],
        )
        .is_err()
    );
}
