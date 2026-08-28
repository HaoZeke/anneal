use anneal_core::descriptor_space::{
    DescriptorBlockKind, DescriptorGeometry, DescriptorVector, UNIVERSAL_DESCRIPTOR_SCHEMA,
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

fn permute(coordinates: &Array1<f64>, species: &[u32], order: &[usize]) -> (Array1<f64>, Vec<u32>) {
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

fn block_values(descriptor: &DescriptorVector, kind: DescriptorBlockKind) -> Vec<&[f64]> {
    descriptor
        .blocks()
        .iter()
        .filter(|block| block.kind() == kind)
        .map(|block| {
            &descriptor.values()[block.offset()..block.offset() + block.len()]
        })
        .collect()
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
                0.0, 0.0, 0.0, 0.7572, 0.5865, 0.0, -0.7572, 0.5865, 0.0, 2.8, 0.1, 0.0, 3.5572,
                0.6865, 0.0, 2.0428, 0.6865, 0.0,
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
                0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.8, 0.8, 1.7, 1.6,
                0.8, 1.7,
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
        0.0, 0.0, 0.0, 1.1, 0.2, -0.1, -0.3, 1.3, 0.4, 0.4, -0.5, 1.5, -1.0, -0.7, 0.2,
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
fn periodic_descriptor_is_invariant_to_a_unimodular_cell_basis_change() {
    let sheared = DescriptorGeometry::new(
        1.0,
        Some([4.0, 0.0, 0.0, 3.8, 1.0, 0.0, 0.0, 0.0, 8.0]),
        [true; 3],
    )
    .unwrap();
    let reduced = DescriptorGeometry::new(
        1.0,
        Some([4.0, 0.0, 0.0, -0.2, 1.0, 0.0, 0.0, 0.0, 8.0]),
        [true; 3],
    )
    .unwrap();
    let coordinates = Array1::from_vec(vec![
        0.1, 0.1, 0.2, 3.9, 0.55, 0.2, 1.7, 0.72, 3.1,
    ]);
    let species = [14, 8, 14];
    let left = universal_descriptor_space(sheared)
        .describe(coordinates.view(), Some(&species))
        .unwrap();
    let right = universal_descriptor_space(reduced)
        .describe(coordinates.view(), Some(&species))
        .unwrap();

    assert!(distance(left.values(), right.values()) < 1e-8);
}

#[test]
fn slab_geometry_wraps_only_its_periodic_axes() {
    let geometry = DescriptorGeometry::new(
        1.0,
        Some([6.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0, 0.0, 12.0]),
        [true, true, false],
    )
    .unwrap();
    let descriptor_space = universal_descriptor_space(geometry);
    let coordinates = Array1::from_vec(vec![
        0.2, 0.3, 0.4, 1.4, 0.5, 0.7, 0.8, 1.7, 1.1, 2.1, 1.9, 2.0,
    ]);
    let species = [78, 78, 78, 1];
    let mut in_plane_image = coordinates.clone();
    in_plane_image[3] += 7.0;
    in_plane_image[4] += 5.0;
    let mut vacuum_shift = coordinates.clone();
    vacuum_shift[5] += 12.0;

    let reference = descriptor_space
        .describe(coordinates.view(), Some(&species))
        .unwrap();
    let wrapped = descriptor_space
        .describe(in_plane_image.view(), Some(&species))
        .unwrap();
    let displaced = descriptor_space
        .describe(vacuum_shift.view(), Some(&species))
        .unwrap();

    assert!(distance(reference.values(), wrapped.values()) < 1e-8);
    assert!(distance(reference.values(), displaced.values()) > 1e-3);
}

#[test]
fn equivalent_periodic_primitive_and_supercells_have_the_same_descriptor() {
    let primitive = DescriptorGeometry::new(
        1.0,
        Some([2.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0]),
        [true; 3],
    )
    .unwrap();
    let supercell = DescriptorGeometry::new(
        1.0,
        Some([4.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0]),
        [true; 3],
    )
    .unwrap();
    let primitive_descriptor = universal_descriptor_space(primitive)
        .describe(Array1::from_vec(vec![0.0, 0.0, 0.0]).view(), Some(&[18]))
        .unwrap();
    let supercell_descriptor = universal_descriptor_space(supercell)
        .describe(
            Array1::from_vec(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]).view(),
            Some(&[18, 18]),
        )
        .unwrap();

    assert!(
        primitive_descriptor
            .blocks()
            .iter()
            .filter(|block| block.kind() == DescriptorBlockKind::PairRadial)
            .all(|block| block.raw_norm() > 0.0)
    );
    assert!(
        primitive_descriptor
            .blocks()
            .iter()
            .filter(|block| block.kind() == DescriptorBlockKind::InvariantSoapMean)
            .all(|block| block.raw_norm() > 0.0)
    );
    assert!(
        distance(
            primitive_descriptor.values(),
            supercell_descriptor.values()
        ) < 1e-8
    );
}

#[test]
fn universal_descriptor_is_reflection_invariant() {
    let coordinates = Array1::from_vec(vec![
        0.0, 0.0, 0.0, 1.1, 0.2, -0.1, -0.3, 1.3, 0.4, 0.4, -0.5, 1.5, -1.0, -0.7, 0.2,
    ]);
    let mut reflected = coordinates.clone();
    for atom in 0..coordinates.len() / 3 {
        reflected[3 * atom] = -reflected[3 * atom];
    }
    let species = [6, 8, 6, 8, 6];
    let descriptor_space = finite(1.0);
    let original = descriptor_space
        .describe(coordinates.view(), Some(&species))
        .unwrap();
    let mirror = descriptor_space
        .describe(reflected.view(), Some(&species))
        .unwrap();

    assert!(distance(original.values(), mirror.values()) < 1e-8);
}

#[test]
fn graph_block_is_continuous_at_every_radial_scale() {
    let descriptor_space = finite(1.0);
    for threshold in 1..=6 {
        let epsilon = 1e-8;
        let inside = descriptor_space
            .describe(
                Array1::from_vec(vec![0.0, 0.0, 0.0, threshold as f64 - epsilon, 0.0, 0.0])
                    .view(),
                Some(&[6, 6]),
            )
            .unwrap();
        let outside = descriptor_space
            .describe(
                Array1::from_vec(vec![0.0, 0.0, 0.0, threshold as f64 + epsilon, 0.0, 0.0])
                    .view(),
                Some(&[6, 6]),
            )
            .unwrap();
        let inside_graph = block_values(&inside, DescriptorBlockKind::GraphTopology);
        let outside_graph = block_values(&outside, DescriptorBlockKind::GraphTopology);

        assert_eq!(inside_graph.len(), 1);
        assert_eq!(outside_graph.len(), 1);
        assert!(
            distance(inside_graph[0], outside_graph[0]) < 1e-5,
            "graph block jumps at radial scale {threshold}"
        );
    }
}

#[test]
fn descriptor_is_continuous_when_the_last_neighbor_crosses_a_cutoff() {
    let descriptor_space = finite(1.0);
    let epsilon = 1e-8;
    let inside = descriptor_space
        .describe(
            Array1::from_vec(vec![0.0, 0.0, 0.0, 6.0 - epsilon, 0.0, 0.0]).view(),
            Some(&[8, 1]),
        )
        .unwrap();
    let outside = descriptor_space
        .describe(
            Array1::from_vec(vec![0.0, 0.0, 0.0, 6.0 + epsilon, 0.0, 0.0]).view(),
            Some(&[8, 1]),
        )
        .unwrap();

    assert!(distance(inside.values(), outside.values()) < 1e-5);
}

#[test]
fn zero_signal_blocks_stay_finite_under_soft_normalization() {
    let descriptor = finite(1.0)
        .describe(Array1::from_vec(vec![0.0, 0.0, 0.0]).view(), Some(&[2]))
        .unwrap();

    assert!(descriptor.values().iter().all(|value| value.is_finite()));
    for block in descriptor.blocks() {
        assert_eq!(block.normalization(), "soft-l2-eps-v1");
        let values = &descriptor.values()[block.offset()..block.offset() + block.len()];
        if block.raw_norm() == 0.0 {
            assert!(values.iter().all(|&value| value == 0.0));
        } else {
            assert!(norm(values) <= 1.0);
        }
    }
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
