#![cfg(feature = "ira")]

use anneal_core::pes_exploration::{ExactStructureWitness, StructureContext, StructureView};
use anneal_core::shape::IraStructureWitness;
use ndarray::{Array1, array, s};

fn witness() -> IraStructureWitness {
    IraStructureWitness {
        kmax_factor: 1.8,
        radius: 0.1,
    }
}

fn tetrahedron() -> Array1<f64> {
    array![0.0, 0.0, 0.0, 1.3, 0.1, 0.0, 0.2, 1.7, 0.3, 0.1, 0.3, 2.1]
}

#[test]
fn repeated_comparisons_prepare_each_coordinate_set_once() {
    let cached = witness().with_pair_cache(8192);
    let left = tetrahedron();
    let right = &left * 3.0;
    for _ in 0..5 {
        assert!(!cached.equivalent(left.view(), right.view()));
    }
    let stats = cached.cache_stats();
    assert_eq!(stats.preparations, 2);
    assert_eq!(stats.hits, 8);
    assert_eq!(stats.entries, 2);
    assert!(stats.payload_bytes <= 8192);
}

#[test]
fn changing_a_coordinate_buffer_cannot_reuse_its_cached_spectrum() {
    let cached = witness().with_pair_cache(8192);
    let left = tetrahedron();
    let mut right = &left * 3.0;
    assert!(!cached.equivalent(left.view(), right.view()));
    right.assign(&left);
    assert!(cached.equivalent(left.view(), right.view()));
    assert_eq!(cached.cache_stats().preparations, 2);
    right[3] += 2.0;
    assert!(!cached.equivalent(left.view(), right.view()));
    assert_eq!(cached.cache_stats().preparations, 3);
}

#[test]
fn cache_eviction_changes_cost_but_not_exact_relations() {
    let cached = witness().with_pair_cache(256);
    let left = tetrahedron();
    for scale in [1.0, 1.01, 3.0, 1.0, 1.01] {
        let right = &left * scale;
        assert_eq!(
            cached.relation(left.view(), right.view()),
            witness().relation(left.view(), right.view())
        );
        assert!(cached.cache_stats().payload_bytes <= 256);
    }
    assert!(cached.cache_stats().preparations > 3);
}

#[test]
fn a_disabled_cache_retains_no_coordinate_payload() {
    let cached = witness().with_pair_cache(0);
    let left = tetrahedron();
    for _ in 0..2 {
        assert!(cached.equivalent(left.view(), left.view()));
    }
    let stats = cached.cache_stats();
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.payload_bytes, 0);
    assert_eq!(stats.hits, 0);
}

#[test]
fn cached_spectra_preserve_strided_coordinates_and_rigid_permutations() {
    let cached = witness().with_pair_cache(8192);
    let left = tetrahedron();
    let mut interleaved = Array1::zeros(24);
    for atom in 0..4 {
        let source = 3 - atom;
        interleaved[6 * atom] = -left[3 * source + 1] + 3.0;
        interleaved[6 * atom + 2] = left[3 * source] - 2.0;
        interleaved[6 * atom + 4] = left[3 * source + 2] + 1.0;
    }
    let right = interleaved.slice(s![..;2]);
    assert!(cached.equivalent(left.view(), right));
    assert_eq!(
        cached.relation(left.view(), right),
        witness().relation(left.view(), right)
    );
    assert_eq!(cached.cache_stats().preparations, 2);
}

#[test]
fn cached_geometry_cannot_bypass_identity_context() {
    let cached = witness().with_pair_cache(8192);
    let coords = tetrahedron();
    assert!(cached.equivalent(coords.view(), coords.view()));
    let left_context = StructureContext::new(Some(vec![1; 4]), None, Some("left".into()));
    let right_context = StructureContext::new(Some(vec![1; 4]), None, Some("right".into()));
    let left = StructureView {
        coordinates: coords.view(),
        context: &left_context,
    };
    let right = StructureView {
        coordinates: coords.view(),
        context: &right_context,
    };
    assert_eq!(
        cached.relation_structures(left, right),
        witness().relation_structures(left, right)
    );
    assert!(!cached.equivalent_structures(left, right));
    assert!(cached.equivalent_structures(left, left));
}

#[test]
fn matching_pair_spectra_do_not_certify_homometric_identity() {
    let line = |points: &[f64]| Array1::from_iter(points.iter().flat_map(|&x| [x, 0.0, 0.0]));
    let left = line(&[0.0, 1.0, 4.0, 10.0, 12.0, 17.0]);
    let right = line(&[0.0, 1.0, 8.0, 11.0, 13.0, 17.0]);
    let cached = witness().with_pair_cache(8192);
    assert!(!cached.equivalent(left.view(), right.view()));
    assert_eq!(
        cached.relation(left.view(), right.view()),
        witness().relation(left.view(), right.view())
    );
}
