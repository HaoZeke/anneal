use anneal_core::soap::{SoapSpec, packing_mean_nu3, push_away_clouds, push_away_means};
use ndarray::{Array1, ArrayView1, array};

fn structure() -> Array1<f64> {
    array![0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.3]
}

fn distance(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    (&a - &b).mapv(|value| value * value).sum().sqrt()
}

#[test]
fn a_nearby_peer_on_either_side_is_repulsive_not_a_descriptor_origin() {
    let x = structure();
    let spec = SoapSpec::default();
    let here = packing_mean_nu3(x.view(), spec, None, None);
    let cap = 1e-4;
    for scale in [0.97, 1.03] {
        let peer = &x * scale;
        let held = packing_mean_nu3(peer.view(), spec, None, None);
        let before = distance(here.view(), held.view());
        assert!(before > 0.0 && before.is_finite());
        let proposal = push_away_means(x.view(), &[held.to_vec()], spec, cap)
            .expect("a nonzero nearby descriptor difference supplies a separation direction");
        assert!(proposal.iter().all(|value| value.is_finite()));
        let rmsd = distance(proposal.view(), x.view()) / (x.len() as f64 / 3.0).sqrt();
        assert!(rmsd > 0.0 && rmsd <= cap * (1.0 + 1e-10));
        let moved = packing_mean_nu3(proposal.view(), spec, None, None);
        let after = distance(moved.view(), held.view());
        assert!(
            after > before,
            "peer scale {scale}: repulsion must increase descriptor distance, {before} -> {after}"
        );
    }
}

#[test]
fn coincident_descriptors_yield_to_the_independent_escape() {
    let x = structure();
    let spec = SoapSpec::default();
    let here = packing_mean_nu3(x.view(), spec, None, None);
    assert!(push_away_means(x.view(), &[here.to_vec()], spec, 0.01).is_none());
    assert!(push_away_clouds(x.view(), &[x.to_vec()], spec, 0.01).is_none());
}

#[test]
fn a_nonfinite_reference_cannot_supply_a_repulsive_proposal() {
    let x = structure();
    let spec = SoapSpec::default();
    let here = packing_mean_nu3(x.view(), spec, None, None);
    assert!(push_away_means(x.view(), &[vec![f64::NAN; here.len()]], spec, 0.01).is_none());
    assert!(push_away_means(x.view(), &[vec![f64::INFINITY; here.len()]], spec, 0.01).is_none());
}

#[test]
fn coordinate_and_prepared_mean_entries_use_the_same_separation() {
    let x = structure();
    let peer = &x * 1.03;
    let spec = SoapSpec::default();
    let held = packing_mean_nu3(peer.view(), spec, None, None);
    let prepared = push_away_means(x.view(), &[held.to_vec()], spec, 1e-4);
    let coordinates = push_away_clouds(x.view(), &[peer.to_vec()], spec, 1e-4);
    assert_eq!(prepared, coordinates);
    assert!(prepared.is_some());
}

#[test]
fn absent_and_incompatible_references_have_no_direction() {
    let x = structure();
    let spec = SoapSpec::default();
    assert!(push_away_means(x.view(), &[], spec, 0.01).is_none());
    assert!(push_away_means(x.view(), &[vec![0.0]], spec, 0.01).is_none());
    assert!(push_away_clouds(x.view(), &[vec![0.0]], spec, 0.01).is_none());
}
