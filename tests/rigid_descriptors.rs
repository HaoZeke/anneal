use anneal_core::bias::Fingerprint;
use anneal_core::methods::cluster_hopping::{ClusterFingerprint, Config};
use anneal_core::potentials::Tip4pCluster;
use ndarray::Array1;

fn dimer() -> Array1<f64> {
    Array1::from(vec![0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
}

#[test]
fn basin_descriptor_distinguishes_orientation_dependent_water_energies() {
    let x = dimer();
    let mut turned = x.clone();
    turned[10] = 1.2;
    let pot = Tip4pCluster::new(2);
    assert!((pot.value_and_gradient(x.view()).0 - pot.value_and_gradient(turned.view()).0).abs() > 1.0);
    let fingerprint = ClusterFingerprint::of_config(&Config::for_tip4p(2), &x);
    let difference = &fingerprint.describe(x.view()) - &fingerprint.describe(turned.view());
    assert!(difference.dot(&difference) > 0.01, "a rigid rotation with a different energy must not alias the same basin descriptor");
}

#[test]
fn rigid_basin_descriptor_respects_global_motion_and_molecule_permutation() {
    let mut x = dimer();
    x[8] = 0.4;
    let fingerprint = ClusterFingerprint::of_config(&Config::for_tip4p(2), &x);
    let original = fingerprint.describe(x.view());
    let mut moved = x.clone();
    let angle = 0.7_f64;
    for molecule in 0..2 {
        moved[3 * molecule] = angle.cos() * x[3 * molecule] - angle.sin() * x[3 * molecule + 1] + 5.0;
        moved[3 * molecule + 1] = angle.sin() * x[3 * molecule] + angle.cos() * x[3 * molecule + 1] - 2.0;
        moved[3 * molecule + 2] += 4.0;
        moved[6 + 3 * molecule + 2] += angle;
    }
    let delta = fingerprint.describe(moved.view()) - &original;
    assert!(delta.dot(&delta) < 1e-20);
    for offset in [0, 6] {
        for axis in 0..3 { moved.swap(offset + axis, offset + 3 + axis); }
    }
    let delta = fingerprint.describe(moved.view()) - original;
    assert!(delta.dot(&delta) < 1e-20);
}

#[test]
fn equivalent_rigid_rotation_charts_have_the_same_descriptor() {
    let mut x = dimer();
    x[10] = 0.7;
    let fingerprint = ClusterFingerprint::of_config(&Config::for_tip4p(2), &x);
    let original = fingerprint.describe(x.view());
    x[10] += std::f64::consts::TAU;
    let delta = fingerprint.describe(x.view()) - original;
    assert!(delta.dot(&delta) < 1e-20);
}

#[test]
fn exchanging_equivalent_hydrogen_sites_does_not_change_the_basin_descriptor() {
    let mut x = dimer();
    let fingerprint = ClusterFingerprint::of_config(&Config::for_tip4p(2), &x);
    let original = fingerprint.describe(x.view());
    x[8] = std::f64::consts::PI;
    let delta = fingerprint.describe(x.view()) - original;
    assert!(delta.dot(&delta) < 1e-20);
}
