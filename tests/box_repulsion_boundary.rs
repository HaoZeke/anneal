//! Test the production geometric correction independently of objective callbacks.

use anneal_core::shared_bias;
use ndarray::{Array1, array};
use rand::{Rng, SeedableRng, rngs::StdRng};

#[path = "../src/methods/box_hopping/repulsion.rs"]
mod repulsion;
use repulsion::{PeerSamples, Separation};

fn face_separation(fixed_axis: bool) {
    let (point, peer, widths) = if fixed_axis {
        (array![0.0, 0.0, 0.5], array![0.0, 0.1, 0.5], array![0.0, 1.0, 1.0])
    } else {
        (array![0.0, 0.5], array![0.1, 0.5], array![1.0, 1.0])
    };
    let mut samples = PeerSamples::new(2);
    samples.receive(1, peer.clone());
    let mut rng = StdRng::seed_from_u64(7);
    let mut untouched = rng.clone();
    let (separation, overlaps) = samples.separate(
        point.view(), point.view(), widths.view(), 1.0, 0.2, 1.0, &mut rng,
    ).expect("a foreign sample is present");
    assert!(overlaps);
    let Separation::Moved(moved) = separation else {
        panic!("clipped radial motion must not hide a feasible sideways separation");
    };
    assert!(moved.iter().all(|&x| (0.0..=1.0).contains(&x)));
    assert_eq!(moved[0], 0.0);
    if fixed_axis { assert_eq!(moved[1], 0.0); }
    let initial = &point - &peer;
    let final_gap = &moved - &peer;
    assert!(final_gap.dot(&final_gap) > initial.dot(&initial));
    let displacement = &moved - &point;
    assert!(displacement.dot(&displacement) <= 0.1_f64.powi(2));
    assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
}

#[test]
fn a_box_face_allows_sideways_separation() { face_separation(false); }

#[test]
fn sideways_separation_preserves_fixed_coordinates() { face_separation(true); }

#[test]
fn non_improving_feasible_endpoints_remain_constrained() {
    let mut samples = PeerSamples::new(2);
    samples.receive(1, array![0.9]);
    let point = array![0.0];
    let widths = array![1.0];
    let mut rng = StdRng::seed_from_u64(7);
    assert!(matches!(samples.separate(
        point.view(), point.view(), widths.view(), 1.0, 2.0, 1.0, &mut rng,
    ), Some((Separation::Constrained, true))));
}

#[test]
fn an_admissible_radial_correction_keeps_its_direction_and_random_state() {
    let mut samples = PeerSamples::new(2);
    samples.receive(1, array![0.4, 0.5]);
    let point = array![0.5, 0.5];
    let mut rng = StdRng::seed_from_u64(7);
    let mut untouched = rng.clone();
    let (separation, overlaps) = samples.separate(
        point.view(), point.view(), Array1::ones(2).view(), 1.0, 0.2, 1.0, &mut rng,
    ).unwrap();
    assert!(overlaps);
    let Separation::Moved(moved) = separation else { panic!("radial separation is feasible"); };
    assert!(moved[0] > point[0]);
    assert_eq!(moved[1], point[1]);
    assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
}
