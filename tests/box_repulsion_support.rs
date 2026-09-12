use anneal_core::shared_bias;
use ndarray::array;
use rand::{Rng, SeedableRng, rngs::StdRng};

#[path = "../src/methods/box_hopping/repulsion.rs"]
mod repulsion;
use repulsion::{PeerSamples, Separation};

#[test]
fn immobile_nonzero_coordinates_retain_their_position_and_step_cap() {
    let point = array![0.4, 0.5];
    let anchor = array![0.4, 0.4];
    let mut peers = PeerSamples::new(2);
    peers.receive(1, point.clone());
    let mut rng = StdRng::seed_from_u64(7);
    let mut untouched = rng.clone();
    let (separation, overlap) = peers
        .separate(
            point.view(),
            anchor.view(),
            array![0.0, 1.0].view(),
            1.0,
            0.25,
            1.0,
            &mut rng,
        )
        .unwrap();
    assert!(overlap);
    let Separation::Moved(moved) = separation else {
        panic!("the active coordinate admits separation");
    };
    assert_eq!(moved[0], point[0]);
    assert_eq!(moved[1], 0.75);
    assert!(peers.clearance(moved.view()).unwrap() > peers.clearance(point.view()).unwrap());
    let displacement = &moved - &point;
    assert!(displacement.dot(&displacement) <= 0.25_f64.powi(2));
    assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
}

#[test]
fn an_empty_move_support_cannot_displace_a_nonzero_point() {
    let point = array![0.4, 0.5];
    let mut peers = PeerSamples::new(2);
    peers.receive(1, point.clone());
    let mut rng = StdRng::seed_from_u64(7);
    assert!(matches!(
        peers.separate(
            point.view(),
            array![0.4, 0.4].view(),
            array![0.0, 0.0].view(),
            1.0,
            0.25,
            1.0,
            &mut rng,
        ),
        Some((Separation::Constrained, true))
    ));
}
