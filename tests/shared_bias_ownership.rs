use anneal_core::bias::{BasinBias, Bias, Fingerprint};
use anneal_core::shared_bias::{SharedDeposits, visit_deltas};
use ndarray::{Array1, ArrayView1, array};

struct Coordinates;

impl Fingerprint for Coordinates {
    fn describe(&self, x: ArrayView1<f64>) -> Array1<f64> {
        x.to_owned()
    }
}

fn bias() -> BasinBias<Coordinates> {
    let mut bias = BasinBias::new(Coordinates, 0.05, 0.1, 5.0);
    bias.entropic = false;
    bias
}

fn publishable(bias: &BasinBias<Coordinates>, seen: &mut Vec<u64>) -> Vec<(Array1<f64>, u64)> {
    visit_deltas(
        |i| bias.index().centre(i),
        |i| bias.index().visits(i),
        bias.n_basins(),
        seen,
    )
}

#[test]
fn imported_visits_raise_repulsion_without_recording_local_arrivals() {
    let point = array![0.37, -1.29];
    let mut receiver = bias();
    let mut reference = bias();
    receiver.deposit_scaled_n(point.view(), 0.8, 0.25, 3);
    for _ in 0..3 {
        reference.deposit_scaled(point.view(), 0.8, 0.25);
    }
    assert_eq!(
        receiver.potential(point.view()).to_bits(),
        reference.potential(point.view()).to_bits()
    );
    assert!(receiver.potential(point.view()) > 0.0);
    assert_eq!(
        receiver.index().visits(0),
        0,
        "importing a descriptor is not a local search arrival"
    );
    assert_eq!(reference.index().visits(0), 3);
}

#[test]
fn repeated_exchange_cannot_manufacture_search_effort() {
    let mut walkers = [bias(), bias(), bias()];
    let mut seen = [Vec::new(), Vec::new(), Vec::new()];
    let mut exchange = SharedDeposits::new(walkers.len());
    let points = [array![0.2, 0.4], array![1.2, 0.7], array![-1.2, 1.7]];
    for (walker, point) in walkers.iter_mut().zip(&points) {
        walker.deposit(point.view(), 0.8);
    }
    for round in 0..4 {
        for (index, walker) in walkers.iter().enumerate() {
            exchange.publish(index, publishable(walker, &mut seen[index]));
        }
        for (index, walker) in walkers.iter_mut().enumerate() {
            for (centre, count) in exchange.drain(index) {
                walker.deposit_scaled_n(centre.view(), 0.8, 0.25, count);
            }
        }
        assert_eq!(
            exchange.counts(),
            (3, 6),
            "round {round}: three actual arrivals reach two peers each"
        );
        assert_eq!(exchange.retained(), 0);
    }
    for walker in &walkers {
        assert_eq!(walker.n_basins(), 3);
        for point in &points {
            assert!(walker.potential(point.view()) > 0.0);
        }
    }
}

#[test]
fn a_local_arrival_in_an_imported_region_is_published_once() {
    let point = array![0.37, -1.29];
    let mut receiver = bias();
    let mut seen = Vec::new();
    receiver.deposit_scaled_n(point.view(), 0.8, 0.25, 7);
    assert!(publishable(&receiver, &mut seen).is_empty());
    receiver.deposit(point.view(), 0.8);
    assert_eq!(publishable(&receiver, &mut seen), vec![(point, 1)]);
    assert!(publishable(&receiver, &mut seen).is_empty());
}
