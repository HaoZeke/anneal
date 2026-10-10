use anneal_core::shared_bias::{SAMPLE_WINDOW, SharedDeposits};
use ndarray::array;

#[test]
fn samples_retain_their_source_without_creating_visit_deltas() {
    let mut exchange = SharedDeposits::new(3);
    assert!(exchange.publish_sample(0, array![0.25, 0.5]));
    assert!(!exchange.publish_sample(0, array![0.25, 0.5]));
    exchange.publish(0, vec![(array![0.0, 0.0], 1)]);
    assert!(exchange.drain_samples(0).is_empty());
    assert_eq!(exchange.drain_samples(1), vec![(0, array![0.25, 0.5])]);
    assert!(exchange.drain_samples(1).is_empty());
    assert_eq!(exchange.drain(1), vec![(array![0.0, 0.0], 1)]);
    assert!(exchange.drain(1).is_empty());
    assert_eq!(exchange.counts(), (1, 1));
    assert_eq!(exchange.drain_samples(2), vec![(0, array![0.25, 0.5])]);
    assert_eq!(exchange.counts(), (1, 1));
}

#[test]
fn silent_receivers_do_not_make_the_sample_exchange_unbounded() {
    let mut exchange = SharedDeposits::new(3);
    exchange.publish_sample(1, array![-0.5]);
    for sample in 0..(SAMPLE_WINDOW * 4) {
        exchange.publish_sample(0, array![sample as f64]);
    }
    let received = exchange.drain_samples(2);
    assert_eq!(received.len(), SAMPLE_WINDOW + 1);
    assert_eq!(received[0], (0, array![(SAMPLE_WINDOW * 3) as f64]));
    assert_eq!(
        received[SAMPLE_WINDOW - 1],
        (0, array![(SAMPLE_WINDOW * 4 - 1) as f64])
    );
    assert_eq!(received[SAMPLE_WINDOW], (1, array![-0.5]));
    assert!(exchange.drain_samples(2).is_empty());
    assert_eq!(exchange.counts(), (0, 0));
}

#[test]
fn consuming_visits_does_not_consume_or_republish_samples() {
    let mut exchange = SharedDeposits::new(2);
    exchange.publish_sample(0, array![0.2]);
    assert!(exchange.drain(1).is_empty());
    assert_eq!(exchange.drain_samples(1), vec![(0, array![0.2])]);
    assert!(exchange.drain_samples(0).is_empty());
    assert_eq!(exchange.sample_counts(), (1, 1));
}
