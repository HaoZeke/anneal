use anneal_core::shared_bias::SharedDeposits;
use ndarray::array;

#[test]
fn a_retired_reader_does_not_pin_delivered_visits() {
    let mut exchange = SharedDeposits::new(3);
    exchange.retire_reader(0);
    for step in 0..128 {
        let deposit = (array![step as f64], 1);
        exchange.publish(1, vec![deposit.clone()]);
        assert!(exchange.drain(1).is_empty());
        assert_eq!(exchange.drain(2), vec![deposit]);
        assert_eq!(exchange.retained(), 0);
    }
    assert_eq!(exchange.counts(), (128, 128));
}

#[test]
fn a_slow_live_reader_keeps_every_pending_visit() {
    let mut exchange = SharedDeposits::new(3);
    for step in 0..128 {
        exchange.publish(1, vec![(array![step as f64], 1)]);
        assert!(exchange.drain(1).is_empty());
        assert_eq!(exchange.drain(2), vec![(array![step as f64], 1)]);
        assert_eq!(exchange.retained(), step + 1);
    }
    let expected: Vec<_> = (0..128).map(|step| (array![step as f64], 1)).collect();
    assert_eq!(exchange.drain(0), expected);
    assert_eq!(exchange.retained(), 0);
    assert_eq!(exchange.counts(), (128, 256));
}

#[test]
fn retiring_one_reader_preserves_pending_delivery_to_live_readers() {
    let mut exchange = SharedDeposits::new(3);
    exchange.publish(0, vec![(array![1.0], 5)]);
    assert_eq!(exchange.drain(1), vec![(array![1.0], 5)]);
    exchange.retire_reader(0);
    assert_eq!(exchange.retained(), 1);
    exchange.publish(1, vec![(array![2.0], 7)]);
    assert!(exchange.drain(1).is_empty());
    assert_eq!(exchange.retained(), 2);
    assert_eq!(exchange.drain(2), vec![(array![1.0], 5), (array![2.0], 7)]);
    assert!(exchange.drain(2).is_empty());
    assert_eq!(exchange.retained(), 0);
    assert_eq!(exchange.counts(), (12, 17));
}

#[test]
fn retirement_keeps_the_producers_samples_for_live_receivers() {
    let mut exchange = SharedDeposits::new(3);
    assert!(exchange.publish_sample(0, array![3.0]));
    exchange.publish(0, vec![(array![4.0], 2)]);
    exchange.retire_reader(0);
    exchange.retire_reader(0);
    for reader in [1, 2] {
        assert_eq!(exchange.drain_samples(reader), vec![(0, array![3.0])]);
        assert_eq!(exchange.drain(reader), vec![(array![4.0], 2)]);
        assert!(exchange.drain_samples(reader).is_empty());
        assert!(exchange.drain(reader).is_empty());
    }
    assert_eq!(exchange.sample_counts(), (1, 2));
    assert_eq!(exchange.counts(), (2, 4));
    assert_eq!(exchange.retained(), 0);
}

#[test]
fn retired_readers_do_not_receive_or_reactivate() {
    let mut exchange = SharedDeposits::new(2);
    exchange.retire_reader(1);
    exchange.publish(0, vec![(array![1.0], 3)]);
    assert!(exchange.publish_sample(0, array![2.0]));
    assert!(exchange.drain(1).is_empty());
    assert!(exchange.drain_samples(1).is_empty());
    assert_eq!(exchange.counts(), (3, 0));
    assert_eq!(exchange.sample_counts(), (1, 0));
    exchange.retire_reader(0);
    assert_eq!(exchange.retained(), 0);
    exchange.retire_reader(0);
    exchange.publish(0, vec![(array![3.0], 4)]);
    assert!(exchange.drain(0).is_empty());
    assert!(exchange.drain(1).is_empty());
    assert_eq!(exchange.retained(), 0);
    assert_eq!(exchange.counts(), (7, 0));
}
