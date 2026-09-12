use std::collections::BTreeSet;

use anneal_core::shared_bias::{SAMPLE_WINDOW, SharedDeposits};
use ndarray::array;

fn expected_sources(walkers: usize, reader: usize, neighbors: usize) -> Vec<usize> {
    let steps = if neighbors == 0 { walkers - 1 } else { neighbors.min(walkers - 1) };
    let mut sources = BTreeSet::new();
    for step in 1..=steps {
        sources.insert((reader + step) % walkers);
        sources.insert((reader + walkers - step) % walkers);
    }
    sources.remove(&reader);
    sources.into_iter().collect()
}

#[test]
fn both_streams_deliver_exactly_the_direct_ring_neighbors_once() {
    for walkers in 1..=8 {
        for neighbors in [0, 1, 2, usize::MAX] {
            let mut exchange = SharedDeposits::with_neighbors(walkers, neighbors);
            for source in 0..walkers {
                assert!(exchange.publish_sample(source, array![source as f64]));
                exchange.publish(source, vec![(array![source as f64], source as u64 + 1)]);
            }
            let mut delivered_samples = 0;
            let mut delivered_visits = 0;
            for reader in 0..walkers {
                let sources = expected_sources(walkers, reader, neighbors);
                let samples: Vec<_> = sources.iter().map(|&source| (source, array![source as f64])).collect();
                let visits: Vec<_> = sources.iter().map(|&source| (array![source as f64], source as u64 + 1)).collect();
                assert_eq!(exchange.drain_samples(reader), samples, "walkers={walkers}, reader={reader}, neighbors={neighbors}");
                assert_eq!(exchange.drain(reader), visits);
                assert!(exchange.drain_samples(reader).is_empty());
                assert!(exchange.drain(reader).is_empty());
                delivered_samples += sources.len() as u64;
                delivered_visits += visits.iter().map(|(_, count)| count).sum::<u64>();
            }
            assert_eq!(exchange.sample_counts(), (walkers as u64, delivered_samples));
            assert_eq!(exchange.counts(), (((walkers * (walkers + 1)) / 2) as u64, delivered_visits));
            assert_eq!(exchange.retained(), 0);
        }
    }
}

#[test]
fn retirement_keeps_ring_identity_and_neighbor_backlog() {
    let mut exchange = SharedDeposits::with_neighbors(5, 1);
    exchange.publish_sample(0, array![0.25]);
    exchange.publish(0, vec![(array![0.5], 7)]);
    exchange.retire_reader(0);
    exchange.retire_reader(0);
    for reader in [2, 3, 0] {
        assert!(exchange.drain_samples(reader).is_empty());
        assert!(exchange.drain(reader).is_empty());
    }
    assert_eq!(exchange.retained(), 1);
    for reader in [1, 4] {
        assert_eq!(exchange.drain_samples(reader), vec![(0, array![0.25])]);
        assert_eq!(exchange.drain(reader), vec![(array![0.5], 7)]);
    }
    assert_eq!(exchange.retained(), 0);
    assert_eq!(exchange.sample_counts(), (1, 2));
    assert_eq!(exchange.counts(), (7, 14));
}

#[test]
fn excluded_events_advance_cursors_without_delivery_or_forwarding() {
    let mut exchange = SharedDeposits::with_neighbors(5, 1);
    for step in 0..(2 * SAMPLE_WINDOW) {
        exchange.publish_sample(2, array![step as f64]);
        exchange.publish(2, vec![(array![step as f64], 1)]);
        assert!(exchange.drain_samples(0).is_empty());
        assert!(exchange.drain(0).is_empty());
    }
    assert_eq!(exchange.sample_counts(), ((2 * SAMPLE_WINDOW) as u64, 0));
    assert_eq!(exchange.counts(), ((2 * SAMPLE_WINDOW) as u64, 0));
    assert_eq!(exchange.drain_samples(1).len(), SAMPLE_WINDOW);
    assert_eq!(exchange.drain(1).len(), 2 * SAMPLE_WINDOW);
    assert!(exchange.drain_samples(0).is_empty());
    assert!(exchange.drain(0).is_empty());
    exchange.publish_sample(4, array![-1.0]);
    exchange.publish(4, vec![(array![-2.0], 3)]);
    assert_eq!(exchange.drain_samples(0), vec![(4, array![-1.0])]);
    assert_eq!(exchange.drain(0), vec![(array![-2.0], 3)]);
    assert!(exchange.drain_samples(0).is_empty());
    assert!(exchange.drain(0).is_empty());
    for reader in 0..5 { exchange.retire_reader(reader); }
    assert_eq!(exchange.retained(), 0);
}

#[test]
fn explicit_zero_preserves_the_default_exchange() {
    let mut default = SharedDeposits::new(5);
    let mut explicit = SharedDeposits::with_neighbors(5, 0);
    for step in 0..16 {
        let source = step % 5;
        for exchange in [&mut default, &mut explicit] {
            exchange.publish_sample(source, array![step as f64]);
            exchange.publish(source, vec![(array![step as f64], step as u64 + 1)]);
        }
        for reader in (0..5).rev() {
            assert_eq!(default.drain_samples(reader), explicit.drain_samples(reader));
            assert_eq!(default.drain(reader), explicit.drain(reader));
        }
        assert_eq!(default.counts(), explicit.counts());
        assert_eq!(default.sample_counts(), explicit.sample_counts());
        assert_eq!(default.retained(), explicit.retained());
    }
}
