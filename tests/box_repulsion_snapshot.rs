//! Frozen proposal geometry has no exchange ownership or callback dependency.

use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use anneal_core::methods::box_hopping::BoxEnsembleConfig;
use anneal_core::{bias, descriptor_space, pes_exploration, shared_bias};
use eindir_core::Bounds;
use ndarray::{Array1, array};
use rand::{Rng, SeedableRng, rngs::StdRng};

#[path = "../src/methods/minima_hopping.rs"]
pub mod minima_hopping_source;
mod methods {
    pub use crate::minima_hopping_source as minima_hopping;
    pub use anneal_core::methods::{cluster_hopping, ensemble};
}

#[path = "../src/methods/box_hopping/coverage.rs"]
mod coverage;
#[path = "../src/methods/box_hopping/repulsion.rs"]
mod repulsion;
use coverage::{BoxCoverageConfig, Coverage};

fn cloud() -> Coverage {
    let bounds = Bounds::new(array![0.0, 7.0, 0.0], array![1.0, 7.0, 1.0], 0.0);
    let mut coverage = Coverage::new(
        &bounds,
        3,
        &BoxCoverageConfig {
            radius: 0.35,
            ..BoxCoverageConfig::default()
        },
        1,
    );
    coverage.sample(1, array![0.4, 7.0, 0.5].view(), 1.0);
    coverage.sample(2, array![0.7, 7.0, 0.6].view(), 2.0);
    coverage.hear(0, 1.0);
    coverage
}

#[test]
fn detached_repulsion_matches_geometry_random_stream_and_counters() {
    let mut reference = cloud();
    let mut detached = cloud();
    let mut snapshot = detached.repulsion_snapshot(0);
    let mut rng = StdRng::seed_from_u64(17);
    let mut snapshot_rng = rng.clone();
    for position in [
        array![0.4, 7.0, 0.5],
        array![0.0, 7.0, 0.5],
        array![0.5, 7.0, 0.5],
        array![1.0, 7.0, 0.0],
        array![0.7, 7.0, 0.6],
    ] {
        let mut expected = position.clone();
        let mut actual = position.clone();
        reference.repel(0, position.view(), &mut expected, &mut rng);
        snapshot.repel(position.view(), &mut actual, &mut snapshot_rng);
        assert_eq!(actual, expected);
        assert_eq!(actual[1], 7.0);
        assert_eq!(rng.random::<u64>(), snapshot_rng.random::<u64>());
    }
    detached.record_repulsion(snapshot.take_stats());
    let expected = reference.finish().0;
    let actual = detached.finish().0;
    assert!(actual.repelled_proposals > 0);
    assert_eq!(actual, expected);
    assert_eq!(snapshot.take_stats(), Default::default());
}

#[test]
fn frozen_repulsion_does_not_observe_uncheckpointed_peer_delivery() {
    let mut coverage = cloud();
    let mut snapshot = coverage.repulsion_snapshot(0);
    let position = array![0.0, 7.0, 0.0];
    coverage.sample(1, position.view(), 0.0);
    coverage.hear(0, 1.0);
    let mut expected = position.clone();
    let mut actual = position.clone();
    let mut rng = StdRng::seed_from_u64(17);
    let mut snapshot_rng = rng.clone();
    coverage.repel(0, position.view(), &mut expected, &mut rng);
    snapshot.repel(position.view(), &mut actual, &mut snapshot_rng);
    assert_eq!(actual, position);
    assert_ne!(expected, position);
    assert_eq!(snapshot.take_stats().repelled_proposals, 0);
    assert!(coverage.finish().0.repelled_proposals > 0);
}

#[test]
fn frozen_geometry_runs_while_the_exchange_is_locked() {
    let coverage = Arc::new(Mutex::new(cloud()));
    let mut snapshot = coverage.lock().unwrap().repulsion_snapshot(0);
    let held = coverage.lock().unwrap();
    let (send, receive) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let position = array![0.4, 7.0, 0.5];
        let mut proposal = position.clone();
        let mut rng = StdRng::seed_from_u64(17);
        snapshot.repel(position.view(), &mut proposal, &mut rng);
        send.send((proposal, snapshot.take_stats())).unwrap();
    });
    let result = receive.recv_timeout(Duration::from_secs(10));
    drop(held);
    worker.join().unwrap();
    let (proposal, stats): (Array1<f64>, _) = result.expect("geometry needs no exchange lock");
    assert_ne!(proposal, array![0.4, 7.0, 0.5]);
    assert_eq!(stats.repelled_proposals, 1);
}
