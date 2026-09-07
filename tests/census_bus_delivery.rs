#![cfg(feature = "bank-rpc")]

use std::net::{Ipv4Addr, TcpListener};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anneal_core::census_bus::{CensusBus, PeerMinimum};
use nng::options::Options;
use nng::options::protocol::pubsub::Subscribe;
use nng::{PipeEvent, Protocol, Socket};

const COORDINATES: [f64; 6] = [0.0, 0.0, 0.0, 1.2, 0.0, 0.0];
const ENERGY: f64 = -1.0;
static SOCKET_SETUP: Mutex<()> = Mutex::new(());

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(3)
}

fn wait_until(deadline: Instant, description: &str, mut condition: impl FnMut() -> bool) {
    loop {
        if condition() {
            return;
        }
        assert!(Instant::now() < deadline, "timed out: {description}");
        thread::sleep(Duration::from_millis(5));
    }
}

fn adjacent_ports(deadline: Instant) -> (u16, TcpListener, TcpListener) {
    assert_ne!(
        std::env::var("CENSUS_BUS_IPC").ok().as_deref(),
        Some("1"),
        "these localhost delivery tests require the default TCP transport"
    );
    loop {
        assert!(
            Instant::now() < deadline,
            "reserving adjacent localhost ports"
        );
        let first = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let base = first.local_addr().unwrap().port();
        if let Some(next) = base.checked_add(1)
            && let Ok(second) = TcpListener::bind((Ipv4Addr::LOCALHOST, next))
        {
            return (base, first, second);
        }
    }
}

fn raw_frame(replica: u32, hops: u64, coordinates: &[f64]) -> Vec<u8> {
    let mut bytes = format!("census/{replica:03}\n").into_bytes();
    bytes.extend_from_slice(&replica.to_le_bytes());
    bytes.extend_from_slice(&hops.to_le_bytes());
    bytes.extend_from_slice(&ENERGY.to_le_bytes());
    bytes.extend_from_slice(&u32::try_from(coordinates.len()).unwrap().to_le_bytes());
    for value in coordinates {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn raw_publisher_and_receiver(deadline: Instant) -> (Socket, CensusBus) {
    let _setup = SOCKET_SETUP.lock().unwrap();
    let (base, receiver_port, publisher_port) = adjacent_ports(deadline);
    let publisher = Socket::new(Protocol::Pub0).unwrap();
    drop(publisher_port);
    publisher
        .listen(&format!("tcp://127.0.0.1:{}", u32::from(base) + 1))
        .unwrap();
    drop(receiver_port);
    let receiver = CensusBus::new(0, base, 2).unwrap();
    (publisher, receiver)
}

fn deliver_to_latest(
    publisher: &Socket,
    receiver: &mut CensusBus,
    hops: u64,
    coordinates: &[f64],
    deadline: Instant,
) -> Vec<PeerMinimum> {
    let frame = raw_frame(1, hops, coordinates);
    let mut notifications = Vec::new();
    wait_until(
        deadline,
        "raw publication must reach the retained census",
        || {
            match publisher.try_send(frame.as_slice()) {
                Ok(()) | Err((_, nng::Error::TryAgain)) => {}
                Err((_, error)) => panic!("raw publisher failed: {error}"),
            }
            notifications.extend(receiver.poll());
            receiver.peers().any(|peer| {
                peer.replica == 1
                    && peer.hops == hops
                    && peer.energy == ENERGY
                    && peer.coordinates == coordinates
            })
        },
    );
    notifications
}

#[test]
fn equal_energy_changed_geometry_is_fresh_after_actual_delivery() {
    let deadline = deadline();
    let (publisher, mut receiver) = raw_publisher_and_receiver(deadline);
    let initial = deliver_to_latest(&publisher, &mut receiver, 7, &COORDINATES, deadline);
    assert_eq!(initial.len(), 1);
    assert_eq!(initial[0].coordinates, COORDINATES);

    let changed = [0.0, 0.0, 0.0, 0.0, 1.3, 0.0];
    let fresh = deliver_to_latest(&publisher, &mut receiver, 8, &changed, deadline);
    assert_eq!(receiver.peer_count(), 1);
    assert_eq!(
        fresh.len(),
        1,
        "a delivered equal-energy, equal-length geometry change must notify its consumer"
    );
    assert_eq!(fresh[0].replica, 1);
    assert_eq!(fresh[0].hops, 8);
    assert_eq!(fresh[0].energy, ENERGY);
    assert_eq!(fresh[0].coordinates, changed);
}

#[test]
fn equal_energy_changed_geometry_is_published_without_waiting_for_refresh() {
    let deadline = deadline();
    let (mut publisher, base, _unused_peer_port) = {
        let _setup = SOCKET_SETUP.lock().unwrap();
        let (base, publisher_port, unused_peer_port) = adjacent_ports(deadline);
        drop(publisher_port);
        (CensusBus::new(0, base, 2).unwrap(), base, unused_peer_port)
    };
    let subscriber = Socket::new(Protocol::Sub0).unwrap();
    subscriber
        .set_opt::<Subscribe>(b"census/".to_vec())
        .unwrap();
    let connected = Arc::new(AtomicBool::new(false));
    let connection = Arc::clone(&connected);
    subscriber
        .pipe_notify(move |_, event| {
            if matches!(event, PipeEvent::AddPost) {
                connection.store(true, Ordering::SeqCst);
            }
        })
        .unwrap();
    subscriber
        .dial_async(&format!("tcp://127.0.0.1:{base}"))
        .unwrap();
    wait_until(deadline, "subscriber must establish its pipe", || {
        connected.load(Ordering::SeqCst)
    });

    publisher.publish(7, ENERGY, &COORDINATES);
    wait_until(
        deadline,
        "subscriber must receive the initial state",
        || match subscriber.try_recv() {
            Ok(message) => {
                let bytes: &[u8] = &message;
                assert_eq!(bytes, raw_frame(0, 7, &COORDINATES));
                true
            }
            Err(nng::Error::TryAgain) => false,
            Err(error) => panic!("subscriber failed: {error}"),
        },
    );
    assert!(matches!(subscriber.try_recv(), Err(nng::Error::TryAgain)));

    let changed = [0.0, 0.0, 0.0, 0.0, 1.3, 0.0];
    publisher.publish(8, ENERGY, &changed);
    wait_until(
        deadline,
        "a single equal-energy geometry change must publish without refresh calls",
        || match subscriber.try_recv() {
            Ok(message) => {
                let bytes: &[u8] = &message;
                assert_eq!(bytes, raw_frame(0, 8, &changed));
                true
            }
            Err(nng::Error::TryAgain) => false,
            Err(error) => panic!("subscriber failed: {error}"),
        },
    );
}

#[test]
fn unchanged_publication_refreshes_a_subscriber_that_joins_late() {
    let deadline = deadline();
    let (mut publisher, base, _unused_peer_port) = {
        let _setup = SOCKET_SETUP.lock().unwrap();
        let (base, publisher_port, unused_peer_port) = adjacent_ports(deadline);
        drop(publisher_port);
        (CensusBus::new(0, base, 2).unwrap(), base, unused_peer_port)
    };
    publisher.publish(7, ENERGY, &COORDINATES);

    let subscriber = Socket::new(Protocol::Sub0).unwrap();
    subscriber
        .set_opt::<Subscribe>(b"census/".to_vec())
        .unwrap();
    let connected = Arc::new(AtomicBool::new(false));
    let connection = Arc::clone(&connected);
    subscriber
        .pipe_notify(move |_, event| {
            if matches!(event, PipeEvent::AddPost) {
                connection.store(true, Ordering::SeqCst);
            }
        })
        .unwrap();
    subscriber
        .dial_async(&format!("tcp://127.0.0.1:{base}"))
        .unwrap();
    wait_until(deadline, "late subscriber must establish its pipe", || {
        connected.load(Ordering::SeqCst)
    });

    let mut hops = 7_u64;
    let mut received = None;
    wait_until(
        deadline,
        "unchanged state must refresh a connected late subscriber",
        || {
            hops += 1;
            publisher.publish(hops, ENERGY, &COORDINATES);
            match subscriber.try_recv() {
                Ok(message) => {
                    let bytes: &[u8] = &message;
                    let topic_length = b"census/000\n".len();
                    assert_eq!(bytes.len(), raw_frame(0, 7, &COORDINATES).len());
                    let delivered_hops = u64::from_le_bytes(
                        bytes[topic_length + 4..topic_length + 12]
                            .try_into()
                            .unwrap(),
                    );
                    assert_eq!(bytes, raw_frame(0, delivered_hops, &COORDINATES));
                    if delivered_hops <= 7 {
                        return false;
                    }
                    received = Some(message);
                    true
                }
                Err(nng::Error::TryAgain) => false,
                Err(error) => panic!("late subscriber failed: {error}"),
            }
        },
    );
    let received = received.unwrap();
    let bytes: &[u8] = &received;
    let topic_length = b"census/000\n".len();
    assert_eq!(bytes.len(), raw_frame(0, 7, &COORDINATES).len());
    let received_hops = u64::from_le_bytes(
        bytes[topic_length + 4..topic_length + 12]
            .try_into()
            .unwrap(),
    );
    assert!(
        received_hops > 7 && received_hops <= hops,
        "the late subscriber must receive a repeated publication"
    );
    assert_eq!(bytes, raw_frame(0, received_hops, &COORDINATES));
}

#[test]
fn repeated_geometry_updates_latest_hops_without_duplicate_fresh_events() {
    let deadline = deadline();
    let (publisher, mut receiver) = raw_publisher_and_receiver(deadline);
    let initial = deliver_to_latest(&publisher, &mut receiver, 7, &COORDINATES, deadline);
    assert_eq!(initial.len(), 1);

    for hops in 8..=10 {
        let fresh = deliver_to_latest(&publisher, &mut receiver, hops, &COORDINATES, deadline);
        assert!(
            fresh.is_empty(),
            "a delivered unchanged geometry at hop {hops} must not duplicate fresh evidence"
        );
        let latest = receiver.peers().next().unwrap();
        assert_eq!(latest.hops, hops);
        assert_eq!(latest.coordinates, COORDINATES);
        assert_eq!(receiver.peer_count(), 1);
    }
}
