#![cfg(feature = "bank-rpc")]

use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::net::{Ipv4Addr, TcpListener};
use std::path::PathBuf;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use anneal_core::census_bus::CensusBus;

const REPLICAS: u16 = 5;
const CHILD_SCENARIO: &str = "ANNEAL_CENSUS_TOPOLOGY_CHILD";
const TEST_NAME: &str = "census_transports_respect_neighborhoods";

struct EndpointReservation {
    base: u16,
    ports: Vec<Option<TcpListener>>,
    lock_path: PathBuf,
    _lock: File,
}

impl Drop for EndpointReservation {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.lock_path);
    }
}

fn ipc_path(base: u16, replica: u16) -> PathBuf {
    PathBuf::from(format!("/tmp/anneal-census-{base}-{replica:03}"))
}

fn reserve_endpoints(deadline: Instant) -> EndpointReservation {
    loop {
        assert!(Instant::now() < deadline, "reserving census endpoints");
        let first = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let base = first.local_addr().unwrap().port();
        if base.checked_add(REPLICAS - 1).is_none() {
            continue;
        }
        let mut ports = vec![Some(first)];
        for replica in 1..REPLICAS {
            let Ok(port) = TcpListener::bind((Ipv4Addr::LOCALHOST, base + replica)) else {
                break;
            };
            ports.push(Some(port));
        }
        if ports.len() != usize::from(REPLICAS) {
            continue;
        }
        let lock_path = PathBuf::from(format!("/tmp/anneal-census-topology-{base}.lock"));
        let lock = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
        {
            Ok(lock) => lock,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => panic!("reserving census namespace: {error}"),
        };
        let reservation = EndpointReservation {
            base,
            ports,
            lock_path,
            _lock: lock,
        };
        if (0..REPLICAS).all(|replica| !ipc_path(base, replica).try_exists().unwrap()) {
            return reservation;
        }
    }
}

fn coordinates(replica: u32) -> [f64; 6] {
    [0.0, 0.0, 0.0, 1.2 + f64::from(replica) / 10.0, 0.0, 0.0]
}

fn expected_peers(replica: usize, ring: bool) -> BTreeSet<u32> {
    if ring {
        const NEIGHBORS: [[u32; 2]; 5] = [[1, 4], [0, 2], [1, 3], [2, 4], [0, 3]];
        NEIGHBORS[replica].into_iter().collect()
    } else {
        let replica = u32::try_from(replica).unwrap();
        (0..u32::from(REPLICAS))
            .filter(|&peer| peer != replica)
            .collect()
    }
}

fn publish_and_check(buses: &mut [CensusBus], hops: u64, expected: &[BTreeSet<u32>]) -> bool {
    for (replica, bus) in buses.iter_mut().enumerate() {
        let replica = u32::try_from(replica).unwrap();
        bus.publish(hops, -f64::from(replica + 1), &coordinates(replica));
    }
    let mut all_connected = true;
    for (replica, bus) in buses.iter_mut().enumerate() {
        bus.poll();
        let actual: BTreeSet<_> = bus.peers().map(|peer| peer.replica).collect();
        assert!(
            actual.is_subset(&expected[replica]),
            "replica {replica} retained {actual:?}, but only {:?} may reach it",
            expected[replica]
        );
        assert_eq!(bus.peer_count(), actual.len());
        for peer in bus.peers() {
            assert_eq!(peer.energy, -f64::from(peer.replica + 1));
            assert_eq!(peer.coordinates, coordinates(peer.replica));
            assert!(peer.hops <= hops);
        }
        all_connected &= actual == expected[replica];
    }
    all_connected
}

fn exercise_scenario(scenario: &str) {
    let (ipc, ring) = match scenario {
        "tcp-default" => (false, false),
        "tcp-ring" => (false, true),
        "ipc-all" => (true, false),
        "ipc-ring" => (true, true),
        other => panic!("unknown census topology scenario: {other}"),
    };
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut reservation = reserve_endpoints(deadline);
    let mut buses = Vec::new();
    for replica in 0..REPLICAS {
        if !ipc {
            drop(reservation.ports[usize::from(replica)].take());
        }
        buses.push(
            CensusBus::new(u32::from(replica), reservation.base, u32::from(REPLICAS)).unwrap(),
        );
        assert_eq!(
            ipc_path(reservation.base, replica).try_exists().unwrap(),
            ipc,
            "the selected transport must own exactly its expected endpoint"
        );
    }
    let expected: Vec<_> = (0..usize::from(REPLICAS))
        .map(|replica| expected_peers(replica, ring))
        .collect();
    let mut hops = 0;
    loop {
        hops += 1;
        if publish_and_check(&mut buses, hops, &expected) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "{scenario}: every publisher must reach every selected neighbor"
        );
        thread::sleep(Duration::from_millis(5));
    }

    // Every publisher has demonstrated delivery before non-neighbor exclusion
    // is checked across multiple unchanged-state refresh opportunities.
    let refresh_hops = hops + 1;
    for _ in 0..64 {
        hops += 1;
        assert!(publish_and_check(&mut buses, hops, &expected));
        thread::sleep(Duration::from_millis(5));
    }
    loop {
        hops += 1;
        assert!(publish_and_check(&mut buses, hops, &expected));
        if buses
            .iter()
            .all(|bus| bus.peers().all(|peer| peer.hops >= refresh_hops))
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "{scenario}: connected publishers must deliver repeated refreshes"
        );
        thread::sleep(Duration::from_millis(5));
    }
    drop(buses);
    drop(reservation);
}

#[test]
fn census_transports_respect_neighborhoods() {
    if let Ok(scenario) = std::env::var(CHILD_SCENARIO) {
        exercise_scenario(&scenario);
    } else {
        let executable = std::env::current_exe().unwrap();
        for scenario in ["tcp-default", "tcp-ring", "ipc-all", "ipc-ring"] {
            let mut command = Command::new(&executable);
            command
                .args(["--exact", TEST_NAME, "--nocapture"])
                .env(CHILD_SCENARIO, scenario)
                .env_remove("CENSUS_BUS_IPC")
                .env_remove("CENSUS_BUS_NEIGHBORS");
            if scenario.starts_with("ipc-") {
                command.env("CENSUS_BUS_IPC", "1");
            }
            if scenario.ends_with("ring") {
                command.env("CENSUS_BUS_NEIGHBORS", "1");
            } else if scenario == "ipc-all" {
                command.env("CENSUS_BUS_NEIGHBORS", "0");
            }
            let output = command.output().unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.success(),
                "{scenario} failed:\nstdout:\n{}\nstderr:\n{}",
                stdout,
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                stdout.contains("running 1 test") && stdout.contains("1 passed"),
                "{scenario} must execute exactly one transport scenario: {stdout}"
            );
        }
    }
}
