//! nng pub/sub census bus between replicas of one cooperative run.
//!
//! Each replica publishes its current minimum (replica id, hop count,
//! energy, Cartesian coordinates) on a `Pub0` socket at every checkpoint and
//! reads every peer's latest publication from one `Sub0` socket dialled to
//! all peers. Nothing here goes through the coordinator: the population's
//! live positions, which are what the crowd count, the repulsion references
//! and the shared-bias deposits need, arrive peer to peer with no round
//! trip and no barrier. Receives never block; a slow peer costs nothing and
//! a lost message is replaced by the next publication.
//!
//! Wire format: topic `census/NNN\n` then little-endian `u32 replica`,
//! `u64 hops`, `f64 energy`, `u32 n_coords`, `f64 * n_coords`.

use nng::options::Options;
use nng::options::protocol::pubsub::Subscribe;
use nng::{Protocol, Socket};
use std::collections::HashMap;

/// Transport failure; the search continues uncoupled.
#[derive(Debug, thiserror::Error)]
#[error("census bus: {0}")]
pub struct CensusBusError(String);

/// A peer's latest published minimum.
#[derive(Debug, Clone)]
pub struct PeerMinimum {
    /// Replica identifier carried by the publication.
    pub replica: u32,
    /// Search hop count at publication.
    pub hops: u64,
    /// Published objective value at the minimum.
    pub energy: f64,
    /// Flattened Cartesian coordinates of the minimum.
    pub coordinates: Vec<f64>,
}

/// Nonblocking peer publications and the locally retained census.
pub struct CensusBus {
    replica: u32,
    publisher: Socket,
    subscriber: Socket,
    latest: HashMap<u32, PeerMinimum>,
}

fn url(base_port: u16, replica: u32) -> String {
    format!("tcp://127.0.0.1:{}", u32::from(base_port) + replica)
}

impl CensusBus {
    /// Binds this replica's publisher at `base_port + replica` and dials
    /// every other replica in `0..replicas`.
    pub fn new(replica: u32, base_port: u16, replicas: u32) -> Result<Self, CensusBusError> {
        let publisher = Socket::new(Protocol::Pub0).map_err(|e| CensusBusError(format!("pub: {e}")))?;
        publisher
            .listen(&url(base_port, replica))
            .map_err(|e| CensusBusError(format!("listen {}: {e}", url(base_port, replica))))?;
        let subscriber =
            Socket::new(Protocol::Sub0).map_err(|e| CensusBusError(format!("sub: {e}")))?;
        subscriber
            .set_opt::<Subscribe>(b"census/".to_vec())
            .map_err(|e| CensusBusError(format!("subscribe: {e}")))?;
        for peer in 0..replicas {
            if peer == replica {
                continue;
            }
            // Non-blocking dial: peers that have not bound yet are retried
            // by nng in the background.
            let _ = subscriber.dial_async(&url(base_port, peer));
        }
        Ok(Self {
            replica,
            publisher,
            subscriber,
            latest: HashMap::new(),
        })
    }

    /// Publishes this replica's current minimum. Never blocks.
    pub fn publish(&self, hops: u64, energy: f64, coordinates: &[f64]) {
        let mut frame = format!("census/{:03}\n", self.replica).into_bytes();
        frame.extend_from_slice(&self.replica.to_le_bytes());
        frame.extend_from_slice(&hops.to_le_bytes());
        frame.extend_from_slice(&energy.to_le_bytes());
        frame.extend_from_slice(&(coordinates.len() as u32).to_le_bytes());
        for v in coordinates {
            frame.extend_from_slice(&v.to_le_bytes());
        }
        let mut message = nng::Message::new();
        message.push_back(&frame);
        let _ = self.publisher.try_send(message);
    }

    /// Drains every waiting publication and returns the peers whose latest
    /// minimum changed in this poll. Never blocks.
    pub fn poll(&mut self) -> Vec<PeerMinimum> {
        let mut changed = Vec::new();
        while let Ok(message) = self.subscriber.try_recv() {
            let bytes: &[u8] = &message;
            let Some(end) = bytes.iter().position(|b| *b == b'\n') else {
                continue;
            };
            let Some(peer) = decode(&bytes[end + 1..]) else {
                continue;
            };
            if peer.replica == self.replica {
                continue;
            }
            let fresh = self
                .latest
                .get(&peer.replica)
                .is_none_or(|held| held.hops != peer.hops || held.energy != peer.energy);
            if fresh {
                changed.push(peer.clone());
            }
            self.latest.insert(peer.replica, peer);
        }
        changed
    }

    /// Latest minimum of every peer heard so far.
    pub fn peers(&self) -> impl Iterator<Item = &PeerMinimum> {
        self.latest.values()
    }

    /// Number of distinct peer identifiers retained in the local census.
    pub fn peer_count(&self) -> usize {
        self.latest.len()
    }
}

fn decode(bytes: &[u8]) -> Option<PeerMinimum> {
    let mut at = 0usize;
    let mut take = |n: usize| -> Option<&[u8]> {
        let slice = bytes.get(at..at + n)?;
        at += n;
        Some(slice)
    };
    let replica = u32::from_le_bytes(take(4)?.try_into().ok()?);
    let hops = u64::from_le_bytes(take(8)?.try_into().ok()?);
    let energy = f64::from_le_bytes(take(8)?.try_into().ok()?);
    let n = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
    let mut coordinates = Vec::with_capacity(n);
    for _ in 0..n {
        coordinates.push(f64::from_le_bytes(take(8)?.try_into().ok()?));
    }
    Some(PeerMinimum {
        replica,
        hops,
        energy,
        coordinates,
    })
}
