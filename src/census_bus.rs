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
        let publisher =
            Socket::new(Protocol::Pub0).map_err(|e| CensusBusError(format!("pub: {e}")))?;
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

fn decode_frame(_bytes: &[u8], _replicas: u32) -> Option<PeerMinimum> {
    unimplemented!("census sender admission")
}

fn decode(bytes: &[u8]) -> Option<PeerMinimum> {
    let header = bytes.get(..24)?;
    let replica = u32::from_le_bytes(header[..4].try_into().ok()?);
    let hops = u64::from_le_bytes(header[4..12].try_into().ok()?);
    let energy = f64::from_le_bytes(header[12..20].try_into().ok()?);
    let n = usize::try_from(u32::from_le_bytes(header[20..24].try_into().ok()?)).ok()?;
    if !energy.is_finite() || n == 0 || !n.is_multiple_of(3) {
        return None;
    }
    // The complete Cartesian payload must exist before its declared count
    // can reserve memory. Exact length also excludes trailing frame bytes.
    let expected_len = 24_usize.checked_add(n.checked_mul(std::mem::size_of::<f64>())?)?;
    if bytes.len() != expected_len {
        return None;
    }
    for chunk in bytes[24..].chunks_exact(8) {
        let coordinate = f64::from_le_bytes(chunk.try_into().ok()?);
        if !coordinate.is_finite() {
            return None;
        }
    }
    let mut coordinates = Vec::with_capacity(n);
    for chunk in bytes[24..].chunks_exact(8) {
        coordinates.push(f64::from_le_bytes(chunk.try_into().ok()?));
    }
    Some(PeerMinimum {
        replica,
        hops,
        energy,
        coordinates,
    })
}

#[cfg(test)]
mod tests {
    use super::decode;

    const COORDINATES: [f64; 6] = [0.0, 0.0, 0.0, 1.2, 0.0, 0.0];

    fn frame(energy: f64, coordinates: &[f64]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&7_u64.to_le_bytes());
        bytes.extend_from_slice(&energy.to_le_bytes());
        bytes.extend_from_slice(&u32::try_from(coordinates.len()).unwrap().to_le_bytes());
        for coordinate in coordinates {
            bytes.extend_from_slice(&coordinate.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn finite_cartesian_minimum_round_trips_without_a_socket() {
        let minimum = decode(&frame(-1.0, &COORDINATES)).unwrap();
        assert_eq!(minimum.replica, 1);
        assert_eq!(minimum.hops, 7);
        assert_eq!(minimum.energy, -1.0);
        assert_eq!(minimum.coordinates, COORDINATES);
    }

    #[test]
    fn nonfinite_energy_is_not_a_peer_minimum() {
        for energy in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
            assert!(
                decode(&frame(energy, &COORDINATES)).is_none(),
                "nonfinite energy {energy:?} must not enter the census"
            );
        }
    }

    #[test]
    fn nonfinite_coordinates_are_not_a_peer_minimum() {
        for value in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
            for index in 0..COORDINATES.len() {
                let mut coordinates = COORDINATES;
                coordinates[index] = value;
                assert!(
                    decode(&frame(-1.0, &coordinates)).is_none(),
                    "nonfinite coordinate {index}={value:?} must not enter the census"
                );
            }
        }
    }

    #[test]
    fn empty_geometry_is_not_a_peer_minimum() {
        assert!(decode(&frame(-1.0, &[])).is_none());
    }

    #[test]
    fn incomplete_cartesian_triplets_are_not_a_peer_minimum() {
        for length in [1, 2, 4, 5] {
            assert!(
                decode(&frame(-1.0, &COORDINATES[..length])).is_none(),
                "{length} coordinates do not describe complete Cartesian atoms"
            );
        }
    }

    #[test]
    fn trailing_bytes_are_not_part_of_a_census_frame() {
        for trailing in [&[0_u8][..], &[0_u8; 8][..]] {
            let mut bytes = frame(-1.0, &COORDINATES);
            bytes.extend_from_slice(trailing);
            assert!(decode(&bytes).is_none());
        }
    }

    #[test]
    fn every_truncated_frame_is_rejected() {
        let bytes = frame(-1.0, &COORDINATES);
        for end in 0..bytes.len() {
            assert!(decode(&bytes[..end]).is_none(), "truncated at byte {end}");
        }
    }

    #[test]
    fn coordinate_count_must_match_the_payload_exactly() {
        for declared in [0_u32, 3, 7, 12, 1024] {
            let mut bytes = frame(-1.0, &COORDINATES);
            bytes[20..24].copy_from_slice(&declared.to_le_bytes());
            assert!(
                decode(&bytes).is_none(),
                "{declared} declared coordinates do not match the six-coordinate payload"
            );
        }
    }

    fn framed_minimum(topic_replica: u32, payload_replica: u32) -> Vec<u8> {
        let mut bytes = format!("census/{topic_replica:03}\n").into_bytes();
        let mut payload = frame(-1.0, &COORDINATES);
        payload[..4].copy_from_slice(&payload_replica.to_le_bytes());
        bytes.extend_from_slice(&payload);
        bytes
    }

    #[test]
    fn sender_admission_preserves_known_topic_and_payload_identity() {
        for replica in [0, 1, 999, 1000] {
            let minimum = super::decode_frame(&framed_minimum(replica, replica), 1001).unwrap();
            assert_eq!(minimum.replica, replica);
            assert_eq!(minimum.hops, 7);
            assert_eq!(minimum.energy, -1.0);
            assert_eq!(minimum.coordinates, COORDINATES);
        }
    }

    #[test]
    fn sender_admission_rejects_ids_outside_the_configured_cohort() {
        for replica in [4, 17, u32::MAX] {
            assert!(
                super::decode_frame(&framed_minimum(replica, replica), 4).is_none(),
                "replica {replica} does not belong to a four-replica census"
            );
        }
    }

    #[test]
    fn sender_admission_rejects_an_empty_cohort() {
        assert!(super::decode_frame(&framed_minimum(0, 0), 0).is_none());
    }

    #[test]
    fn sender_admission_rejects_topic_and_payload_identity_mismatch() {
        for (topic_replica, payload_replica) in [(1, 2), (2, 1)] {
            assert!(
                super::decode_frame(&framed_minimum(topic_replica, payload_replica), 4).is_none(),
                "topic {topic_replica} cannot identify payload replica {payload_replica}"
            );
        }
    }
}
