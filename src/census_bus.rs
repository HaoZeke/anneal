//! nng pub/sub census bus between replicas of one cooperative run.
//!
//! Each replica publishes its current minimum (replica id, hop count,
//! energy, Cartesian coordinates) on a `Pub0` socket when it changes, with
//! an unchanged-state refresh every eight eligible checkpoints. One `Sub0`
//! socket reads direct neighbours selected by the configured topology.
//! Nothing here goes through the coordinator: the neighbourhood's
//! live positions, which are what the crowd count, the repulsion references
//! and the shared-bias deposits need, arrive peer to peer with no round
//! trip and no barrier. Receives never block; periodic best-effort refreshes
//! let late subscribers and receivers of lost messages recover current state.
//! Peer records are not forwarded, so this is not a global gossip aggregate.
//!
//! Wire format: topic `census/NNN\n` then little-endian `u32 replica`,
//! `u64 hops`, `f64 energy`, `u32 n_coords`, `f64 * n_coords`.

use ndarray::Array1;
use nng::options::Options;
use nng::options::protocol::pubsub::Subscribe;
use nng::{Protocol, Socket};
use std::collections::HashMap;

const REFRESH_CHECKPOINTS: u8 = 8;

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
    replicas: u32,
    publisher: Socket,
    subscriber: Socket,
    latest: HashMap<u32, PeerMinimum>,
    last_published: Option<PeerMinimum>,
    checkpoints_since_publication: u8,
    /// Peer well tables received since the last [`CensusBus::poll_wells`].
    pending_wells: Vec<(u32, Vec<(Array1<f64>, f64)>)>,
    /// Whether each peer's latest minimum lies on this replica's side of
    /// the packing map, recomputed only when the peer's minimum or this
    /// replica's own minimum changes.
    pub nearby: HashMap<u32, bool>,
}

/// Transport for the bus: TCP loopback, or ipc (Unix domain sockets) when
/// every replica of the run lives on one node, which is the HyperQueue
/// and Slurm layout used here. ipc avoids the loopback TCP stack and the
/// port range; `CENSUS_BUS_IPC=1` selects it.
fn url(base_port: u16, replica: u32) -> String {
    if std::env::var("CENSUS_BUS_IPC").is_ok_and(|v| v == "1") {
        format!("ipc:///tmp/anneal-census-{}-{:03}", base_port, replica)
    } else {
        format!("tcp://127.0.0.1:{}", u32::from(base_port) + replica)
    }
}

/// Direct-neighbour topology. `CENSUS_BUS_NEIGHBORS=k` subscribes a replica
/// to ring neighbours within distance `k` only; 0 (default) is all-to-all.
/// This restricts census traffic, not coordinator-mediated adoption. A ring
/// has a growing diameter and supplies no logarithmic mixing guarantee.
fn ring_distance(a: u32, b: u32, n: u32) -> u32 {
    let d = a.abs_diff(b);
    d.min(n - d)
}

impl CensusBus {
    /// Binds this replica's publisher at `base_port + replica` and dials
    /// the configured neighbours in `0..replicas`.
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
        subscriber
            .set_opt::<Subscribe>(b"wells/".to_vec())
            .map_err(|e| CensusBusError(format!("subscribe wells: {e}")))?;
        let neighbors: u32 = std::env::var("CENSUS_BUS_NEIGHBORS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        for peer in 0..replicas {
            if peer == replica {
                continue;
            }
            if neighbors > 0 && ring_distance(peer, replica, replicas) > neighbors {
                continue;
            }
            // Non-blocking dial: peers that have not bound yet are retried
            // by nng in the background.
            let _ = subscriber.dial_async(&url(base_port, peer));
        }
        Ok(Self {
            replica,
            replicas,
            publisher,
            subscriber,
            latest: HashMap::new(),
            last_published: None,
            checkpoints_since_publication: 0,
            pending_wells: Vec::new(),
            nearby: HashMap::new(),
        })
    }

    /// Publishes a well table for gossip: `(centre, depth)` pairs, deepest
    /// first as [`crate::bias::BasinBias::deepest_wells`] returns them.
    ///
    /// Anti-entropy over the bus: the table is the state, a late subscriber
    /// gets the whole of it on the next round, and nothing is forwarded.
    /// Never blocks; a failed send is retried by the next round.
    pub fn publish_wells(&mut self, wells: &[(Array1<f64>, f64)]) -> bool {
        let Some(dim) = wells.first().map(|(centre, _)| centre.len()) else {
            return false;
        };
        if wells
            .iter()
            .any(|(centre, depth)| centre.len() != dim || !depth.is_finite())
        {
            return false;
        }
        let mut frame = format!("wells/{:03}\n", self.replica).into_bytes();
        frame.extend_from_slice(&self.replica.to_le_bytes());
        frame.extend_from_slice(&(wells.len() as u32).to_le_bytes());
        frame.extend_from_slice(&(dim as u32).to_le_bytes());
        for (centre, depth) in wells {
            frame.extend_from_slice(&depth.to_le_bytes());
            for v in centre {
                frame.extend_from_slice(&v.to_le_bytes());
            }
        }
        let mut message = nng::Message::new();
        message.push_back(&frame);
        self.publisher.try_send(message).is_ok()
    }

    /// Peer well tables that arrived since the last call, oldest first.
    pub fn poll_wells(&mut self) -> Vec<(u32, Vec<(Array1<f64>, f64)>)> {
        self.drain();
        std::mem::take(&mut self.pending_wells)
    }

    /// Reads every waiting message into the census or the wells queue.
    fn drain(&mut self) -> Vec<PeerMinimum> {
        let mut changed = Vec::new();
        while let Ok(message) = self.subscriber.try_recv() {
            let bytes: &[u8] = &message;
            if bytes.starts_with(b"wells/") {
                if let Some((peer, wells)) = decode_wells_frame(bytes, self.replicas)
                    && peer != self.replica
                {
                    self.pending_wells.push((peer, wells));
                }
                continue;
            }
            let Some(peer) = decode_frame(bytes, self.replicas) else {
                continue;
            };
            if peer.replica == self.replica {
                continue;
            }
            // Fresh means the minimum changed, not that the peer hopped: a
            // replica sitting on the shelf refreshes the same structure
            // without costing its peers a packing-map comparison each time.
            let fresh = self.latest.get(&peer.replica).is_none_or(|held| {
                held.energy != peer.energy || held.coordinates != peer.coordinates
            });
            if fresh {
                changed.push(peer.clone());
            }
            self.latest.insert(peer.replica, peer);
        }
        changed
    }

    /// Publishes changed minima immediately and refreshes unchanged minima
    /// every eight calls. Failed sends remain eligible for retry. Never blocks.
    pub fn publish(&mut self, hops: u64, energy: f64, coordinates: &[f64]) {
        self.checkpoints_since_publication = self.checkpoints_since_publication.saturating_add(1);
        // Energy is not a geometry key. Refreshes repair best-effort delivery
        // without making every checkpoint perform descriptor comparisons.
        if self
            .last_published
            .as_ref()
            .is_some_and(|last| last.energy == energy && last.coordinates == coordinates)
            && self.checkpoints_since_publication < REFRESH_CHECKPOINTS
        {
            return;
        }
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
        if self.publisher.try_send(message).is_ok() {
            self.last_published = Some(PeerMinimum {
                replica: self.replica,
                hops,
                energy,
                coordinates: coordinates.to_vec(),
            });
            self.checkpoints_since_publication = 0;
        }
    }

    /// Drains every waiting publication and returns the peers whose latest
    /// minimum changed in this poll. Never blocks. Well tables read in the
    /// same drain wait for [`CensusBus::poll_wells`].
    pub fn poll(&mut self) -> Vec<PeerMinimum> {
        self.drain()
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

fn decode_frame(bytes: &[u8], replicas: u32) -> Option<PeerMinimum> {
    let end = bytes.iter().position(|byte| *byte == b'\n')?;
    let payload = bytes.get(end + 1..)?;
    let replica = u32::from_le_bytes(payload.get(..4)?.try_into().ok()?);
    if replica >= replicas {
        return None;
    }
    // The publication topic must identify the same configured sender as
    // its body before any coordinate payload is admitted.
    let topic = format!("census/{replica:03}");
    if &bytes[..end] != topic.as_bytes() {
        return None;
    }
    decode(payload)
}

fn decode_wells_frame(bytes: &[u8], replicas: u32) -> Option<(u32, Vec<(Array1<f64>, f64)>)> {
    let end = bytes.iter().position(|byte| *byte == b'\n')?;
    let payload = bytes.get(end + 1..)?;
    let replica = u32::from_le_bytes(payload.get(..4)?.try_into().ok()?);
    if replica >= replicas || &bytes[..end] != format!("wells/{replica:03}").as_bytes() {
        return None;
    }
    decode_wells(payload).map(|wells| (replica, wells))
}

/// Decodes a well table: replica, count, dimension, then depth and centre
/// per well; exact length and finite values or nothing.
fn decode_wells(bytes: &[u8]) -> Option<Vec<(Array1<f64>, f64)>> {
    let header = bytes.get(..12)?;
    let count = usize::try_from(u32::from_le_bytes(header[4..8].try_into().ok()?)).ok()?;
    let dim = usize::try_from(u32::from_le_bytes(header[8..12].try_into().ok()?)).ok()?;
    if count == 0 || dim == 0 {
        return None;
    }
    let per_well = dim
        .checked_add(1)?
        .checked_mul(std::mem::size_of::<f64>())?;
    let expected = 12_usize.checked_add(count.checked_mul(per_well)?)?;
    if bytes.len() != expected {
        return None;
    }
    let values: Vec<f64> = bytes[12..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| f64::from_le_bytes(*chunk))
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(
        values
            .chunks_exact(dim + 1)
            .map(|well| (Array1::from(well[1..].to_vec()), well[0]))
            .collect(),
    )
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
    for chunk in bytes[24..].as_chunks::<8>().0 {
        let coordinate = f64::from_le_bytes(*chunk);
        if !coordinate.is_finite() {
            return None;
        }
    }
    let mut coordinates = Vec::with_capacity(n);
    for chunk in bytes[24..].as_chunks::<8>().0 {
        coordinates.push(f64::from_le_bytes(*chunk));
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
    use super::{CensusBus, REFRESH_CHECKPOINTS, decode};
    use nng::{Protocol, Socket};
    use std::collections::HashMap;

    const COORDINATES: [f64; 6] = [0.0, 0.0, 0.0, 1.2, 0.0, 0.0];

    fn unconnected_bus() -> CensusBus {
        CensusBus {
            replica: 0,
            replicas: 2,
            publisher: Socket::new(Protocol::Pub0).unwrap(),
            subscriber: Socket::new(Protocol::Sub0).unwrap(),
            latest: HashMap::new(),
            last_published: None,
            checkpoints_since_publication: 0,
            pending_wells: Vec::new(),
            nearby: HashMap::new(),
        }
    }

    #[test]
    fn a_well_table_round_trips_and_a_truncated_one_is_refused() {
        use ndarray::array;
        let wells = vec![(array![1.0, 2.0, 3.0], 0.5), (array![4.0, 5.0, 6.0], 0.25)];
        let mut frame = b"wells/001\n".to_vec();
        frame.extend_from_slice(&1u32.to_le_bytes());
        frame.extend_from_slice(&2u32.to_le_bytes());
        frame.extend_from_slice(&3u32.to_le_bytes());
        for (centre, depth) in &wells {
            frame.extend_from_slice(&depth.to_le_bytes());
            for v in centre {
                frame.extend_from_slice(&v.to_le_bytes());
            }
        }
        let (peer, decoded) = super::decode_wells_frame(&frame, 4).unwrap();
        assert_eq!(peer, 1);
        assert_eq!(decoded, wells);
        assert!(super::decode_wells_frame(&frame[..frame.len() - 1], 4).is_none());
        assert!(
            super::decode_wells_frame(&frame, 1).is_none(),
            "replica out of range"
        );
        let mut bus = unconnected_bus();
        assert!(!bus.publish_wells(&[]), "an empty table is not published");
        assert!(bus.poll_wells().is_empty());
    }

    #[test]
    fn failed_initial_publication_preserves_retry_eligibility() {
        let mut bus = unconnected_bus();
        bus.publisher.close();
        assert!(matches!(
            bus.publisher.try_send(&b"probe"[..]),
            Err((_, nng::Error::Closed))
        ));

        bus.publish(7, -1.0, &COORDINATES);
        assert!(bus.last_published.is_none());
        assert_eq!(bus.checkpoints_since_publication, 1);

        bus.publisher = Socket::new(Protocol::Pub0).unwrap();
        bus.publish(8, -1.0, &COORDINATES);
        let published = bus.last_published.as_ref().unwrap();
        assert_eq!(published.replica, 0);
        assert_eq!(published.hops, 8);
        assert_eq!(published.energy, -1.0);
        assert_eq!(published.coordinates, COORDINATES);
        assert_eq!(bus.checkpoints_since_publication, 0);
    }

    #[test]
    fn failed_unchanged_refresh_preserves_successful_state_and_retries_next_call() {
        let mut bus = unconnected_bus();
        bus.publish(7, -1.0, &COORDINATES);
        assert_eq!(bus.last_published.as_ref().unwrap().hops, 7);
        assert_eq!(bus.checkpoints_since_publication, 0);

        bus.publisher.close();
        assert!(matches!(
            bus.publisher.try_send(&b"probe"[..]),
            Err((_, nng::Error::Closed))
        ));
        for checkpoint in 1..=REFRESH_CHECKPOINTS {
            bus.publish(7 + u64::from(checkpoint), -1.0, &COORDINATES);
            let published = bus.last_published.as_ref().unwrap();
            assert_eq!(published.replica, 0);
            assert_eq!(published.hops, 7);
            assert_eq!(published.energy, -1.0);
            assert_eq!(published.coordinates, COORDINATES);
            assert_eq!(bus.checkpoints_since_publication, checkpoint);
        }

        bus.publisher = Socket::new(Protocol::Pub0).unwrap();
        let retry_hops = 8 + u64::from(REFRESH_CHECKPOINTS);
        bus.publish(retry_hops, -1.0, &COORDINATES);
        let published = bus.last_published.as_ref().unwrap();
        assert_eq!(published.replica, 0);
        assert_eq!(published.hops, retry_hops);
        assert_eq!(published.energy, -1.0);
        assert_eq!(published.coordinates, COORDINATES);
        assert_eq!(bus.checkpoints_since_publication, 0);
    }

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
