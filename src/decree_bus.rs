//! nng transport for consensus traffic between server brains.
//!
//! Every brain owns one bus-protocol socket: it listens on its own
//! address and dials every peer, and a send is a broadcast the
//! receivers filter by the envelope's `to` field. Addressed delivery
//! through a broadcast medium costs nothing at the handful of brains
//! an ensemble runs and keeps the wiring one socket per process. All
//! receives are bounded by a short timeout, so a poll never blocks the
//! chain that hosts it; consensus tolerates the loss, reordering, and
//! duplication a bus can produce, which is why nothing here retries.

use crate::raft::NodeId;
use nng::options::{Options, RecvTimeout};
use nng::{Protocol, Socket};
use std::time::Duration;

/// Transport failure; consensus survives losses, so callers log and
/// continue rather than unwind.
#[derive(Debug, thiserror::Error)]
#[error("decree bus: {0}")]
pub struct BusError(String);

/// One brain's connection to the others.
pub struct DecreeBus {
    socket: Socket,
    id: NodeId,
}

impl DecreeBus {
    /// Listen on `listen_url`, dial every peer, filter as `id`.
    ///
    /// Dials are asynchronous: a peer that has not started yet is
    /// dialed again by nng when it appears, so start order between
    /// brains does not matter.
    pub fn new(id: NodeId, listen_url: &str, peer_urls: &[String]) -> Result<Self, BusError> {
        let socket = Socket::new(Protocol::Bus0).map_err(|e| BusError(e.to_string()))?;
        socket
            .set_opt::<RecvTimeout>(Some(Duration::from_millis(1)))
            .map_err(|e| BusError(e.to_string()))?;
        socket
            .listen(listen_url)
            .map_err(|e| BusError(format!("listen {listen_url}: {e}")))?;
        for url in peer_urls {
            socket
                .dial_async(url)
                .map_err(|e| BusError(format!("dial {url}: {e}")))?;
        }
        Ok(Self { socket, id })
    }

    /// Broadcast one encoded envelope.
    pub fn send(&self, frame: &[u8]) -> Result<(), BusError> {
        self.socket
            .send(frame)
            .map_err(|(_, e)| BusError(e.to_string()))
    }

    /// Drain every frame currently queued that is addressed to this
    /// brain. Never blocks longer than the socket's short timeout.
    pub fn poll(&self) -> Vec<Vec<u8>> {
        let mut frames = Vec::new();
        while let Ok(message) = self.socket.recv() {
            let bytes = message.as_slice().to_vec();
            if let Ok((_, to, _)) = crate::raft::wire::decode_envelope(&bytes)
                && to == self.id
            {
                frames.push(bytes);
            }
        }
        frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raft::RaftMessage;
    use crate::raft::wire::{decode_envelope, encode_envelope};

    #[test]
    fn two_brains_exchange_addressed_envelopes_over_loopback() {
        let a = DecreeBus::new(
            0,
            "tcp://127.0.0.1:47331",
            &["tcp://127.0.0.1:47332".to_owned()],
        )
        .unwrap();
        let b = DecreeBus::new(
            1,
            "tcp://127.0.0.1:47332",
            &["tcp://127.0.0.1:47331".to_owned()],
        )
        .unwrap();
        // Dials settle asynchronously.
        std::thread::sleep(std::time::Duration::from_millis(200));
        let to_b = RaftMessage::VoteReply {
            term: 3,
            granted: true,
        };
        let to_a = RaftMessage::VoteReply {
            term: 3,
            granted: false,
        };
        a.send(&encode_envelope(0, 1, &to_b)).unwrap();
        b.send(&encode_envelope(1, 0, &to_a)).unwrap();
        let mut got_b = Vec::new();
        let mut got_a = Vec::new();
        for _ in 0..50 {
            got_b.extend(b.poll());
            got_a.extend(a.poll());
            if !got_b.is_empty() && !got_a.is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert_eq!(decode_envelope(&got_b[0]).unwrap(), (0, 1, to_b));
        assert_eq!(decode_envelope(&got_a[0]).unwrap(), (1, 0, to_a));
        // A frame addressed elsewhere is dropped by the filter.
        a.send(&encode_envelope(
            0,
            9,
            &RaftMessage::VoteReply {
                term: 4,
                granted: true,
            },
        ))
        .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(b.poll().is_empty());
    }
}
