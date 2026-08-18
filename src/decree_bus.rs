//! Transport for consensus traffic between server brains.
//!
//! Occupancy talking is one brain per replica, and what travels is the
//! capnp envelope [`crate::raft::wire`] encodes. Two planes carry it,
//! each on the nng protocol that matches its shape:
//!
//! - Consensus is directed. Raft addresses every message to one node,
//!   so a brain owns a `Pull0` inbox that peers push into and one
//!   `Push0` pipe per peer. A message goes to the node it names.
//!   Broadcasting instead would send each `AppendEntries` to every
//!   brain for all but one to discard it.
//! - Observation is published. A separate `Pub0` socket carries a copy
//!   of everything the brain sends, under a topic a `Sub0` reader
//!   filters on by prefix. Consensus never waits on a reader: nng
//!   drops for a subscriber that cannot keep up, which is the right
//!   failure for analysis and the wrong one for agreement, so the two
//!   never share a socket.
//!
//! Receives never block. Consensus tolerates loss, reordering, and
//! duplication, which is why nothing here retries a refused send.

use crate::raft::wire::{decode_envelope, encode_envelope};
use crate::raft::{NodeId, RaftMessage};
use nng::options::Options;
use nng::options::protocol::pubsub::Subscribe;
use nng::{Protocol, Socket};
use std::sync::Mutex;

/// Transport failure; consensus survives losses, so callers log and
/// continue rather than unwind.
#[derive(Debug, thiserror::Error)]
#[error("decree bus: {0}")]
pub struct BusError(String);

/// nng wants a scheme; a brain is configured with `host:port`.
fn bus_url(url: &str) -> String {
    if url.contains("://") {
        url.to_owned()
    } else {
        format!("tcp://{url}")
    }
}

/// Topic a published envelope carries, ahead of the capnp bytes.
///
/// Subscriptions match a byte prefix, so the fields are fixed width
/// and ordered coarse to fine: `raft/007/` is everything node seven
/// sent, and `raft/007/003/` is what it sent to node three. Kinds
/// split on outcome rather than on message type alone, so a refused
/// vote and a rejected append are each their own subscription. The
/// newline ends the topic and never occurs inside it.
fn topic(from: NodeId, to: NodeId, message: &RaftMessage) -> Vec<u8> {
    let kind = match message {
        RaftMessage::RequestVote { .. } => "vote_request",
        RaftMessage::VoteReply { granted: true, .. } => "vote_granted",
        RaftMessage::VoteReply { .. } => "vote_refused",
        RaftMessage::AppendEntries { entries, .. } if entries.is_empty() => "heartbeat",
        RaftMessage::AppendEntries { .. } => "append",
        RaftMessage::AppendReply { success: true, .. } => "append_accepted",
        RaftMessage::AppendReply { .. } => "append_rejected",
    };
    format!("raft/{from:03}/{to:03}/{kind}\n").into_bytes()
}

/// Split a published message into its topic and its envelope.
fn strip_topic(bytes: &[u8]) -> Option<&[u8]> {
    let end = bytes.iter().position(|byte| *byte == b'\n')?;
    bytes.get(end + 1..)
}

/// Take every message a socket holds without waiting for one.
fn drain(socket: &Socket) -> Vec<Vec<u8>> {
    let mut frames = Vec::new();
    while let Ok(message) = socket.try_recv() {
        frames.push(message[..].to_vec());
    }
    frames
}

/// One peer's outbound pipe.
struct Peer {
    id: NodeId,
    url: String,
    push: Socket,
    dialed: bool,
}

impl Peer {
    /// Dial on first use and again after a peer restarts, so start
    /// order between brains does not matter.
    fn ready(&mut self) -> bool {
        if !self.dialed {
            self.dialed = self.push.dial(&self.url).is_ok();
        }
        self.dialed
    }
}

/// One brain's connection to the others.
pub struct DecreeBus {
    id: NodeId,
    inbox: Socket,
    peers: Mutex<Vec<Peer>>,
    publish: Option<Socket>,
}

impl DecreeBus {
    /// Listen on `listen_url` and open a pipe to each `(id, url)` peer.
    pub fn new(id: NodeId, listen_url: &str, peers: &[(NodeId, String)]) -> Result<Self, BusError> {
        let inbox =
            Socket::new(Protocol::Pull0).map_err(|error| BusError(format!("inbox: {error}")))?;
        inbox
            .listen(&bus_url(listen_url))
            .map_err(|error| BusError(format!("listen {listen_url}: {error}")))?;
        let mut pipes = Vec::with_capacity(peers.len());
        for (peer_id, url) in peers {
            let push = Socket::new(Protocol::Push0)
                .map_err(|error| BusError(format!("peer {peer_id}: {error}")))?;
            let mut peer = Peer {
                id: *peer_id,
                url: bus_url(url),
                push,
                dialed: false,
            };
            peer.ready();
            pipes.push(peer);
        }
        Ok(Self {
            id,
            inbox,
            peers: Mutex::new(pipes),
            publish: None,
        })
    }

    /// Also publish a copy of everything this brain sends, on `url`.
    ///
    /// A separate address from the inbox on purpose, so a reader
    /// attaches to the observation plane and never to consensus.
    pub fn publishing(mut self, url: &str) -> Result<Self, BusError> {
        let socket =
            Socket::new(Protocol::Pub0).map_err(|error| BusError(format!("publish: {error}")))?;
        socket
            .listen(&bus_url(url))
            .map_err(|error| BusError(format!("publish {url}: {error}")))?;
        self.publish = Some(socket);
        Ok(self)
    }

    /// Deliver one message to the node it is addressed to.
    ///
    /// The bus owns the wire format so a caller holds raft state and
    /// nothing else: nowhere above here encodes an envelope, and
    /// nowhere above here decodes one.
    pub fn send(&self, to: NodeId, message: &RaftMessage) -> Result<(), BusError> {
        let frame = encode_envelope(self.id, to, message);
        if let Some(publish) = self.publish.as_ref() {
            let mut published = topic(self.id, to, message);
            published.extend_from_slice(&frame);
            // Observation is lossy by design.
            let _ = publish.try_send(&published[..]);
        }
        if let Ok(mut peers) = self.peers.lock()
            && let Some(peer) = peers.iter_mut().find(|peer| peer.id == to)
            && peer.ready()
        {
            // A refused push is a lost message, which consensus
            // survives; holding it for a peer that may never return
            // would not help.
            let _ = peer.push.try_send(&frame[..]);
        }
        Ok(())
    }

    /// Every message waiting for this brain, and who sent it. Never
    /// blocks on a missing peer.
    pub fn poll(&self) -> Vec<(NodeId, RaftMessage)> {
        drain(&self.inbox)
            .into_iter()
            .filter_map(|bytes| decode_envelope(&bytes).ok())
            .filter(|(_, to, _)| *to == self.id)
            .map(|(from, _, message)| (from, message))
            .collect()
    }
}

/// A reader of the consensus stream that is not a peer of it.
///
/// Subscribes to the brains' observation plane, so every message they
/// send decodes here whoever it was addressed to. Consensus does not
/// know it exists and does not wait for it.
pub struct DecreeObserver {
    socket: Socket,
}

impl DecreeObserver {
    /// Watch every publisher in `publish_urls`, keeping traffic whose
    /// topic starts with `prefix`. An empty prefix is all of it;
    /// `raft/007/` is one node's outbound traffic; `raft/007/003/` is
    /// one direction of one pair. The filter runs in the transport, so
    /// an unsubscribed message is never carried at all.
    pub fn new(publish_urls: &[String], prefix: &str) -> Result<Self, BusError> {
        let socket =
            Socket::new(Protocol::Sub0).map_err(|error| BusError(format!("observer: {error}")))?;
        socket
            .set_opt::<Subscribe>(prefix.as_bytes().to_vec())
            .map_err(|error| BusError(format!("subscribe {prefix}: {error}")))?;
        for url in publish_urls {
            // A publisher that has not started yet is simply not seen.
            // An observer is not worth failing a brain over.
            let _ = socket.dial(&bus_url(url));
        }
        Ok(Self { socket })
    }

    /// Every message seen since the last call, with both ends named.
    /// Unlike a brain's poll this keeps traffic addressed to anyone,
    /// which is the point of watching rather than participating.
    pub fn poll(&self) -> Vec<(NodeId, NodeId, RaftMessage)> {
        drain(&self.socket)
            .into_iter()
            .filter_map(|bytes| {
                let envelope = strip_topic(&bytes)?;
                decode_envelope(envelope).ok()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// nng dials settle asynchronously; give them a moment.
    fn settle() {
        std::thread::sleep(std::time::Duration::from_millis(300));
    }

    #[test]
    fn two_brains_exchange_addressed_envelopes_over_loopback() {
        let a = DecreeBus::new(
            0,
            "tcp://127.0.0.1:47331",
            &[(1, "tcp://127.0.0.1:47332".into())],
        )
        .unwrap();
        let b = DecreeBus::new(
            1,
            "tcp://127.0.0.1:47332",
            &[(0, "tcp://127.0.0.1:47331".into())],
        )
        .unwrap();
        settle();

        a.send(
            1,
            &RaftMessage::VoteReply {
                term: 3,
                granted: true,
            },
        )
        .unwrap();
        settle();

        let seen = b.poll();
        assert_eq!(seen.len(), 1, "one message was sent to node one");
        assert_eq!(seen[0].0, 0, "node zero sent it");
        assert!(matches!(
            seen[0].1,
            RaftMessage::VoteReply {
                term: 3,
                granted: true
            }
        ));
        // Directed delivery: the sender's own inbox stays empty.
        assert!(
            a.poll().is_empty(),
            "a directed message reached a bystander"
        );
    }

    #[test]
    fn an_observer_reads_traffic_addressed_to_someone_else() {
        let brain = DecreeBus::new(
            4,
            "tcp://127.0.0.1:47341",
            &[(5, "tcp://127.0.0.1:47342".into())],
        )
        .unwrap()
        .publishing("tcp://127.0.0.1:47343")
        .unwrap();
        let watcher = DecreeObserver::new(&["tcp://127.0.0.1:47343".into()], "raft/004/").unwrap();
        settle();

        brain
            .send(
                5,
                &RaftMessage::AppendReply {
                    term: 9,
                    success: false,
                    match_index: 2,
                },
            )
            .unwrap();
        settle();

        let seen = watcher.poll();
        assert_eq!(seen.len(), 1, "the observer sees traffic it is not part of");
        assert_eq!(seen[0].0, 4);
        assert_eq!(seen[0].1, 5, "the addressee is kept, not filtered away");
        assert!(matches!(
            seen[0].2,
            RaftMessage::AppendReply { term: 9, .. }
        ));
    }

    #[test]
    fn a_subscription_prefix_selects_one_sender() {
        let brain = DecreeBus::new(
            6,
            "tcp://127.0.0.1:47351",
            &[(7, "tcp://127.0.0.1:47352".into())],
        )
        .unwrap()
        .publishing("tcp://127.0.0.1:47353")
        .unwrap();
        // Node six publishes; a reader that wants node nine sees none.
        let elsewhere =
            DecreeObserver::new(&["tcp://127.0.0.1:47353".into()], "raft/009/").unwrap();
        settle();

        brain
            .send(
                7,
                &RaftMessage::RequestVote {
                    term: 1,
                    last_log_index: 0,
                    last_log_term: 0,
                },
            )
            .unwrap();
        settle();

        assert!(
            elsewhere.poll().is_empty(),
            "the transport carried a topic nobody subscribed to"
        );
    }
}
