//! nng transport for consensus traffic between server brains.
//!
//! What travels is the capnp envelope [`crate::raft::wire`] encodes,
//! and the bus owns that: a caller holds raft state and never a frame.
//!
//! Two planes, each on the nng protocol that matches its shape.
//! Consensus is directed, because raft addresses every message to one
//! node, so a brain owns a `Pull0` inbox that peers push into and one
//! `Push0` pipe per peer. Observation is published on a separate
//! `Pub0` socket under a topic a `Sub0` reader filters by prefix, so a
//! reader can decode the whole stream without becoming a peer of it
//! and a slow one cannot backpressure agreement.
//!
//! Receives never block. Consensus tolerates loss, reordering and
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

fn bus_url(url: &str) -> String {
    if url.contains("://") {
        url.to_owned()
    } else {
        format!("tcp://{url}")
    }
}

/// Topic a published envelope carries, ahead of the capnp bytes.
///
/// Fixed width and coarse to fine, so a subscription prefix is
/// unambiguous: `raft/007/` is everything node seven sent and
/// `raft/007/003/` is what it sent to node three. Kinds split on
/// outcome, so a refused vote and a rejected append each have their
/// own subscription.
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

fn strip_topic(bytes: &[u8]) -> Option<&[u8]> {
    let end = bytes.iter().position(|byte| *byte == b'\n')?;
    bytes.get(end + 1..)
}

fn drain(socket: &Socket) -> Vec<Vec<u8>> {
    let mut frames = Vec::new();
    while let Ok(message) = socket.try_recv() {
        frames.push(message[..].to_vec());
    }
    frames
}

struct Peer {
    id: NodeId,
    url: String,
    push: Socket,
    dialed: bool,
}

impl Peer {
    fn ready(&mut self) -> bool {
        if !self.dialed {
            self.dialed = self.push.dial(&self.url).is_ok();
        }
        self.dialed
    }
}

/// One brain's connection to the others over nng.
pub struct DecreeBus {
    id: NodeId,
    inbox: Socket,
    peers: Mutex<Vec<Peer>>,
    publish: Option<Socket>,
}

impl DecreeBus {
    /// Listen on `listen_url` and pipe to each `(id, url)` peer.
    pub fn new(id: NodeId, listen_url: &str, peers: &[(NodeId, String)]) -> Result<Self, BusError> {
        let inbox = Socket::new(Protocol::Pull0).map_err(|e| BusError(format!("inbox: {e}")))?;
        inbox
            .listen(&bus_url(listen_url))
            .map_err(|e| BusError(format!("listen {listen_url}: {e}")))?;
        let mut pipes = Vec::with_capacity(peers.len());
        for (peer_id, url) in peers {
            let push = Socket::new(Protocol::Push0).map_err(|e| BusError(format!("peer: {e}")))?;
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
        let socket = Socket::new(Protocol::Pub0).map_err(|e| BusError(format!("publish: {e}")))?;
        socket
            .listen(&bus_url(url))
            .map_err(|e| BusError(format!("publish {url}: {e}")))?;
        self.publish = Some(socket);
        Ok(self)
    }

    /// Deliver one message to the node it is addressed to.
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
            let _ = peer.push.try_send(&frame[..]);
        }
        Ok(())
    }

    /// Every message waiting for this brain, and who sent it.
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
pub struct DecreeObserver {
    socket: Socket,
}

impl DecreeObserver {
    /// Watch every publisher, keeping traffic whose topic starts
    /// with `prefix`. The filter runs in the transport, so an
    /// unsubscribed message is never carried at all.
    pub fn new(publish_urls: &[String], prefix: &str) -> Result<Self, BusError> {
        let socket = Socket::new(Protocol::Sub0).map_err(|e| BusError(format!("observer: {e}")))?;
        socket
            .set_opt::<Subscribe>(prefix.as_bytes().to_vec())
            .map_err(|e| BusError(format!("subscribe {prefix}: {e}")))?;
        for url in publish_urls {
            let _ = socket.dial(&bus_url(url));
        }
        Ok(Self { socket })
    }

    /// Every message seen since the last call, both ends named.
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

    fn settle() {
        std::thread::sleep(std::time::Duration::from_millis(300));
    }

    /// Loopback endpoints no other process on the host is holding.
    ///
    /// The fixed 473xx ports collide the moment two test runs share a
    /// node, which a Slurm box guarantees eventually: measured, one
    /// suite run failed on "Address in use" at 47332 while an unrelated
    /// job held it. Each test takes a block keyed by its own PID
    /// instead, so concurrent suites bind disjoint ports and a rerun of
    /// a crashed suite does not trip over its own corpse either.
    fn endpoints<const N: usize>(block: u16) -> [String; N] {
        let base = 20000 + (std::process::id() % 20000) as u16 + block * 8;
        std::array::from_fn(|slot| format!("tcp://127.0.0.1:{}", base + slot as u16))
    }

    #[test]
    fn two_brains_exchange_addressed_envelopes_over_loopback() {
        let [ea, eb] = endpoints::<2>(0);
        let a = DecreeBus::new(0, &ea, &[(1, eb.clone())]).unwrap();
        let b = DecreeBus::new(1, &eb, &[(0, ea.clone())]).unwrap();
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
        // Directed delivery: the sender's own inbox stays empty.
        assert!(
            a.poll().is_empty(),
            "a directed message reached a bystander"
        );
    }

    #[test]
    fn two_brains_exchange_over_job_scoped_ipc_endpoints() {
        let prefix = format!("/tmp/anneal-decree-ipc-{}", std::process::id());
        let first = format!("ipc://{prefix}-0.sock");
        let second = format!("ipc://{prefix}-1.sock");
        let a = DecreeBus::new(0, &first, &[(1, second.clone())]).unwrap();
        let b = DecreeBus::new(1, &second, &[(0, first.clone())]).unwrap();
        settle();

        a.send(
            1,
            &RaftMessage::VoteReply {
                term: 7,
                granted: true,
            },
        )
        .unwrap();
        settle();

        let seen = b.poll();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].0, 0);
        assert_eq!(
            seen[0].1,
            RaftMessage::VoteReply {
                term: 7,
                granted: true,
            }
        );
    }

    #[test]
    fn an_observer_reads_traffic_addressed_to_someone_else() {
        let [eb, epeer, epub] = endpoints::<3>(1);
        let brain = DecreeBus::new(4, &eb, &[(5, epeer)])
            .unwrap()
            .publishing(&epub)
            .unwrap();
        let watcher = DecreeObserver::new(&[epub.clone()], "raft/004/").unwrap();
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
        assert_eq!(seen[0].1, 5, "the addressee is kept, not filtered away");
    }

    #[test]
    fn a_subscription_prefix_selects_one_sender() {
        let [eb, epeer, epub] = endpoints::<3>(2);
        let brain = DecreeBus::new(6, &eb, &[(7, epeer)])
            .unwrap()
            .publishing(&epub)
            .unwrap();
        let elsewhere = DecreeObserver::new(&[epub.clone()], "raft/009/").unwrap();
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
