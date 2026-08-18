//! Transport for consensus traffic between server brains.
//!
//! What travels is the capnp envelope [`crate::raft::wire`] encodes,
//! and the bus owns that: a caller holds raft state and never a frame.
//! Two transports carry it and a build takes one:
//!
//! - Raw TCP by default. Each brain listens on its own address, dials
//!   every peer, and writes an addressed frame to the peer that frame
//!   names. It needs nothing but the standard library, which is the
//!   deciding property on a cluster whose compute nodes have no
//!   outbound network and a hand-assembled toolchain: a transport that
//!   pulls a C library and builds it through cmake cannot be the only
//!   way to run the occupancy campaign.
//! - nng under `nng-transport`. A `Pull0` inbox with a `Push0` pipe per
//!   peer for consensus, and a separate `Pub0` socket for observation
//!   that a `Sub0` reader filters by topic prefix. Better semantics,
//!   at the cost of a dependency the cluster build cannot always take.
//!
//! Receives never block on either. Consensus tolerates loss,
//! reordering and duplication, which is why neither retries a refused
//! send.

use crate::raft::wire::{decode_envelope, encode_envelope};
use crate::raft::{NodeId, RaftMessage};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Mutex;

/// Transport failure; consensus survives losses, so callers log and
/// continue rather than unwind.
#[derive(Debug, thiserror::Error)]
#[error("decree bus: {0}")]
pub struct BusError(String);

/// The transport this build carries.
#[cfg(not(feature = "nng-transport"))]
pub type DecreeBus = TcpDecreeBus;
#[cfg(feature = "nng-transport")]
pub use nng_bus::{DecreeBus, DecreeObserver};

struct PeerLink {
    id: NodeId,
    addr: SocketAddr,
    stream: Option<TcpStream>,
}

struct Inbound {
    stream: TcpStream,
    buf: Vec<u8>,
}

/// One brain's connection to the others over raw TCP.
pub struct TcpDecreeBus {
    id: NodeId,
    listener: TcpListener,
    peers: Mutex<Vec<PeerLink>>,
    inbound: Mutex<Vec<Inbound>>,
}

impl TcpDecreeBus {
    /// Listen on `listen_url` and dial each `(id, url)` peer.
    ///
    /// Dials are asynchronous: a peer that has not started yet is
    /// dialed again on the next poll, so start order does not matter.
    pub fn new(id: NodeId, listen_url: &str, peers: &[(NodeId, String)]) -> Result<Self, BusError> {
        let listen = parse_tcp_url(listen_url)?;
        let listener =
            TcpListener::bind(listen).map_err(|e| BusError(format!("listen {listen_url}: {e}")))?;
        listener
            .set_nonblocking(true)
            .map_err(|e| BusError(e.to_string()))?;
        let mut links = Vec::with_capacity(peers.len());
        for (peer, url) in peers {
            links.push(PeerLink {
                id: *peer,
                addr: parse_tcp_url(url)?,
                stream: None,
            });
        }
        Ok(Self {
            id,
            listener,
            peers: Mutex::new(links),
            inbound: Mutex::new(Vec::new()),
        })
    }

    /// Accepted for API parity with the nng transport, which is the one
    /// that can publish. Raw TCP has no observation plane, so naming an
    /// address here changes nothing rather than pretending.
    pub fn publishing(self, _url: &str) -> Result<Self, BusError> {
        Ok(self)
    }

    /// Deliver one message to the node it is addressed to.
    pub fn send(&self, to: NodeId, message: &RaftMessage) -> Result<(), BusError> {
        self.connect_peers();
        let packet = encode_frame(&encode_envelope(self.id, to, message));
        if let Ok(mut peers) = self.peers.lock()
            && let Some(peer) = peers.iter_mut().find(|peer| peer.id == to)
            && let Some(stream) = peer.stream.as_mut()
            && stream.write_all(&packet).is_err()
        {
            peer.stream = None;
        }
        Ok(())
    }

    /// Every message waiting for this brain, and who sent it.
    pub fn poll(&self) -> Vec<(NodeId, RaftMessage)> {
        self.accept_inbound();
        self.connect_peers();
        let mut frames = Vec::new();
        if let Ok(mut inbound) = self.inbound.lock() {
            let mut index = 0;
            while index < inbound.len() {
                match read_available(&mut inbound[index]) {
                    Ok(mut got) => {
                        frames.append(&mut got);
                        index += 1;
                    }
                    Err(()) => {
                        inbound.swap_remove(index);
                    }
                }
            }
        }
        frames
            .into_iter()
            .filter_map(|bytes| decode_envelope(&bytes).ok())
            .filter(|(_, to, _)| *to == self.id)
            .map(|(from, _, message)| (from, message))
            .collect()
    }

    fn accept_inbound(&self) {
        let Ok(mut inbound) = self.inbound.lock() else {
            return;
        };
        while let Ok((stream, _)) = self.listener.accept() {
            let Ok(()) = stream.set_nonblocking(true) else {
                continue;
            };
            inbound.push(Inbound {
                stream,
                buf: Vec::new(),
            });
        }
    }

    fn connect_peers(&self) {
        let Ok(mut peers) = self.peers.lock() else {
            return;
        };
        for peer in peers.iter_mut() {
            if peer.stream.is_some() {
                continue;
            }
            let Ok(stream) = TcpStream::connect(peer.addr) else {
                continue;
            };
            if stream.set_nonblocking(true).is_err() {
                continue;
            }
            peer.stream = Some(stream);
        }
    }
}

fn parse_tcp_url(url: &str) -> Result<SocketAddr, BusError> {
    let rest = url.strip_prefix("tcp://").unwrap_or(url);
    rest.parse()
        .map_err(|e| BusError(format!("brain url must be host:port, {url}: {e}")))
}

fn encode_frame(frame: &[u8]) -> Vec<u8> {
    let len = u32::try_from(frame.len()).unwrap_or(u32::MAX);
    let mut packet = Vec::with_capacity(4 + frame.len());
    packet.extend_from_slice(&len.to_le_bytes());
    packet.extend_from_slice(frame);
    packet
}

fn read_available(link: &mut Inbound) -> Result<Vec<Vec<u8>>, ()> {
    let mut scratch = [0u8; 4096];
    loop {
        match link.stream.read(&mut scratch) {
            Ok(0) => return Err(()),
            Ok(n) => link.buf.extend_from_slice(&scratch[..n]),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(()),
        }
    }
    let mut frames = Vec::new();
    loop {
        if link.buf.len() < 4 {
            break;
        }
        let len = u32::from_le_bytes(link.buf[..4].try_into().expect("four header bytes")) as usize;
        if link.buf.len() < 4 + len {
            break;
        }
        frames.push(link.buf[4..4 + len].to_vec());
        link.buf.drain(..4 + len);
    }
    Ok(frames)
}

/// nng transport: directed consensus, published observation.
#[cfg(feature = "nng-transport")]
mod nng_bus {
    use super::BusError;
    use crate::raft::wire::{decode_envelope, encode_envelope};
    use crate::raft::{NodeId, RaftMessage};
    use nng::options::Options;
    use nng::options::protocol::pubsub::Subscribe;
    use nng::{Protocol, Socket};
    use std::sync::Mutex;

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
        pub fn new(
            id: NodeId,
            listen_url: &str,
            peers: &[(NodeId, String)],
        ) -> Result<Self, BusError> {
            let inbox =
                Socket::new(Protocol::Pull0).map_err(|e| BusError(format!("inbox: {e}")))?;
            inbox
                .listen(&bus_url(listen_url))
                .map_err(|e| BusError(format!("listen {listen_url}: {e}")))?;
            let mut pipes = Vec::with_capacity(peers.len());
            for (peer_id, url) in peers {
                let push =
                    Socket::new(Protocol::Push0).map_err(|e| BusError(format!("peer: {e}")))?;
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
                Socket::new(Protocol::Pub0).map_err(|e| BusError(format!("publish: {e}")))?;
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
            let socket =
                Socket::new(Protocol::Sub0).map_err(|e| BusError(format!("observer: {e}")))?;
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
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The default transport must need nothing but the standard
    /// library, because the cluster the campaigns run on has no
    /// outbound network on its compute nodes and assembles its
    /// toolchain by hand. A build that reaches for cmake to talk
    /// between brains does not run there.
    #[test]
    fn the_default_transport_carries_no_build_dependency() {
        let bus = DecreeBus::new(9, "tcp://127.0.0.1:47361", &[]).unwrap();
        assert!(bus.poll().is_empty());
        // Naming a publish address is accepted and inert here, so the
        // caller is identical on both transports.
        let bus = bus.publishing("tcp://127.0.0.1:47362").unwrap();
        assert!(bus.poll().is_empty());
    }
}
