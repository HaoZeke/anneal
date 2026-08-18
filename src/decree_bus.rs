//! Transport for consensus traffic between server brains.
//!
//! Occupancy talking is one brain per replica, and what travels is the
//! capnp envelope [`crate::raft::wire`] encodes. Two transports carry
//! it, chosen by the `nng-transport` feature and sharing one API, so a
//! caller names neither:
//!
//! - Default: raw TCP. Each brain listens on its own address, dials
//!   every peer, and filters by the envelope `to` field.
//! - `nng-transport`: an nng `Bus0` socket. The protocol already is a
//!   broadcast mesh with reconnection, so the transport stops being
//!   hand-rolled, and a reader that only listens can join the same bus
//!   and decode the whole stream. That is what
//!   [`DecreeObserver`] is: analysis over live consensus traffic
//!   without becoming a peer of it.
//!
//! Receives never block on either. Consensus tolerates loss,
//! reordering, and duplication, which is why nothing here retries a
//! failed write.

use crate::raft::NodeId;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Mutex;

/// Transport failure; consensus survives losses, so callers log and
/// continue rather than unwind.
#[derive(Debug, thiserror::Error)]
#[error("decree bus: {0}")]
pub struct BusError(String);

struct PeerLink {
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

/// The transport this build carries.
#[cfg(not(feature = "nng-transport"))]
pub type DecreeBus = TcpDecreeBus;

impl TcpDecreeBus {
    /// Listen on `listen_url`, dial every peer, filter as `id`.
    ///
    /// Dials are asynchronous: a peer that has not started yet is
    /// dialed again on the next poll, so start order between brains
    /// does not matter.
    ///
    /// Peer identity is taken but not routed on here. This transport
    /// broadcasts and filters on receipt, which is what the deployed
    /// occupancy build runs; the nng transport routes by it. Changing
    /// the send path of a bus a campaign is talking over is not worth
    /// the saved writes.
    pub fn new(
        id: NodeId,
        listen_url: &str,
        peers: &[(NodeId, String)],
    ) -> Result<Self, BusError> {
        let listen = parse_tcp_url(listen_url)?;
        let listener =
            TcpListener::bind(listen).map_err(|e| BusError(format!("listen {listen_url}: {e}")))?;
        listener
            .set_nonblocking(true)
            .map_err(|e| BusError(e.to_string()))?;
        let mut links = Vec::with_capacity(peers.len());
        for (_, url) in peers {
            links.push(PeerLink {
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

    /// Broadcast one encoded envelope.
    pub fn send(&self, frame: &[u8]) -> Result<(), BusError> {
        self.connect_peers();
        let packet = encode_frame(frame);
        if let Ok(mut peers) = self.peers.lock() {
            for peer in peers.iter_mut() {
                let Some(stream) = peer.stream.as_mut() else {
                    continue;
                };
                if stream.write_all(&packet).is_err() {
                    peer.stream = None;
                }
            }
        }
        Ok(())
    }

    /// Drain every frame currently queued that is addressed to this
    /// brain. Never blocks on a missing peer.
    pub fn poll(&self) -> Vec<Vec<u8>> {
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
        frames.retain(|bytes| {
            matches!(
                crate::raft::wire::decode_envelope(bytes),
                Ok((_, to, _)) if to == self.id
            )
        });
        frames
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
            };
            peer.stream = Some(stream);
        }
    }
}

fn parse_tcp_url(url: &str) -> Result<SocketAddr, BusError> {
    let rest = url
        .strip_prefix("tcp://")
        .ok_or_else(|| BusError(format!("brain url must be tcp://host:port, not {url}")))?;
    rest.parse().map_err(|e| BusError(format!("{url}: {e}")))
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

/// nng transport: directed consensus traffic, published observation.
#[cfg(feature = "nng-transport")]
mod nng_bus {
    use super::BusError;
    use crate::raft::wire::decode_envelope;
    use crate::raft::{NodeId, RaftMessage};
    use nng::options::Options;
    use nng::options::protocol::pubsub::Subscribe;
    use nng::{Protocol, Socket};
    use std::sync::Mutex;

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
    /// Subscriptions in nng match a byte prefix, so the fields are
    /// fixed width and ordered coarse to fine: `raft/007/` is every
    /// message node seven sent, and `raft/007/003/` is what it sent to
    /// node three. The newline ends the topic and never appears in it.
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

    /// Split a published message back into its topic and its envelope.
    fn strip_topic(bytes: &[u8]) -> Option<&[u8]> {
        let end = bytes.iter().position(|byte| *byte == b'\n')?;
        bytes.get(end + 1..)
    }

    /// One peer's outbound pipe.
    struct Peer {
        id: NodeId,
        url: String,
        push: Socket,
        dialed: bool,
    }

    impl Peer {
        /// Dial on first use and after a peer restarts, so start order
        /// between brains does not matter.
        fn ready(&mut self) -> bool {
            if !self.dialed {
                self.dialed = self.push.dial(&self.url).is_ok();
            }
            self.dialed
        }
    }

    /// One brain's connection to the others.
    ///
    /// The consensus plane is a `Pull0` inbox that every peer pushes
    /// into, and one `Push0` pipe per peer. Raft addresses each message
    /// to one node, so this delivers it to that node rather than
    /// broadcasting to all and discarding what the others were not
    /// meant to read.
    ///
    /// The observation plane is a separate `Pub0` socket, opened only
    /// when [`DecreeBus::publishing`] names an address. Consensus never
    /// waits on it: a subscriber that cannot keep up loses messages
    /// instead of applying backpressure to the brains.
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
            let inbox = Socket::new(Protocol::Pull0)
                .map_err(|error| BusError(format!("inbox socket: {error}")))?;
            inbox
                .listen(&bus_url(listen_url))
                .map_err(|error| BusError(format!("listen {listen_url}: {error}")))?;
            let mut pipes = Vec::with_capacity(peers.len());
            for (peer_id, url) in peers {
                let push = Socket::new(Protocol::Push0)
                    .map_err(|error| BusError(format!("peer socket: {error}")))?;
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

        /// Also publish every envelope this brain sends, on `url`.
        ///
        /// Separate from the consensus address on purpose, so a reader
        /// attaches to the observation plane and never to the inbox.
        pub fn publishing(mut self, url: &str) -> Result<Self, BusError> {
            let socket = Socket::new(Protocol::Pub0)
                .map_err(|error| BusError(format!("publish socket: {error}")))?;
            socket
                .listen(&bus_url(url))
                .map_err(|error| BusError(format!("publish {url}: {error}")))?;
            self.publish = Some(socket);
            Ok(self)
        }

        /// Deliver one encoded envelope to the node it names.
        pub fn send(&self, frame: &[u8]) -> Result<(), BusError> {
            let Ok((from, to, message)) = decode_envelope(frame) else {
                return Err(BusError("send: envelope does not decode".into()));
            };
            if let Some(publish) = self.publish.as_ref() {
                let mut published = topic(from, to, &message);
                published.extend_from_slice(frame);
                // Observation is lossy by design: a slow reader must
                // never hold up consensus.
                let _ = publish.try_send(&published[..]);
            }
            if let Ok(mut peers) = self.peers.lock()
                && let Some(peer) = peers.iter_mut().find(|peer| peer.id == to)
                && peer.ready()
            {
                // A refused push is a lost message, which consensus
                // survives; queueing it behind a peer that may never
                // return would not help.
                let _ = peer.push.try_send(frame);
            }
            Ok(())
        }

        /// Drain the inbox. Never blocks on a missing peer.
        pub fn poll(&self) -> Vec<Vec<u8>> {
            let mut frames = Vec::new();
            while let Ok(message) = self.inbox.try_recv() {
                frames.push(message[..].to_vec());
            }
            frames.retain(|bytes| matches!(decode_envelope(bytes), Ok((_, to, _)) if to == self.id));
            frames
        }
    }

    /// A reader of the consensus stream that is not a peer of it.
    ///
    /// Subscribes to the brains' observation plane, so every envelope
    /// they send decodes here whoever it was addressed to. Consensus
    /// does not know it exists and does not wait for it.
    pub struct DecreeObserver {
        socket: Socket,
    }

    impl DecreeObserver {
        /// Watch every publisher in `publish_urls`, keeping the traffic
        /// whose topic starts with `prefix`. An empty prefix is all of
        /// it; `raft/007/` is one node's outbound traffic.
        pub fn new(publish_urls: &[String], prefix: &str) -> Result<Self, BusError> {
            let socket = Socket::new(Protocol::Sub0)
                .map_err(|error| BusError(format!("observer socket: {error}")))?;
            socket
                .set_opt::<Subscribe>(prefix.as_bytes().to_vec())
                .map_err(|error| BusError(format!("subscribe {prefix}: {error}")))?;
            for url in publish_urls {
                // A publisher that has not started yet is simply not
                // seen; an observer is not worth failing a brain over.
                let _ = socket.dial(&bus_url(url));
            }
            Ok(Self { socket })
        }

        /// Every envelope seen since the last call, decoded. Unlike a
        /// brain's poll this keeps traffic addressed to anyone, which
        /// is the point of watching rather than participating.
        pub fn poll(&self) -> Vec<(NodeId, NodeId, RaftMessage)> {
            let mut seen = Vec::new();
            while let Ok(message) = self.socket.try_recv() {
                if let Some(envelope) = strip_topic(&message[..])
                    && let Ok(decoded) = decode_envelope(envelope)
                {
                    seen.push(decoded);
                }
            }
            seen
        }
    }
}

#[cfg(feature = "nng-transport")]
pub use nng_bus::{DecreeBus, DecreeObserver};

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
            &[(1, "tcp://127.0.0.1:47332".to_owned())],
        )
        .unwrap();
        let b = DecreeBus::new(
            1,
            "tcp://127.0.0.1:47332",
            &[(0, "tcp://127.0.0.1:47331".to_owned())],
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
