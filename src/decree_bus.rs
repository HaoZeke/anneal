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
    pub fn new(id: NodeId, listen_url: &str, peer_urls: &[String]) -> Result<Self, BusError> {
        let listen = parse_tcp_url(listen_url)?;
        let listener =
            TcpListener::bind(listen).map_err(|e| BusError(format!("listen {listen_url}: {e}")))?;
        listener
            .set_nonblocking(true)
            .map_err(|e| BusError(e.to_string()))?;
        let mut peers = Vec::with_capacity(peer_urls.len());
        for url in peer_urls {
            peers.push(PeerLink {
                addr: parse_tcp_url(url)?,
                stream: None,
            });
        }
        Ok(Self {
            id,
            listener,
            peers: Mutex::new(peers),
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

/// nng `Bus0` transport, and a read-only view of the same bus.
#[cfg(feature = "nng-transport")]
mod nng_bus {
    use super::BusError;
    use crate::raft::{NodeId, RaftMessage};
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

    /// Dial every peer that is not connected yet, and report which
    /// stayed down. A peer that has not started is dialed again on the
    /// next call, so start order between brains does not matter.
    fn dial_pending(socket: &Socket, pending: &mut Vec<String>) {
        pending.retain(|url| socket.dial(url).is_err());
    }

    /// One brain's connection to the others over an nng bus.
    pub struct DecreeBus {
        id: NodeId,
        socket: Socket,
        pending: Mutex<Vec<String>>,
    }

    impl DecreeBus {
        /// Listen on `listen_url`, dial every peer, filter as `id`.
        pub fn new(id: NodeId, listen_url: &str, peer_urls: &[String]) -> Result<Self, BusError> {
            let socket = Socket::new(Protocol::Bus0)
                .map_err(|error| BusError(format!("bus socket: {error}")))?;
            socket
                .listen(&bus_url(listen_url))
                .map_err(|error| BusError(format!("listen {listen_url}: {error}")))?;
            let mut pending: Vec<String> = peer_urls.iter().map(|url| bus_url(url)).collect();
            dial_pending(&socket, &mut pending);
            Ok(Self {
                id,
                socket,
                pending: Mutex::new(pending),
            })
        }

        /// Broadcast one encoded envelope. A bus message is already
        /// framed, so nothing here prefixes a length.
        pub fn send(&self, frame: &[u8]) -> Result<(), BusError> {
            self.redial();
            // A refused send is a lost message, which consensus
            // survives, and the message nng hands back is dropped with
            // it rather than queued behind a peer that may never come.
            let _ = self.socket.try_send(frame);
            Ok(())
        }

        /// Drain every frame currently queued that is addressed to this
        /// brain. Never blocks on a missing peer.
        pub fn poll(&self) -> Vec<Vec<u8>> {
            self.redial();
            drain(&self.socket)
                .into_iter()
                .filter(|bytes| {
                    matches!(
                        crate::raft::wire::decode_envelope(bytes),
                        Ok((_, to, _)) if to == self.id
                    )
                })
                .collect()
        }

        fn redial(&self) {
            if let Ok(mut pending) = self.pending.lock()
                && !pending.is_empty()
            {
                dial_pending(&self.socket, &mut pending);
            }
        }
    }

    /// A reader of the consensus stream that is not a peer of it.
    ///
    /// The observer joins the same bus and never sends, so every
    /// envelope the brains exchange decodes here, addressed to whoever
    /// it was addressed to. Consensus does not know it exists and does
    /// not wait for it.
    pub struct DecreeObserver {
        socket: Socket,
        pending: Mutex<Vec<String>>,
    }

    impl DecreeObserver {
        /// Dial every brain, listening only.
        pub fn new(peer_urls: &[String]) -> Result<Self, BusError> {
            let socket = Socket::new(Protocol::Bus0)
                .map_err(|error| BusError(format!("observer socket: {error}")))?;
            let mut pending: Vec<String> = peer_urls.iter().map(|url| bus_url(url)).collect();
            dial_pending(&socket, &mut pending);
            Ok(Self {
                socket,
                pending: Mutex::new(pending),
            })
        }

        /// Every envelope seen since the last call, decoded. Unlike a
        /// brain's poll this keeps traffic addressed to anyone, which
        /// is the point of watching rather than participating.
        pub fn poll(&self) -> Vec<(NodeId, NodeId, RaftMessage)> {
            if let Ok(mut pending) = self.pending.lock()
                && !pending.is_empty()
            {
                dial_pending(&self.socket, &mut pending);
            }
            drain(&self.socket)
                .into_iter()
                .filter_map(|bytes| crate::raft::wire::decode_envelope(&bytes).ok())
                .collect()
        }
    }

    /// Take every message the socket has without waiting for one.
    fn drain(socket: &Socket) -> Vec<Vec<u8>> {
        let mut frames = Vec::new();
        while let Ok(message) = socket.try_recv() {
            frames.push(message[..].to_vec());
        }
        frames
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
