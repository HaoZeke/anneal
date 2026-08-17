//! TCP transport for consensus traffic between server brains.
//!
//! Occupancy talking is one brain per replica. The occupancy build is
//! `featomic,ira,bank-rpc` and does not carry `nng-transport`, so the
//! bus is raw TCP: each brain listens on its own address, dials every
//! peer, and filters by the envelope `to` field. Receives are
//! non-blocking. Consensus tolerates loss, reordering, and
//! duplication, which is why nothing here retries a failed write.

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

/// One brain's connection to the others.
pub struct DecreeBus {
    id: NodeId,
    listener: TcpListener,
    peers: Mutex<Vec<PeerLink>>,
    inbound: Mutex<Vec<Inbound>>,
}

impl DecreeBus {
    /// Listen on `listen_url`, dial every peer, filter as `id`.
    ///
    /// Dials are asynchronous: a peer that has not started yet is
    /// dialed again on the next poll, so start order between brains
    /// does not matter.
    pub fn new(id: NodeId, listen_url: &str, peer_urls: &[String]) -> Result<Self, BusError> {
        let listen = parse_tcp_url(listen_url)?;
        let listener = TcpListener::bind(listen)
            .map_err(|e| BusError(format!("listen {listen_url}: {e}")))?;
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
    rest.parse()
        .map_err(|e| BusError(format!("{url}: {e}")))
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
