//! Timeout-bounded client for one isolated catalog coordinator.

use std::collections::BTreeMap;
use std::io::Write;
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

use capnp::message::ReaderOptions;
use capnp::serialize;

use super::{
    AcceptedReply, CatalogIdentity, CatalogOperation, CatalogReply, CatalogRequest,
    CatalogSnapshot, PROTOCOL_VERSION, ProtocolError, ProtocolRejection, decode_reply_reader,
    encode_request,
};
use crate::Catalog_capnp::catalog_reply;

/// Connection and I/O deadlines for a catalog client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClientConfig {
    /// TCP connection deadline.
    pub connect_timeout: Duration,
    /// Read and write deadline.
    pub io_timeout: Duration,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(2),
            io_timeout: Duration::from_secs(5),
        }
    }
}

/// Transport, wire, or typed coordinator rejection.
#[derive(Debug, thiserror::Error)]
pub enum CatalogClientError {
    /// TCP or stream I/O failed.
    #[error("catalog transport failed: {0}")]
    Transport(#[from] std::io::Error),
    /// Cap'n Proto encoding or decoding failed.
    #[error("catalog protocol failed: {0}")]
    Protocol(#[from] ProtocolError),
    /// Coordinator rejected the request without mutation.
    #[error("catalog coordinator rejected request: {0:?}")]
    Rejected(ProtocolRejection),
}

impl PartialEq for CatalogClientError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Rejected(left), Self::Rejected(right)) => left == right,
            (Self::Protocol(left), Self::Protocol(right)) => left == right,
            (Self::Transport(left), Self::Transport(right)) => left.kind() == right.kind(),
            _ => false,
        }
    }
}

/// Version and replay classification for one accepted mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MutationReceipt {
    /// Coordinator snapshot version after the mutation.
    pub version: u64,
    /// Whether the coordinator recognized an identical replay.
    pub duplicate: bool,
}

/// Persistent client bound to one replica identity.
pub struct CatalogClient {
    stream: TcpStream,
    identity: CatalogIdentity,
    snapshot_version: u64,
    requests: BTreeMap<u64, CatalogRequest>,
}

impl CatalogClient {
    /// Connect to a coordinator with explicit deadlines.
    pub fn connect(
        addr: SocketAddr,
        identity: CatalogIdentity,
        config: ClientConfig,
    ) -> Result<Self, CatalogClientError> {
        let stream = TcpStream::connect_timeout(&addr, config.connect_timeout)?;
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(config.io_timeout))?;
        stream.set_write_timeout(Some(config.io_timeout))?;
        Ok(Self {
            stream,
            identity,
            snapshot_version: 0,
            requests: BTreeMap::new(),
        })
    }

    /// Read the current coordinator snapshot.
    pub fn snapshot(&mut self, event_sequence: u64) -> Result<CatalogSnapshot, CatalogClientError> {
        Ok(self
            .call(event_sequence, CatalogOperation::Snapshot)?
            .snapshot)
    }

    /// Record one exact census observation.
    pub fn record_visit(
        &mut self,
        event_sequence: u64,
        basin_id: u64,
        created: bool,
        descriptor: Vec<f64>,
    ) -> Result<MutationReceipt, CatalogClientError> {
        let reply = self.call(
            event_sequence,
            CatalogOperation::RecordVisit {
                basin_id,
                created,
                descriptor,
            },
        )?;
        Ok(MutationReceipt {
            version: reply.snapshot.version,
            duplicate: reply.duplicate,
        })
    }

    fn call(
        &mut self,
        event_sequence: u64,
        operation: CatalogOperation,
    ) -> Result<AcceptedReply, CatalogClientError> {
        let request = self
            .requests
            .entry(event_sequence)
            .or_insert_with(|| CatalogRequest {
                protocol_version: PROTOCOL_VERSION,
                identity: self.identity.clone(),
                event_sequence,
                snapshot_version: self.snapshot_version,
                operation,
            })
            .clone();
        self.stream.write_all(&encode_request(&request)?)?;
        self.stream.flush()?;
        let message = serialize::read_message(&mut self.stream, ReaderOptions::new())
            .map_err(|error| ProtocolError::Malformed(error.to_string()))?;
        let root = message
            .get_root::<catalog_reply::Reader>()
            .map_err(|error| ProtocolError::Malformed(error.to_string()))?;
        match decode_reply_reader(root)? {
            CatalogReply::Accepted(reply) => {
                self.snapshot_version = self.snapshot_version.max(reply.snapshot.version);
                Ok(reply)
            }
            CatalogReply::Rejected { reason, .. } => Err(CatalogClientError::Rejected(reason)),
        }
    }
}
