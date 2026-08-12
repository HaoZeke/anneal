//! Serialized coordinator for one campaign ensemble and system signature.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use capnp::message::ReaderOptions;
use capnp::serialize;

use super::{
    AcceptedReply, CatalogIdentity, CatalogOperation, CatalogReply, CatalogRequest,
    CatalogSnapshot, PROTOCOL_VERSION, ProtocolRejection, decode_request_reader, encode_reply,
};
use crate::Catalog_capnp::catalog_request;

/// Immutable identity and allowed replicas for one coordinator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerConfig {
    campaign: String,
    ensemble: String,
    signature_digest: [u8; 32],
    replicas: BTreeSet<u32>,
}

impl ServerConfig {
    /// Construct an isolated ensemble configuration.
    pub fn new(
        campaign: impl Into<String>,
        ensemble: impl Into<String>,
        signature_digest: [u8; 32],
        replicas: impl IntoIterator<Item = u32>,
    ) -> Result<Self, CatalogServerError> {
        let campaign = campaign.into();
        let ensemble = ensemble.into();
        let replicas = replicas.into_iter().collect::<BTreeSet<_>>();
        if campaign.is_empty() || ensemble.is_empty() || replicas.is_empty() {
            return Err(CatalogServerError::InvalidConfiguration);
        }
        Ok(Self {
            campaign,
            ensemble,
            signature_digest,
            replicas,
        })
    }
}

/// Run-header evidence that the coordinator owns a new empty state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerHeader {
    /// Campaign identity.
    pub campaign: String,
    /// Ensemble identity.
    pub ensemble: String,
    /// Allowed replica identities.
    pub replicas: Vec<u32>,
    /// Snapshot version at construction.
    pub initial_snapshot_version: u64,
    /// Whether catalog, census, replay map, and counters were empty.
    pub empty_state_proof: bool,
}

/// Coordinator startup failure.
#[derive(Debug, thiserror::Error)]
pub enum CatalogServerError {
    /// Campaign, ensemble, or replica set is empty.
    #[error("catalog server configuration is incomplete")]
    InvalidConfiguration,
    /// Listener setup failed.
    #[error("catalog server I/O failed: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Default)]
struct CoordinatorState {
    snapshot_version: u64,
    census_visits: u64,
    active_entries: u32,
    requests: BTreeMap<(u32, u64), CatalogRequest>,
    maximum_sequence: BTreeMap<u32, u64>,
}

/// Running localhost or remote coordinator.
pub struct CatalogServer {
    addr: SocketAddr,
    header: ServerHeader,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl CatalogServer {
    /// Bind and start a coordinator for one isolated ensemble.
    pub fn start(addr: &str, config: ServerConfig) -> Result<Self, CatalogServerError> {
        let listener = TcpListener::bind(addr)?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;
        let header = ServerHeader {
            campaign: config.campaign.clone(),
            ensemble: config.ensemble.clone(),
            replicas: config.replicas.iter().copied().collect(),
            initial_snapshot_version: 0,
            empty_state_proof: true,
        };
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let state = Arc::new(Mutex::new(CoordinatorState::default()));
        let thread = thread::spawn(move || {
            while !thread_stop.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let state = Arc::clone(&state);
                        let config = config.clone();
                        thread::spawn(move || {
                            let _ = handle_connection(stream, &config, state);
                        });
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(_) => break,
                }
            }
        });
        Ok(Self {
            addr,
            header,
            stop,
            thread: Some(thread),
        })
    }

    /// Bound socket address.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Immutable empty-state run header.
    pub fn header(&self) -> &ServerHeader {
        &self.header
    }
}

impl Drop for CatalogServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn handle_connection(
    mut stream: TcpStream,
    config: &ServerConfig,
    state: Arc<Mutex<CoordinatorState>>,
) -> Result<(), String> {
    stream
        .set_nodelay(true)
        .map_err(|error| error.to_string())?;
    loop {
        let message = match serialize::read_message(&mut stream, ReaderOptions::new()) {
            Ok(message) => message,
            Err(_) => return Ok(()),
        };
        let root = match message.get_root::<catalog_request::Reader>() {
            Ok(root) => root,
            Err(_) => return Ok(()),
        };
        let request = match decode_request_reader(root) {
            Ok(request) => request,
            Err(_) => {
                write_reply(
                    &mut stream,
                    CatalogReply::Rejected {
                        event_sequence: 0,
                        snapshot_version: 0,
                        reason: ProtocolRejection::Malformed,
                    },
                )?;
                continue;
            }
        };
        let reply = process_request(config, &state, request);
        write_reply(&mut stream, reply)?;
    }
}

fn process_request(
    config: &ServerConfig,
    state: &Arc<Mutex<CoordinatorState>>,
    request: CatalogRequest,
) -> CatalogReply {
    let mut state = state.lock().expect("catalog coordinator mutex poisoned");
    let rejection = identity_rejection(config, &request.identity).or_else(|| {
        (request.snapshot_version > state.snapshot_version)
            .then_some(ProtocolRejection::SnapshotRegression)
    });
    if let Some(reason) = rejection {
        return rejected(&state, request.event_sequence, reason);
    }
    let key = (request.identity.replica, request.event_sequence);
    if let Some(stored) = state.requests.get(&key) {
        return if stored == &request {
            accepted(&state, request.event_sequence, true)
        } else {
            rejected(
                &state,
                request.event_sequence,
                ProtocolRejection::SequenceReplay,
            )
        };
    }
    if state
        .maximum_sequence
        .get(&request.identity.replica)
        .is_some_and(|maximum| request.event_sequence <= *maximum)
    {
        return rejected(
            &state,
            request.event_sequence,
            ProtocolRejection::SequenceRegression,
        );
    }
    match &request.operation {
        CatalogOperation::Snapshot
        | CatalogOperation::Sample { .. }
        | CatalogOperation::DescriptorHole { .. } => {}
        CatalogOperation::RecordVisit { .. } => {
            state.census_visits = state.census_visits.saturating_add(1);
            state.snapshot_version = state.snapshot_version.saturating_add(1);
        }
        CatalogOperation::OfferCandidate { .. } => {
            state.active_entries = state.active_entries.saturating_add(1);
            state.snapshot_version = state.snapshot_version.saturating_add(1);
        }
        CatalogOperation::LedgerEvent { .. } => {
            state.snapshot_version = state.snapshot_version.saturating_add(1);
        }
    }
    state
        .maximum_sequence
        .insert(request.identity.replica, request.event_sequence);
    state.requests.insert(key, request);
    accepted(&state, key.1, false)
}

fn identity_rejection(
    config: &ServerConfig,
    identity: &CatalogIdentity,
) -> Option<ProtocolRejection> {
    if identity.campaign != config.campaign {
        Some(ProtocolRejection::CampaignMismatch)
    } else if identity.ensemble != config.ensemble {
        Some(ProtocolRejection::EnsembleMismatch)
    } else if identity.signature_digest != config.signature_digest {
        Some(ProtocolRejection::SignatureMismatch)
    } else if !config.replicas.contains(&identity.replica) {
        Some(ProtocolRejection::ReplicaMismatch)
    } else {
        None
    }
}

fn accepted(state: &CoordinatorState, event_sequence: u64, duplicate: bool) -> CatalogReply {
    CatalogReply::Accepted(AcceptedReply {
        event_sequence,
        duplicate,
        snapshot: CatalogSnapshot {
            version: state.snapshot_version,
            census_visits: state.census_visits,
            active_entries: state.active_entries,
        },
    })
}

fn rejected(
    state: &CoordinatorState,
    event_sequence: u64,
    reason: ProtocolRejection,
) -> CatalogReply {
    CatalogReply::Rejected {
        event_sequence,
        snapshot_version: state.snapshot_version,
        reason,
    }
}

fn write_reply(stream: &mut TcpStream, reply: CatalogReply) -> Result<(), String> {
    let bytes = encode_reply(reply).map_err(|error| error.to_string())?;
    stream
        .write_all(&bytes)
        .map_err(|error| error.to_string())?;
    stream.flush().map_err(|error| error.to_string())
}
