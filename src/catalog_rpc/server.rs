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
use ndarray::ArrayView1;

use super::{
    decode_request_reader, encode_reply, AcceptedReply, CatalogCandidate, CatalogIdentity,
    CatalogOperation, CatalogReply, CatalogRequest, CatalogSnapshot, ProtocolError,
    ProtocolRejection,
};
use crate::catalog::{
    BasinCatalog, BasinCensus, CandidateRecord, CandidateValidator, FreshEvaluation, QuenchStatus,
    SystemSignature, ValidatedCandidate, ValidatorConfig,
};
use crate::descriptor_space::DescriptorSpace;
use crate::Catalog_capnp::catalog_request;

type FreshEvaluator = dyn Fn(&[f64]) -> Result<FreshEvaluation, String> + Send + Sync;

/// Immutable identity and allowed replicas for one coordinator.
#[derive(Clone)]
pub struct ServerConfig {
    campaign: String,
    ensemble: String,
    signature_digest: [u8; 32],
    replicas: BTreeSet<u32>,
    scientific: Option<ScientificConfig>,
}

#[derive(Clone)]
struct ScientificConfig {
    signature: SystemSignature,
    descriptor_space: DescriptorSpace,
    validator: ValidatorConfig,
    catalog_capacity: usize,
    census_radius: f64,
    total_charged_work: u64,
    evaluate: Arc<FreshEvaluator>,
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
            scientific: None,
        })
    }

    /// Attach the scientific state and receiving-side engine validation.
    #[allow(clippy::too_many_arguments)]
    pub fn with_scientific_state<F>(
        mut self,
        signature: SystemSignature,
        descriptor_space: DescriptorSpace,
        validator: ValidatorConfig,
        catalog_capacity: usize,
        census_radius: f64,
        total_charged_work: u64,
        evaluate: F,
    ) -> Result<Self, CatalogServerError>
    where
        F: Fn(&[f64]) -> Result<FreshEvaluation, String> + Send + Sync + 'static,
    {
        if signature.digest() != self.signature_digest
            || signature.descriptor.schema != descriptor_space.schema().name()
            || signature.descriptor.version != descriptor_space.schema().version()
            || usize::try_from(signature.coordinate_dim).ok()
                != Some(validator.reference_coordinates.len())
            || catalog_capacity > u32::MAX as usize
        {
            return Err(CatalogServerError::InvalidScientificConfiguration);
        }
        let descriptor = descriptor_space
            .describe(
                ArrayView1::from(&validator.reference_coordinates),
                Some(&signature.atomic_numbers),
            )
            .map_err(|_| CatalogServerError::InvalidScientificConfiguration)?;
        if descriptor.values().len() != validator.descriptor_dim {
            return Err(CatalogServerError::InvalidScientificConfiguration);
        }
        BasinCensus::new(validator.descriptor_dim, census_radius)
            .map_err(|_| CatalogServerError::InvalidScientificConfiguration)?;
        BasinCatalog::new(catalog_capacity, census_radius, total_charged_work)
            .map_err(|_| CatalogServerError::InvalidScientificConfiguration)?;
        self.scientific = Some(ScientificConfig {
            signature,
            descriptor_space,
            validator,
            catalog_capacity,
            census_radius,
            total_charged_work,
            evaluate: Arc::new(evaluate),
        });
        Ok(self)
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
    /// Scientific signature, descriptor, validator, or catalog settings disagree.
    #[error("catalog scientific configuration is inconsistent")]
    InvalidScientificConfiguration,
    /// Listener setup failed.
    #[error("catalog server I/O failed: {0}")]
    Io(#[from] std::io::Error),
}

struct ScientificState {
    signature: SystemSignature,
    descriptor_space: DescriptorSpace,
    validator: CandidateValidator,
    census: BasinCensus,
    catalog: BasinCatalog,
    evaluate: Arc<FreshEvaluator>,
}

struct CoordinatorState {
    snapshot_version: u64,
    census_visits: u64,
    active_entries: u32,
    requests: BTreeMap<(u32, u64), CatalogRequest>,
    maximum_sequence: BTreeMap<u32, u64>,
    scientific: Option<ScientificState>,
}

impl CoordinatorState {
    fn new(config: &ServerConfig) -> Result<Self, CatalogServerError> {
        let scientific = config
            .scientific
            .as_ref()
            .map(|scientific| {
                Ok::<ScientificState, CatalogServerError>(ScientificState {
                    signature: scientific.signature.clone(),
                    descriptor_space: scientific.descriptor_space.clone(),
                    validator: CandidateValidator::new(
                        scientific.signature.clone(),
                        scientific.validator.clone(),
                    ),
                    census: BasinCensus::new(
                        scientific.validator.descriptor_dim,
                        scientific.census_radius,
                    )
                    .map_err(|_| CatalogServerError::InvalidScientificConfiguration)?,
                    catalog: BasinCatalog::new(
                        scientific.catalog_capacity,
                        scientific.census_radius,
                        scientific.total_charged_work,
                    )
                    .map_err(|_| CatalogServerError::InvalidScientificConfiguration)?,
                    evaluate: Arc::clone(&scientific.evaluate),
                })
            })
            .transpose()?;
        Ok(Self {
            snapshot_version: 0,
            census_visits: 0,
            active_entries: 0,
            requests: BTreeMap::new(),
            maximum_sequence: BTreeMap::new(),
            scientific,
        })
    }
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
        let state = Arc::new(Mutex::new(CoordinatorState::new(&config)?));
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
            Err(error) => {
                let event_sequence = root.get_event_sequence();
                let state = match state.lock() {
                    Ok(state) => state,
                    Err(poisoned) => poisoned.into_inner(),
                };
                write_reply(
                    &mut stream,
                    rejected(&state, event_sequence, rejection_for_protocol_error(&error)),
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
    let mut state = match state.lock() {
        Ok(state) => state,
        Err(poisoned) => poisoned.into_inner(),
    };
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
        CatalogOperation::RecordVisit { candidate } => {
            let census_visits = if let Some(scientific) = state.scientific.as_mut() {
                let Ok(validated) = validate_candidate(
                    scientific,
                    &request.identity,
                    request.event_sequence,
                    candidate,
                ) else {
                    return rejected(
                        &state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                let Ok(observation) = scientific.census.observe(&validated.candidate.descriptor)
                else {
                    return rejected(
                        &state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                observation.total_visits
            } else {
                let Some(census_visits) = state.census_visits.checked_add(1) else {
                    return rejected(
                        &state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                census_visits
            };
            let Some(snapshot_version) = state.snapshot_version.checked_add(1) else {
                return rejected(
                    &state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            state.census_visits = census_visits;
            state.snapshot_version = snapshot_version;
        }
        CatalogOperation::OfferCandidate { candidate } => {
            let (census_visits, active_entries) = if let Some(scientific) =
                state.scientific.as_mut()
            {
                let Ok(validated) = validate_candidate(
                    scientific,
                    &request.identity,
                    request.event_sequence,
                    candidate,
                ) else {
                    return rejected(
                        &state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                let Ok(observation) = scientific.census.observe(&validated.candidate.descriptor)
                else {
                    return rejected(
                        &state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                scientific
                    .catalog
                    .admit(observation.basin_id, observation.basin_visits, validated);
                (
                    observation.total_visits,
                    u32::try_from(scientific.catalog.len())
                        .expect("catalog capacity is checked against u32"),
                )
            } else {
                let Some(active_entries) = state.active_entries.checked_add(1) else {
                    return rejected(
                        &state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                let Some(census_visits) = state.census_visits.checked_add(1) else {
                    return rejected(
                        &state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                (census_visits, active_entries)
            };
            let Some(snapshot_version) = state.snapshot_version.checked_add(1) else {
                return rejected(
                    &state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            state.census_visits = census_visits;
            state.active_entries = active_entries;
            state.snapshot_version = snapshot_version;
        }
        CatalogOperation::LedgerEvent { .. } => {
            let Some(snapshot_version) = state.snapshot_version.checked_add(1) else {
                return rejected(
                    &state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            state.snapshot_version = snapshot_version;
        }
    }
    state
        .maximum_sequence
        .insert(request.identity.replica, request.event_sequence);
    state.requests.insert(key, request);
    accepted(&state, key.1, false)
}

fn rejection_for_protocol_error(error: &ProtocolError) -> ProtocolRejection {
    match error {
        ProtocolError::UnsupportedVersion { .. } => ProtocolRejection::UnsupportedVersion,
        ProtocolError::CampaignMismatch => ProtocolRejection::CampaignMismatch,
        ProtocolError::EnsembleMismatch => ProtocolRejection::EnsembleMismatch,
        ProtocolError::ReplicaMismatch => ProtocolRejection::ReplicaMismatch,
        ProtocolError::SignatureMismatch => ProtocolRejection::SignatureMismatch,
        ProtocolError::SignatureDigestLength { .. } | ProtocolError::Malformed(_) => {
            ProtocolRejection::Malformed
        }
    }
}

fn validate_candidate(
    scientific: &ScientificState,
    identity: &CatalogIdentity,
    event_sequence: u64,
    candidate: &CatalogCandidate,
) -> Result<ValidatedCandidate, ()> {
    if candidate.producer_replica != identity.replica
        || candidate.event_sequence != event_sequence
        || candidate.descriptor_schema_version != scientific.signature.descriptor.version
    {
        return Err(());
    }
    let recomputed = scientific
        .descriptor_space
        .describe(
            ArrayView1::from(&candidate.coordinates),
            Some(&scientific.signature.atomic_numbers),
        )
        .map_err(|_| ())?;
    if recomputed.values().len() != candidate.descriptor.len()
        || recomputed
            .values()
            .iter()
            .zip(&candidate.descriptor)
            .any(|(expected, actual)| (expected - actual).abs() > 1e-12)
    {
        return Err(());
    }
    let record = CandidateRecord {
        signature: scientific.signature.clone(),
        producer_replica: candidate.producer_replica,
        coordinates: candidate.coordinates.clone(),
        cell: candidate.cell,
        energy: candidate.energy,
        forces: candidate.forces.clone(),
        gradient_norm: candidate.gradient_norm,
        descriptor: candidate.descriptor.clone(),
        descriptor_schema_version: candidate.descriptor_schema_version,
        quench_status: if candidate.quench_converged {
            QuenchStatus::Converged
        } else {
            QuenchStatus::Unconverged
        },
        charged_work: candidate.charged_work,
        event_sequence: candidate.event_sequence,
        seed: candidate.seed,
    };
    scientific
        .validator
        .validate(&record, |coordinates| (scientific.evaluate)(coordinates))
        .map_err(|_| ())
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
