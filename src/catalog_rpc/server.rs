//! Serialized coordinator for one campaign ensemble and system signature.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use capnp::message::ReaderOptions;
use capnp::serialize;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;

use super::{
    AcceptedPayload, AcceptedReply, CatalogCandidate, CatalogIdentity, CatalogOperation,
    CatalogRelation, CatalogReply, CatalogRequest, CatalogSnapshot, DescriptorHoleProposal,
    PolicyState, ProtocolError, ProtocolRejection, decode_request, decode_request_reader,
    encode_reply, encode_request,
};
use crate::Catalog_capnp::catalog_request;
use crate::catalog::{
    BasinCatalog, BasinCensus, CandidateRecord, CandidateValidator, FreshEvaluation, QuenchStatus,
    SystemSignature, ValidatedCandidate, ValidatorConfig,
};
use crate::catalog_policy::proposal::farthest_hole;
use crate::cooperative_search::ledger::{ChargeKind, CooperativeLedger, ReplicaLedgerEvent};
use crate::descriptor_space::DescriptorSpace;

type FreshEvaluator = dyn Fn(&[f64]) -> Result<FreshEvaluation, String> + Send + Sync;

/// Immutable identity and allowed replicas for one coordinator.
#[derive(Clone)]
pub struct ServerConfig {
    campaign: String,
    ensemble: String,
    signature_digest: [u8; 32],
    replicas: BTreeSet<u32>,
    scientific: Option<ScientificConfig>,
    per_replica_budget: Option<u64>,
    state_directory: Option<PathBuf>,
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
            per_replica_budget: None,
            state_directory: None,
        })
    }

    /// Attach an equal charged-work budget for every configured replica.
    pub fn with_ledger_budget(
        mut self,
        per_replica_budget: u64,
    ) -> Result<Self, CatalogServerError> {
        CooperativeLedger::new(self.replicas.iter().copied(), per_replica_budget)
            .map_err(|_| CatalogServerError::InvalidConfiguration)?;
        self.per_replica_budget = Some(per_replica_budget);
        Ok(self)
    }

    /// Persist accepted requests under one isolated ensemble directory.
    pub fn with_state_directory(
        mut self,
        directory: impl Into<PathBuf>,
    ) -> Result<Self, CatalogServerError> {
        let directory = directory.into();
        if directory.as_os_str().is_empty() {
            return Err(CatalogServerError::InvalidConfiguration);
        }
        self.state_directory = Some(directory);
        Ok(self)
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
        let replica_count = u64::try_from(self.replicas.len())
            .map_err(|_| CatalogServerError::InvalidScientificConfiguration)?;
        if !total_charged_work.is_multiple_of(replica_count) {
            return Err(CatalogServerError::InvalidScientificConfiguration);
        }
        self = self
            .with_ledger_budget(total_charged_work / replica_count)
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
    /// A durable request journal is truncated, corrupt, or inconsistent.
    #[error("catalog request journal is invalid: {0}")]
    InvalidJournal(String),
}

#[derive(Clone)]
struct ScientificState {
    signature: SystemSignature,
    descriptor_space: DescriptorSpace,
    validator: CandidateValidator,
    census: BasinCensus,
    catalog: BasinCatalog,
    evaluate: Arc<FreshEvaluator>,
}

#[derive(Clone)]
struct CoordinatorState {
    snapshot_version: u64,
    census_visits: u64,
    active_entries: u32,
    requests: BTreeMap<(u32, u64), (CatalogRequest, AcceptedPayload)>,
    maximum_sequence: BTreeMap<u32, u64>,
    ledger: Option<CooperativeLedger>,
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
        let ledger = config
            .per_replica_budget
            .map(|budget| CooperativeLedger::new(config.replicas.iter().copied(), budget))
            .transpose()
            .map_err(|_| CatalogServerError::InvalidConfiguration)?;
        Ok(Self {
            snapshot_version: 0,
            census_visits: 0,
            active_entries: 0,
            requests: BTreeMap::new(),
            maximum_sequence: BTreeMap::new(),
            ledger,
            scientific,
        })
    }
}

const JOURNAL_FILE: &str = "catalog-requests-v4.bin";
const MAX_JOURNAL_FRAME: usize = 256 * 1024 * 1024;

fn journal_path(config: &ServerConfig) -> Option<PathBuf> {
    config
        .state_directory
        .as_ref()
        .map(|directory| directory.join(JOURNAL_FILE))
}

fn append_journal(
    config: &ServerConfig,
    request: &CatalogRequest,
) -> Result<(), CatalogServerError> {
    let Some(path) = journal_path(config) else {
        return Ok(());
    };
    let directory = path
        .parent()
        .ok_or_else(|| CatalogServerError::InvalidJournal("journal path has no parent".into()))?;
    fs::create_dir_all(directory)?;
    let bytes = encode_request(request)
        .map_err(|error| CatalogServerError::InvalidJournal(error.to_string()))?;
    let length = u64::try_from(bytes.len())
        .map_err(|_| CatalogServerError::InvalidJournal("journal frame is too large".into()))?;
    let mut journal = OpenOptions::new().create(true).append(true).open(path)?;
    journal.write_all(&length.to_le_bytes())?;
    journal.write_all(&bytes)?;
    journal.flush()?;
    Ok(())
}

fn replay_journal(
    config: &ServerConfig,
    state: &mut CoordinatorState,
) -> Result<(), CatalogServerError> {
    let Some(path) = journal_path(config) else {
        return Ok(());
    };
    fs::create_dir_all(
        path.parent().ok_or_else(|| {
            CatalogServerError::InvalidJournal("journal path has no parent".into())
        })?,
    )?;
    if !Path::new(&path).exists() {
        return Ok(());
    }
    let mut journal = File::open(path)?;
    loop {
        let mut length_bytes = [0u8; 8];
        let first = journal.read(&mut length_bytes)?;
        if first == 0 {
            return Ok(());
        }
        if first != length_bytes.len() {
            return Err(CatalogServerError::InvalidJournal(
                "truncated frame length".into(),
            ));
        }
        let length = usize::try_from(u64::from_le_bytes(length_bytes))
            .map_err(|_| CatalogServerError::InvalidJournal("frame length overflow".into()))?;
        if length > MAX_JOURNAL_FRAME {
            return Err(CatalogServerError::InvalidJournal(format!(
                "frame length {length} exceeds the journal limit"
            )));
        }
        let mut bytes = vec![0u8; length];
        journal.read_exact(&mut bytes).map_err(|error| {
            CatalogServerError::InvalidJournal(format!("truncated request frame: {error}"))
        })?;
        let request = decode_request(&bytes)
            .map_err(|error| CatalogServerError::InvalidJournal(error.to_string()))?;
        if !matches!(
            apply_request(config, state, request),
            CatalogReply::Accepted(AcceptedReply {
                duplicate: false,
                ..
            })
        ) {
            return Err(CatalogServerError::InvalidJournal(
                "persisted request cannot be replayed".into(),
            ));
        }
    }
}

fn state_is_empty(state: &CoordinatorState) -> bool {
    state.snapshot_version == 0
        && state.census_visits == 0
        && state.active_entries == 0
        && state.requests.is_empty()
        && state.maximum_sequence.is_empty()
        && state
            .ledger
            .as_ref()
            .is_none_or(|ledger| ledger.ensemble_total() == 0 && ledger.event_count() == 0)
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
        let mut initial_state = CoordinatorState::new(&config)?;
        replay_journal(&config, &mut initial_state)?;
        let listener = TcpListener::bind(addr)?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;
        let header = ServerHeader {
            campaign: config.campaign.clone(),
            ensemble: config.ensemble.clone(),
            replicas: config.replicas.iter().copied().collect(),
            initial_snapshot_version: initial_state.snapshot_version,
            empty_state_proof: state_is_empty(&initial_state),
        };
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let state = Arc::new(Mutex::new(initial_state));
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
        let reply = process_request(config, &state, request)?;
        write_reply(&mut stream, reply)?;
    }
}

fn process_request(
    config: &ServerConfig,
    state: &Arc<Mutex<CoordinatorState>>,
    request: CatalogRequest,
) -> Result<CatalogReply, String> {
    let mut state = match state.lock() {
        Ok(state) => state,
        Err(poisoned) => poisoned.into_inner(),
    };
    let mut next = state.clone();
    let reply = apply_request(config, &mut next, request.clone());
    if matches!(
        reply,
        CatalogReply::Accepted(AcceptedReply {
            duplicate: false,
            ..
        })
    ) {
        append_journal(config, &request).map_err(|error| error.to_string())?;
        *state = next;
    }
    Ok(reply)
}

fn apply_request(
    config: &ServerConfig,
    state: &mut CoordinatorState,
    request: CatalogRequest,
) -> CatalogReply {
    let rejection = identity_rejection(config, &request.identity).or_else(|| {
        (request.snapshot_version > state.snapshot_version)
            .then_some(ProtocolRejection::SnapshotRegression)
    });
    if let Some(reason) = rejection {
        return rejected(&state, request.event_sequence, reason);
    }
    let key = (request.identity.replica, request.event_sequence);
    if let Some((stored, payload)) = state.requests.get(&key) {
        return if stored == &request {
            accepted_with_payload(&state, request.event_sequence, true, payload.clone())
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
    let mut payload = AcceptedPayload::None;
    match &request.operation {
        CatalogOperation::Snapshot => {}
        CatalogOperation::Sample { draw } => {
            if let Some(scientific) = state.scientific.as_ref()
                && !scientific.catalog.is_empty()
            {
                let index = usize::try_from(*draw % scientific.catalog.len() as u64)
                    .expect("sample index is bounded by catalog length");
                payload = AcceptedPayload::Candidate(candidate_from_validated(
                    scientific.catalog.entries()[index].validated(),
                ));
            }
        }
        CatalogOperation::DescriptorHole {
            current,
            samples,
            draw,
        } => {
            let Some(scientific) = state.scientific.as_ref() else {
                return rejected(
                    &state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let catalog = scientific
                .catalog
                .entries()
                .iter()
                .map(|entry| Array1::from_vec(entry.descriptor().to_vec()))
                .collect::<Vec<_>>();
            let Ok(sample_count) = usize::try_from(*samples) else {
                return rejected(
                    &state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let mut rng = rand::rngs::StdRng::seed_from_u64(*draw);
            let Ok(hole) = farthest_hole(
                &Array1::from_vec(current.clone()),
                &catalog,
                sample_count,
                &mut rng,
            ) else {
                return rejected(
                    &state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            payload = AcceptedPayload::DescriptorHole(DescriptorHoleProposal {
                target: hole.target().to_vec(),
                increment: hole.increment().to_vec(),
                nearest_catalog_distance: hole.nearest_catalog_distance(),
            });
        }
        CatalogOperation::PolicyState { descriptor, energy } => {
            let Some(scientific) = state.scientific.as_ref() else {
                return rejected(
                    &state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            if !energy.is_finite() {
                return rejected(
                    &state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            }
            let Ok(local_basin) = scientific.census.basin_for(descriptor) else {
                return rejected(
                    &state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let local_basin_visits = local_basin
                .and_then(|id| scientific.census.entry(id))
                .map_or(0, |entry| entry.visits());
            let relation = match scientific.catalog.incumbent() {
                None => CatalogRelation::Empty,
                Some(incumbent) if local_basin.is_some_and(|id| id == incumbent.census_id()) => {
                    CatalogRelation::Incumbent
                }
                Some(_) if local_basin.is_some_and(|id| scientific.catalog.entry(id).is_some()) => {
                    CatalogRelation::SameBasin
                }
                Some(incumbent) if incumbent.energy() < *energy => {
                    CatalogRelation::UnrelatedLowerAnchor
                }
                Some(_) => CatalogRelation::UnrelatedNoAnchor,
            };
            payload = AcceptedPayload::PolicyState(PolicyState {
                total_visits: scientific.census.total_visits(),
                singleton_basins: scientific.census.singleton_count(),
                local_basin_visits,
                globally_saturated: scientific.census.is_saturated(),
                relation,
                aggregate_charged: state
                    .ledger
                    .as_ref()
                    .map_or(0, CooperativeLedger::ensemble_total),
                aggregate_budget: state
                    .ledger
                    .as_ref()
                    .map_or(0, CooperativeLedger::aggregate_budget),
            });
        }
        CatalogOperation::RecordVisit { candidate } => {
            let census_visits = if let Some(scientific) = state.scientific.as_mut() {
                let Ok(validated) = validate_candidate(scientific, &request.identity, candidate)
                else {
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
                let Ok(validated) = validate_candidate(scientific, &request.identity, candidate)
                else {
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
        CatalogOperation::LedgerEvent {
            kind,
            charged_calls,
            cumulative_charged,
        } => {
            let Some(kind) = ChargeKind::from_wire_code(*kind) else {
                return rejected(
                    &state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let Some(ledger) = state.ledger.as_mut() else {
                return rejected(
                    &state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            if ledger
                .record(ReplicaLedgerEvent {
                    replica: request.identity.replica,
                    sequence: request.event_sequence,
                    kind,
                    charged_calls: *charged_calls,
                    cumulative_charged: *cumulative_charged,
                })
                .is_err()
            {
                return rejected(
                    &state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            }
            let aggregate_charged = ledger.ensemble_total();
            if let Some(scientific) = state.scientific.as_mut() {
                scientific.catalog.update_threshold(aggregate_charged);
            }
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
    state.requests.insert(key, (request, payload.clone()));
    accepted_with_payload(&state, key.1, false, payload)
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
    candidate: &CatalogCandidate,
) -> Result<ValidatedCandidate, ()> {
    if candidate.producer_replica != identity.replica
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

fn candidate_from_validated(validated: &ValidatedCandidate) -> CatalogCandidate {
    CatalogCandidate {
        producer_replica: validated.candidate.producer_replica,
        coordinates: validated.candidate.coordinates.clone(),
        cell: validated.candidate.cell,
        energy: validated.fresh.energy,
        forces: validated.fresh.forces.clone(),
        gradient_norm: validated
            .fresh
            .forces
            .iter()
            .map(|force| force * force)
            .sum::<f64>()
            .sqrt(),
        descriptor: validated.candidate.descriptor.clone(),
        descriptor_schema_version: validated.candidate.descriptor_schema_version,
        quench_converged: true,
        charged_work: validated.candidate.charged_work,
        event_sequence: validated.candidate.event_sequence,
        seed: validated.candidate.seed,
    }
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

fn accepted_with_payload(
    state: &CoordinatorState,
    event_sequence: u64,
    duplicate: bool,
    payload: AcceptedPayload,
) -> CatalogReply {
    CatalogReply::Accepted(AcceptedReply {
        event_sequence,
        duplicate,
        snapshot: CatalogSnapshot {
            version: state.snapshot_version,
            census_visits: state.census_visits,
            active_entries: state.active_entries,
            aggregate_charged: state
                .ledger
                .as_ref()
                .map_or(0, CooperativeLedger::ensemble_total),
            aggregate_budget: state
                .ledger
                .as_ref()
                .map_or(0, CooperativeLedger::aggregate_budget),
        },
        payload,
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
