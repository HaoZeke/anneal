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
    AcceptedPayload, AcceptedReply, BoundaryCrossingRecord, CatalogCandidate, CatalogIdentity,
    CatalogMutation, CatalogMutationKind, CatalogOperation, CatalogRelation, CatalogReply,
    CatalogRequest, CatalogSnapshot, DescriptorHoleProposal, PolicyState, PopulationEpochState,
    PopulationPlan, ProtocolError, ProtocolRejection, TransitionDestination, decode_request,
    decode_request_reader, encode_reply, encode_request,
};
use crate::Catalog_capnp::catalog_request;
use crate::catalog::{
    AdmissionOutcome, AdmissionRejection, BasinCatalog, BasinCensus, BasinId, CandidateRecord,
    CandidateValidator, FreshEvaluation, QuenchStatus, SystemSignature, ValidatedCandidate,
    ValidatorConfig, euclidean_gradient_norm,
};
use crate::catalog_policy::proposal::farthest_hole;
use crate::cooperative_search::ledger::{ChargeKind, CooperativeLedger, ReplicaLedgerEvent};
use crate::descriptor_space::DescriptorSpace;
use crate::methods::feynman_kac::{
    EpochSubmissionOutcome, PopulationMember, SelectionCoefficients, SynchronousPopulation,
};
use crate::region_assignment::{RegionCandidate, RegionUtility, diversity_constrained_assignment};
use crate::transition_graph::{AttractionRegionConfig, TransitionGraph, TransitionOutcome};

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
    transition_graph: TransitionGraph,
    transition_nodes: BTreeMap<BasinId, usize>,
    last_basin_by_replica: BTreeMap<u32, BasinId>,
    last_candidate_by_replica: BTreeMap<u32, CatalogCandidate>,
    boundary_crossings: Vec<BoundaryCrossingRecord>,
    transition_capacity: usize,
    population: SynchronousPopulation,
    population_candidates: BTreeMap<u64, BTreeMap<u32, CatalogCandidate>>,
    population_plans: BTreeMap<u64, PopulationPlan>,
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
                    transition_graph: TransitionGraph::new(),
                    transition_nodes: BTreeMap::new(),
                    last_basin_by_replica: BTreeMap::new(),
                    last_candidate_by_replica: BTreeMap::new(),
                    boundary_crossings: Vec::new(),
                    transition_capacity: scientific.catalog_capacity,
                    population: SynchronousPopulation::new(
                        config.replicas.iter().copied(),
                        SelectionCoefficients::default(),
                        config.replicas.len().div_ceil(2),
                        u64::from_le_bytes(
                            config.signature_digest[..8]
                                .try_into()
                                .expect("signature prefix has eight bytes"),
                        ),
                    )
                    .map_err(|_| CatalogServerError::InvalidScientificConfiguration)?,
                    population_candidates: BTreeMap::new(),
                    population_plans: BTreeMap::new(),
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

const JOURNAL_FILE: &str = "catalog-requests-v5.bin";
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
        && state.scientific.as_ref().is_none_or(|scientific| {
            scientific.transition_nodes.is_empty()
                && scientific.last_basin_by_replica.is_empty()
                && scientific.population_candidates.is_empty()
                && scientific.population_plans.is_empty()
        })
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
    // Cap'n Proto framing expects a blocking stream. Listener polling remains
    // nonblocking, while every accepted connection waits for its next frame.
    stream
        .set_nonblocking(false)
        .map_err(|error| error.to_string())?;
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
        return rejected(state, request.event_sequence, reason);
    }
    let key = (request.identity.replica, request.event_sequence);
    if let Some((stored, payload)) = state.requests.get(&key) {
        return if stored == &request {
            accepted_with_payload(state, request.event_sequence, true, payload.clone())
        } else {
            rejected(
                state,
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
            state,
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
                let entry = &scientific.catalog.entries()[index];
                payload = AcceptedPayload::Candidate(candidate_from_validated(
                    entry.validated(),
                    Some(entry.census_id()),
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
                    state,
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
                    state,
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
                    state,
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
        CatalogOperation::BoundaryCrossing { current, draw } => {
            let Some(scientific) = state.scientific.as_ref() else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            if let Some(crossing) = sample_boundary_crossing(scientific, current, *draw) {
                payload = AcceptedPayload::BoundaryCrossing(crossing);
            }
        }
        CatalogOperation::PolicyState { descriptor, energy } => {
            let Some(scientific) = state.scientific.as_ref() else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            if !energy.is_finite() {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            }
            let Ok(local_basin) = scientific.census.basin_for(descriptor) else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let local_basin_visits = local_basin
                .and_then(|id| scientific.census.entry(id))
                .map_or(0, |entry| entry.visits());
            let local_basin_distance = local_basin
                .and_then(|id| scientific.census.entry(id))
                .map_or(0.0, |entry| descriptor_distance(descriptor, entry.medoid()));
            let novelty = local_basin.map_or_else(
                || nearest_census_distance(&scientific.census, descriptor).unwrap_or(0.0),
                |id| nearest_other_census_distance(&scientific.census, id, descriptor),
            );
            let transition_uncertainty =
                local_basin.map_or_else(|| 1.0, |id| transition_uncertainty(scientific, id));
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
                local_basin: local_basin.map(BasinId::as_raw),
                local_basin_distance,
                novelty,
                transition_uncertainty,
            });
        }
        CatalogOperation::PopulationSubmit { epoch, candidate } => {
            let Some(scientific) = state.scientific.as_mut() else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let Ok(validated) = validate_candidate(scientific, &request.identity, candidate) else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let Ok(Some(basin_id)) = scientific.census.basin_for(&validated.candidate.descriptor)
            else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let basin_visits = scientific
                .census
                .entry(basin_id)
                .expect("classified census basin exists")
                .visits();
            let novelty = nearest_other_census_distance(
                &scientific.census,
                basin_id,
                &validated.candidate.descriptor,
            );
            let canonical = candidate_from_validated(&validated, Some(basin_id));
            let epoch_candidates = scientific.population_candidates.entry(*epoch).or_default();
            let inserted = match epoch_candidates.get(&request.identity.replica) {
                Some(stored) if stored == &canonical => false,
                Some(_) => {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                }
                None => {
                    epoch_candidates.insert(request.identity.replica, canonical);
                    true
                }
            };
            let residual_uncertainty = transition_uncertainty(scientific, basin_id);
            let Ok(member) = PopulationMember::new_with_uncertainty(
                request.identity.replica,
                validated.fresh.energy,
                novelty,
                basin_visits as f64,
                residual_uncertainty,
            ) else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let Ok(outcome) = scientific.population.submit(*epoch, member) else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let required = u32::try_from(config.replicas.len())
                .expect("replica count is bounded by the protocol");
            payload = match outcome {
                EpochSubmissionOutcome::Pending { submitted, .. } => {
                    AcceptedPayload::PopulationEpoch(PopulationEpochState {
                        epoch: *epoch,
                        submitted: u32::try_from(submitted)
                            .expect("submission count is bounded by replica count"),
                        required,
                        plan: None,
                    })
                }
                EpochSubmissionOutcome::Ready(plan) => {
                    let source_candidates = scientific
                        .population_candidates
                        .get(epoch)
                        .expect("complete epoch retains every source candidate");
                    let (parents, weights) = region_population_assignment(
                        scientific,
                        plan.destinations(),
                        source_candidates,
                        config.replicas.len().div_ceil(2),
                    )
                    .unwrap_or_else(|| (plan.parents().to_vec(), plan.weights().to_vec()));
                    let parent_candidates = parents
                        .iter()
                        .map(|parent| {
                            source_candidates
                                .get(parent)
                                .expect("parent replica submitted a validated candidate")
                                .clone()
                        })
                        .collect::<Vec<_>>();
                    let diagnostics = population_diagnostics(&weights, &parents, &config.replicas);
                    let wire_plan = PopulationPlan {
                        epoch: plan.epoch(),
                        destinations: plan.destinations().to_vec(),
                        parents,
                        weights,
                        effective_sample_size: diagnostics.0,
                        unique_parents: u32::try_from(diagnostics.1)
                            .expect("unique parents are bounded by replica count"),
                        max_family_size: u32::try_from(diagnostics.2)
                            .expect("family size is bounded by replica count"),
                        offspring_variance: diagnostics.3,
                        parent_candidates,
                    };
                    scientific
                        .population_plans
                        .insert(*epoch, wire_plan.clone());
                    AcceptedPayload::PopulationEpoch(PopulationEpochState {
                        epoch: *epoch,
                        submitted: required,
                        required,
                        plan: Some(wire_plan),
                    })
                }
            };
            if inserted {
                let Some(snapshot_version) = state.snapshot_version.checked_add(1) else {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                state.snapshot_version = snapshot_version;
            }
        }
        CatalogOperation::PopulationPlan { epoch } => {
            let Some(scientific) = state.scientific.as_ref() else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let required = u32::try_from(config.replicas.len())
                .expect("replica count is bounded by the protocol");
            if let Some(plan) = scientific.population_plans.get(epoch) {
                payload = AcceptedPayload::PopulationEpoch(PopulationEpochState {
                    epoch: *epoch,
                    submitted: required,
                    required,
                    plan: Some(plan.clone()),
                });
            } else if *epoch == scientific.population.open_epoch() {
                let submitted = scientific
                    .population_candidates
                    .get(epoch)
                    .map_or(0, BTreeMap::len);
                payload = AcceptedPayload::PopulationEpoch(PopulationEpochState {
                    epoch: *epoch,
                    submitted: u32::try_from(submitted)
                        .expect("submission count is bounded by replica count"),
                    required,
                    plan: None,
                });
            } else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            }
        }
        CatalogOperation::RecordVisit { candidate } => {
            let census_visits = if let Some(scientific) = state.scientific.as_mut() {
                let Ok(validated) = validate_candidate(scientific, &request.identity, candidate)
                else {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                let Ok(observation) = scientific.census.observe(&validated.candidate.descriptor)
                else {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                scientific
                    .last_basin_by_replica
                    .insert(request.identity.replica, observation.basin_id);
                scientific.last_candidate_by_replica.insert(
                    request.identity.replica,
                    candidate_from_validated(&validated, Some(observation.basin_id)),
                );
                let _ = transition_node(scientific, observation.basin_id);
                observation.total_visits
            } else {
                let Some(census_visits) = state.census_visits.checked_add(1) else {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                census_visits
            };
            let Some(snapshot_version) = state.snapshot_version.checked_add(1) else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            state.census_visits = census_visits;
            state.snapshot_version = snapshot_version;
        }
        CatalogOperation::OfferCandidate { candidate } => {
            let (census_visits, active_entries, mutation) = if let Some(scientific) =
                state.scientific.as_mut()
            {
                let Ok(validated) = validate_candidate(scientific, &request.identity, candidate)
                else {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                let Ok(observation) = scientific.census.observe(&validated.candidate.descriptor)
                else {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                let outcome = scientific.catalog.admit(
                    observation.basin_id,
                    observation.basin_visits,
                    validated,
                );
                let incumbent = scientific
                    .catalog
                    .incumbent()
                    .map(|entry| entry.census_id());
                (
                    observation.total_visits,
                    u32::try_from(scientific.catalog.len())
                        .expect("catalog capacity is checked against u32"),
                    Some(catalog_mutation(outcome, observation.basin_id, incumbent)),
                )
            } else {
                let Some(active_entries) = state.active_entries.checked_add(1) else {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                let Some(census_visits) = state.census_visits.checked_add(1) else {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                (census_visits, active_entries, None)
            };
            let Some(snapshot_version) = state.snapshot_version.checked_add(1) else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            state.census_visits = census_visits;
            state.active_entries = active_entries;
            state.snapshot_version = snapshot_version;
            if let Some(mutation) = mutation {
                payload = AcceptedPayload::CatalogMutation(mutation);
            }
        }
        CatalogOperation::RecordTransition {
            action,
            destination,
            adopted,
        } => {
            let Some(scientific) = state.scientific.as_mut() else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            if action.is_empty() {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            }
            let Some(source_basin) = scientific
                .last_basin_by_replica
                .get(&request.identity.replica)
                .copied()
            else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let Some(source_node) = transition_node(scientific, source_basin) else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let source_candidate = scientific
                .last_candidate_by_replica
                .get(&request.identity.replica)
                .cloned();
            let outcome = match destination {
                TransitionDestination::Unresolved => TransitionOutcome::Unresolved,
                TransitionDestination::Resolved(candidate) => {
                    let Ok(validated) =
                        validate_candidate(scientific, &request.identity, candidate)
                    else {
                        return rejected(
                            state,
                            request.event_sequence,
                            ProtocolRejection::ValidationRejected,
                        );
                    };
                    let Ok(observation) =
                        scientific.census.observe(&validated.candidate.descriptor)
                    else {
                        return rejected(
                            state,
                            request.event_sequence,
                            ProtocolRejection::ValidationRejected,
                        );
                    };
                    let Some(destination_node) = transition_node(scientific, observation.basin_id)
                    else {
                        return rejected(
                            state,
                            request.event_sequence,
                            ProtocolRejection::ValidationRejected,
                        );
                    };
                    if *adopted {
                        scientific
                            .last_basin_by_replica
                            .insert(request.identity.replica, observation.basin_id);
                        let destination_candidate =
                            candidate_from_validated(&validated, Some(observation.basin_id));
                        scientific.last_candidate_by_replica.insert(
                            request.identity.replica,
                            destination_candidate.clone(),
                        );
                        if scientific.transition_capacity > 0
                            && source_basin != observation.basin_id
                            && let Some(source_candidate) = source_candidate.as_ref()
                        {
                            if scientific.boundary_crossings.len()
                                == scientific.transition_capacity
                            {
                                scientific.boundary_crossings.remove(0);
                            }
                            scientific.boundary_crossings.push(BoundaryCrossingRecord {
                                action: action.clone(),
                                from: source_candidate.coordinates.clone(),
                                to: destination_candidate.coordinates,
                                source_basin: source_basin.as_raw(),
                                destination_basin: observation.basin_id.as_raw(),
                            });
                        }
                    }
                    state.census_visits = observation.total_visits;
                    TransitionOutcome::Resolved(destination_node)
                }
            };
            if scientific
                .transition_graph
                .observe(action.clone(), source_node, outcome)
                .is_err()
            {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            }
            let Some(snapshot_version) = state.snapshot_version.checked_add(1) else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            state.snapshot_version = snapshot_version;
        }
        CatalogOperation::LedgerEvent {
            kind,
            charged_calls,
            cumulative_charged,
        } => {
            let Some(kind) = ChargeKind::from_wire_code(*kind) else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let Some(ledger) = state.ledger.as_mut() else {
                return rejected(
                    state,
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
                    state,
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
                    state,
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
    accepted_with_payload(state, key.1, false, payload)
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
        || candidate.census_basin.is_some()
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

fn candidate_from_validated(
    validated: &ValidatedCandidate,
    census_basin: Option<BasinId>,
) -> CatalogCandidate {
    CatalogCandidate {
        producer_replica: validated.candidate.producer_replica,
        coordinates: validated.candidate.coordinates.clone(),
        cell: validated.candidate.cell,
        energy: validated.fresh.energy,
        forces: validated.fresh.forces.clone(),
        gradient_norm: euclidean_gradient_norm(&validated.fresh.forces),
        descriptor: validated.candidate.descriptor.clone(),
        descriptor_schema_version: validated.candidate.descriptor_schema_version,
        quench_converged: true,
        charged_work: validated.candidate.charged_work,
        event_sequence: validated.candidate.event_sequence,
        seed: validated.candidate.seed,
        census_basin: census_basin.map(BasinId::as_raw),
    }
}

fn catalog_mutation(
    outcome: AdmissionOutcome,
    offered_basin: BasinId,
    incumbent_basin: Option<BasinId>,
) -> CatalogMutation {
    let (kind, evicted) = match outcome {
        AdmissionOutcome::Added { .. } => (CatalogMutationKind::Added, Vec::new()),
        AdmissionOutcome::ReplacedSameBasin { .. } => {
            (CatalogMutationKind::ReplacedSameBasin, Vec::new())
        }
        AdmissionOutcome::ReplacedConflicts { evicted, .. } => (
            CatalogMutationKind::ReplacedConflicts,
            evicted.into_iter().map(BasinId::as_raw).collect(),
        ),
        AdmissionOutcome::ReplacedCapacity { evicted, .. } => (
            CatalogMutationKind::ReplacedCapacity,
            vec![evicted.as_raw()],
        ),
        AdmissionOutcome::Rejected {
            reason: AdmissionRejection::SameBasinNotLower,
        } => (CatalogMutationKind::RejectedSameBasin, Vec::new()),
        AdmissionOutcome::Rejected {
            reason: AdmissionRejection::ConflictNotLower,
        } => (CatalogMutationKind::RejectedConflict, Vec::new()),
        AdmissionOutcome::Rejected {
            reason: AdmissionRejection::CapacityNotLower,
        } => (CatalogMutationKind::RejectedCapacity, Vec::new()),
    };
    CatalogMutation {
        basin_id: offered_basin.as_raw(),
        kind,
        evicted,
        incumbent_basin: incumbent_basin.map(BasinId::as_raw),
    }
}

fn region_population_assignment(
    scientific: &ScientificState,
    destinations: &[u32],
    source_candidates: &BTreeMap<u32, CatalogCandidate>,
    max_family_size: usize,
) -> Option<(Vec<u32>, Vec<f64>)> {
    let region_config = AttractionRegionConfig {
        probe_action: "probe".into(),
        concentration: 0.5,
        diffusion_steps: 2,
        maximum_distance: 0.35,
        minimum_probes: 8,
    };
    let regions = scientific
        .transition_graph
        .attraction_regions(&region_config)
        .ok()?;
    let mut node_region = vec![usize::MAX; scientific.transition_graph.node_count()];
    for (region, nodes) in regions.iter().enumerate() {
        for node in nodes {
            node_region[*node] = region;
        }
    }
    let mut fallback_regions = BTreeMap::new();
    let mut next_region = regions.len();
    let source_regions = destinations
        .iter()
        .map(|replica| {
            let candidate = source_candidates.get(replica)?;
            let basin = BasinId::from_raw(candidate.census_basin?);
            if let Some(node) = scientific.transition_nodes.get(&basin)
                && node_region.get(*node).copied().unwrap_or(usize::MAX) != usize::MAX
            {
                return Some(node_region[*node]);
            }
            Some(*fallback_regions.entry(basin).or_insert_with(|| {
                let region = next_region;
                next_region += 1;
                region
            }))
        })
        .collect::<Option<Vec<_>>>()?;
    let mut occupancy = BTreeMap::<usize, usize>::new();
    for region in &source_regions {
        *occupancy.entry(*region).or_default() += 1;
    }
    let probe = scientific
        .transition_graph
        .posterior_matrix("probe", 0.5)
        .ok()?;
    let candidates = destinations
        .iter()
        .enumerate()
        .map(|(index, replica)| {
            let region = source_regions[index];
            let basin = BasinId::from_raw(source_candidates.get(replica)?.census_basin?);
            let node = scientific.transition_nodes.get(&basin).copied();
            let transition_uncertainty = node
                .and_then(|node| scientific.transition_graph.uncertainty("probe", node, 0.5))
                .unwrap_or(1.0);
            let outgoing_frontier = node.map_or(0.0, |source| {
                (0..scientific.transition_graph.node_count())
                    .filter(|destination| {
                        node_region.get(*destination).copied().unwrap_or(usize::MAX) != region
                    })
                    .map(|destination| probe[[source, destination]])
                    .sum::<f64>()
            });
            RegionCandidate::new(
                *replica,
                region,
                RegionUtility {
                    transition_uncertainty,
                    inverse_occupancy: 1.0 / occupancy[&region] as f64,
                    outgoing_frontier,
                    geometry_compatibility: 1.0,
                    access_cost: 0.0,
                },
            )
            .ok()
        })
        .collect::<Option<Vec<_>>>()?;
    let parents =
        diversity_constrained_assignment(&candidates, destinations.len(), max_family_size).ok()?;
    let scores = candidates
        .iter()
        .map(|candidate| candidate.score())
        .collect::<Vec<_>>();
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut weights = scores
        .iter()
        .map(|score| (score - maximum).clamp(-4.0, 0.0).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>();
    for weight in &mut weights {
        *weight /= total;
    }
    Some((parents, weights))
}

fn sample_boundary_crossing(
    scientific: &ScientificState,
    current: &[f64],
    draw: u64,
) -> Option<BoundaryCrossingRecord> {
    let query_basin = scientific.census.basin_for(current).ok().flatten()?;
    let query_node = *scientific.transition_nodes.get(&query_basin)?;
    let regions = scientific
        .transition_graph
        .attraction_regions(&AttractionRegionConfig {
            probe_action: "probe".into(),
            concentration: 0.5,
            diffusion_steps: 2,
            maximum_distance: 0.35,
            minimum_probes: 8,
        })
        .ok()?;
    let mut node_region = vec![usize::MAX; scientific.transition_graph.node_count()];
    for (region, nodes) in regions.iter().enumerate() {
        for node in nodes {
            node_region[*node] = region;
        }
    }
    let query_region = *node_region.get(query_node)?;
    if query_region == usize::MAX {
        return None;
    }
    let eligible = scientific
        .boundary_crossings
        .iter()
        .filter(|crossing| {
            let source = BasinId::from_raw(crossing.source_basin);
            let destination = BasinId::from_raw(crossing.destination_basin);
            let Some(source_node) = scientific.transition_nodes.get(&source).copied() else {
                return false;
            };
            let Some(destination_node) = scientific.transition_nodes.get(&destination).copied()
            else {
                return false;
            };
            node_region.get(source_node).copied() == Some(query_region)
                && node_region.get(destination_node).copied() != Some(query_region)
        })
        .collect::<Vec<_>>();
    if eligible.is_empty() {
        return None;
    }
    let index = usize::try_from(draw % eligible.len() as u64).ok()?;
    Some(eligible[index].clone())
}

fn population_diagnostics(
    weights: &[f64],
    parents: &[u32],
    replicas: &BTreeSet<u32>,
) -> (f64, usize, usize, f64) {
    let effective_sample_size = 1.0 / weights.iter().map(|weight| weight * weight).sum::<f64>();
    let mut counts = BTreeMap::<u32, usize>::new();
    for parent in parents {
        *counts.entry(*parent).or_default() += 1;
    }
    let unique_parents = counts.len();
    let max_family_size = counts.values().copied().max().unwrap_or(0);
    let mean = parents.len() as f64 / replicas.len() as f64;
    let offspring_variance = replicas
        .iter()
        .map(|replica| {
            let difference = counts.get(replica).copied().unwrap_or(0) as f64 - mean;
            difference * difference
        })
        .sum::<f64>()
        / replicas.len() as f64;
    (
        effective_sample_size,
        unique_parents,
        max_family_size,
        offspring_variance,
    )
}

fn transition_node(scientific: &mut ScientificState, basin: BasinId) -> Option<usize> {
    if let Some(node) = scientific.transition_nodes.get(&basin) {
        return Some(*node);
    }
    if scientific.transition_nodes.len() >= scientific.transition_capacity {
        return None;
    }
    let node = scientific.transition_nodes.len();
    scientific.transition_nodes.insert(basin, node);
    Some(node)
}

fn transition_uncertainty(scientific: &ScientificState, basin: BasinId) -> f64 {
    scientific.transition_nodes.get(&basin).map_or_else(
        || 1.0,
        |node| {
            scientific
                .transition_graph
                .uncertainty("probe", *node, 0.5)
                .unwrap_or(1.0)
        },
    )
}

fn nearest_other_census_distance(census: &BasinCensus, local: BasinId, descriptor: &[f64]) -> f64 {
    census
        .entries()
        .iter()
        .filter(|entry| entry.id() != local)
        .map(|entry| descriptor_distance(descriptor, entry.medoid()))
        .fold(None, |nearest, distance| {
            Some(nearest.map_or(distance, |current: f64| current.min(distance)))
        })
        .unwrap_or(0.0)
}

fn nearest_census_distance(census: &BasinCensus, descriptor: &[f64]) -> Option<f64> {
    census
        .entries()
        .iter()
        .map(|entry| descriptor_distance(descriptor, entry.medoid()))
        .reduce(f64::min)
}

fn descriptor_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let delta = left - right;
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
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
