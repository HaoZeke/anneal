//! Serialized coordinator for one campaign ensemble and system signature.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
    CatalogRequest, CatalogRideConnection, CatalogRideOutcome, CatalogRideSaddleEvidence,
    CatalogRideWork, CatalogSnapshot, DescriptorHoleProposal, PolicyState, PopulationEpochState,
    PopulationPlan, PopulationSelection, ProtocolError, ProtocolRejection, TransitionDestination,
    decode_request, decode_request_reader, encode_reply, encode_request,
};
use crate::Catalog_capnp::catalog_request;
use crate::catalog::{
    AdmissionOutcome, AdmissionRejection, Archive, AttractorStrength, BasinCatalog, BasinCensus,
    BasinId, CHAMPION_RANK, CandidateRecord, CandidateValidator, CensusObservation, Curiosity,
    DEFAULT_MIN_OCCUPIED_FAMILIES, FreshEvaluation, GoodTuringSample, INTERFACE_HORIZON,
    InterfaceSeat, MixingEvidence, PRODUCTION_MINIMUM_VISITS, PackingBook, PackingRole,
    QuenchStatus, REDUCTION_FACTOR, SystemSignature, ValidatedCandidate, ValidatorConfig,
    WalkRecord, euclidean_gradient_norm, explore_must_leave, invert_mixing,
    leftover_dwell_from_census, leftover_esty_stable, leftover_esty_upper, leftover_lambda,
    occupancy_ei_exhausted, occupancy_family_floor, occupancy_fes_delta, occupancy_landfold_split,
    occupancy_min_families, occupancy_ring_profile, occupancy_ring_split,
    occupancy_sparsify_packing, occupant_rhat, packing_role, promote_one_sided, prune,
    retis_exchange_adjacent, same_packing, seat_extras,
};
use crate::catalog_policy::proposal::farthest_hole;
use crate::cooperative_search::ledger::{ChargeKind, CooperativeLedger, ReplicaLedgerEvent};
use crate::descriptor_space::{DescriptorSpace, UNIVERSAL_LOCAL_ENVIRONMENT_RADIUS};
use crate::discovery_roster::{
    DiscoveryCoverage, DiscoveryEffort, assign_discovery_roles, coverage_allocation_weight,
};
use crate::methods::feynman_kac::{
    EpochSubmissionOutcome, PackingOccupant, PopulationEpochPlan, PopulationMember,
    SelectionCoefficients, SynchronousPopulation, assign_parents_by_packing,
};
use crate::methods::landscape_graph::LandscapeGraph;
use crate::methods::neus_bridge::{BridgeString, EntryLists, WeightLedger};
use crate::pes_exploration::{
    ExactStructureWitness, PesExplorationConfig, PesSurface, RideMethod, StationaryIndex,
    StructureContext, StructureView, stationary_index_cartesian,
};
use crate::region_assignment::{RegionCandidate, RegionUtility, diversity_constrained_assignment};
use crate::ride_ledger::{
    EnvironmentBook, RideLedger, RideOutcome, RidePortfolio, RideSource,
    SADDLE_COVERAGE_MINIMUM_OBSERVATIONS,
};
use crate::soap::local_nu3_z;
use crate::transition_graph::{AttractionRegionConfig, TransitionGraph, TransitionOutcome};

type FreshEvaluator = dyn Fn(&[f64]) -> Result<FreshEvaluation, String> + Send + Sync;
type StructuralWitness = dyn ExactStructureWitness + Send + Sync;

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
    attraction_regions: AttractionRegionConfig,
    evaluate: Arc<FreshEvaluator>,
    exact_witness: Option<Arc<StructuralWitness>>,
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
            attraction_regions: AttractionRegionConfig {
                probe_action: "probe".into(),
                concentration: 0.5,
                diffusion_steps: 2,
                maximum_distance: 0.35,
                minimum_probes: 8,
            },
            evaluate: Arc::new(evaluate),
            exact_witness: None,
        });
        Ok(self)
    }

    /// Bind the symmetry-aware witness that makes final basin-identity decisions.
    pub fn with_exact_structure_witness<W>(mut self, witness: W) -> Result<Self, CatalogServerError>
    where
        W: ExactStructureWitness + Send + Sync + 'static,
    {
        let Some(scientific) = self.scientific.as_mut() else {
            return Err(CatalogServerError::InvalidScientificConfiguration);
        };
        scientific.exact_witness = Some(Arc::new(witness));
        Ok(self)
    }

    /// Configure the fixed-probe posterior used to define attraction regions.
    pub fn with_attraction_region_config(
        mut self,
        config: AttractionRegionConfig,
    ) -> Result<Self, CatalogServerError> {
        TransitionGraph::new()
            .attraction_regions(&config)
            .map_err(|_| CatalogServerError::InvalidScientificConfiguration)?;
        let Some(scientific) = self.scientific.as_mut() else {
            return Err(CatalogServerError::InvalidScientificConfiguration);
        };
        scientific.attraction_regions = config;
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

#[derive(Clone, Copy, PartialEq, Eq)]
struct OccupancyGtKey {
    leftover_n: u64,
    leftover_n1: u64,
    packing_n: u64,
    packing_n1: u64,
    families: u32,
    min_families: u32,
    landfold_floor: usize,
    ring_floor: usize,
    fes_delta_bits: Option<u64>,
    stop: bool,
    communities: usize,
    holes: bool,
    fes_minima: usize,
}

/// Shared frontier ladder capacity. A ring: newest posts displace the
/// oldest, so staleness is bounded by churn without wall clocks.
const FRONTIER_POOL_CAP: usize = 256;

#[derive(Clone)]
struct CertifiedRideSaddle {
    candidate: CatalogCandidate,
    lowest_curvature: f64,
    lowest_mode: Vec<f64>,
    negative_modes: usize,
    source_basins: BTreeSet<u64>,
}

#[derive(Clone)]
struct ScientificState {
    signature: SystemSignature,
    descriptor_space: DescriptorSpace,
    structure_context: StructureContext,
    exact_witness: Arc<StructuralWitness>,
    validator: CandidateValidator,
    census: BasinCensus,
    catalog: BasinCatalog,
    transition_graph: TransitionGraph,
    attraction_regions: AttractionRegionConfig,
    transition_nodes: BTreeMap<BasinId, usize>,
    landscape: LandscapeGraph,
    bridges: BTreeMap<u64, BridgeServerState>,
    next_bridge: u64,
    last_basin_by_replica: BTreeMap<u32, BasinId>,
    last_candidate_by_replica: BTreeMap<u32, CatalogCandidate>,
    best_candidate_by_replica: BTreeMap<u32, CatalogCandidate>,
    boundary_crossings: Vec<BoundaryCrossingRecord>,
    transition_capacity: usize,
    population: SynchronousPopulation,
    population_candidates: BTreeMap<u64, BTreeMap<u32, CatalogCandidate>>,
    population_plans: BTreeMap<u64, PopulationPlan>,
    packing: PackingBook,
    ride_environments: EnvironmentBook,
    ride_ledger: RideLedger,
    ride_candidates: BTreeMap<u64, CatalogCandidate>,
    ride_saddles: BTreeMap<u64, CertifiedRideSaddle>,
    next_ride_saddle: u64,
    discovery_replicas: Vec<u32>,
    energy_history: BTreeMap<u32, VecDeque<f64>>,
    family_history: BTreeMap<u32, VecDeque<f64>>,
    trial_hops: BTreeMap<u32, u64>,
    pending_reseed: BTreeSet<u32>,
    leftover_lambda_by_replica: BTreeMap<u32, f64>,
    interface_seat_by_replica: BTreeMap<u32, crate::catalog::InterfaceSeat>,
    leftover_arrivals: BTreeMap<u64, u64>,
    /// Basin each replica last arrived in, for crediting one well
    /// visit per arrival. Kept apart from `last_basin_by_replica`,
    /// which names the source a recorded transition departs from.
    arrival_basin_by_replica: BTreeMap<u32, BasinId>,
    /// Per-family curiosity, in the MAP-Elites sense: a family is paid
    /// when a replica sent there produced a candidate the catalog kept
    /// and charged when it did not, so effort follows what has worked
    /// rather than whichever cell happens to be emptiest.
    curiosity: Curiosity,
    /// Family each replica was last handed a representative of, so the
    /// reward can be attributed to the cell the start came from.
    drawn_from_by_replica: BTreeMap<u32, usize>,
    /// Archive of descriptor cells at an annealed radius: conformational
    /// space annealing's Dcut rule doing the niching, and
    /// return-then-explore choosing which cell a Leave goes back to.
    archive: Archive,
    /// Fraction of the ensemble budget spent, which is what the archive
    /// radius anneals against.
    archive_progress: f64,
    /// Folded book kept beside the version it was folded from.
    ///
    /// Single linkage over the cells is quadratic in their number and the
    /// policy response asks for the fold four times, so a coordinator
    /// serving 48 replicas spends its core folding a book that has not
    /// changed. Recomputed only when the book moves.
    sparsified: Option<(u64, crate::catalog::OccupancyBookMap)>,
    /// EI verdict kept beside the FunnelModel version it was read from.
    /// The sweep predicts at every observed site and is cubic in their
    /// number; perf on a live 48-replica coordinator put
    /// FunnelModel::predict at 74 percent of its cycles.
    ei_verdict: Option<(u64, bool)>,
    /// Worthwhile-community count beside the book and funnel versions it
    /// was computed from. The count clones the folded map and every
    /// occupied histogram and runs an EI predict per community; perf put
    /// that at a fifth of the coordinator's cycles once the EI sweep
    /// itself was cached.
    worthwhile: Option<(u64, u64, usize)>,
    /// Sizes of the candidate and catalog stores the funnel was last fed
    /// from. Feeding it walks every candidate and every catalog entry and
    /// clones a histogram per family into a map; perf put that map's clone
    /// and drop at a quarter of the coordinator's cycles, with the
    /// allocator behind it at two fifths. The observations are idempotent,
    /// so while the stores have not grown there is nothing to feed.
    fed_from: Option<(usize, usize, usize)>,
    /// Landfold split beside the book version it was folded from. The
    /// split clones every occupied histogram and runs a Torgerson MDS,
    /// and occupancy_floor asks for it once per policy request.
    landfold: Option<(u64, (usize, usize, usize))>,
    last_gt_report: Option<OccupancyGtKey>,
    /// Leftover occupancy sample whose saturation state has been counted
    /// toward the retirement dwell.
    last_leftover_dwell_sample: Option<(u64, u64, u64)>,
    leftover_sat_streak: u32,
    leftover_dwell: bool,
    funnel: crate::funnel_bo::FunnelModel,
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
    /// Raw frontier excursion posts, shared across every replica.
    frontier: std::collections::VecDeque<crate::catalog_rpc::CatalogFrontierPost>,
    /// A journal append failed after the live state had already moved,
    /// so a replay of the log no longer reproduces this coordinator.
    journal_broken: bool,
}

impl CoordinatorState {
    fn new(config: &ServerConfig) -> Result<Self, CatalogServerError> {
        let scientific = config
            .scientific
            .as_ref()
            .map(|scientific| {
                let exact_witness = scientific
                    .exact_witness
                    .as_ref()
                    .ok_or(CatalogServerError::InvalidScientificConfiguration)?;
                Ok::<ScientificState, CatalogServerError>(ScientificState {
                    signature: scientific.signature.clone(),
                    descriptor_space: scientific.descriptor_space.clone(),
                    structure_context: StructureContext::new(
                        Some(scientific.signature.atomic_numbers.clone()),
                        scientific.descriptor_space.geometry(),
                        Some(format!("{:02x?}", scientific.signature.digest())),
                    ),
                    exact_witness: Arc::clone(exact_witness),
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
                    attraction_regions: scientific.attraction_regions.clone(),
                    transition_nodes: BTreeMap::new(),
                    landscape: LandscapeGraph::new(),
                    bridges: BTreeMap::new(),
                    next_bridge: 0,
                    last_basin_by_replica: BTreeMap::new(),
                    last_candidate_by_replica: BTreeMap::new(),
                    best_candidate_by_replica: BTreeMap::new(),
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
                    packing: PackingBook::default(),
                    ride_environments: EnvironmentBook::new(
                        if scientific.descriptor_space.geometry().is_some() {
                            UNIVERSAL_LOCAL_ENVIRONMENT_RADIUS
                        } else {
                            crate::catalog::packing::ENVIRONMENT_RADIUS
                        },
                    )
                    .map_err(|_| CatalogServerError::InvalidScientificConfiguration)?,
                    ride_ledger: RideLedger::new(
                        RidePortfolio::new(2, vec![RideMethod::Dimer, RideMethod::Lanczos])
                            .map_err(|_| CatalogServerError::InvalidScientificConfiguration)?,
                    ),
                    ride_candidates: BTreeMap::new(),
                    ride_saddles: BTreeMap::new(),
                    next_ride_saddle: 0,
                    discovery_replicas: config.replicas.iter().copied().collect(),
                    energy_history: BTreeMap::new(),
                    family_history: BTreeMap::new(),
                    trial_hops: BTreeMap::new(),
                    pending_reseed: BTreeSet::new(),
                    leftover_lambda_by_replica: BTreeMap::new(),
                    interface_seat_by_replica: BTreeMap::new(),
                    leftover_arrivals: BTreeMap::new(),
                    arrival_basin_by_replica: BTreeMap::new(),
                    curiosity: Curiosity::default(),
                    drawn_from_by_replica: BTreeMap::new(),
                    archive: Archive::new(scientific.census_radius.max(1e-6), ARCHIVE_RADIUS_FLOOR),
                    archive_progress: 0.0,
                    sparsified: None,
                    ei_verdict: None,
                    worthwhile: None,
                    fed_from: None,
                    landfold: None,
                    last_gt_report: None,
                    last_leftover_dwell_sample: None,
                    leftover_sat_streak: 0,
                    leftover_dwell: false,
                    funnel: crate::funnel_bo::FunnelModel::new(0.15, 20.0, 1e-2),
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
            frontier: std::collections::VecDeque::new(),
            journal_broken: false,
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
            apply_request(config, state, request, None),
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
        // The expensive part of validation -- re-deriving the engine's own
        // energy and descriptor for the posted coordinates -- has no
        // dependency on anything CoordinatorState holds mutably: it is a
        // pure function of the candidate and the run's fixed validator and
        // evaluate closure. Cloning those two out under a lock this thread
        // holds only for the clone, then calling the closure with no lock
        // held at all, is what turns 48 chains serialized behind one core
        // back into 48 chains actually running concurrently. The commit
        // that follows still locks and still sees the live, current state;
        // only the CPU-bound recompute moves off the lock.
        let precomputed = candidate_needing_validation(&request.operation)
            .map(|candidate| precompute_validation(&state, &request.identity, candidate));
        let reply = process_request(config, &state, request, precomputed)?;
        write_reply(&mut stream, reply)?;
    }
}

/// The candidate one request's operation will send through
/// [`validate_candidate`], if any -- the set of operations whose
/// expensive descriptor recomputation this thread can do before ever
/// taking the coordinator's lock for the request itself.
fn candidate_needing_validation(operation: &CatalogOperation) -> Option<&CatalogCandidate> {
    match operation {
        CatalogOperation::PopulationSubmit { candidate, .. }
        | CatalogOperation::RecordVisit { candidate }
        | CatalogOperation::OfferCandidate { candidate } => Some(candidate),
        CatalogOperation::RecordTransition {
            destination: TransitionDestination::Resolved(candidate),
            ..
        } => Some(candidate),
        _ => None,
    }
}

/// Clone what validation needs under a brief lock, then run the
/// expensive evaluation and the admission math with the lock released.
/// `CandidateValidator`, the system signature, and the evaluate closure
/// are all fixed for the run's lifetime, so nothing here can observe a
/// state change another replica makes while this thread holds no lock,
/// and the caller's later commit step still reads and writes the
/// current shared state, not a stale copy of it.
fn precompute_validation(
    state: &Arc<Mutex<CoordinatorState>>,
    identity: &CatalogIdentity,
    candidate: &CatalogCandidate,
) -> Result<ValidatedCandidate, ()> {
    let (signature, descriptor_space, validator, evaluate) = {
        let locked = match state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        let scientific = locked.scientific.as_ref().ok_or(())?;
        (
            scientific.signature.clone(),
            scientific.descriptor_space.clone(),
            scientific.validator.clone(),
            Arc::clone(&scientific.evaluate),
        )
    };
    validate_candidate(
        &signature,
        &descriptor_space,
        &validator,
        evaluate.as_ref(),
        identity,
        candidate,
    )
}

fn process_request(
    config: &ServerConfig,
    state: &Arc<Mutex<CoordinatorState>>,
    request: CatalogRequest,
    mut precomputed: Option<Result<ValidatedCandidate, ()>>,
) -> Result<CatalogReply, String> {
    let mut state = match state.lock() {
        Ok(state) => state,
        Err(poisoned) => poisoned.into_inner(),
    };
    if state.journal_broken {
        return Err("catalog request journal is behind the coordinator state".to_owned());
    }
    if matches!(request.operation, CatalogOperation::ObserverStatus) {
        // Observation is not history: the reply is assembled from the live
        // state, never journaled, and never advances a replica's sequence.
        // An observer need not occupy a replica slot, but it must identify the
        // same system as the coordinator so live state cannot cross PESes.
        if let Some(reason) = system_identity_rejection(config, &request.identity) {
            return Ok(rejected(&state, request.event_sequence, reason));
        }
        return Ok(observer_status_reply(
            config,
            &state,
            request.event_sequence,
        ));
    }
    // Identity before the replay cache. The cache is keyed by replica
    // and sequence alone, so a caller from another campaign or ensemble
    // that happens to share a replica id collides with a stored request
    // and is told its sequence replayed rather than that it is talking
    // to the wrong coordinator.
    if let Some(reason) = identity_rejection(config, &request.identity) {
        return Ok(rejected(&state, request.event_sequence, reason));
    }
    let key = (request.identity.replica, request.event_sequence);
    if let Some((stored, payload)) = state.requests.get(&key) {
        return Ok(if stored == &request {
            accepted_with_payload(&state, request.event_sequence, true, payload.clone())
        } else {
            rejected(
                &state,
                request.event_sequence,
                ProtocolRejection::SequenceReplay,
            )
        });
    }
    // Every operation applies straight to the live state now, not to a
    // clone taken up front and conditionally swapped in. That clone used
    // to deep-copy the whole scientific state -- catalog, census,
    // packing book, archive, every per-replica history -- on every
    // single accepted request, paid once per request and growing with
    // the run, for a guarantee only the disk-failure case ever used: if
    // the journal append below fails, there is no live state to roll
    // back to, so the coordinator marks itself broken and stops serving
    // instead of answering from a state its own log cannot reproduce.
    // PolicyState took exactly this trade first, because it is the
    // request every replica sends on every checkpoint; every other
    // operation now takes the same trade for the same reason.
    let reply = apply_request(config, &mut state, request.clone(), precomputed);
    if matches!(
        reply,
        CatalogReply::Accepted(AcceptedReply {
            duplicate: false,
            ..
        })
    ) && let Err(error) = append_journal(config, &request)
    {
        state.journal_broken = true;
        return Err(error.to_string());
    }
    Ok(reply)
}

fn apply_request(
    config: &ServerConfig,
    state: &mut CoordinatorState,
    request: CatalogRequest,
    mut precomputed: Option<Result<ValidatedCandidate, ()>>,
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
        // Answered in process_request from the live state; a status that
        // reaches this dispatch has bypassed it, which apply_request treats
        // as a plain snapshot rather than a scientific operation.
        CatalogOperation::ObserverStatus => {}
        CatalogOperation::Snapshot => {}
        CatalogOperation::BridgeAssignment { draw } => {
            if let Some(scientific) = state.scientific.as_mut()
                && let Some((&bridge_id, bridge)) = scientific.bridges.iter_mut().next_back()
            {
                let replica = request.identity.replica;
                let region = match bridge.assignments.get(&replica) {
                    Some(&region) => region,
                    None => {
                        // Fewest-launched interior region; endpoints
                        // belong to the minima, not to walkers.
                        let interior = 1..bridge.string.regions() - 1;
                        let region = interior
                            .min_by_key(|&r| bridge.ledger.launches(r))
                            .unwrap_or(1);
                        bridge.assignments.insert(replica, region);
                        region
                    }
                };
                if bridge.ledger.launch(region).is_err() {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                }
                let entry = bridge
                    .entries
                    .draw(region, *draw)
                    .map(|state| state.to_vec());
                let images = bridge
                    .string
                    .images()
                    .iter()
                    .flat_map(|image| image.iter().copied())
                    .collect();
                payload =
                    AcceptedPayload::BridgeAssignment(crate::catalog_rpc::BridgeAssignmentRecord {
                        bridge: bridge_id,
                        from_basin: bridge.from_basin,
                        to_basin: bridge.to_basin,
                        images,
                        image_count: u32::try_from(bridge.string.regions())
                            .expect("bridge images are bounded by a constant"),
                        region: u32::try_from(region)
                            .expect("bridge region is bounded by image count"),
                        tube_radius: bridge.tube_radius,
                        entry,
                    });
            }
        }
        CatalogOperation::BridgeCrossing { crossing } => {
            let Some(scientific) = state.scientific.as_mut() else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let Some(bridge) = scientific.bridges.get_mut(&crossing.bridge) else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let from = crossing.from_region as usize;
            let to = crossing.to_region as usize;
            if bridge.ledger.crossing(from, to).is_err()
                || bridge
                    .entries
                    .push(to, ndarray::Array1::from(crossing.state.clone()))
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
        CatalogOperation::Sample { draw } => {
            if let Some(scientific) = state.scientific.as_mut()
                && !scientific.catalog.is_empty()
            {
                // u64::MAX is the incumbent draw: exploit a deeper
                // catalog representative rather than a random slot.
                let sparse = if *draw == crate::catalog_rpc::SPARSE_SAMPLE_DRAW {
                    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(
                        draw ^ u64::from(request.identity.replica),
                    );
                    let picked = if scientific.funnel.len() >= 3 {
                        q_ei_family_entry(scientific, request.identity.replica)
                    } else {
                        sparsest_family_entry(scientific, &mut rng)
                    };
                    if let Some((family, _)) = picked {
                        // Remember which cell this start came from, so
                        // the curiosity credit lands on it when the
                        // replica reports back.
                        scientific
                            .drawn_from_by_replica
                            .insert(request.identity.replica, family);
                    }
                    picked.map(|(_, slot)| slot)
                } else {
                    None
                };
                let entry = if let Some(slot) = sparse {
                    &scientific.catalog.entries()[slot]
                } else if *draw == crate::catalog_rpc::INCUMBENT_SAMPLE_DRAW
                    || *draw == crate::catalog_rpc::SPARSE_SAMPLE_DRAW
                {
                    scientific
                        .catalog
                        .incumbent()
                        .expect("nonempty catalog has an incumbent")
                } else {
                    let index = usize::try_from(*draw % scientific.catalog.len() as u64)
                        .expect("sample index is bounded by catalog length");
                    &scientific.catalog.entries()[index]
                };
                payload = AcceptedPayload::Candidate(candidate_from_validated(
                    entry.validated(),
                    Some(entry.census_id()),
                ));
            }
        }
        CatalogOperation::PostFrontier { post } => {
            // A post is banked, never validated as a minimum: it is a
            // live excursion state and the census must not see it. The
            // ring bounds staleness by churn.
            if post.energy.is_finite()
                && post.gap.is_finite()
                && post.gap > 0.0
                && !post.coordinates.is_empty()
            {
                if state.frontier.len() >= FRONTIER_POOL_CAP {
                    state.frontier.pop_front();
                }
                state.frontier.push_back(post.clone());
            }
        }
        CatalogOperation::DrawFrontier { draw } => {
            if !state.frontier.is_empty() {
                let index = usize::try_from(*draw % state.frontier.len() as u64)
                    .expect("frontier index is bounded by pool length");
                payload = AcceptedPayload::FrontierPost(state.frontier[index].clone());
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
        CatalogOperation::PolicyState {
            descriptor,
            energy,
            leftover_lambda,
        } => {
            let ensemble_total = state
                .ledger
                .as_ref()
                .map_or(0, CooperativeLedger::ensemble_total);
            let ensemble_budget = state
                .ledger
                .as_ref()
                .map_or(0, CooperativeLedger::aggregate_budget);
            let Some(scientific) = state.scientific.as_mut() else {
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
            let packing = replica_packing(scientific, request.identity.replica);
            let local_basin_visits = packing
                .as_ref()
                .and_then(|fp| scientific.packing.family_of(fp))
                .map_or_else(
                    || {
                        local_basin
                            .and_then(|id| scientific.census.entry(id))
                            .map_or(0, |entry| entry.visits())
                    },
                    |family| scientific.packing.visits(family),
                );
            let local_basin_distance = local_basin
                .and_then(|id| scientific.census.entry(id))
                .map_or(0.0, |entry| descriptor_distance(descriptor, entry.medoid()));
            let novelty = packing.as_ref().map_or_else(
                || {
                    local_basin.map_or_else(
                        || nearest_census_distance(&scientific.census, descriptor).unwrap_or(0.0),
                        |id| nearest_other_census_distance(&scientific.census, id, descriptor),
                    )
                },
                |fp| scientific.packing.novelty(fp),
            );
            let transition_uncertainty =
                local_basin.map_or_else(|| 1.0, |id| transition_uncertainty(scientific, id));
            let basin_unseen_mass_upper = coverage_allocation_weight(
                scientific.census.total_visits(),
                PRODUCTION_MINIMUM_VISITS,
                leftover_esty_upper(
                    scientific.census.total_visits(),
                    scientific.census.singleton_count(),
                    scientific.census.doubleton_count(),
                ),
            );
            let saddle_coverage = scientific.ride_ledger.saddle_coverage();
            let saddle_unseen_mass_upper = coverage_allocation_weight(
                saddle_coverage.observations,
                SADDLE_COVERAGE_MINIMUM_OBSERVATIONS,
                saddle_coverage.unseen_mass_upper,
            );
            let discovery_epoch = scientific
                .census
                .total_visits()
                .saturating_add(scientific.ride_ledger.completed_attempts());
            let basin_effort = scientific.ledger.charge_summary(ChargeKind::BasinEscape);
            let saddle_effort = scientific.ledger.charge_summary(ChargeKind::SaddleRide);
            let discovery_role = assign_discovery_roles(
                &scientific.discovery_replicas,
                DiscoveryCoverage {
                    basin_unseen_mass_upper,
                    saddle_unseen_mass_upper,
                    basin_effort: DiscoveryEffort {
                        observations: basin_effort.events,
                        charged_calls: basin_effort.charged_calls,
                    },
                    saddle_effort: DiscoveryEffort {
                        observations: saddle_effort.events,
                        charged_calls: saddle_effort.charged_calls,
                    },
                    ride_available: scientific.ride_ledger.has_claimable_work(),
                },
                discovery_epoch,
            )
            .ok()
            .and_then(|assignments| {
                assignments
                    .into_iter()
                    .find(|assignment| assignment.replica == request.identity.replica)
            })
            .map(|assignment| assignment.role);
            let Some(discovery_role) = discovery_role else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let mut mixing = mixing_from_state(scientific);
            mixing.pruned = hyperband_prune(scientific, request.identity.replica);
            let relation = packing_or_region_relation(
                scientific,
                request.identity.replica,
                local_basin,
                *energy,
                mixing,
            );
            if mixing.pruned {
                reset_trial(scientific, request.identity.replica);
            }
            if scientific
                .population
                .mark_live(request.identity.replica)
                .is_err()
            {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            }
            // Fold the book once for this response; the four consumers
            // below would otherwise each pay for it.
            // Packings an extra can actually be sent to: communities a walk
            // has arrived in. The raw community count reaches two at the
            // second cell, which is single linkage on a pair rather than on
            // a cloud, and it sends extras between icosahedral cells that
            // have simply not chained yet.
            let occupied_packing_communities = worthwhile_communities(scientific);
            let (seat, frame_lambda) = assign_leftover_interfaces(
                scientific,
                request.identity.replica,
                descriptor,
                *leftover_lambda,
                relation,
            );
            // The archive radius anneals against the ensemble's spend,
            // so it needs to know it. This is the only place the
            // coordinator sees both halves.
            if ensemble_budget > 0 {
                scientific.archive_progress = ensemble_total as f64 / ensemble_budget as f64;
            }
            payload = AcceptedPayload::PolicyState(PolicyState {
                total_visits: scientific.census.total_visits(),
                singleton_basins: scientific.census.singleton_count(),
                local_basin_visits,
                globally_saturated: scientific.census.is_saturated(),
                relation,
                aggregate_charged: ensemble_total,
                aggregate_budget: ensemble_budget,
                local_basin: local_basin.map(BasinId::as_raw),
                local_basin_distance,
                novelty,
                transition_uncertainty,
                explore_collapsed: mixing.explore_collapsed,
                certified_attractor: mixing.certified_attractor,
                pruned: mixing.pruned,
                leftover_lambda: frame_lambda,
                interface_rank: seat.rank,
                interface_threshold: seat.threshold,
                interface_count: scientific
                    .interface_seat_by_replica
                    .values()
                    .filter(|seat| seat.rank != CHAMPION_RANK)
                    .count() as u32,
                occupied_family_count: occupied_packing_communities as u32,
                packing_saturated: packing_census_saturated(scientific),
                leftover_dwell: leftover_census_dwell(scientific),
                ei_exhausted: occupancy_funnel_ei_exhausted(scientific),
                min_families: occupancy_floor(scientific) as u32,
                discovery_role,
                discovery_epoch,
                basin_unseen_mass_upper,
                saddle_unseen_mass_upper,
                saddle_coverage_saturated: saddle_coverage.saturated,
            });
            report_occupancy_gt(scientific);
        }
        CatalogOperation::PopulationSubmit { epoch, candidate } => {
            let Some(scientific) = state.scientific.as_mut() else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let Ok(validated) =
                resolve_validation(&mut precomputed, scientific, &request.identity, candidate)
            else {
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
            observe_packing(
                scientific,
                request.identity.replica,
                &validated.candidate.coordinates,
                None,
                false,
            );
            // The tessellation learns from the same structures the
            // packing book does, so its cells sit where the search has
            // been rather than where a box was drawn.
            observe_descriptor(scientific, &validated.candidate.descriptor);
            record_energy(scientific, request.identity.replica, validated.fresh.energy);
            let packing = scientific
                .packing
                .histogram(&validated.candidate.coordinates);
            let basin_visits = packing
                .as_ref()
                .and_then(|fp| scientific.packing.family_of(fp))
                .map_or_else(
                    || {
                        scientific
                            .census
                            .entry(basin_id)
                            .expect("classified census basin exists")
                            .visits()
                    },
                    |family| scientific.packing.visits(family),
                );
            let novelty = packing.as_ref().map_or_else(
                || {
                    nearest_other_census_distance(
                        &scientific.census,
                        basin_id,
                        &validated.candidate.descriptor,
                    )
                },
                |fp| scientific.packing.novelty(fp),
            );
            let canonical = candidate_from_validated(&validated, Some(basin_id));
            if observe_ride_source(scientific, &canonical).is_err() {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            }
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
            payload = match outcome {
                EpochSubmissionOutcome::Pending {
                    submitted,
                    required,
                    ..
                } => AcceptedPayload::PopulationEpoch(PopulationEpochState {
                    epoch: *epoch,
                    submitted: u32::try_from(submitted)
                        .expect("submission count is bounded by replica count"),
                    required: u32::try_from(required)
                        .expect("requirement is bounded by replica count"),
                    plan: None,
                }),
                EpochSubmissionOutcome::Ready(plan) => {
                    let participants = u32::try_from(plan.destinations().len())
                        .expect("participants are bounded by replica count");
                    realize_population_plan(scientific, config, *epoch, &plan, participants)
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
        CatalogOperation::PopulationJoin { epoch } => {
            let Some(scientific) = state.scientific.as_mut() else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            // Barrier membership is a lookup: the member is formed from the
            // best candidate this coordinator has already fresh-validated for
            // the replica, so nothing is re-shipped or re-validated at the
            // barrier and a replica whose current point is mid-hop can still
            // join with its best minimum. A retry reuses the entry already
            // recorded for this epoch, so an improved best between retries
            // cannot turn a repeat into a conflicting submission.
            let recorded = scientific
                .population_candidates
                .get(epoch)
                .and_then(|entries| entries.get(&request.identity.replica))
                .cloned();
            let Some(member_candidate) = recorded.or_else(|| {
                scientific
                    .best_candidate_by_replica
                    .get(&request.identity.replica)
                    .cloned()
            }) else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let Ok(Some(basin_id)) = scientific.census.basin_for(&member_candidate.descriptor)
            else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let packing = scientific.packing.histogram(&member_candidate.coordinates);
            let basin_visits = packing
                .as_ref()
                .and_then(|fp| scientific.packing.family_of(fp))
                .map_or_else(
                    || {
                        scientific
                            .census
                            .entry(basin_id)
                            .expect("classified census basin exists")
                            .visits()
                    },
                    |family| scientific.packing.visits(family),
                );
            let novelty = packing.as_ref().map_or_else(
                || {
                    nearest_other_census_distance(
                        &scientific.census,
                        basin_id,
                        &member_candidate.descriptor,
                    )
                },
                |fp| scientific.packing.novelty(fp),
            );
            let residual_uncertainty = transition_uncertainty(scientific, basin_id);
            let Ok(member) = PopulationMember::new_with_uncertainty(
                request.identity.replica,
                member_candidate.energy,
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
            scientific
                .population_candidates
                .entry(*epoch)
                .or_default()
                .entry(request.identity.replica)
                .or_insert(member_candidate);
            if scientific
                .population
                .mark_live(request.identity.replica)
                .is_err()
            {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            }
            let Ok(outcome) = scientific.population.submit(*epoch, member) else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            payload = match outcome {
                EpochSubmissionOutcome::Pending {
                    submitted,
                    required,
                    ..
                } => AcceptedPayload::PopulationEpoch(PopulationEpochState {
                    epoch: *epoch,
                    submitted: u32::try_from(submitted)
                        .expect("submission count is bounded by replica count"),
                    required: u32::try_from(required)
                        .expect("requirement is bounded by replica count"),
                    plan: None,
                }),
                EpochSubmissionOutcome::Ready(plan) => {
                    let participants = u32::try_from(plan.destinations().len())
                        .expect("participants are bounded by replica count");
                    realize_population_plan(scientific, config, *epoch, &plan, participants)
                }
            };
        }
        CatalogOperation::PopulationAbstain { epoch } => {
            let Some(scientific) = state.scientific.as_mut() else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let Ok(outcome) = scientific
                .population
                .abstain(*epoch, request.identity.replica)
            else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let _ = scientific.population.retire(request.identity.replica);
            payload = match outcome {
                EpochSubmissionOutcome::Pending {
                    submitted,
                    required,
                    ..
                } => AcceptedPayload::PopulationEpoch(PopulationEpochState {
                    epoch: *epoch,
                    submitted: u32::try_from(submitted)
                        .expect("submission count is bounded by replica count"),
                    required: u32::try_from(required)
                        .expect("requirement is bounded by replica count"),
                    plan: None,
                }),
                EpochSubmissionOutcome::Ready(plan) => {
                    let participants = u32::try_from(plan.destinations().len())
                        .expect("participants are bounded by replica count");
                    realize_population_plan(scientific, config, *epoch, &plan, participants)
                }
            };
        }
        CatalogOperation::PopulationPlan { epoch } => {
            let Some(scientific) = state.scientific.as_ref() else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            if let Some(plan) = scientific.population_plans.get(epoch) {
                // A completed epoch's population is its participants, which
                // abstentions may have made smaller than the configured
                // replica set; reporting the configured count here would fail
                // every reader's consistency check against the plan vectors.
                let participants = u32::try_from(plan.destinations.len())
                    .expect("participants are bounded by replica count");
                payload = AcceptedPayload::PopulationEpoch(PopulationEpochState {
                    epoch: *epoch,
                    submitted: participants,
                    required: participants,
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
                    required: u32::try_from(scientific.population.open_requirement())
                        .expect("requirement is bounded by replica count"),
                    plan: None,
                });
            } else if *epoch < scientific.population.open_epoch() {
                // Closed with no stored plan: every replica abstained from
                // it. Zero of zero is the vacant-close answer a poller needs
                // to advance past the epoch instead of wedging on it.
                payload = AcceptedPayload::PopulationEpoch(PopulationEpochState {
                    epoch: *epoch,
                    submitted: 0,
                    required: 0,
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
                let Ok(validated) =
                    resolve_validation(&mut precomputed, scientific, &request.identity, candidate)
                else {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                let Ok(observation) = observe_exact_basin(scientific, &validated) else {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                let previous = scientific
                    .last_basin_by_replica
                    .insert(request.identity.replica, observation.basin_id);
                // The census-visit stream is the referee's evidence: one
                // replica occupying basin A and then basin B is one
                // observed transition across their seam.
                scientific
                    .landscape
                    .observe_basin(observation.basin_id.as_raw());
                if let Some(previous) = previous
                    && previous != observation.basin_id
                {
                    scientific.landscape.observe_crossing(
                        previous.as_raw(),
                        observation.basin_id.as_raw(),
                        1.0,
                    );
                }
                let canonical = candidate_from_validated(&validated, Some(observation.basin_id));
                let deeper = scientific
                    .best_candidate_by_replica
                    .get(&request.identity.replica)
                    .is_none_or(|stored| canonical.energy < stored.energy);
                if deeper {
                    scientific
                        .best_candidate_by_replica
                        .insert(request.identity.replica, canonical.clone());
                }
                if observe_ride_source(scientific, &canonical).is_err() {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                }
                remember_candidate(scientific, request.identity.replica, canonical);
                let arrival = scientific
                    .arrival_basin_by_replica
                    .insert(request.identity.replica, observation.basin_id)
                    != Some(observation.basin_id);
                observe_packing(
                    scientific,
                    request.identity.replica,
                    &validated.candidate.coordinates,
                    Some(observation.basin_id),
                    arrival,
                );
                // The tessellation learns from the same structures the
                // packing book does, so its cells sit where the search has
                // been rather than where a box was drawn.
                observe_descriptor(scientific, &validated.candidate.descriptor);
                record_energy(scientific, request.identity.replica, validated.fresh.energy);
                if scientific
                    .population
                    .mark_live(request.identity.replica)
                    .is_err()
                {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                }
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
                let Ok(validated) =
                    resolve_validation(&mut precomputed, scientific, &request.identity, candidate)
                else {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                let Ok(observation) = observe_exact_basin(scientific, &validated) else {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                // An offer is validated exactly as a visit is, so it also
                // refreshes the best candidate the population join reads;
                // without this a replica whose only validated states arrive
                // as offers has nothing on file at every barrier.
                let canonical = candidate_from_validated(&validated, Some(observation.basin_id));
                let deeper = scientific
                    .best_candidate_by_replica
                    .get(&request.identity.replica)
                    .is_none_or(|stored| canonical.energy < stored.energy);
                if deeper {
                    scientific
                        .best_candidate_by_replica
                        .insert(request.identity.replica, canonical.clone());
                }
                if observe_ride_source(scientific, &canonical).is_err() {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                }
                remember_candidate(scientific, request.identity.replica, canonical);
                let arrival = scientific
                    .arrival_basin_by_replica
                    .insert(request.identity.replica, observation.basin_id)
                    != Some(observation.basin_id);
                observe_packing(
                    scientific,
                    request.identity.replica,
                    &validated.candidate.coordinates,
                    Some(observation.basin_id),
                    arrival,
                );
                // The tessellation learns from the same structures the
                // packing book does, so its cells sit where the search has
                // been rather than where a box was drawn.
                observe_descriptor(scientific, &validated.candidate.descriptor);
                record_energy(scientific, request.identity.replica, validated.fresh.energy);
                if scientific
                    .population
                    .mark_live(request.identity.replica)
                    .is_err()
                {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                }
                let outcome = scientific.catalog.admit(
                    observation.basin_id,
                    observation.basin_visits,
                    validated,
                );
                // Curiosity credit for the cell this replica's start was
                // drawn from. A start that led somewhere the catalog
                // kept makes that cell worth drawing from again; one
                // that led nowhere makes it less so, without ever
                // writing the cell off, because a descriptor can be
                // wrong where a cell is not.
                if let Some(family) = scientific
                    .drawn_from_by_replica
                    .get(&request.identity.replica)
                    .copied()
                {
                    scientific.curiosity.ensure(family + 1);
                    if matches!(outcome, AdmissionOutcome::Rejected { .. }) {
                        scientific.curiosity.penalise(family);
                        scientific.archive.penalise(family);
                    } else {
                        scientific.curiosity.reward(family);
                        scientific.archive.reward(family);
                    }
                }
                let incumbent = scientific
                    .catalog
                    .incumbent()
                    .map(|entry| entry.census_id());
                (
                    observation.total_visits,
                    u32::try_from(scientific.catalog.len())
                        .expect("catalog capacity is checked against u32"),
                    Some(catalog_mutation(
                        outcome,
                        observation.basin_id,
                        observation.created,
                        incumbent,
                    )),
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
                    let Ok(validated) = resolve_validation(
                        &mut precomputed,
                        scientific,
                        &request.identity,
                        candidate,
                    ) else {
                        return rejected(
                            state,
                            request.event_sequence,
                            ProtocolRejection::ValidationRejected,
                        );
                    };
                    let Ok(observation) = observe_exact_basin(scientific, &validated) else {
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
                    let destination_candidate =
                        candidate_from_validated(&validated, Some(observation.basin_id));
                    if observe_ride_source(scientific, &destination_candidate).is_err() {
                        return rejected(
                            state,
                            request.event_sequence,
                            ProtocolRejection::ValidationRejected,
                        );
                    }
                    if *adopted {
                        scientific
                            .last_basin_by_replica
                            .insert(request.identity.replica, observation.basin_id);
                        remember_candidate(
                            scientific,
                            request.identity.replica,
                            destination_candidate.clone(),
                        );
                        let arrival = scientific
                            .arrival_basin_by_replica
                            .insert(request.identity.replica, observation.basin_id)
                            != Some(observation.basin_id);
                        observe_packing(
                            scientific,
                            request.identity.replica,
                            &destination_candidate.coordinates,
                            Some(observation.basin_id),
                            arrival,
                        );
                        record_energy(
                            scientific,
                            request.identity.replica,
                            destination_candidate.energy,
                        );
                        if scientific.transition_capacity > 0
                            && source_basin != observation.basin_id
                            && let Some(source_candidate) = source_candidate.as_ref()
                        {
                            if scientific.boundary_crossings.len() == scientific.transition_capacity
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
        CatalogOperation::ClaimRide { seed } => {
            let Some(scientific) = state.scientific.as_mut() else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let mut ride_ledger = scientific.ride_ledger.clone();
            if let Some(order) = ride_ledger.claim(request.identity.replica, *seed) {
                let Some(source) = scientific
                    .ride_candidates
                    .get(&order.arm.source_basin)
                    .cloned()
                else {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                let avoid_saddles = scientific
                    .ride_saddles
                    .values()
                    .filter(|saddle| saddle.source_basins.contains(&order.arm.source_basin))
                    .map(|saddle| saddle.candidate.clone())
                    .collect();
                let Some(snapshot_version) = state.snapshot_version.checked_add(1) else {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                scientific.ride_ledger = ride_ledger;
                payload = AcceptedPayload::RideWork(CatalogRideWork {
                    order,
                    source,
                    avoid_saddles,
                });
                state.snapshot_version = snapshot_version;
            }
        }
        CatalogOperation::ReportRide { report } => {
            let Some(scientific) = state.scientific.as_mut() else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let mut next_scientific = scientific.clone();
            let Some(order) = next_scientific.ride_ledger.active_order(report.work) else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            if order.replica != request.identity.replica {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            }
            let source_basin = order.arm.source_basin;
            let (outcome, receiving_evaluations) = match &report.outcome {
                CatalogRideOutcome::Certified(connection) => certify_ride_connection(
                    &mut next_scientific,
                    &request.identity,
                    source_basin,
                    connection,
                ),
                CatalogRideOutcome::Unresolved(evidence) => certify_unresolved_ride_saddle(
                    &mut next_scientific,
                    &request.identity,
                    source_basin,
                    evidence,
                ),
                CatalogRideOutcome::Failed(failure) => (RideOutcome::Failed(*failure), 0),
            };
            let Some(charged_evaluations) = report
                .charged_evaluations
                .checked_add(receiving_evaluations)
            else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let Ok(credit) = next_scientific.ride_ledger.report(
                request.identity.replica,
                report.work,
                charged_evaluations,
                outcome.clone(),
            ) else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            if let RideOutcome::Certified { endpoints, .. } = &outcome {
                let left = BasinId::from_raw(endpoints[0]);
                let right = BasinId::from_raw(endpoints[1]);
                let Some(left_node) = transition_node(&mut next_scientific, left) else {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                let Some(right_node) = transition_node(&mut next_scientific, right) else {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                if next_scientific
                    .transition_graph
                    .observe(
                        "certified_ride",
                        left_node,
                        TransitionOutcome::Resolved(right_node),
                    )
                    .is_err()
                    || next_scientific
                        .transition_graph
                        .observe(
                            "certified_ride",
                            right_node,
                            TransitionOutcome::Resolved(left_node),
                        )
                        .is_err()
                {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                }
                next_scientific
                    .landscape
                    .observe_crossing(endpoints[0], endpoints[1], 1.0);
            }
            let Some(snapshot_version) = state.snapshot_version.checked_add(1) else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let census_visits = next_scientific.census.total_visits();
            let active_entries = u32::try_from(next_scientific.catalog.len())
                .expect("catalog capacity is checked against u32");
            *scientific = next_scientific;
            payload = AcceptedPayload::RideCredit(credit);
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
        CatalogOperation::LedgerBatch { events } => {
            let ordered = !events.is_empty()
                && events.first().is_some_and(|event| {
                    state
                        .maximum_sequence
                        .get(&request.identity.replica)
                        .is_none_or(|maximum| event.sequence > *maximum)
                })
                && events
                    .windows(2)
                    .all(|pair| pair[0].sequence.checked_add(1) == Some(pair[1].sequence))
                && events
                    .last()
                    .is_some_and(|event| event.sequence == request.event_sequence);
            if !ordered {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            }
            let Some(ledger) = state.ledger.as_ref() else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let mut staged = ledger.clone();
            for event in events {
                let Some(kind) = ChargeKind::from_wire_code(event.kind) else {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                };
                if staged
                    .record(ReplicaLedgerEvent {
                        replica: request.identity.replica,
                        sequence: event.sequence,
                        kind,
                        charged_calls: event.charged_calls,
                        cumulative_charged: event.cumulative_charged,
                    })
                    .is_err()
                {
                    return rejected(
                        state,
                        request.event_sequence,
                        ProtocolRejection::ValidationRejected,
                    );
                }
            }
            let aggregate_charged = staged.ensemble_total();
            let Ok(event_count) = u64::try_from(events.len()) else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            let Some(snapshot_version) = state.snapshot_version.checked_add(event_count) else {
                return rejected(
                    state,
                    request.event_sequence,
                    ProtocolRejection::ValidationRejected,
                );
            };
            state.ledger = Some(staged);
            if let Some(scientific) = state.scientific.as_mut() {
                scientific.catalog.update_threshold(aggregate_charged);
            }
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

/// The precomputed evaluation if this request had one queued before the
/// lock was taken, otherwise the original synchronous path: journal
/// replay at startup calls `apply_request` directly with no separate
/// precompute step, single-threaded, so recomputing here costs nothing
/// it does not already pay and stays correct without one.
fn resolve_validation(
    precomputed: &mut Option<Result<ValidatedCandidate, ()>>,
    scientific: &ScientificState,
    identity: &CatalogIdentity,
    candidate: &CatalogCandidate,
) -> Result<ValidatedCandidate, ()> {
    precomputed.take().unwrap_or_else(|| {
        validate_candidate(
            &scientific.signature,
            &scientific.descriptor_space,
            &scientific.validator,
            scientific.evaluate.as_ref(),
            identity,
            candidate,
        )
    })
}

fn validate_candidate<F>(
    signature: &SystemSignature,
    descriptor_space: &DescriptorSpace,
    validator: &CandidateValidator,
    evaluate: &F,
    identity: &CatalogIdentity,
    candidate: &CatalogCandidate,
) -> Result<ValidatedCandidate, ()>
where
    F: Fn(&[f64]) -> Result<FreshEvaluation, String> + Send + Sync + ?Sized,
{
    let reject = |reason: &str| {
        eprintln!(
            "catalog candidate rejected: replica={} event={} reason={reason}",
            identity.replica, candidate.event_sequence
        );
    };
    if candidate.producer_replica != identity.replica {
        reject("producer replica does not match request identity");
        return Err(());
    }
    if candidate.descriptor_schema_version != signature.descriptor.version {
        reject("descriptor schema version does not match system signature");
        return Err(());
    }
    if candidate.census_basin.is_some() {
        reject("producer supplied a coordinator-owned census basin");
        return Err(());
    }
    // The worker already evaluated leftover SOAP. The book merges the
    // posted vector; the validator still checks length and finite
    // values. What follows used to recompute SOAP under the
    // coordinator's lock, serializing every replica behind one
    // descriptor call; the caller now runs this with no lock held.
    let record = CandidateRecord {
        signature: signature.clone(),
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
    let mut validated = validator
        .validate(&record, |coordinates| evaluate(coordinates))
        .map_err(|error| {
            eprintln!(
                "catalog candidate rejected: replica={} event={} reason={error}",
                identity.replica, candidate.event_sequence
            );
        })?;
    let descriptor = descriptor_space
        .describe(
            ArrayView1::from(&validated.candidate.coordinates),
            Some(&signature.atomic_numbers),
        )
        .map_err(|error| {
            eprintln!(
                "catalog candidate rejected: replica={} event={} reason=descriptor recomputation failed: {error}",
                identity.replica, candidate.event_sequence
            );
        })?;
    if descriptor.values().len() != validated.candidate.descriptor.len()
        || descriptor.schema_version() != signature.descriptor.version
    {
        reject("recomputed descriptor shape does not match candidate record");
        return Err(());
    }
    validated.candidate.descriptor = descriptor.values().to_vec();
    validated.candidate.descriptor_schema_version = descriptor.schema_version();
    Ok(validated)
}

struct ReceivingRideSurface<'a, F: ?Sized>(&'a F);

impl<F> PesSurface for ReceivingRideSurface<'_, F>
where
    F: Fn(&[f64]) -> Result<FreshEvaluation, String> + Send + Sync + ?Sized,
{
    type Error = String;

    fn evaluate(
        &self,
        coordinates: ArrayView1<'_, f64>,
    ) -> Result<(f64, Array1<f64>), Self::Error> {
        let point = coordinates.to_vec();
        let fresh = (self.0)(&point)?;
        // FreshEvaluation carries physical forces, while PesSurface and
        // rgsaddle consume the Cartesian gradient of the potential energy.
        let gradient = Array1::from_iter(fresh.forces.into_iter().map(|force| -force));
        Ok((fresh.energy, gradient))
    }
}

fn certify_ride_connection(
    scientific: &mut ScientificState,
    identity: &CatalogIdentity,
    source_basin: u64,
    connection: &CatalogRideConnection,
) -> (RideOutcome, u64) {
    let signature = scientific.signature.clone();
    let descriptor_space = scientific.descriptor_space.clone();
    let validator = scientific.validator.clone();
    let evaluate = Arc::clone(&scientific.evaluate);
    let receiving_evaluations = AtomicU64::new(0);
    let counted_evaluate = |coordinates: &[f64]| {
        receiving_evaluations.fetch_add(1, Ordering::Relaxed);
        evaluate(coordinates)
    };
    let result = (|| -> Result<RideOutcome, crate::ride_ledger::RideFailure> {
        let (saddle, index) = certify_ride_saddle_candidate(
            &signature,
            &descriptor_space,
            &validator,
            &counted_evaluate,
            identity,
            &connection.saddle,
        )?;
        let endpoints = [
            validate_candidate(
                &signature,
                &descriptor_space,
                &validator,
                &counted_evaluate,
                identity,
                &connection.endpoints[0],
            )
            .map_err(|_| crate::ride_ledger::RideFailure::Surface)?,
            validate_candidate(
                &signature,
                &descriptor_space,
                &validator,
                &counted_evaluate,
                identity,
                &connection.endpoints[1],
            )
            .map_err(|_| crate::ride_ledger::RideFailure::Surface)?,
        ];

        let mut staged = scientific.clone();
        let saddle_id = observe_certified_ride_saddle(&mut staged, source_basin, &saddle, &index)?;

        let mut endpoint_ids = [0_u64; 2];
        for (slot, validated) in endpoints.into_iter().enumerate() {
            let observation = observe_exact_basin(&mut staged, &validated)
                .map_err(|_| crate::ride_ledger::RideFailure::Surface)?;
            let canonical = candidate_from_validated(&validated, Some(observation.basin_id));
            observe_ride_source(&mut staged, &canonical)
                .map_err(|_| crate::ride_ledger::RideFailure::Surface)?;
            staged
                .catalog
                .admit(observation.basin_id, observation.basin_visits, validated);
            endpoint_ids[slot] = observation.basin_id.as_raw();
        }
        if let Some(stored) = staged.ride_saddles.get_mut(&saddle_id) {
            stored.source_basins.extend(endpoint_ids);
        }
        *scientific = staged;
        Ok(RideOutcome::Certified {
            saddle: saddle_id,
            endpoints: endpoint_ids,
        })
    })();
    let charged = receiving_evaluations.load(Ordering::Relaxed);
    (result.unwrap_or_else(RideOutcome::Failed), charged)
}

fn certify_unresolved_ride_saddle(
    scientific: &mut ScientificState,
    identity: &CatalogIdentity,
    source_basin: u64,
    evidence: &CatalogRideSaddleEvidence,
) -> (RideOutcome, u64) {
    let receiving_evaluations = AtomicU64::new(0);
    let evaluate = Arc::clone(&scientific.evaluate);
    let counted_evaluate = |coordinates: &[f64]| {
        receiving_evaluations.fetch_add(1, Ordering::Relaxed);
        evaluate(coordinates)
    };
    let result = (|| -> Result<RideOutcome, crate::ride_ledger::RideFailure> {
        if !matches!(
            evidence.failure,
            crate::ride_ledger::RideFailure::CollapsedConnection
                | crate::ride_ledger::RideFailure::IrcNotConverged
                | crate::ride_ledger::RideFailure::DisconnectedConnection
        ) {
            return Err(crate::ride_ledger::RideFailure::Surface);
        }
        let (saddle, index) = certify_ride_saddle_candidate(
            &scientific.signature,
            &scientific.descriptor_space,
            &scientific.validator,
            &counted_evaluate,
            identity,
            &evidence.saddle,
        )?;
        let mut staged = scientific.clone();
        let saddle_id = observe_certified_ride_saddle(&mut staged, source_basin, &saddle, &index)?;
        *scientific = staged;
        Ok(RideOutcome::Unresolved {
            saddle: saddle_id,
            failure: evidence.failure,
        })
    })();
    let charged = receiving_evaluations.load(Ordering::Relaxed);
    (result.unwrap_or_else(RideOutcome::Failed), charged)
}

fn certify_ride_saddle_candidate<F>(
    signature: &SystemSignature,
    descriptor_space: &DescriptorSpace,
    validator: &CandidateValidator,
    evaluate: &F,
    identity: &CatalogIdentity,
    candidate: &CatalogCandidate,
) -> Result<(ValidatedCandidate, StationaryIndex), crate::ride_ledger::RideFailure>
where
    F: Fn(&[f64]) -> Result<FreshEvaluation, String> + Send + Sync + ?Sized,
{
    let saddle = validate_candidate(
        signature,
        descriptor_space,
        validator,
        evaluate,
        identity,
        candidate,
    )
    .map_err(|_| crate::ride_ledger::RideFailure::Surface)?;
    let config = PesExplorationConfig::default();
    let index = stationary_index_cartesian(
        &ReceivingRideSurface(evaluate),
        ArrayView1::from(&saddle.candidate.coordinates),
        &signature.frozen_mask,
        signature.periodic,
        config.hessian_step,
        config.negative_curvature_tolerance,
    )
    .map_err(|_| crate::ride_ledger::RideFailure::Surface)?;
    match index.negative_modes {
        0 => return Err(crate::ride_ledger::RideFailure::NoNegativeMode),
        1 => {}
        _ => return Err(crate::ride_ledger::RideFailure::HigherIndex),
    }
    Ok((saddle, index))
}

fn observe_certified_ride_saddle(
    scientific: &mut ScientificState,
    source_basin: u64,
    saddle: &ValidatedCandidate,
    index: &StationaryIndex,
) -> Result<u64, crate::ride_ledger::RideFailure> {
    let candidate = candidate_from_validated(saddle, None);
    let mut ordered = scientific
        .ride_saddles
        .iter()
        .filter_map(|(&id, stored)| {
            (stored.candidate.descriptor.len() == candidate.descriptor.len()).then(|| {
                let distance = stored
                    .candidate
                    .descriptor
                    .iter()
                    .zip(&candidate.descriptor)
                    .map(|(left, right)| (left - right).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (distance, id)
            })
        })
        .collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    let existing = ordered.into_iter().find_map(|(_, id)| {
        let stored = scientific.ride_saddles.get(&id)?;
        scientific
            .exact_witness
            .equivalent_structures(
                StructureView {
                    coordinates: ArrayView1::from(&stored.candidate.coordinates),
                    context: &scientific.structure_context,
                },
                StructureView {
                    coordinates: ArrayView1::from(&candidate.coordinates),
                    context: &scientific.structure_context,
                },
            )
            .then_some(id)
    });
    if let Some(id) = existing {
        let stored = scientific
            .ride_saddles
            .get_mut(&id)
            .ok_or(crate::ride_ledger::RideFailure::Surface)?;
        stored.source_basins.insert(source_basin);
        if candidate.gradient_norm < stored.candidate.gradient_norm {
            stored.candidate = candidate;
            stored.lowest_curvature = index.eigenvalues[0];
            stored.lowest_mode = index.lowest_mode.to_vec();
            stored.negative_modes = index.negative_modes;
        }
        return Ok(id);
    }
    let id = scientific.next_ride_saddle;
    scientific.next_ride_saddle = id
        .checked_add(1)
        .ok_or(crate::ride_ledger::RideFailure::Surface)?;
    scientific.ride_saddles.insert(
        id,
        CertifiedRideSaddle {
            candidate,
            lowest_curvature: index.eigenvalues[0],
            lowest_mode: index.lowest_mode.to_vec(),
            negative_modes: index.negative_modes,
            source_basins: BTreeSet::from([source_basin]),
        },
    );
    Ok(id)
}

fn observe_exact_basin(
    scientific: &mut ScientificState,
    validated: &ValidatedCandidate,
) -> Result<CensusObservation, ()> {
    let assigned = {
        let mut representatives = scientific
            .ride_candidates
            .iter()
            .map(|(&basin, stored)| {
                let distance_squared = stored
                    .descriptor
                    .iter()
                    .zip(&validated.candidate.descriptor)
                    .map(|(left, right)| {
                        let delta = left - right;
                        delta * delta
                    })
                    .sum::<f64>();
                (basin, distance_squared, stored)
            })
            .collect::<Vec<_>>();
        representatives.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        representatives.into_iter().find_map(|(basin, _, stored)| {
            scientific
                .exact_witness
                .equivalent_structures(
                    StructureView {
                        coordinates: ArrayView1::from(&stored.coordinates),
                        context: &scientific.structure_context,
                    },
                    StructureView {
                        coordinates: ArrayView1::from(&validated.candidate.coordinates),
                        context: &scientific.structure_context,
                    },
                )
                .then_some(BasinId::from_raw(basin))
        })
    };
    scientific
        .census
        .observe_assigned(&validated.candidate.descriptor, assigned)
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
    new_basin: bool,
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
        new_basin,
        kind,
        evicted,
        incumbent_basin: incumbent_basin.map(BasinId::as_raw),
    }
}

/// One commissioned bridge: the string, the region weights, the stored
/// entry states, and which replica holds which region.
#[derive(Clone)]
struct BridgeServerState {
    string: BridgeString,
    ledger: WeightLedger,
    entries: EntryLists,
    from_basin: u64,
    to_basin: u64,
    tube_radius: f64,
    assignments: BTreeMap<u32, usize>,
}

/// Interior images of a commissioned bridge string.
const BRIDGE_IMAGES: usize = 12;
/// Weight fraction transferred per attempted exit.
const BRIDGE_TRANSFER: f64 = 0.1;
/// Stored entry states per region.
const BRIDGE_ENTRY_CAPACITY: usize = 8;
/// Bridge tube radius, in census radii.
const BRIDGE_TUBE_RADII: f64 = 4.0;

/// Commission a bridge across the referee's seam when none is active.
///
/// The seam names the two most weakly coupled communities; the bridge
/// connects their best-anchored catalog entries through descriptor
/// space, so the crossing evidence the confined segments produce is
/// exactly the flux the landscape graph is missing. One bridge at a
/// time: the machinery prices one seam, and a second seam only becomes
/// visible after the first is crossed.
fn commission_bridge(scientific: &mut ScientificState, census_radius: f64) {
    if !scientific.bridges.is_empty() {
        return;
    }
    let Ok(split) = scientific.landscape.spectral_split() else {
        return;
    };
    if split.conductance >= 0.1 {
        return;
    }
    let descriptor_of = |basin: u64| {
        scientific
            .catalog
            .entries()
            .iter()
            .find(|entry| entry.census_id().as_raw() == basin)
            .map(|entry| Array1::from(entry.descriptor().to_vec()))
    };
    let (Some(from), Some(to)) = (
        descriptor_of(split.representatives.0),
        descriptor_of(split.representatives.1),
    ) else {
        return;
    };
    let Ok(string) = BridgeString::chord(&from, &to, BRIDGE_IMAGES) else {
        return;
    };
    let regions = string.regions();
    let Ok(ledger) = WeightLedger::new(regions, BRIDGE_TRANSFER) else {
        return;
    };
    let bridge = scientific.next_bridge;
    scientific.next_bridge += 1;
    scientific.bridges.insert(
        bridge,
        BridgeServerState {
            string,
            ledger,
            entries: EntryLists::new(regions, BRIDGE_ENTRY_CAPACITY),
            from_basin: split.representatives.0,
            to_basin: split.representatives.1,
            tube_radius: BRIDGE_TUBE_RADII * census_radius,
            assignments: BTreeMap::new(),
        },
    );
}

/// Keep every weakly coupled community of the landscape represented in
/// the parent map.
///
/// Resampling concentrates offspring on the deepest submissions, and on
/// a landscape whose explored graph splits into two communities that
/// rarely exchange, that concentration abandons whichever community is
/// currently shallower, after which nothing ever samples it again. When
/// the referee sees such a split and the chosen parents all sit on one
/// side while a submission exists on the other, the plan gives one slot
/// of the most crowded family to the deepest candidate of the abandoned
/// side. A well-mixed landscape passes through untouched.
fn referee_community_coverage(
    scientific: &ScientificState,
    mut parents: Vec<u32>,
    source_candidates: &BTreeMap<u32, CatalogCandidate>,
) -> Vec<u32> {
    let Ok(split) = scientific.landscape.spectral_split() else {
        return parents;
    };
    if split.conductance >= 0.1 {
        return parents;
    }
    let side = |replica: &u32| -> Option<bool> {
        let basin = source_candidates.get(replica)?.census_basin?;
        if split.left.contains(&basin) {
            Some(true)
        } else if split.right.contains(&basin) {
            Some(false)
        } else {
            None
        }
    };
    let mut has_left = false;
    let mut has_right = false;
    for parent in &parents {
        match side(parent) {
            Some(true) => has_left = true,
            Some(false) => has_right = true,
            None => {}
        }
    }
    if has_left == has_right {
        return parents;
    }
    let Some(replacement) = source_candidates
        .keys()
        .filter(|replica| side(replica) == Some(!has_left))
        .min_by(|a, b| {
            source_candidates[a]
                .energy
                .total_cmp(&source_candidates[b].energy)
        })
        .copied()
    else {
        return parents;
    };
    let mut counts = BTreeMap::<u32, usize>::new();
    for parent in &parents {
        *counts.entry(*parent).or_default() += 1;
    }
    let Some((&crowded, &family)) = counts.iter().max_by_key(|(_, count)| **count) else {
        return parents;
    };
    if family < 2 {
        return parents;
    }
    if let Some(slot) = parents.iter().rposition(|parent| *parent == crowded) {
        parents[slot] = replacement;
    }
    parents
}

fn region_population_assignment(
    scientific: &ScientificState,
    destinations: &[u32],
    source_candidates: &BTreeMap<u32, CatalogCandidate>,
    max_family_size: usize,
) -> Option<(Vec<u32>, Vec<f64>)> {
    let regions = scientific
        .transition_graph
        .attraction_regions(&scientific.attraction_regions)
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
        .posterior_matrix(
            &scientific.attraction_regions.probe_action,
            scientific.attraction_regions.concentration,
        )
        .ok()?;
    let candidates = destinations
        .iter()
        .enumerate()
        .map(|(index, replica)| {
            let region = source_regions[index];
            let basin = BasinId::from_raw(source_candidates.get(replica)?.census_basin?);
            let node = scientific.transition_nodes.get(&basin).copied();
            let transition_uncertainty = node
                .and_then(|node| {
                    scientific.transition_graph.uncertainty(
                        &scientific.attraction_regions.probe_action,
                        node,
                        scientific.attraction_regions.concentration,
                    )
                })
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
        .attraction_regions(&scientific.attraction_regions)
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
    scientific.transition_graph.register_node(node).ok()?;
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

/// Build, store, and report the plan for one completed epoch.
///
/// Reached from a submission that completes the barrier and from an
/// abstention that completes it, so the two cannot drift apart in how a
/// plan is selected, counted, or retained for later polls.
fn realize_population_plan(
    scientific: &mut ScientificState,
    config: &ServerConfig,
    epoch: u64,
    plan: &PopulationEpochPlan,
    required: u32,
) -> AcceptedPayload {
    if let Some(census_radius) = config
        .scientific
        .as_ref()
        .map(|scientific| scientific.census_radius)
    {
        commission_bridge(scientific, census_radius);
    }
    let source_candidates = scientific
        .population_candidates
        .get(&epoch)
        .expect("complete epoch retains every source candidate");
    let family_cap = usize::try_from(REDUCTION_FACTOR).unwrap_or(3);
    let (fallback_parents, weights, selection) = match region_population_assignment(
        scientific,
        plan.destinations(),
        source_candidates,
        family_cap,
    ) {
        Some((parents, weights)) => (parents, weights, PopulationSelection::RegionCovering),
        None => (
            plan.parents().to_vec(),
            plan.weights().to_vec(),
            PopulationSelection::SystematicResampling,
        ),
    };
    let fallback_parents =
        referee_community_coverage(scientific, fallback_parents, source_candidates);
    let occupants = plan
        .destinations()
        .iter()
        .map(|replica| {
            let candidate = source_candidates.get(replica);
            PackingOccupant {
                replica: *replica,
                family: replica_family_index(scientific, *replica, candidate),
                energy: candidate
                    .map(|candidate| candidate.energy)
                    .or_else(|| {
                        scientific
                            .last_candidate_by_replica
                            .get(replica)
                            .map(|candidate| candidate.energy)
                    })
                    .unwrap_or(f64::INFINITY),
            }
        })
        .collect::<Vec<_>>();
    let parents = if occupants.iter().any(|occupant| occupant.family.is_some()) {
        assign_parents_by_packing(&occupants, family_cap)
    } else {
        fallback_parents
    };
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
        weights: weights.to_vec(),
        effective_sample_size: diagnostics.0,
        unique_parents: u32::try_from(diagnostics.1)
            .expect("unique parents are bounded by replica count"),
        max_family_size: u32::try_from(diagnostics.2)
            .expect("family size is bounded by replica count"),
        offspring_variance: diagnostics.3,
        parent_candidates,
        selection,
    };
    scientific.population_plans.insert(epoch, wire_plan.clone());
    AcceptedPayload::PopulationEpoch(PopulationEpochState {
        epoch,
        submitted: required,
        required,
        plan: Some(wire_plan),
    })
}

fn observe_packing(
    scientific: &mut ScientificState,
    replica: u32,
    coordinates: &[f64],
    basin: Option<BasinId>,
    arrival: bool,
) {
    if let Some(index) = scientific.packing.observe(coordinates) {
        if arrival {
            scientific.packing.credit_well(index);
            if let Some(basin) = basin {
                *scientific
                    .leftover_arrivals
                    .entry(basin.as_raw())
                    .or_insert(0) += 1;
            }
        }
        let changed = scientific
            .family_history
            .get(&replica)
            .and_then(|history| history.back().copied())
            .is_some_and(|previous| previous as usize != index);
        if changed {
            scientific.energy_history.remove(&replica);
        }
        let history = scientific.family_history.entry(replica).or_default();
        history.push_back(index as f64);
        while history.len() > 64 {
            history.pop_front();
        }
    }
}

fn observe_ride_source(
    scientific: &mut ScientificState,
    candidate: &CatalogCandidate,
) -> Result<(), ()> {
    let basin = candidate.census_basin.ok_or(())?;
    if let Some(stored) = scientific.ride_candidates.get_mut(&basin) {
        if candidate.energy < stored.energy {
            *stored = candidate.clone();
        }
        return Ok(());
    }
    let local = if scientific.descriptor_space.geometry().is_some() {
        scientific
            .descriptor_space
            .describe_local(
                ArrayView1::from(&candidate.coordinates),
                Some(&scientific.signature.atomic_numbers),
            )
            .map_err(|_| ())?
    } else {
        local_nu3_z(
            ArrayView1::from(&candidate.coordinates),
            crate::catalog::packing::PACKING_SPEC,
            Some(&scientific.signature.atomic_numbers),
        )
    };
    let environments = scientific
        .ride_environments
        .observe(local.view())
        .map_err(|_| ())?;
    scientific
        .ride_ledger
        .register_source(RideSource {
            basin,
            energy: candidate.energy,
            environments,
        })
        .map_err(|_| ())?;
    let replace = scientific
        .ride_candidates
        .get(&basin)
        .is_none_or(|stored| candidate.energy < stored.energy);
    if replace {
        scientific.ride_candidates.insert(basin, candidate.clone());
    }
    Ok(())
}

/// Catalog representative of the packing family the fewest live
/// replicas occupy.
///
/// Occupancy is counted over the last candidate each replica reported,
/// through the query path so that counting cannot grow the codebook.
/// A family with no catalog representative cannot be drawn from, and a
/// family no replica stands on counts zero, which is what makes it the
/// one to hand out.
/// Archive cell a descriptor belongs to.
///
/// The tessellation once it exists, and the leader-clustered family
/// until then. Cells from a tessellation are fixed in number and equal
/// by construction, so occupancy over them is a distribution rather
/// than a count over a support that grows while it is measured.
fn archive_cell(
    scientific: &ScientificState,
    descriptor: &[f64],
    coordinates: &[f64],
) -> Option<usize> {
    // Nearest, not within-radius. An archive with cells answers for
    // every descriptor in its own numbering, because the fallback
    // numbering is the packing families and the two use the same
    // integers for unrelated cells: mixing them credits occupancy and
    // reward to whichever cell shares the number. The families answer
    // only while the archive is empty, when there is nothing to
    // collide with.
    if let Some(cell) = scientific.archive.nearest(descriptor) {
        return Some(cell);
    }
    let histogram = scientific.packing.histogram(coordinates)?;
    scientific.packing.family_of(&histogram)
}

/// Floor of the archive radius as a fraction of where it starts.
///
/// The same floor conformational space annealing uses for Dcut, so the
/// archive coarsens and refines on a schedule the bank already trusts
/// rather than on a second one invented here.
const ARCHIVE_RADIUS_FLOOR: f64 = 0.4;

/// Feed a descriptor to the tessellation, and relax one once there is
/// enough to relax.
///
/// Built once and then left alone: cells that moved under the search
/// would make coverage incomparable between one moment and the next,
/// which is the property the tessellation exists to provide.
fn observe_descriptor(scientific: &mut ScientificState, descriptor: &[f64]) {
    if descriptor.is_empty() {
        return;
    }
    scientific.archive.observe(descriptor);
    // The radius follows the ensemble's spend, so the archive is coarse
    // while the budget is young and fine once it is not: asked to be
    // different first and better second, which is the annealing half of
    // conformational space annealing applied to diversity.
    let progress = scientific.archive_progress.clamp(0.0, 1.0);
    scientific.archive.anneal(progress);
}

/// Fraction of archive cells any live replica occupies.
///
/// A fixed denominator, which is what makes this a coverage rather
/// than a count. Falls back to nothing when no tessellation exists yet.
fn archive_coverage(scientific: &ScientificState) -> Option<f64> {
    if scientific.archive.cells() == 0 {
        return None;
    }
    let descriptors: Vec<&[f64]> = scientific
        .last_candidate_by_replica
        .values()
        .map(|candidate| candidate.descriptor.as_slice())
        .collect();
    Some(scientific.archive.coverage(descriptors))
}

/// Family and catalog slot to hand a replica that is leaving a crowded
/// packing.
///
/// Indices rather than a borrow, so the caller can record the draw
/// before reading the entry. The selection is deterministic in the
/// draw the client sent, because this runs again on journal replay and
/// a coordinator that rebuilt itself differently would not be the same
/// coordinator.
fn sparsest_family_entry<R: rand::Rng + ?Sized>(
    scientific: &ScientificState,
    rng: &mut R,
) -> Option<(usize, usize)> {
    let mut occupancy: BTreeMap<usize, usize> = BTreeMap::new();
    for candidate in scientific.last_candidate_by_replica.values() {
        if let Some(cell) = archive_cell(scientific, &candidate.descriptor, &candidate.coordinates)
        {
            *occupancy.entry(cell).or_insert(0) += 1;
        }
    }
    // Where the ensemble already is, for the novelty of a candidate
    // start measured against it.
    let occupied: Vec<Vec<f64>> = scientific
        .last_candidate_by_replica
        .values()
        .map(|candidate| candidate.descriptor.clone())
        .collect();
    // One representative per family, the deepest, which is the elite of
    // that cell.
    let mut elites: BTreeMap<usize, (usize, usize, f64)> = BTreeMap::new();
    for (slot, entry) in scientific.catalog.entries().iter().enumerate() {
        let Some(cell) = archive_cell(scientific, entry.descriptor(), entry.coordinates()) else {
            continue;
        };
        let crowd = occupancy.get(&cell).copied().unwrap_or(0);
        // The elite of a cell is its most novel deep structure, not
        // simply its deepest: two entries of equal depth in one cell
        // are worth different amounts to a replica being sent away
        // from where the ensemble already stands.
        let score = -crate::catalog::novelty(entry.descriptor(), &occupied, 4);
        elites
            .entry(cell)
            .and_modify(|held| {
                if entry.energy() < held.2 || (entry.energy() == held.2 && score < held.2) {
                    *held = (crowd, slot, entry.energy());
                }
            })
            .or_insert((crowd, slot, entry.energy()));
    }
    if elites.is_empty() {
        return None;
    }
    // Under-occupied first: the point of the draw is to send a replica
    // where replicas are not. Among those, curiosity decides, because
    // always taking the emptiest cell keeps choosing one nothing can
    // reach as soon as the descriptor is noisy.
    let least = elites
        .values()
        .map(|(crowd, _, _)| *crowd)
        .min()
        .unwrap_or(0);
    let allowed: Vec<usize> = elites
        .iter()
        .filter(|(_, (crowd, _, _))| *crowd == least)
        .map(|(family, _)| *family)
        .collect();
    // Families the table has not seen score neutral, so a cell
    // discovered since the last credit still competes.
    let mut curiosity = scientific.curiosity.clone();
    curiosity.ensure(allowed.iter().copied().max().unwrap_or(0) + 1);
    // Return then explore: the cell is chosen on what it has produced,
    // discounted by how heavily it has already been visited, so a thin
    // cell is worth going back to before it has any record at all.
    let chosen = scientific.archive.select(&allowed, rng).or_else(|| {
        let mut curiosity = scientific.curiosity.clone();
        curiosity.ensure(allowed.iter().copied().max().unwrap_or(0) + 1);
        curiosity.select(&allowed, rng)
    })?;
    elites.get(&chosen).map(|(_, slot, _)| (chosen, *slot))
}

fn highest_ei_family_entry(scientific: &mut ScientificState) -> Option<(usize, usize)> {
    let sites: Vec<(usize, Vec<f64>, Vec<f64>)> = scientific
        .catalog
        .entries()
        .iter()
        .enumerate()
        .map(|(slot, entry)| {
            (
                slot,
                entry.coordinates().to_vec(),
                entry.descriptor().to_vec(),
            )
        })
        .collect();
    let mut best: Option<(f64, usize, usize)> = None;
    for (slot, coordinates, descriptor) in &sites {
        let Some(histogram) = scientific.packing.histogram(coordinates) else {
            continue;
        };
        let ei = scientific
            .funnel
            .expected_improvement(ndarray::Array1::from(histogram).view());
        if !ei.is_finite() {
            continue;
        }
        let family = archive_cell(scientific, descriptor, coordinates).unwrap_or(*slot);
        if best.is_none_or(|(held, _, _)| ei > held) {
            best = Some((ei, family, *slot));
        }
    }
    best.map(|(_, family, slot)| (family, slot))
}

/// WAVE batch: greedy q-EI over packing families, then this replica
/// takes assignment[replica % q]. Independent highest-EI would send
/// every extra to the same family.
fn q_ei_family_entry(scientific: &mut ScientificState, replica: u32) -> Option<(usize, usize)> {
    let mut by_family: std::collections::BTreeMap<usize, (usize, Vec<f64>)> =
        std::collections::BTreeMap::new();
    let sites: Vec<(usize, Vec<f64>, Vec<f64>)> = scientific
        .catalog
        .entries()
        .iter()
        .enumerate()
        .map(|(slot, entry)| {
            (
                slot,
                entry.coordinates().to_vec(),
                entry.descriptor().to_vec(),
            )
        })
        .collect();
    for (slot, coordinates, descriptor) in &sites {
        let Some(histogram) = scientific.packing.histogram(coordinates) else {
            continue;
        };
        let family = archive_cell(scientific, descriptor, coordinates).unwrap_or(*slot);
        by_family.entry(family).or_insert((*slot, histogram));
    }
    if by_family.is_empty() {
        return highest_ei_family_entry(scientific);
    }
    let candidates: Vec<(usize, Vec<f64>)> = by_family
        .iter()
        .map(|(family, (_, histogram))| (*family, histogram.clone()))
        .collect();
    let q = scientific.last_candidate_by_replica.len().max(1);
    let order = scientific.funnel.assign_q_ei(&candidates, q);
    if order.is_empty() {
        return highest_ei_family_entry(scientific);
    }
    let family = order[replica as usize % order.len()];
    by_family.get(&family).map(|(slot, _)| (family, *slot))
}

/// The folded book, recomputed only when the book has changed.
fn sparsified_book(scientific: &mut ScientificState) -> &crate::catalog::OccupancyBookMap {
    let version = scientific.packing.version();
    let stale = scientific
        .sparsified
        .as_ref()
        .is_none_or(|(held, _)| *held != version);
    if stale {
        let folded = occupancy_sparsify_packing(&scientific.packing);
        scientific.sparsified = Some((version, folded));
    }
    &scientific
        .sparsified
        .as_ref()
        .expect("the fold was just stored")
        .1
}

fn packing_census_saturated(scientific: &mut ScientificState) -> bool {
    // Packing completeness is Chao1 on the landfold-sparsified book.
    // Leftover DECAF wells of one packing collapse to one community,
    // so extras do not walk the force budget after that compacted
    // census closes. Leftover-SOAP arrivals stay the hole generator
    // while the sparsified book still has holes.
    !sparsified_book(scientific).holes
}

/// Occupied packings the ensemble still has reason to station a replica in.
///
/// `packing_communities >= 2` was the test for "the surface is divided, so
/// send the extra to the other half". Measured on LJ75 it is true from the
/// second cell of every run and stays true: the walks quench into genuinely
/// distinct amorphous minima and each is its own community at the packing
/// grain, 10 and 11 of them by a few hundred funnel observations. Dividing
/// an ensemble between an icosahedral funnel and ten amorphous minima 5 to
/// 25 eps above its floor is dilution, not division of labour.
///
/// A community worth occupying is one the FunnelModel still expects
/// improvement from. Jones, Schonlau and Welch (*J. Global Optim.* **1998**,
/// *13*, 455) give \(\mathrm{EI}\to\max(f_{\min}-\mu,0)\) as
/// \(\sigma\to0\), so a packing that has been observed and mined out
/// scores at or below the model noise and drops out, while an unexplored one
/// does not. When fewer than two qualify there is nothing to divide and the
/// extra walks.
fn worthwhile_communities(scientific: &mut ScientificState) -> usize {
    let book = scientific.packing.version();
    let funnel = scientific.funnel.version();
    if let Some((held_book, held_funnel, count)) = scientific.worthwhile
        && held_book == book
        && held_funnel == funnel
    {
        return count;
    }
    let map = sparsified_book(scientific).clone();
    let noise = scientific.funnel.noise;
    let histograms: BTreeMap<usize, Vec<f64>> = scientific
        .packing
        .occupied_histograms()
        .into_iter()
        .collect();
    let mut open: BTreeSet<usize> = BTreeSet::new();
    for point in &map.points {
        if point.wells == 0 || open.contains(&point.community) {
            continue;
        }
        let Some(histogram) = histograms.get(&point.family) else {
            continue;
        };
        let ei = scientific
            .funnel
            .expected_improvement(ndarray::Array1::from(histogram.clone()).view());
        if ei.is_finite() && ei > noise {
            open.insert(point.community);
        }
    }
    let count = open.len();
    scientific.worthwhile = Some((book, funnel, count));
    count
}

fn occupancy_funnel_ei_exhausted(scientific: &mut ScientificState) -> bool {
    let sizes = (
        scientific.last_candidate_by_replica.len(),
        scientific.best_candidate_by_replica.len(),
        scientific.catalog.entries().len(),
    );
    if scientific.fed_from == Some(sizes)
        && let Some((held, verdict)) = scientific.ei_verdict
        && held == scientific.funnel.version()
    {
        return verdict;
    }
    scientific.fed_from = Some(sizes);
    let mut best: BTreeMap<usize, (Vec<f64>, f64)> = BTreeMap::new();
    {
        let packing = &scientific.packing;
        let mut consider = |coordinates: &[f64], energy: f64| {
            if !energy.is_finite() {
                return;
            }
            let Some(histogram) = packing.histogram(coordinates) else {
                return;
            };
            let Some(family) = packing.family_of(&histogram) else {
                return;
            };
            match best.get(&family) {
                Some((_, held)) if energy >= *held - 1e-15 => {}
                _ => {
                    best.insert(family, (histogram, energy));
                }
            }
        };
        for candidate in scientific.last_candidate_by_replica.values() {
            consider(&candidate.coordinates, candidate.energy);
        }
        for candidate in scientific.best_candidate_by_replica.values() {
            consider(&candidate.coordinates, candidate.energy);
        }
        for entry in scientific.catalog.entries() {
            consider(entry.coordinates(), entry.energy());
        }
    }
    for (histogram, energy) in best.values() {
        scientific
            .funnel
            .observe(ndarray::Array1::from(histogram.clone()).view(), *energy);
    }
    let version = scientific.funnel.version();
    if let Some((held, verdict)) = scientific.ei_verdict
        && held == version
    {
        return verdict;
    }
    let max_ei = scientific.funnel.max_expected_improvement_at_data();
    let verdict = occupancy_ei_exhausted(max_ei, scientific.funnel.len(), scientific.funnel.noise);
    scientific.ei_verdict = Some((version, verdict));
    verdict
}

fn basin_packing_family(scientific: &ScientificState, basin: u64) -> Option<usize> {
    let coordinates = scientific
        .last_candidate_by_replica
        .values()
        .chain(scientific.best_candidate_by_replica.values())
        .find(|candidate| candidate.census_basin == Some(basin))
        .map(|candidate| candidate.coordinates.as_slice())
        .or_else(|| {
            scientific
                .catalog
                .entries()
                .iter()
                .find(|entry| entry.census_id().as_raw() == basin)
                .map(|entry| entry.coordinates())
        })?;
    let histogram = scientific.packing.histogram(coordinates)?;
    scientific.packing.family_of(&histogram)
}

fn occupancy_seam_floor(
    scientific: &ScientificState,
) -> (usize, Option<f64>, Option<f64>, usize, usize, usize) {
    match scientific.landscape.spectral_split() {
        Ok(split) => {
            let live: BTreeSet<u64> = scientific
                .last_candidate_by_replica
                .values()
                .filter_map(|candidate| candidate.census_basin)
                .collect();
            let left_live: Vec<u64> = split
                .left
                .iter()
                .copied()
                .filter(|basin| live.contains(basin))
                .collect();
            let right_live: Vec<u64> = split
                .right
                .iter()
                .copied()
                .filter(|basin| live.contains(basin))
                .collect();
            let mut families = BTreeSet::new();
            for basin in left_live.iter().chain(right_live.iter()) {
                if let Some(family) = basin_packing_family(scientific, *basin) {
                    families.insert(family);
                }
            }
            (
                occupancy_family_floor(
                    Some(split.conductance),
                    Some(split.algebraic_connectivity),
                    left_live.len(),
                    right_live.len(),
                    families.len() >= 2,
                ),
                Some(split.conductance),
                Some(split.algebraic_connectivity),
                split.left.len(),
                split.right.len(),
                families.len(),
            )
        }
        Err(_) => (DEFAULT_MIN_OCCUPIED_FAMILIES, None, None, 0, 0, 0),
    }
}

fn occupancy_landfold_from_book(scientific: &mut ScientificState) -> (usize, usize, usize) {
    let version = scientific.packing.version();
    if let Some((held, split)) = scientific.landfold
        && held == version
    {
        return split;
    }
    let split = occupancy_landfold_uncached(scientific);
    scientific.landfold = Some((version, split));
    split
}

fn occupancy_landfold_uncached(scientific: &ScientificState) -> (usize, usize, usize) {
    let occupied = scientific.packing.occupied_histograms();
    let histograms: Vec<Vec<f64>> = occupied
        .iter()
        .map(|(_, histogram)| histogram.clone())
        .collect();
    let families: Vec<usize> = occupied.iter().map(|(index, _)| *index).collect();
    occupancy_landfold_split(&histograms, &families)
}

fn occupancy_ring_from_book(scientific: &ScientificState) -> (usize, usize, usize) {
    let occupied: BTreeSet<usize> = scientific
        .packing
        .occupied_histograms()
        .into_iter()
        .map(|(index, _)| index)
        .collect();
    let mut best: BTreeMap<usize, (Vec<f64>, f64)> = BTreeMap::new();
    let mut consider = |coordinates: &[f64], energy: f64| {
        if !energy.is_finite() {
            return;
        }
        let Some(histogram) = scientific.packing.histogram(coordinates) else {
            return;
        };
        let Some(family) = scientific.packing.family_of(&histogram) else {
            return;
        };
        if !occupied.contains(&family) {
            return;
        }
        match best.get(&family) {
            Some((_, held)) if energy >= *held - 1e-15 => {}
            _ => {
                best.insert(family, (coordinates.to_vec(), energy));
            }
        }
    };
    for candidate in scientific.last_candidate_by_replica.values() {
        consider(&candidate.coordinates, candidate.energy);
    }
    for candidate in scientific.best_candidate_by_replica.values() {
        consider(&candidate.coordinates, candidate.energy);
    }
    for entry in scientific.catalog.entries() {
        consider(entry.coordinates(), entry.energy());
    }
    let mut profiles = Vec::new();
    for (coordinates, _) in best.values() {
        if let Some(profile) = occupancy_ring_profile(coordinates) {
            profiles.push(profile);
        }
    }
    occupancy_ring_split(&profiles)
}

fn occupancy_floor(scientific: &mut ScientificState) -> usize {
    if std::env::var("CATALOG_MIN_FAMILIES")
        .ok()
        .and_then(|value| value.parse().ok())
        .is_some_and(|count: usize| count >= 1)
    {
        return occupancy_min_families();
    }
    let fiedler = occupancy_seam_floor(scientific).0;
    let landfold = occupancy_landfold_from_book(scientific).0;
    // The book's community count is a hurdle, not a quota. Every other
    // measured floor here is one or two, and single linkage can name many
    // more packings than an ensemble has replicas to sit in: asking the live
    // ensemble to occupy every packing the book has ever recorded is a floor
    // no run can meet.
    let peeled = sparsified_book(scientific).communities.clamp(1, 2);
    fiedler.max(landfold).max(peeled)
}

fn leftover_census_dwell(scientific: &mut ScientificState) -> bool {
    let leftover = GoodTuringSample::from_counts(scientific.leftover_arrivals.values().copied());
    let sample = (leftover.n, leftover.n1, leftover.n2);
    if scientific.last_leftover_dwell_sample != Some(sample) {
        scientific.last_leftover_dwell_sample = Some(sample);
        if leftover.saturated() {
            scientific.leftover_sat_streak = scientific.leftover_sat_streak.saturating_add(1);
        } else {
            scientific.leftover_sat_streak = 0;
        }
    }
    scientific.leftover_dwell = leftover_dwell_from_census(
        leftover.saturated(),
        leftover_esty_stable(
            leftover.n,
            leftover.n1,
            leftover.n2,
            crate::catalog::PRODUCTION_MAX_UNSEEN_MASS,
        ),
        scientific.leftover_sat_streak,
    );
    scientific.leftover_dwell
}

fn report_occupancy_gt(scientific: &mut ScientificState) {
    let leftover = GoodTuringSample::from_counts(scientific.leftover_arrivals.values().copied());
    let packing = scientific.packing.well_sample();
    let families = scientific.packing.occupied_packings_among(
        scientific
            .last_candidate_by_replica
            .values()
            .map(|candidate| candidate.coordinates.as_slice()),
    ) as u32;
    let (measured_floor, conductance, algebraic, seam_left, seam_right, seam_packings) =
        occupancy_seam_floor(scientific);
    let (landfold_floor, landfold_left, landfold_right) = occupancy_landfold_from_book(scientific);
    let sparsified = sparsified_book(scientific).clone();
    let sparsified_sample = sparsified.sample();
    let (ring_floor, ring_distinct, ring_n) = occupancy_ring_from_book(scientific);
    let fes_delta = occupancy_fes_delta(&scientific.packing.occupied_well_counts());
    let fes_minima = sparsified.fes_minima;
    let min_families = occupancy_floor(scientific) as u32;
    let leftover_sat = leftover.saturated();
    let packing_sat = sparsified.saturated();
    let sparsified_sat = packing_sat;
    let cells = sparsified.cells;
    // Whether a one-packing book is allowed to Walk. Without this on the
    // record there is no way to tell a run where the gate fired from one
    // where it never did, and the two look identical from outside.
    let funnel_obs = scientific.funnel.len();
    let occupied_communities = sparsified.occupied_communities;
    let worthwhile = worthwhile_communities(scientific);
    let ei_exhausted = occupancy_funnel_ei_exhausted(scientific);
    let stop = packing_sat && families >= min_families;
    let key = OccupancyGtKey {
        leftover_n: leftover.n,
        leftover_n1: leftover.n1,
        packing_n: packing.n,
        packing_n1: packing.n1,
        families,
        min_families,
        landfold_floor,
        ring_floor,
        fes_delta_bits: fes_delta.map(f64::to_bits),
        stop,
        communities: sparsified.communities,
        holes: sparsified.holes,
        fes_minima,
    };
    if scientific.last_gt_report == Some(key) {
        return;
    }
    scientific.last_gt_report = Some(key);
    let leftover_p0 = leftover
        .unseen()
        .map(|mass| format!("{mass:.4}"))
        .unwrap_or_else(|| "null".to_owned());
    let packing_p0 = packing
        .unseen()
        .map(|mass| format!("{mass:.4}"))
        .unwrap_or_else(|| "null".to_owned());
    let cell_p0 = cells
        .unseen()
        .map(|mass| format!("{mass:.4}"))
        .unwrap_or_else(|| "null".to_owned());
    let conductance_s = conductance
        .map(|value| format!("{value:.4}"))
        .unwrap_or_else(|| "null".to_owned());
    let algebraic_s = algebraic
        .map(|value| format!("{value:.4}"))
        .unwrap_or_else(|| "null".to_owned());
    let fes_delta_s = fes_delta
        .map(|value| format!("{value:.4}"))
        .unwrap_or_else(|| "null".to_owned());
    println!(
        "{{\"kind\":\"occupancy_gt\",\"leftover_n\":{},\"leftover_n1\":{},\"leftover_p0\":{},\"leftover_sat\":{},\"leftover_dwell\":{},\"packing_n\":{},\"packing_n1\":{},\"packing_p0\":{},\"packing_sat\":{},\"families\":{},\"min_families\":{},\"n_floor\":{},\"p0_ceiling\":{},\"conductance\":{},\"algebraic_connectivity\":{},\"seam_left\":{},\"seam_right\":{},\"seam_packings\":{},\"measured_floor\":{},\"landfold_floor\":{},\"landfold_left\":{},\"landfold_right\":{},\"ring_floor\":{},\"ring_distinct\":{},\"ring_n\":{},\"fes_delta\":{},\"fes_minima\":{},\"sparsified_n\":{},\"sparsified_n1\":{},\"sparsified_sat\":{},\"cell_n\":{},\"cell_n1\":{},\"cell_p0\":{},\"funnel_obs\":{},\"ei_exhausted\":{},\"occupied_communities\":{},\"worthwhile_communities\":{},\"landfold_holes\":{},\"landfold_communities\":{},\"shannon\":{:.4},\"shannon_ceiling\":{:.4},\"stop\":{}}}",
        leftover.n,
        leftover.n1,
        leftover_p0,
        leftover_sat,
        scientific.leftover_dwell,
        packing.n,
        packing.n1,
        packing_p0,
        packing_sat,
        families,
        min_families,
        crate::catalog::PRODUCTION_MINIMUM_VISITS,
        crate::catalog::PRODUCTION_MAX_UNSEEN_MASS,
        conductance_s,
        algebraic_s,
        seam_left,
        seam_right,
        seam_packings,
        measured_floor,
        landfold_floor,
        landfold_left,
        landfold_right,
        ring_floor,
        ring_distinct,
        ring_n,
        fes_delta_s,
        fes_minima,
        sparsified_sample.n,
        sparsified_sample.n1,
        sparsified_sat,
        cells.n,
        cells.n1,
        cell_p0,
        funnel_obs,
        ei_exhausted,
        occupied_communities,
        worthwhile,
        sparsified.holes,
        sparsified.communities,
        // How the arrivals are spread, not only how many communities were
        // reached. A run can hold every community on the book and still
        // put all but a handful of its arrivals in one of them, which is
        // what a double funnel looks like from inside. The ceiling is the
        // even spread.
        sparsified.entropy(),
        sparsified.entropy_ceiling(),
        stop,
    );
    report_occupancy_landfold(&sparsified);
    let _ = std::io::Write::flush(&mut std::io::stdout());
}

fn report_occupancy_landfold(map: &crate::catalog::OccupancyBookMap) {
    let mut points = String::new();
    for (index, point) in map.points.iter().enumerate() {
        if index > 0 {
            points.push(',');
        }
        points.push_str(&format!(
            "{{\"family\":{},\"community\":{},\"x\":{:.6},\"y\":{:.6},\"wells\":{}}}",
            point.family, point.community, point.xy[0], point.xy[1], point.wells
        ));
    }
    let sample = map.sample();
    println!(
        "{{\"kind\":\"occupancy_landfold\",\"floor\":{},\"left\":{},\"right\":{},\"communities\":{},\"holes\":{},\"fes_minima\":{},\"sparsified_n\":{},\"sparsified_n1\":{},\"points\":[{}]}}",
        map.floor,
        map.left,
        map.right,
        map.communities,
        map.holes,
        map.fes_minima,
        sample.n,
        sample.n1,
        points
    );
}

fn record_energy(scientific: &mut ScientificState, replica: u32, energy: f64) {
    if !energy.is_finite() {
        return;
    }
    *scientific.trial_hops.entry(replica).or_insert(0) += 1;
    let history = scientific.energy_history.entry(replica).or_default();
    history.push_back(energy);
    while history.len() > 64 {
        history.pop_front();
    }
}

fn hyperband_max_resource() -> u64 {
    std::env::var("CATALOG_MAX_HOPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&hops| hops >= crate::catalog::MIN_RESOURCE)
        .unwrap_or(crate::catalog::DEFAULT_MAX_RESOURCE)
}

fn hyperband_prune(scientific: &ScientificState, replica: u32) -> bool {
    if scientific.pending_reseed.contains(&replica) {
        return true;
    }
    let max_resource = hyperband_max_resource();
    let walks: Vec<WalkRecord> = scientific
        .last_candidate_by_replica
        .iter()
        .map(|(id, candidate)| {
            let family = replica_family_index(scientific, *id, Some(candidate));
            WalkRecord {
                id: *id,
                resource: scientific.trial_hops.get(id).copied().unwrap_or(0),
                energy: candidate.energy,
                family,
            }
        })
        .collect();
    prune(&walks, replica, max_resource)
}

fn reset_trial(scientific: &mut ScientificState, replica: u32) {
    scientific.energy_history.remove(&replica);
    scientific.family_history.remove(&replica);
    scientific.trial_hops.remove(&replica);
    scientific.last_candidate_by_replica.remove(&replica);
    scientific.best_candidate_by_replica.remove(&replica);
    scientific.pending_reseed.insert(replica);
}

fn remember_candidate(scientific: &mut ScientificState, replica: u32, candidate: CatalogCandidate) {
    scientific.pending_reseed.remove(&replica);
    scientific
        .last_candidate_by_replica
        .insert(replica, candidate);
}

fn replica_series(scientific: &ScientificState, replica: u32) -> Vec<f64> {
    scientific
        .energy_history
        .get(&replica)
        .map(|history| history.iter().copied().collect())
        .unwrap_or_default()
}

fn mixing_from_state(scientific: &ScientificState) -> MixingEvidence {
    struct Family {
        energy: f64,
        occupancy: usize,
        series: Vec<Vec<f64>>,
    }
    let mut families: BTreeMap<usize, Family> = BTreeMap::new();
    let mut assigned = BTreeSet::new();
    for (replica, candidate) in &scientific.last_candidate_by_replica {
        assigned.insert(*replica);
        let series = replica_series(scientific, *replica);
        let family = replica_family_index(scientific, *replica, Some(candidate));
        let Some(index) = family else {
            continue;
        };
        let entry = families.entry(index).or_insert(Family {
            energy: candidate.energy,
            occupancy: 0,
            series: Vec::new(),
        });
        entry.occupancy += 1;
        entry.energy = entry.energy.min(candidate.energy);
        if series.len() >= 2 {
            entry.series.push(series);
        }
    }
    let attractors: Vec<AttractorStrength> = families
        .values()
        .map(|family| AttractorStrength {
            energy: family.energy,
            occupancy: family.occupancy,
            occupant_rhat: occupant_rhat(&family.series),
        })
        .collect();
    let deepest = attractors
        .iter()
        .enumerate()
        .filter(|(_, attractor)| attractor.energy.is_finite())
        .min_by(|(_, left), (_, right)| {
            left.energy
                .partial_cmp(&right.energy)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, attractor)| (index, attractor.energy));
    let deepest_unique = deepest.is_some_and(|(index, energy)| {
        attractors
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .all(|(_, attractor)| attractor.energy > energy + 1e-10)
    });
    let deepest_energy = deepest.map(|(_, energy)| energy);
    let mut explore = Vec::new();
    if families.len() <= 1 || !deepest_unique {
        for history in scientific.energy_history.values() {
            let series: Vec<f64> = history.iter().copied().collect();
            if series.len() >= 2 {
                explore.push(series);
            }
        }
    } else if let Some(floor) = deepest_energy {
        for family in families.values() {
            if family.energy > floor + 1e-10 {
                explore.extend(family.series.iter().cloned());
            }
        }
        for (replica, history) in &scientific.energy_history {
            if assigned.contains(replica) {
                continue;
            }
            let series: Vec<f64> = history.iter().copied().collect();
            if series.len() >= 2 {
                explore.push(series);
            }
        }
    }
    let mut evidence = invert_mixing(&attractors, &explore);
    if explore_must_leave(&explore, n_on_incumbent_packing(scientific), assigned.len()) {
        evidence.explore_collapsed = true;
    }
    evidence
}

fn family_champion_replicas(scientific: &ScientificState) -> BTreeSet<u32> {
    let mut best: BTreeMap<usize, (u32, f64)> = BTreeMap::new();
    for (id, candidate) in &scientific.last_candidate_by_replica {
        let Some(histogram) = scientific.packing.histogram(&candidate.coordinates) else {
            continue;
        };
        let Some(family) = scientific.packing.family_of(&histogram) else {
            continue;
        };
        match best.get(&family) {
            None => {
                best.insert(family, (*id, candidate.energy));
            }
            Some((_, energy)) if candidate.energy < *energy - 1e-12 => {
                best.insert(family, (*id, candidate.energy));
            }
            _ => {}
        }
    }
    best.values().map(|(id, _)| *id).collect()
}

fn n_on_incumbent_packing(scientific: &ScientificState) -> usize {
    let Some(incumbent) = scientific.catalog.incumbent() else {
        return 0;
    };
    let Some(incumbent_hist) = scientific.packing.histogram(incumbent.coordinates()) else {
        return 0;
    };
    scientific
        .last_candidate_by_replica
        .values()
        .filter(|candidate| {
            scientific
                .packing
                .histogram(&candidate.coordinates)
                .is_some_and(|histogram| same_packing(&histogram, &incumbent_hist))
        })
        .count()
}

fn replica_packing(scientific: &ScientificState, replica: u32) -> Option<Vec<f64>> {
    scientific
        .last_candidate_by_replica
        .get(&replica)
        .and_then(|candidate| scientific.packing.histogram(&candidate.coordinates))
}

fn replica_family_index(
    scientific: &ScientificState,
    replica: u32,
    candidate: Option<&CatalogCandidate>,
) -> Option<usize> {
    let coordinates = candidate
        .map(|candidate| candidate.coordinates.as_slice())
        .or_else(|| {
            scientific
                .last_candidate_by_replica
                .get(&replica)
                .map(|candidate| candidate.coordinates.as_slice())
        });
    if let Some(family) = coordinates
        .and_then(|coordinates| scientific.packing.histogram(coordinates))
        .and_then(|histogram| scientific.packing.family_of(&histogram))
    {
        return Some(family);
    }
    scientific
        .family_history
        .get(&replica)
        .and_then(|history| history.back().copied())
        .map(|index| index as usize)
}

/// Seat this replica and report the \(\lambda\) of the descriptor it
/// just posted. The seat carries the path maximum, which is the TIS
/// order parameter for interface crossing; the second value is this
/// frame alone, so a client can rank the frames of its own path.
fn assign_leftover_interfaces(
    scientific: &mut ScientificState,
    replica: u32,
    descriptor: &[f64],
    posted_lambda: f64,
    relation: CatalogRelation,
) -> (InterfaceSeat, f64) {
    let centroid = scientific
        .catalog
        .incumbent()
        .map(|entry| entry.descriptor().to_vec());
    let computed = centroid
        .as_ref()
        .map(|mean| leftover_lambda(descriptor, mean))
        .unwrap_or(0.0);
    let lambda = if posted_lambda.is_finite() && posted_lambda > 0.0 {
        posted_lambda.max(computed)
    } else {
        computed
    };
    scientific
        .leftover_lambda_by_replica
        .insert(replica, lambda);
    let champions = family_champion_replicas(scientific);
    if matches!(relation, CatalogRelation::Incumbent)
        || champions.contains(&replica)
        || centroid.is_none()
    {
        let seat = InterfaceSeat {
            replica,
            rank: CHAMPION_RANK,
            threshold: 0.0,
            lambda,
        };
        scientific.interface_seat_by_replica.insert(replica, seat);
        return (seat, computed);
    }
    let extras: Vec<(u32, f64)> = scientific
        .leftover_lambda_by_replica
        .iter()
        .filter(|(id, _)| !champions.contains(*id))
        .map(|(id, value)| (*id, *value))
        .collect();
    let horizon = INTERFACE_HORIZON;
    let held: Vec<InterfaceSeat> = scientific
        .interface_seat_by_replica
        .values()
        .copied()
        .collect();
    let mut seats = seat_extras(&held, &extras, horizon);
    let _ = promote_one_sided(&mut seats);
    let _ = retis_exchange_adjacent(&mut seats);
    scientific.interface_seat_by_replica.clear();
    for id in &champions {
        scientific.interface_seat_by_replica.insert(
            *id,
            InterfaceSeat {
                replica: *id,
                rank: CHAMPION_RANK,
                threshold: 0.0,
                lambda: scientific
                    .leftover_lambda_by_replica
                    .get(id)
                    .copied()
                    .unwrap_or(0.0),
            },
        );
    }
    for seat in seats {
        scientific
            .interface_seat_by_replica
            .insert(seat.replica, seat);
    }
    let seat = scientific
        .interface_seat_by_replica
        .get(&replica)
        .copied()
        .unwrap_or(InterfaceSeat {
            replica,
            rank: 0,
            threshold: horizon,
            lambda,
        });
    (seat, computed)
}

fn packing_or_region_relation(
    scientific: &ScientificState,
    replica: u32,
    local_basin: Option<BasinId>,
    energy: f64,
    mixing: MixingEvidence,
) -> CatalogRelation {
    if let Some(relation) = packing_relation(scientific, replica, energy, mixing) {
        return relation;
    }
    attraction_region_relation(scientific, replica, local_basin)
}

fn packing_relation(
    scientific: &ScientificState,
    replica: u32,
    energy: f64,
    _mixing: MixingEvidence,
) -> Option<CatalogRelation> {
    if scientific.catalog.entries().is_empty() {
        return None;
    }
    let local = replica_packing(scientific, replica)?;
    let mut compared = false;
    let mut same_as_any = false;
    let mut same_as_lower_isomer = false;
    let mut best_of_family: Option<f64> = None;
    for entry in scientific.catalog.entries() {
        let Some(entry_fp) = scientific.packing.histogram(entry.coordinates()) else {
            continue;
        };
        compared = true;
        if same_packing(&local, &entry_fp) {
            same_as_any = true;
            best_of_family =
                Some(best_of_family.map_or(entry.energy(), |best| best.min(entry.energy())));
            if entry.energy() < energy - 1e-10 {
                same_as_lower_isomer = true;
            }
        }
    }
    for candidate in scientific.last_candidate_by_replica.values() {
        let Some(other) = scientific.packing.histogram(&candidate.coordinates) else {
            continue;
        };
        if same_packing(&local, &other) {
            same_as_any = true;
            best_of_family =
                Some(best_of_family.map_or(candidate.energy, |best| best.min(candidate.energy)));
            if candidate.energy < energy - 1e-10 {
                same_as_lower_isomer = true;
            }
        }
    }
    if !compared {
        return None;
    }
    let unique_champion = family_champion_replicas(scientific).contains(&replica);
    match packing_role(same_as_any, energy, best_of_family) {
        PackingRole::NovelFamily => Some(CatalogRelation::UnrelatedNoAnchor),
        PackingRole::FamilyChampion if unique_champion => Some(CatalogRelation::Incumbent),
        PackingRole::FamilyChampion | PackingRole::FamilyExtra if same_as_lower_isomer => {
            Some(CatalogRelation::UnrelatedLowerAnchor)
        }
        PackingRole::FamilyChampion | PackingRole::FamilyExtra => Some(CatalogRelation::SameBasin),
    }
}

fn attraction_region_relation(
    scientific: &ScientificState,
    replica: u32,
    local_basin: Option<BasinId>,
) -> CatalogRelation {
    let Some(local_basin) = local_basin else {
        return CatalogRelation::Empty;
    };
    let energy_relation = || {
        if scientific
            .catalog
            .incumbent()
            .map(|entry| entry.census_id())
            == Some(local_basin)
        {
            CatalogRelation::Incumbent
        } else {
            // A lower energy in another census basin is a different
            // funnel. Exploit is a better isomer of the occupied
            // packing only; packing_relation is what may set
            // UnrelatedLowerAnchor.
            CatalogRelation::UnrelatedNoAnchor
        }
    };
    let Some(local_node) = scientific.transition_nodes.get(&local_basin).copied() else {
        return energy_relation();
    };
    let Ok(regions) = scientific
        .transition_graph
        .attraction_regions(&scientific.attraction_regions)
    else {
        return CatalogRelation::Empty;
    };
    let mut node_region = vec![usize::MAX; scientific.transition_graph.node_count()];
    for (region, nodes) in regions.iter().enumerate() {
        for node in nodes {
            node_region[*node] = region;
        }
    }
    let Some(local_region) = node_region.get(local_node).copied() else {
        return CatalogRelation::Empty;
    };
    let shared = scientific
        .last_basin_by_replica
        .iter()
        .filter(|(other, _)| **other != replica)
        .filter_map(|(_, basin)| scientific.transition_nodes.get(basin))
        .any(|node| node_region.get(*node).copied() == Some(local_region));
    if shared {
        return CatalogRelation::SameBasin;
    }
    energy_relation()
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

/// Aggregate read-only status for an observer.
fn observer_status_reply(
    config: &ServerConfig,
    state: &CoordinatorState,
    event_sequence: u64,
) -> CatalogReply {
    let mut replicas = Vec::new();
    for replica in &config.replicas {
        let charged_work = state
            .ledger
            .as_ref()
            .and_then(|ledger| ledger.replica_total(*replica))
            .unwrap_or(0);
        let best_energy = state
            .scientific
            .as_ref()
            .and_then(|scientific| scientific.best_candidate_by_replica.get(replica))
            .map_or(f64::INFINITY, |candidate| candidate.energy);
        replicas.push(crate::catalog_rpc::ReplicaProgress {
            replica: *replica,
            charged_work,
            best_energy,
        });
    }
    let (open_epoch, epoch_submitted, epoch_required) =
        state.scientific.as_ref().map_or((0, 0, 0), |scientific| {
            let open = scientific.population.open_epoch();
            let submitted = scientific
                .population_candidates
                .get(&open)
                .map_or(0, BTreeMap::len);
            (
                open,
                u32::try_from(submitted).unwrap_or(u32::MAX),
                u32::try_from(scientific.population.open_requirement()).unwrap_or(u32::MAX),
            )
        });
    let landscape_basins = state.scientific.as_ref().map_or(0, |scientific| {
        u32::try_from(scientific.landscape.len()).unwrap_or(u32::MAX)
    });
    let seam = state
        .scientific
        .as_ref()
        .and_then(|scientific| scientific.landscape.spectral_split().ok())
        .map(|split| crate::catalog_rpc::LandscapeSeam {
            algebraic_connectivity: split.algebraic_connectivity,
            conductance: split.conductance,
            community_left: u32::try_from(split.left.len()).unwrap_or(u32::MAX),
            community_right: u32::try_from(split.right.len()).unwrap_or(u32::MAX),
            left_basin: split.representatives.0,
            right_basin: split.representatives.1,
        });
    let status = crate::catalog_rpc::CoordinatorStatus {
        snapshot_version: state.snapshot_version,
        open_epoch,
        epoch_submitted,
        epoch_required,
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
        replicas,
        landscape_basins,
        seam,
    };
    let snapshot = CatalogSnapshot {
        version: state.snapshot_version,
        census_visits: state.census_visits,
        active_entries: state.active_entries,
        aggregate_charged: status.aggregate_charged,
        aggregate_budget: status.aggregate_budget,
    };
    CatalogReply::Accepted(AcceptedReply {
        event_sequence,
        duplicate: true,
        snapshot,
        payload: AcceptedPayload::CoordinatorStatus(status),
    })
}

fn identity_rejection(
    config: &ServerConfig,
    identity: &CatalogIdentity,
) -> Option<ProtocolRejection> {
    if let Some(reason) = system_identity_rejection(config, identity) {
        return Some(reason);
    }
    if !config.replicas.contains(&identity.replica) {
        Some(ProtocolRejection::ReplicaMismatch)
    } else {
        None
    }
}

fn system_identity_rejection(
    config: &ServerConfig,
    identity: &CatalogIdentity,
) -> Option<ProtocolRejection> {
    if identity.campaign != config.campaign {
        Some(ProtocolRejection::CampaignMismatch)
    } else if identity.ensemble != config.ensemble {
        Some(ProtocolRejection::EnsembleMismatch)
    } else if identity.signature_digest != config.signature_digest {
        Some(ProtocolRejection::SignatureMismatch)
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
