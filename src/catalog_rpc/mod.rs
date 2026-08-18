//! Versioned, ensemble-isolated cooperative catalog wire types.

use std::io::Cursor;

use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;

use crate::Catalog_capnp::{
    CatalogMutationKind as WireCatalogMutationKind, CatalogRelation as WireCatalogRelation,
    QuenchStatus as WireQuenchStatus, RejectionKind, accepted_reply, bridge_assignment,
    candidate_record, catalog_mutation_reply, catalog_reply, catalog_request, policy_state_reply,
    population_epoch_reply, transition_record,
};

pub mod client;
pub mod mailbox;
pub mod server;

/// Wire protocol version accepted by this release.
pub const PROTOCOL_VERSION: u16 = 11;
/// `Sample` draw that returns the active-catalog incumbent.
pub const INCUMBENT_SAMPLE_DRAW: u64 = u64::MAX;

/// `Sample` draw that returns a representative of the packing family
/// the fewest live replicas are standing on.
///
/// A replica leaving a crowded packing cannot reach another funnel by
/// perturbing within its own: a move drawn inside a funnel and relaxed
/// downhill lands back in it, which is what a funnel is. Crossing has
/// to be a draw from somewhere else. This is that draw, with the
/// catalog standing in for a fitted model of each funnel.
pub const SPARSE_SAMPLE_DRAW: u64 = u64::MAX - 1;

/// Complete identity carried by every catalog request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogIdentity {
    /// Campaign identity.
    pub campaign: String,
    /// Independent ensemble identity.
    pub ensemble: String,
    /// Replica identity within the ensemble.
    pub replica: u32,
    /// Canonical system-signature digest.
    pub signature_digest: [u8; 32],
}

/// Complete scientific candidate carried under the request identity.
#[derive(Debug, Clone, PartialEq)]
pub struct CatalogCandidate {
    /// Replica that produced the candidate.
    pub producer_replica: u32,
    /// Cartesian coordinates in system-signature order.
    pub coordinates: Vec<f64>,
    /// Row-major cell, or no cell for a finite nonperiodic system.
    pub cell: Option<[f64; 9]>,
    /// Producer-side quenched energy.
    pub energy: f64,
    /// Producer-side forces in coordinate order.
    pub forces: Vec<f64>,
    /// Producer-side Euclidean gradient norm.
    pub gradient_norm: f64,
    /// Descriptor under the declared schema.
    pub descriptor: Vec<f64>,
    /// Descriptor schema version.
    pub descriptor_schema_version: u32,
    /// Whether the producer quench met its convergence contract.
    pub quench_converged: bool,
    /// Producer charged-work counter at this candidate.
    pub charged_work: u64,
    /// Producer event sequence retained inside the immutable record.
    pub event_sequence: u64,
    /// Producer random-seed identity.
    pub seed: u64,
    /// Coordinator-assigned fixed-census basin for returned representatives.
    pub census_basin: Option<u64>,
}

/// Resolved or unresolved result of one action-labelled perturb--quench step.
#[derive(Debug, Clone, PartialEq)]
pub enum TransitionDestination {
    /// The step did not yield a valid classified minimum.
    Unresolved,
    /// The step reached a candidate requiring receiving-side validation.
    Resolved(CatalogCandidate),
}

/// Exact serialized outcome of one active-catalog admission attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogMutationKind {
    /// Candidate filled an available active slot.
    Added,
    /// Candidate improved its basin's active representative.
    ReplacedSameBasin,
    /// Candidate replaced its complete packing-conflict set.
    ReplacedConflicts,
    /// Candidate replaced the eligible entry at capacity.
    ReplacedCapacity,
    /// Same-basin representative had lower or equal energy.
    RejectedSameBasin,
    /// A packing-conflict representative had lower or equal energy.
    RejectedConflict,
    /// The eligible entry at capacity had lower or equal energy.
    RejectedCapacity,
}

impl CatalogMutationKind {
    /// Whether the active catalog changed.
    pub const fn admitted(self) -> bool {
        matches!(
            self,
            Self::Added
                | Self::ReplacedSameBasin
                | Self::ReplacedConflicts
                | Self::ReplacedCapacity
        )
    }

    /// Stable event-stream code.
    pub const fn code(self) -> &'static str {
        match self {
            Self::Added => "added",
            Self::ReplacedSameBasin => "replaced_same_basin",
            Self::ReplacedConflicts => "replaced_conflicts",
            Self::ReplacedCapacity => "replaced_capacity",
            Self::RejectedSameBasin => "rejected_same_basin",
            Self::RejectedConflict => "rejected_conflict",
            Self::RejectedCapacity => "rejected_capacity",
        }
    }
}

/// Catalog identity changes returned by a serialized admission attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogMutation {
    /// Fixed-census basin assigned to the offered candidate.
    pub basin_id: u64,
    /// Exact admission or rejection class.
    pub kind: CatalogMutationKind,
    /// Fixed-census basins removed from the active catalog.
    pub evicted: Vec<u64>,
    /// Active incumbent after the attempt.
    pub incumbent_basin: Option<u64>,
}

/// Mutation or read operation carried by a catalog request.
#[derive(Debug, Clone, PartialEq)]
pub enum CatalogOperation {
    /// Read the current versioned snapshot.
    Snapshot,
    /// Record one exact fixed-census observation.
    RecordVisit {
        /// Candidate that must pass coordinator validation before observation.
        candidate: CatalogCandidate,
    },
    /// Offer one validated candidate to the active catalog.
    OfferCandidate {
        /// Candidate that must pass coordinator validation before admission.
        candidate: CatalogCandidate,
    },
    /// Sample an admissible active representative.
    Sample {
        /// Explicit deterministic random draw.
        draw: u64,
    },
    /// Request a sampled farthest-hole proposal.
    DescriptorHole {
        /// Current replica descriptor.
        current: Vec<f64>,
        /// Number of unit-sphere samples.
        samples: u32,
        /// Explicit deterministic random draw.
        draw: u64,
    },
    /// Request an observed crossing usable from the query's attraction region.
    BoundaryCrossing {
        /// Current replica descriptor under the fixed catalogue schema.
        current: Vec<f64>,
        /// Explicit deterministic random draw.
        draw: u64,
    },
    /// Read exact census and active-catalog evidence for one descriptor.
    PolicyState {
        /// Descriptor to classify against the fixed census and active catalog.
        descriptor: Vec<f64>,
        /// Candidate energy used to compare an unrelated incumbent anchor.
        energy: f64,
        /// Highest leftover-SOAP \(\lambda\) on this replica's Leave path.
        leftover_lambda: f64,
    },
    /// Submit one replay-safe charged-work event.
    LedgerEvent {
        /// Stable charge-kind discriminant.
        kind: u16,
        /// Potential calls charged by this event.
        charged_calls: u64,
        /// Replica counter including this event.
        cumulative_charged: u64,
    },
    /// Submit one validated representative to a synchronous population epoch.
    PopulationSubmit {
        /// Charged-work synchronization epoch.
        epoch: u64,
        /// Representative requiring receiving-side scientific validation.
        candidate: CatalogCandidate,
    },
    /// Poll an immutable synchronous population plan.
    PopulationPlan {
        /// Charged-work synchronization epoch.
        epoch: u64,
    },
    /// Decline to submit to one epoch, releasing the replicas waiting on it.
    PopulationAbstain {
        /// Charged-work synchronization epoch.
        epoch: u64,
    },
    /// Join one epoch by reference to the best validated candidate on file.
    PopulationJoin {
        /// Charged-work synchronization epoch.
        epoch: u64,
    },
    /// Read-only aggregate status, answerable by any observer that names
    /// the right campaign and ensemble.
    ObserverStatus,
    /// Poll for a bridge segment assignment; the argument selects an
    /// entry state from the assigned region's stored entries.
    BridgeAssignment {
        /// Caller-supplied draw for entry selection.
        draw: u64,
    },
    /// Report one attempted exit from a bridge region.
    BridgeCrossing {
        /// Crossing record: bridge, regions, descriptor, state, energy.
        crossing: BridgeCrossingRecord,
    },
    /// Record one action-conditioned transition from the replica's live basin.
    RecordTransition {
        /// Stable target-blind proposal-action identifier.
        action: String,
        /// Resolved validated endpoint or explicit unresolved outcome.
        destination: TransitionDestination,
        /// Whether the reached endpoint became the replica's live state.
        adopted: bool,
    },
}

/// Complete catalog request.
#[derive(Debug, Clone, PartialEq)]
pub struct CatalogRequest {
    /// Requested protocol version.
    pub protocol_version: u16,
    /// Campaign, ensemble, replica, and system identity.
    pub identity: CatalogIdentity,
    /// Monotone replica event sequence.
    pub event_sequence: u64,
    /// Latest snapshot version observed by the replica.
    pub snapshot_version: u64,
    /// Requested operation.
    pub operation: CatalogOperation,
}

/// Typed coordinator rejection returned on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtocolRejection {
    /// Request bytes do not satisfy the schema.
    Malformed,
    /// Protocol version is unsupported.
    UnsupportedVersion,
    /// Campaign identity differs.
    CampaignMismatch,
    /// Ensemble identity differs.
    EnsembleMismatch,
    /// Replica is outside the configured ensemble.
    ReplicaMismatch,
    /// System signature differs.
    SignatureMismatch,
    /// Sequence was already used by different content.
    SequenceReplay,
    /// Sequence moves backward.
    SequenceRegression,
    /// Client snapshot version exceeds coordinator state.
    SnapshotRegression,
    /// Scientific validation rejected the record.
    ValidationRejected,
}

/// Snapshot counters returned by every accepted request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatalogSnapshot {
    /// Monotone coordinator snapshot version.
    pub version: u64,
    /// Exact fixed-census visit total.
    pub census_visits: u64,
    /// Number of finite active entries.
    pub active_entries: u32,
    /// Sum of the latest exact charged counters from all replicas.
    pub aggregate_charged: u64,
    /// Declared ensemble charged-work budget.
    pub aggregate_budget: u64,
}

/// Seeded target-free descriptor-hole result.
#[derive(Debug, Clone, PartialEq)]
pub struct DescriptorHoleProposal {
    /// Selected unit-sphere descriptor target.
    pub target: Vec<f64>,
    /// Descriptor increment from the supplied current point.
    pub increment: Vec<f64>,
    /// Distance from the target to its nearest catalog descriptor.
    pub nearest_catalog_distance: f64,
}

/// One validated adopted crossing shared across cooperative replicas.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryCrossingRecord {
    /// Target-blind move family that generated the crossing.
    pub action: String,
    /// Validated source minimum coordinates.
    pub from: Vec<f64>,
    /// Validated destination minimum coordinates.
    pub to: Vec<f64>,
    /// Fixed-census source basin.
    pub source_basin: u64,
    /// Fixed-census destination basin.
    pub destination_basin: u64,
}

/// Relation between a replica candidate and the active catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogRelation {
    /// The active catalog has no entries.
    Empty,
    /// The candidate belongs to the incumbent basin.
    Incumbent,
    /// The candidate belongs to another active basin.
    SameBasin,
    /// The candidate is unrelated and no lower incumbent anchors exploitation.
    UnrelatedNoAnchor,
    /// The candidate is unrelated and a lower incumbent anchors exploitation.
    UnrelatedLowerAnchor,
}

/// Exact coordinator evidence used by the cooperative policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolicyState {
    /// Exact number of fixed-census observations.
    pub total_visits: u64,
    /// Exact number of singleton basins in the fixed census.
    pub singleton_basins: u64,
    /// Exact visits assigned to the candidate's basin, or zero if unassigned.
    pub local_basin_visits: u64,
    /// Whether the fixed census meets its declared saturation rule.
    pub globally_saturated: bool,
    /// Candidate relation to the active catalog.
    pub relation: CatalogRelation,
    /// Sum of the latest exact charged counters from all replicas.
    pub aggregate_charged: u64,
    /// Declared ensemble charged-work budget.
    pub aggregate_budget: u64,
    /// Stable census-basin identifier, or `None` for an unassigned descriptor.
    pub local_basin: Option<u64>,
    /// Distance from the query descriptor to its assigned immutable medoid.
    pub local_basin_distance: f64,
    /// Distance to the nearest distinct census medoid.
    pub novelty: f64,
    /// Posterior uncertainty of the latent Gaussian basin-transition field.
    pub transition_uncertainty: f64,
    /// Explore-role chains have mixed onto one attractor.
    pub explore_collapsed: bool,
    /// Incumbent attractor won the occupancy contest against competitors.
    pub certified_attractor: bool,
    /// Successive halving discarded this walk at a rung.
    pub pruned: bool,
    /// Leftover-SOAP \(\lambda\) of the descriptor in this request.
    ///
    /// One frame, not the replica's path maximum: the coordinator holds
    /// the maximum for the interface ladder, and a client that tagged
    /// every frame with it would have a non-decreasing series in which
    /// no frame is further from the occupied well than any other.
    pub leftover_lambda: f64,
    /// TIS interface rank. `u32::MAX` is the occupied-packing champion.
    pub interface_rank: u32,
    /// Threshold \(\lambda_i\) this extra must reach.
    pub interface_threshold: f64,
    /// Number of leftover-SOAP interfaces in the current ladder.
    pub interface_count: u32,
    /// Occupied DECAF families on the packing book.
    pub occupied_family_count: u32,
    /// DECAF-family Good--Turing: unseen packing mass is small.
    pub packing_saturated: bool,
}

/// Which selection produced a barrier's parent map.
///
/// The two branches carry different weights, so a reader that cannot
/// tell them apart cannot interpret the weights it is given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PopulationSelection {
    /// Written before the branch was recorded.
    Unspecified,
    /// Systematic resampling of the Feynman-Kac weights.
    SystematicResampling,
    /// Attraction-region coverage before any source is duplicated.
    RegionCovering,
}

impl PopulationSelection {
    /// Stable token used in the replica event stream.
    pub fn as_trace_str(self) -> &'static str {
        match self {
            Self::Unspecified => "unspecified",
            Self::SystematicResampling => "systematic_resampling",
            Self::RegionCovering => "region_covering",
        }
    }
}

/// Replica-addressed fixed-population plan returned by the coordinator.
#[derive(Debug, Clone, PartialEq)]
pub struct PopulationPlan {
    /// Charged-work synchronization epoch.
    pub epoch: u64,
    /// Destination replicas in stable order.
    pub destinations: Vec<u32>,
    /// Parent replica paired with every destination.
    pub parents: Vec<u32>,
    /// Normalized source weights in stable replica order.
    pub weights: Vec<f64>,
    /// Kish effective sample size before resampling.
    pub effective_sample_size: f64,
    /// Number of represented source replicas.
    pub unique_parents: u32,
    /// Largest realized offspring family.
    pub max_family_size: u32,
    /// Population variance of offspring counts.
    pub offspring_variance: f64,
    /// Validated parent record paired with every destination.
    pub parent_candidates: Vec<CatalogCandidate>,
    /// Branch that produced this parent map.
    pub selection: PopulationSelection,
}

/// Barrier state for one synchronous population epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct PopulationEpochState {
    /// Charged-work synchronization epoch.
    pub epoch: u64,
    /// Unique replica submissions received.
    pub submitted: u32,
    /// Complete population size required.
    pub required: u32,
    /// Immutable plan once every replica has submitted.
    pub plan: Option<PopulationPlan>,
}

/// Optional scientific payload returned by an accepted operation.
/// One replica's progress inside a coordinator status report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReplicaProgress {
    /// Replica identifier.
    pub replica: u32,
    /// Force evaluations the replica has recorded.
    pub charged_work: u64,
    /// Lowest validated energy the coordinator holds for the replica.
    pub best_energy: f64,
}

/// One attempted exit from a bridge region, as the confined replica
/// reports it.
#[derive(Debug, Clone, PartialEq)]
pub struct BridgeCrossingRecord {
    /// Bridge identifier.
    pub bridge: u64,
    /// Region the replica was confined to.
    pub from_region: u32,
    /// Region the attempted exit entered.
    pub to_region: u32,
    /// Descriptor at the crossing point.
    pub descriptor: Vec<f64>,
    /// Full state at the crossing point.
    pub state: Vec<f64>,
    /// Energy at the crossing point.
    pub energy: f64,
}

/// A bridge segment assignment: the string, the region to confine to,
/// and an entry state when one is stored.
#[derive(Debug, Clone, PartialEq)]
pub struct BridgeAssignmentRecord {
    /// Bridge identifier.
    pub bridge: u64,
    /// Catalog basin at the A endpoint.
    pub from_basin: u64,
    /// Catalog basin at the B endpoint.
    pub to_basin: u64,
    /// String images, row-major: `image_count` rows of the descriptor
    /// dimension.
    pub images: Vec<f64>,
    /// Number of images.
    pub image_count: u32,
    /// Region index this replica confines to.
    pub region: u32,
    /// Distance from the endpoint chord beyond which a state has left
    /// the bridge tube.
    pub tube_radius: f64,
    /// Stored entry state for the region, when any crossing has
    /// deposited one.
    pub entry: Option<Vec<f64>>,
}

/// The seam between the two most weakly coupled communities of the
/// explored landscape, as the referee reports it to an observer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LandscapeSeam {
    /// Second Laplacian eigenvalue of the basin transition graph.
    pub algebraic_connectivity: f64,
    /// Cut weight over the smaller community's volume.
    pub conductance: f64,
    /// Basin count on the left side.
    pub community_left: u32,
    /// Basin count on the right side.
    pub community_right: u32,
    /// Best-anchored basin of the left community.
    pub left_basin: u64,
    /// Best-anchored basin of the right community.
    pub right_basin: u64,
}

/// Read-only aggregate state for an observer outside the ensemble.
#[derive(Debug, Clone, PartialEq)]
pub struct CoordinatorStatus {
    /// Monotone coordinator snapshot version.
    pub snapshot_version: u64,
    /// Population epoch currently open.
    pub open_epoch: u64,
    /// Members submitted to the open epoch.
    pub epoch_submitted: u32,
    /// Members the open epoch still requires.
    pub epoch_required: u32,
    /// Exact census visit total.
    pub census_visits: u64,
    /// Active catalog entries.
    pub active_entries: u32,
    /// Sum of the replicas' recorded force evaluations.
    pub aggregate_charged: u64,
    /// Declared ensemble budget.
    pub aggregate_budget: u64,
    /// Per-replica progress, in replica order.
    pub replicas: Vec<ReplicaProgress>,
    /// Basins the transition stream has linked into the referee's graph.
    pub landscape_basins: u32,
    /// The referee's seam, when at least two basins have been linked.
    pub seam: Option<LandscapeSeam>,
}

/// Scientific record an accepted reply carries alongside the snapshot.
#[derive(Debug, Clone, PartialEq)]
pub enum AcceptedPayload {
    /// Snapshot or mutation response without an additional scientific record.
    None,
    /// Sampled validated catalog candidate.
    Candidate(CatalogCandidate),
    /// Seeded target-free descriptor-hole proposal.
    DescriptorHole(DescriptorHoleProposal),
    /// Observed crossing selected from the current attraction region.
    BoundaryCrossing(BoundaryCrossingRecord),
    /// Exact census and active-catalog policy evidence.
    PolicyState(PolicyState),
    /// Pending barrier state or a complete synchronous population plan.
    PopulationEpoch(PopulationEpochState),
    /// Exact active-catalog admission result.
    CatalogMutation(CatalogMutation),
    /// Read-only aggregate status for an observer.
    CoordinatorStatus(CoordinatorStatus),
    /// A bridge segment assignment for the requesting replica.
    BridgeAssignment(BridgeAssignmentRecord),
}

/// Accepted coordinator response.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptedReply {
    /// Request sequence being acknowledged.
    pub event_sequence: u64,
    /// Whether an identical request was replayed.
    pub duplicate: bool,
    /// Coordinator state after the request.
    pub snapshot: CatalogSnapshot,
    /// Operation-specific scientific result.
    pub payload: AcceptedPayload,
}

/// Decoded coordinator response.
#[derive(Debug, Clone, PartialEq)]
pub enum CatalogReply {
    /// Request was accepted or replayed idempotently.
    Accepted(AcceptedReply),
    /// Request was rejected without state mutation.
    Rejected {
        /// Request sequence being rejected.
        event_sequence: u64,
        /// Coordinator version at rejection.
        snapshot_version: u64,
        /// Stable rejection reason.
        reason: ProtocolRejection,
    },
}

/// Protocol validation or decoding failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProtocolError {
    /// The request version is not supported.
    #[error("unsupported protocol version {received}; supported version is {supported}")]
    UnsupportedVersion {
        /// Received version.
        received: u16,
        /// Supported version.
        supported: u16,
    },
    /// Campaign identities differ.
    #[error("catalog campaign identity mismatch")]
    CampaignMismatch,
    /// Ensemble identities differ.
    #[error("catalog ensemble identity mismatch")]
    EnsembleMismatch,
    /// Replica identities differ.
    #[error("catalog replica identity mismatch")]
    ReplicaMismatch,
    /// System signature digests differ.
    #[error("catalog system signature mismatch")]
    SignatureMismatch,
    /// Signature digest has a noncanonical length.
    #[error("signature digest length is {actual}, expected 32")]
    SignatureDigestLength {
        /// Received digest length.
        actual: usize,
    },
    /// Cap'n Proto input is malformed or outside the schema.
    #[error("malformed catalog wire message: {0}")]
    Malformed(String),
}

/// Reject a request whose mutation identity differs from the coordinator.
pub fn validate_identity(
    expected: &CatalogIdentity,
    received: &CatalogIdentity,
) -> Result<(), ProtocolError> {
    if expected.campaign != received.campaign {
        return Err(ProtocolError::CampaignMismatch);
    }
    if expected.ensemble != received.ensemble {
        return Err(ProtocolError::EnsembleMismatch);
    }
    if expected.replica != received.replica {
        return Err(ProtocolError::ReplicaMismatch);
    }
    if expected.signature_digest != received.signature_digest {
        return Err(ProtocolError::SignatureMismatch);
    }
    Ok(())
}

/// Encode one checked request as an unpacked Cap'n Proto message.
pub fn encode_request(request: &CatalogRequest) -> Result<Vec<u8>, ProtocolError> {
    check_version(request.protocol_version)?;
    let mut message = Builder::new_default();
    let mut root = message.init_root::<catalog_request::Builder>();
    root.set_protocol_version(request.protocol_version);
    root.set_event_sequence(request.event_sequence);
    root.set_snapshot_version(request.snapshot_version);
    {
        let mut identity = root.reborrow().init_identity();
        identity.set_campaign(request.identity.campaign.as_str());
        identity.set_ensemble(request.identity.ensemble.as_str());
        identity.set_replica(request.identity.replica);
        identity.set_signature_digest(&request.identity.signature_digest);
    }
    let mut operation = root.init_operation();
    match &request.operation {
        CatalogOperation::Snapshot => operation.set_snapshot(()),
        CatalogOperation::RecordVisit { candidate } => {
            fill_candidate(operation.init_record_visit(), candidate);
        }
        CatalogOperation::OfferCandidate { candidate } => {
            fill_candidate(operation.init_offer_candidate(), candidate);
        }
        CatalogOperation::Sample { draw } => operation.set_sample(*draw),
        CatalogOperation::DescriptorHole {
            current,
            samples,
            draw,
        } => {
            let mut hole = operation.init_descriptor_hole();
            fill_f64(hole.reborrow().init_current(current.len() as u32), current);
            hole.set_samples(*samples);
            hole.set_draw(*draw);
        }
        CatalogOperation::BoundaryCrossing { current, draw } => {
            let mut crossing = operation.init_boundary_crossing();
            fill_f64(
                crossing.reborrow().init_current(current.len() as u32),
                current,
            );
            crossing.set_draw(*draw);
        }
        CatalogOperation::PolicyState {
            descriptor,
            energy,
            leftover_lambda,
        } => {
            let mut state = operation.init_policy_state();
            fill_f64(
                state.reborrow().init_descriptor(descriptor.len() as u32),
                descriptor,
            );
            state.set_energy(*energy);
            state.set_leftover_lambda(*leftover_lambda);
        }
        CatalogOperation::LedgerEvent {
            kind,
            charged_calls,
            cumulative_charged,
        } => {
            let mut ledger = operation.init_ledger_event();
            ledger.set_kind(*kind);
            ledger.set_charged_calls(*charged_calls);
            ledger.set_cumulative_charged(*cumulative_charged);
        }
        CatalogOperation::PopulationSubmit { epoch, candidate } => {
            let mut submission = operation.init_population_submit();
            submission.set_epoch(*epoch);
            fill_candidate(submission.init_candidate(), candidate);
        }
        CatalogOperation::PopulationPlan { epoch } => {
            operation.init_population_plan().set_epoch(*epoch);
        }
        CatalogOperation::PopulationAbstain { epoch } => {
            operation.init_population_abstain().set_epoch(*epoch);
        }
        CatalogOperation::PopulationJoin { epoch } => {
            operation.init_population_join().set_epoch(*epoch);
        }
        CatalogOperation::ObserverStatus => {
            operation.set_observer_status(());
        }
        CatalogOperation::BridgeAssignment { draw } => {
            operation.set_bridge_assignment(*draw);
        }
        CatalogOperation::BridgeCrossing { crossing } => {
            let mut wire = operation.init_bridge_crossing();
            wire.set_bridge(crossing.bridge);
            wire.set_from_region(crossing.from_region);
            wire.set_to_region(crossing.to_region);
            fill_f64(
                wire.reborrow()
                    .init_descriptor(crossing.descriptor.len() as u32),
                &crossing.descriptor,
            );
            fill_f64(
                wire.reborrow().init_state(crossing.state.len() as u32),
                &crossing.state,
            );
            wire.set_energy(crossing.energy);
        }
        CatalogOperation::RecordTransition {
            action,
            destination,
            adopted,
        } => {
            let mut transition = operation.init_record_transition();
            transition.set_action(action.as_str());
            transition.set_adopted(*adopted);
            let mut wire_destination = transition.init_destination();
            match destination {
                TransitionDestination::Unresolved => wire_destination.set_unresolved(()),
                TransitionDestination::Resolved(candidate) => {
                    fill_candidate(wire_destination.init_resolved(), candidate);
                }
            }
        }
    }
    let mut bytes = Vec::new();
    serialize::write_message(&mut bytes, &message).map_err(wire_error)?;
    Ok(bytes)
}

/// Decode one request and enforce its protocol version and digest shape.
pub fn decode_request(bytes: &[u8]) -> Result<CatalogRequest, ProtocolError> {
    let mut cursor = Cursor::new(bytes);
    let message = serialize::read_message(&mut cursor, ReaderOptions::new()).map_err(wire_error)?;
    let root = message
        .get_root::<catalog_request::Reader>()
        .map_err(wire_error)?;
    decode_request_reader(root)
}

pub(crate) fn decode_request_reader(
    root: catalog_request::Reader<'_>,
) -> Result<CatalogRequest, ProtocolError> {
    let protocol_version = root.get_protocol_version();
    check_version(protocol_version)?;
    let identity_reader = root.get_identity().map_err(wire_error)?;
    let digest = identity_reader.get_signature_digest().map_err(wire_error)?;
    let signature_digest: [u8; 32] =
        digest
            .try_into()
            .map_err(|_| ProtocolError::SignatureDigestLength {
                actual: digest.len(),
            })?;
    let identity = CatalogIdentity {
        campaign: text_value(identity_reader.get_campaign().map_err(wire_error)?)?,
        ensemble: text_value(identity_reader.get_ensemble().map_err(wire_error)?)?,
        replica: identity_reader.get_replica(),
        signature_digest,
    };
    let operation = match root.get_operation().which().map_err(wire_error)? {
        catalog_request::operation::Snapshot(()) => CatalogOperation::Snapshot,
        catalog_request::operation::RecordVisit(record) => CatalogOperation::RecordVisit {
            candidate: read_candidate(record.map_err(wire_error)?)?,
        },
        catalog_request::operation::OfferCandidate(candidate) => CatalogOperation::OfferCandidate {
            candidate: read_candidate(candidate.map_err(wire_error)?)?,
        },
        catalog_request::operation::Sample(draw) => CatalogOperation::Sample { draw },
        catalog_request::operation::DescriptorHole(hole) => {
            let hole = hole.map_err(wire_error)?;
            CatalogOperation::DescriptorHole {
                current: list_f64(hole.get_current().map_err(wire_error)?),
                samples: hole.get_samples(),
                draw: hole.get_draw(),
            }
        }
        catalog_request::operation::BoundaryCrossing(crossing) => {
            let crossing = crossing.map_err(wire_error)?;
            CatalogOperation::BoundaryCrossing {
                current: list_f64(crossing.get_current().map_err(wire_error)?),
                draw: crossing.get_draw(),
            }
        }
        catalog_request::operation::PolicyState(state) => {
            let state = state.map_err(wire_error)?;
            CatalogOperation::PolicyState {
                descriptor: list_f64(state.get_descriptor().map_err(wire_error)?),
                energy: state.get_energy(),
                leftover_lambda: state.get_leftover_lambda(),
            }
        }
        catalog_request::operation::LedgerEvent(ledger) => {
            let ledger = ledger.map_err(wire_error)?;
            CatalogOperation::LedgerEvent {
                kind: ledger.get_kind(),
                charged_calls: ledger.get_charged_calls(),
                cumulative_charged: ledger.get_cumulative_charged(),
            }
        }
        catalog_request::operation::PopulationSubmit(submission) => {
            let submission = submission.map_err(wire_error)?;
            CatalogOperation::PopulationSubmit {
                epoch: submission.get_epoch(),
                candidate: read_candidate(submission.get_candidate().map_err(wire_error)?)?,
            }
        }
        catalog_request::operation::PopulationPlan(plan) => {
            let plan = plan.map_err(wire_error)?;
            CatalogOperation::PopulationPlan {
                epoch: plan.get_epoch(),
            }
        }
        catalog_request::operation::PopulationAbstain(abstain) => {
            let abstain = abstain.map_err(wire_error)?;
            CatalogOperation::PopulationAbstain {
                epoch: abstain.get_epoch(),
            }
        }
        catalog_request::operation::PopulationJoin(join) => {
            let join = join.map_err(wire_error)?;
            CatalogOperation::PopulationJoin {
                epoch: join.get_epoch(),
            }
        }
        catalog_request::operation::ObserverStatus(()) => CatalogOperation::ObserverStatus,
        catalog_request::operation::BridgeAssignment(draw) => {
            CatalogOperation::BridgeAssignment { draw }
        }
        catalog_request::operation::BridgeCrossing(crossing) => {
            let crossing = crossing.map_err(wire_error)?;
            CatalogOperation::BridgeCrossing {
                crossing: BridgeCrossingRecord {
                    bridge: crossing.get_bridge(),
                    from_region: crossing.get_from_region(),
                    to_region: crossing.get_to_region(),
                    descriptor: list_f64(crossing.get_descriptor().map_err(wire_error)?),
                    state: list_f64(crossing.get_state().map_err(wire_error)?),
                    energy: crossing.get_energy(),
                },
            }
        }
        catalog_request::operation::RecordTransition(transition) => {
            let transition = transition.map_err(wire_error)?;
            let destination = match transition.get_destination().which().map_err(wire_error)? {
                transition_record::destination::Unresolved(()) => TransitionDestination::Unresolved,
                transition_record::destination::Resolved(candidate) => {
                    TransitionDestination::Resolved(read_candidate(candidate.map_err(wire_error)?)?)
                }
            };
            CatalogOperation::RecordTransition {
                action: text_value(transition.get_action().map_err(wire_error)?)?,
                destination,
                adopted: transition.get_adopted(),
            }
        }
    };
    Ok(CatalogRequest {
        protocol_version,
        identity,
        event_sequence: root.get_event_sequence(),
        snapshot_version: root.get_snapshot_version(),
        operation,
    })
}

pub(crate) fn encode_reply(reply: CatalogReply) -> Result<Vec<u8>, ProtocolError> {
    let mut message = Builder::new_default();
    let mut root = message.init_root::<catalog_reply::Builder>();
    root.set_protocol_version(PROTOCOL_VERSION);
    match reply {
        CatalogReply::Accepted(accepted) => {
            root.set_event_sequence(accepted.event_sequence);
            root.set_snapshot_version(accepted.snapshot.version);
            let mut body = root.init_result().init_accepted();
            body.set_duplicate(accepted.duplicate);
            body.set_census_visits(accepted.snapshot.census_visits);
            body.set_active_entries(accepted.snapshot.active_entries);
            body.set_aggregate_charged(accepted.snapshot.aggregate_charged);
            body.set_aggregate_budget(accepted.snapshot.aggregate_budget);
            let mut payload = body.init_payload();
            match &accepted.payload {
                AcceptedPayload::None => payload.set_none(()),
                AcceptedPayload::Candidate(candidate) => {
                    fill_candidate(payload.init_candidate(), candidate);
                }
                AcceptedPayload::DescriptorHole(hole) => {
                    let mut output = payload.init_descriptor_hole();
                    fill_f64(
                        output.reborrow().init_target(hole.target.len() as u32),
                        &hole.target,
                    );
                    fill_f64(
                        output
                            .reborrow()
                            .init_increment(hole.increment.len() as u32),
                        &hole.increment,
                    );
                    output.set_nearest_catalog_distance(hole.nearest_catalog_distance);
                }
                AcceptedPayload::BoundaryCrossing(crossing) => {
                    let mut output = payload.init_boundary_crossing();
                    output.set_action(crossing.action.as_str());
                    fill_f64(
                        output.reborrow().init_from(crossing.from.len() as u32),
                        &crossing.from,
                    );
                    fill_f64(
                        output.reborrow().init_to(crossing.to.len() as u32),
                        &crossing.to,
                    );
                    output.set_source_basin(crossing.source_basin);
                    output.set_destination_basin(crossing.destination_basin);
                }
                AcceptedPayload::CoordinatorStatus(status) => {
                    let mut output = payload.init_coordinator_status();
                    output.set_snapshot_version(status.snapshot_version);
                    output.set_open_epoch(status.open_epoch);
                    output.set_epoch_submitted(status.epoch_submitted);
                    output.set_epoch_required(status.epoch_required);
                    output.set_census_visits(status.census_visits);
                    output.set_active_entries(status.active_entries);
                    output.set_aggregate_charged(status.aggregate_charged);
                    output.set_aggregate_budget(status.aggregate_budget);
                    output.set_landscape_basins(status.landscape_basins);
                    if let Some(seam) = &status.seam {
                        output.set_algebraic_connectivity(seam.algebraic_connectivity);
                        output.set_seam_conductance(seam.conductance);
                        output.set_community_left(seam.community_left);
                        output.set_community_right(seam.community_right);
                        output.set_seam_left_basin(seam.left_basin);
                        output.set_seam_right_basin(seam.right_basin);
                    }
                    let mut replicas = output.init_replicas(status.replicas.len() as u32);
                    for (index, progress) in status.replicas.iter().enumerate() {
                        let mut row = replicas.reborrow().get(index as u32);
                        row.set_replica(progress.replica);
                        row.set_charged_work(progress.charged_work);
                        row.set_best_energy(progress.best_energy);
                    }
                }
                AcceptedPayload::BridgeAssignment(assignment) => {
                    let mut output = payload.init_bridge_assignment();
                    output.set_bridge(assignment.bridge);
                    output.set_from_basin(assignment.from_basin);
                    output.set_to_basin(assignment.to_basin);
                    fill_f64(
                        output
                            .reborrow()
                            .init_images(assignment.images.len() as u32),
                        &assignment.images,
                    );
                    output.set_image_count(assignment.image_count);
                    output.set_region(assignment.region);
                    output.set_tube_radius(assignment.tube_radius);
                    let mut entry = output.init_entry();
                    match &assignment.entry {
                        Some(state) => {
                            fill_f64(entry.init_state(state.len() as u32), state);
                        }
                        None => entry.set_none(()),
                    }
                }
                AcceptedPayload::PolicyState(state) => {
                    let mut output = payload.init_policy_state();
                    output.set_total_visits(state.total_visits);
                    output.set_singleton_basins(state.singleton_basins);
                    output.set_local_basin_visits(state.local_basin_visits);
                    output.set_globally_saturated(state.globally_saturated);
                    output.set_relation(state.relation.into());
                    output.set_aggregate_charged(state.aggregate_charged);
                    output.set_aggregate_budget(state.aggregate_budget);
                    {
                        let mut basin = output.reborrow().init_local_basin();
                        match state.local_basin {
                            Some(identifier) => basin.set_assigned(identifier),
                            None => basin.set_unassigned(()),
                        }
                    }
                    output.set_local_basin_distance(state.local_basin_distance);
                    output.set_novelty(state.novelty);
                    output.set_transition_uncertainty(state.transition_uncertainty);
                    output.set_explore_collapsed(state.explore_collapsed);
                    output.set_certified_attractor(state.certified_attractor);
                    output.set_pruned(state.pruned);
                    output.set_leftover_lambda(state.leftover_lambda);
                    output.set_interface_rank(state.interface_rank);
                    output.set_interface_threshold(state.interface_threshold);
                    output.set_interface_count(state.interface_count);
                    output.set_occupied_family_count(state.occupied_family_count);
                    output.set_packing_saturated(state.packing_saturated);
                }
                AcceptedPayload::PopulationEpoch(state) => {
                    let mut output = payload.init_population_epoch();
                    output.set_epoch(state.epoch);
                    output.set_submitted(state.submitted);
                    output.set_required(state.required);
                    let mut result = output.init_result();
                    if let Some(plan) = &state.plan {
                        let mut wire = result.init_ready();
                        wire.set_epoch(plan.epoch);
                        fill_u32(
                            wire.reborrow()
                                .init_destinations(plan.destinations.len() as u32),
                            &plan.destinations,
                        );
                        fill_u32(
                            wire.reborrow().init_parents(plan.parents.len() as u32),
                            &plan.parents,
                        );
                        fill_f64(
                            wire.reborrow().init_weights(plan.weights.len() as u32),
                            &plan.weights,
                        );
                        wire.set_effective_sample_size(plan.effective_sample_size);
                        wire.set_unique_parents(plan.unique_parents);
                        wire.set_max_family_size(plan.max_family_size);
                        wire.set_offspring_variance(plan.offspring_variance);
                        wire.set_selection(match plan.selection {
                            PopulationSelection::Unspecified => {
                                crate::Catalog_capnp::PopulationSelection::Unspecified
                            }
                            PopulationSelection::SystematicResampling => {
                                crate::Catalog_capnp::PopulationSelection::SystematicResampling
                            }
                            PopulationSelection::RegionCovering => {
                                crate::Catalog_capnp::PopulationSelection::RegionCovering
                            }
                        });
                        let mut candidates = wire
                            .reborrow()
                            .init_parent_candidates(plan.parent_candidates.len() as u32);
                        for (index, candidate) in plan.parent_candidates.iter().enumerate() {
                            fill_candidate(candidates.reborrow().get(index as u32), candidate);
                        }
                    } else {
                        result.set_pending(());
                    }
                }
                AcceptedPayload::CatalogMutation(mutation) => {
                    let mut output = payload.init_catalog_mutation();
                    output.set_basin_id(mutation.basin_id);
                    output.set_kind(mutation.kind.into());
                    fill_u64(
                        output
                            .reborrow()
                            .init_evicted(mutation.evicted.len() as u32),
                        &mutation.evicted,
                    );
                    let mut incumbent = output.init_incumbent_basin();
                    match mutation.incumbent_basin {
                        Some(identifier) => incumbent.set_present(identifier),
                        None => incumbent.set_absent(()),
                    }
                }
            }
        }
        CatalogReply::Rejected {
            event_sequence,
            snapshot_version,
            reason,
        } => {
            root.set_event_sequence(event_sequence);
            root.set_snapshot_version(snapshot_version);
            root.init_result().set_rejected(reason.into());
        }
    }
    let mut bytes = Vec::new();
    serialize::write_message(&mut bytes, &message).map_err(wire_error)?;
    Ok(bytes)
}

pub(crate) fn decode_reply_reader(
    root: catalog_reply::Reader<'_>,
) -> Result<CatalogReply, ProtocolError> {
    check_version(root.get_protocol_version())?;
    let event_sequence = root.get_event_sequence();
    let snapshot_version = root.get_snapshot_version();
    match root.get_result().which().map_err(wire_error)? {
        catalog_reply::result::Accepted(body) => {
            let body = body.map_err(wire_error)?;
            let payload = match body.get_payload().which().map_err(wire_error)? {
                accepted_reply::payload::None(()) => AcceptedPayload::None,
                accepted_reply::payload::Candidate(candidate) => {
                    AcceptedPayload::Candidate(read_candidate(candidate.map_err(wire_error)?)?)
                }
                accepted_reply::payload::DescriptorHole(hole) => {
                    let hole = hole.map_err(wire_error)?;
                    AcceptedPayload::DescriptorHole(DescriptorHoleProposal {
                        target: list_f64(hole.get_target().map_err(wire_error)?),
                        increment: list_f64(hole.get_increment().map_err(wire_error)?),
                        nearest_catalog_distance: hole.get_nearest_catalog_distance(),
                    })
                }
                accepted_reply::payload::BoundaryCrossing(crossing) => {
                    let crossing = crossing.map_err(wire_error)?;
                    AcceptedPayload::BoundaryCrossing(BoundaryCrossingRecord {
                        action: text_value(crossing.get_action().map_err(wire_error)?)?,
                        from: list_f64(crossing.get_from().map_err(wire_error)?),
                        to: list_f64(crossing.get_to().map_err(wire_error)?),
                        source_basin: crossing.get_source_basin(),
                        destination_basin: crossing.get_destination_basin(),
                    })
                }
                accepted_reply::payload::CoordinatorStatus(status) => {
                    let status = status.map_err(wire_error)?;
                    let mut replicas = Vec::new();
                    for row in status.get_replicas().map_err(wire_error)?.iter() {
                        replicas.push(ReplicaProgress {
                            replica: row.get_replica(),
                            charged_work: row.get_charged_work(),
                            best_energy: row.get_best_energy(),
                        });
                    }
                    let seam = (status.get_community_left() > 0
                        && status.get_community_right() > 0)
                        .then(|| LandscapeSeam {
                            algebraic_connectivity: status.get_algebraic_connectivity(),
                            conductance: status.get_seam_conductance(),
                            community_left: status.get_community_left(),
                            community_right: status.get_community_right(),
                            left_basin: status.get_seam_left_basin(),
                            right_basin: status.get_seam_right_basin(),
                        });
                    AcceptedPayload::CoordinatorStatus(CoordinatorStatus {
                        snapshot_version: status.get_snapshot_version(),
                        open_epoch: status.get_open_epoch(),
                        epoch_submitted: status.get_epoch_submitted(),
                        epoch_required: status.get_epoch_required(),
                        census_visits: status.get_census_visits(),
                        active_entries: status.get_active_entries(),
                        aggregate_charged: status.get_aggregate_charged(),
                        aggregate_budget: status.get_aggregate_budget(),
                        replicas,
                        landscape_basins: status.get_landscape_basins(),
                        seam,
                    })
                }
                accepted_reply::payload::BridgeAssignment(assignment) => {
                    let assignment = assignment.map_err(wire_error)?;
                    let entry = match assignment.get_entry().which().map_err(wire_error)? {
                        bridge_assignment::entry::None(()) => None,
                        bridge_assignment::entry::State(state) => {
                            Some(list_f64(state.map_err(wire_error)?))
                        }
                    };
                    AcceptedPayload::BridgeAssignment(BridgeAssignmentRecord {
                        bridge: assignment.get_bridge(),
                        from_basin: assignment.get_from_basin(),
                        to_basin: assignment.get_to_basin(),
                        images: list_f64(assignment.get_images().map_err(wire_error)?),
                        image_count: assignment.get_image_count(),
                        region: assignment.get_region(),
                        tube_radius: assignment.get_tube_radius(),
                        entry,
                    })
                }
                accepted_reply::payload::PolicyState(state) => {
                    let state = state.map_err(wire_error)?;
                    let local_basin = match state.get_local_basin().which().map_err(wire_error)? {
                        policy_state_reply::local_basin::Unassigned(()) => None,
                        policy_state_reply::local_basin::Assigned(identifier) => Some(identifier),
                    };
                    AcceptedPayload::PolicyState(PolicyState {
                        total_visits: state.get_total_visits(),
                        singleton_basins: state.get_singleton_basins(),
                        local_basin_visits: state.get_local_basin_visits(),
                        globally_saturated: state.get_globally_saturated(),
                        relation: state.get_relation().map_err(wire_error)?.into(),
                        aggregate_charged: state.get_aggregate_charged(),
                        aggregate_budget: state.get_aggregate_budget(),
                        local_basin,
                        local_basin_distance: state.get_local_basin_distance(),
                        novelty: state.get_novelty(),
                        transition_uncertainty: state.get_transition_uncertainty(),
                        explore_collapsed: state.get_explore_collapsed(),
                        certified_attractor: state.get_certified_attractor(),
                        pruned: state.get_pruned(),
                        leftover_lambda: state.get_leftover_lambda(),
                        interface_rank: state.get_interface_rank(),
                        interface_threshold: state.get_interface_threshold(),
                        interface_count: state.get_interface_count(),
                        occupied_family_count: state.get_occupied_family_count(),
                        packing_saturated: state.get_packing_saturated(),
                    })
                }
                accepted_reply::payload::PopulationEpoch(state) => {
                    let state = state.map_err(wire_error)?;
                    let plan = match state.get_result().which().map_err(wire_error)? {
                        population_epoch_reply::result::Pending(()) => None,
                        population_epoch_reply::result::Ready(plan) => {
                            let plan = plan.map_err(wire_error)?;
                            let candidates = plan.get_parent_candidates().map_err(wire_error)?;
                            let mut parent_candidates =
                                Vec::with_capacity(candidates.len() as usize);
                            for index in 0..candidates.len() {
                                parent_candidates.push(read_candidate(candidates.get(index))?);
                            }
                            Some(PopulationPlan {
                                epoch: plan.get_epoch(),
                                destinations: list_u32(
                                    plan.get_destinations().map_err(wire_error)?,
                                ),
                                parents: list_u32(plan.get_parents().map_err(wire_error)?),
                                weights: list_f64(plan.get_weights().map_err(wire_error)?),
                                effective_sample_size: plan.get_effective_sample_size(),
                                unique_parents: plan.get_unique_parents(),
                                max_family_size: plan.get_max_family_size(),
                                offspring_variance: plan.get_offspring_variance(),
                                parent_candidates,
                                selection: match plan.get_selection().map_err(wire_error)? {
                                    crate::Catalog_capnp::PopulationSelection::Unspecified => {
                                        PopulationSelection::Unspecified
                                    }
                                    crate::Catalog_capnp::PopulationSelection::SystematicResampling => {
                                        PopulationSelection::SystematicResampling
                                    }
                                    crate::Catalog_capnp::PopulationSelection::RegionCovering => {
                                        PopulationSelection::RegionCovering
                                    }
                                },
                            })
                        }
                    };
                    AcceptedPayload::PopulationEpoch(PopulationEpochState {
                        epoch: state.get_epoch(),
                        submitted: state.get_submitted(),
                        required: state.get_required(),
                        plan,
                    })
                }
                accepted_reply::payload::CatalogMutation(mutation) => {
                    let mutation = mutation.map_err(wire_error)?;
                    let incumbent_basin =
                        match mutation.get_incumbent_basin().which().map_err(wire_error)? {
                            catalog_mutation_reply::incumbent_basin::Absent(()) => None,
                            catalog_mutation_reply::incumbent_basin::Present(identifier) => {
                                Some(identifier)
                            }
                        };
                    AcceptedPayload::CatalogMutation(CatalogMutation {
                        basin_id: mutation.get_basin_id(),
                        kind: mutation.get_kind().map_err(wire_error)?.into(),
                        evicted: list_u64(mutation.get_evicted().map_err(wire_error)?),
                        incumbent_basin,
                    })
                }
            };
            Ok(CatalogReply::Accepted(AcceptedReply {
                event_sequence,
                duplicate: body.get_duplicate(),
                snapshot: CatalogSnapshot {
                    version: snapshot_version,
                    census_visits: body.get_census_visits(),
                    active_entries: body.get_active_entries(),
                    aggregate_charged: body.get_aggregate_charged(),
                    aggregate_budget: body.get_aggregate_budget(),
                },
                payload,
            }))
        }
        catalog_reply::result::Rejected(reason) => Ok(CatalogReply::Rejected {
            event_sequence,
            snapshot_version,
            reason: reason.map_err(wire_error)?.into(),
        }),
    }
}

impl From<ProtocolRejection> for RejectionKind {
    fn from(value: ProtocolRejection) -> Self {
        match value {
            ProtocolRejection::Malformed => Self::Malformed,
            ProtocolRejection::UnsupportedVersion => Self::UnsupportedVersion,
            ProtocolRejection::CampaignMismatch => Self::CampaignMismatch,
            ProtocolRejection::EnsembleMismatch => Self::EnsembleMismatch,
            ProtocolRejection::ReplicaMismatch => Self::ReplicaMismatch,
            ProtocolRejection::SignatureMismatch => Self::SignatureMismatch,
            ProtocolRejection::SequenceReplay => Self::SequenceReplay,
            ProtocolRejection::SequenceRegression => Self::SequenceRegression,
            ProtocolRejection::SnapshotRegression => Self::SnapshotRegression,
            ProtocolRejection::ValidationRejected => Self::ValidationRejected,
        }
    }
}

impl From<RejectionKind> for ProtocolRejection {
    fn from(value: RejectionKind) -> Self {
        match value {
            RejectionKind::Malformed => Self::Malformed,
            RejectionKind::UnsupportedVersion => Self::UnsupportedVersion,
            RejectionKind::CampaignMismatch => Self::CampaignMismatch,
            RejectionKind::EnsembleMismatch => Self::EnsembleMismatch,
            RejectionKind::ReplicaMismatch => Self::ReplicaMismatch,
            RejectionKind::SignatureMismatch => Self::SignatureMismatch,
            RejectionKind::SequenceReplay => Self::SequenceReplay,
            RejectionKind::SequenceRegression => Self::SequenceRegression,
            RejectionKind::SnapshotRegression => Self::SnapshotRegression,
            RejectionKind::ValidationRejected => Self::ValidationRejected,
        }
    }
}

impl From<CatalogRelation> for WireCatalogRelation {
    fn from(value: CatalogRelation) -> Self {
        match value {
            CatalogRelation::Empty => Self::Empty,
            CatalogRelation::Incumbent => Self::Incumbent,
            CatalogRelation::SameBasin => Self::SameBasin,
            CatalogRelation::UnrelatedNoAnchor => Self::UnrelatedNoAnchor,
            CatalogRelation::UnrelatedLowerAnchor => Self::UnrelatedLowerAnchor,
        }
    }
}

impl From<WireCatalogRelation> for CatalogRelation {
    fn from(value: WireCatalogRelation) -> Self {
        match value {
            WireCatalogRelation::Empty => Self::Empty,
            WireCatalogRelation::Incumbent => Self::Incumbent,
            WireCatalogRelation::SameBasin => Self::SameBasin,
            WireCatalogRelation::UnrelatedNoAnchor => Self::UnrelatedNoAnchor,
            WireCatalogRelation::UnrelatedLowerAnchor => Self::UnrelatedLowerAnchor,
        }
    }
}

impl From<CatalogMutationKind> for WireCatalogMutationKind {
    fn from(value: CatalogMutationKind) -> Self {
        match value {
            CatalogMutationKind::Added => Self::Added,
            CatalogMutationKind::ReplacedSameBasin => Self::ReplacedSameBasin,
            CatalogMutationKind::ReplacedConflicts => Self::ReplacedConflicts,
            CatalogMutationKind::ReplacedCapacity => Self::ReplacedCapacity,
            CatalogMutationKind::RejectedSameBasin => Self::RejectedSameBasin,
            CatalogMutationKind::RejectedConflict => Self::RejectedConflict,
            CatalogMutationKind::RejectedCapacity => Self::RejectedCapacity,
        }
    }
}

impl From<WireCatalogMutationKind> for CatalogMutationKind {
    fn from(value: WireCatalogMutationKind) -> Self {
        match value {
            WireCatalogMutationKind::Added => Self::Added,
            WireCatalogMutationKind::ReplacedSameBasin => Self::ReplacedSameBasin,
            WireCatalogMutationKind::ReplacedConflicts => Self::ReplacedConflicts,
            WireCatalogMutationKind::ReplacedCapacity => Self::ReplacedCapacity,
            WireCatalogMutationKind::RejectedSameBasin => Self::RejectedSameBasin,
            WireCatalogMutationKind::RejectedConflict => Self::RejectedConflict,
            WireCatalogMutationKind::RejectedCapacity => Self::RejectedCapacity,
        }
    }
}

fn check_version(received: u16) -> Result<(), ProtocolError> {
    if received == PROTOCOL_VERSION {
        Ok(())
    } else {
        Err(ProtocolError::UnsupportedVersion {
            received,
            supported: PROTOCOL_VERSION,
        })
    }
}

fn fill_f64(mut output: capnp::primitive_list::Builder<'_, f64>, values: &[f64]) {
    for (index, value) in values.iter().copied().enumerate() {
        output.set(index as u32, value);
    }
}

fn fill_u32(mut output: capnp::primitive_list::Builder<'_, u32>, values: &[u32]) {
    for (index, value) in values.iter().copied().enumerate() {
        output.set(index as u32, value);
    }
}

fn fill_u64(mut output: capnp::primitive_list::Builder<'_, u64>, values: &[u64]) {
    for (index, value) in values.iter().copied().enumerate() {
        output.set(index as u32, value);
    }
}

fn fill_candidate(mut output: candidate_record::Builder<'_>, candidate: &CatalogCandidate) {
    output.set_producer_replica(candidate.producer_replica);
    fill_f64(
        output
            .reborrow()
            .init_coordinates(candidate.coordinates.len() as u32),
        &candidate.coordinates,
    );
    {
        let mut cell = output.reborrow().init_cell();
        match candidate.cell {
            Some(values) => fill_f64(cell.reborrow().init_present(9), &values),
            None => cell.set_absent(()),
        }
    }
    output.set_energy(candidate.energy);
    fill_f64(
        output.reborrow().init_forces(candidate.forces.len() as u32),
        &candidate.forces,
    );
    output.set_gradient_norm(candidate.gradient_norm);
    fill_f64(
        output
            .reborrow()
            .init_descriptor(candidate.descriptor.len() as u32),
        &candidate.descriptor,
    );
    output.set_descriptor_schema_version(candidate.descriptor_schema_version);
    output.set_quench_status(if candidate.quench_converged {
        WireQuenchStatus::Converged
    } else {
        WireQuenchStatus::Unconverged
    });
    output.set_charged_work(candidate.charged_work);
    output.set_event_sequence(candidate.event_sequence);
    output.set_seed(candidate.seed);
    let mut basin = output.init_census_basin();
    match candidate.census_basin {
        Some(identifier) => basin.set_assigned(identifier),
        None => basin.set_unassigned(()),
    }
}

fn read_candidate(input: candidate_record::Reader<'_>) -> Result<CatalogCandidate, ProtocolError> {
    let cell = match input.get_cell().which().map_err(wire_error)? {
        candidate_record::cell::Absent(()) => None,
        candidate_record::cell::Present(values) => {
            let values = list_f64(values.map_err(wire_error)?);
            Some(values.try_into().map_err(|values: Vec<f64>| {
                ProtocolError::Malformed(format!("cell length is {}, expected 9", values.len()))
            })?)
        }
    };
    let census_basin = match input.get_census_basin().which().map_err(wire_error)? {
        candidate_record::census_basin::Unassigned(()) => None,
        candidate_record::census_basin::Assigned(identifier) => Some(identifier),
    };
    Ok(CatalogCandidate {
        producer_replica: input.get_producer_replica(),
        coordinates: list_f64(input.get_coordinates().map_err(wire_error)?),
        cell,
        energy: input.get_energy(),
        forces: list_f64(input.get_forces().map_err(wire_error)?),
        gradient_norm: input.get_gradient_norm(),
        descriptor: list_f64(input.get_descriptor().map_err(wire_error)?),
        descriptor_schema_version: input.get_descriptor_schema_version(),
        quench_converged: match input.get_quench_status().map_err(wire_error)? {
            WireQuenchStatus::Converged => true,
            WireQuenchStatus::Unconverged => false,
        },
        charged_work: input.get_charged_work(),
        event_sequence: input.get_event_sequence(),
        seed: input.get_seed(),
        census_basin,
    })
}

fn list_f64(input: capnp::primitive_list::Reader<'_, f64>) -> Vec<f64> {
    input.iter().collect()
}

fn list_u32(input: capnp::primitive_list::Reader<'_, u32>) -> Vec<u32> {
    input.iter().collect()
}

fn list_u64(input: capnp::primitive_list::Reader<'_, u64>) -> Vec<u64> {
    input.iter().collect()
}

fn text_value(input: capnp::text::Reader<'_>) -> Result<String, ProtocolError> {
    input
        .to_str()
        .map(str::to_owned)
        .map_err(|error| ProtocolError::Malformed(error.to_string()))
}

fn wire_error(error: impl std::fmt::Display) -> ProtocolError {
    ProtocolError::Malformed(error.to_string())
}
