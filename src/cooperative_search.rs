//! Cooperative search composition and exact aggregate work accounting.

pub mod ledger;

#[cfg(feature = "bank-rpc")]
mod run {
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    use crate::catalog::MixingEvidence;
    use crate::catalog_policy::{
        ActiveCatalogRelation, AggregateProgress, CatalogPolicy, CatalogPolicyInput,
        CensusEvidence, PolicyAction, PolicyDecision, PolicyInputError, ValidationState,
    };
    use crate::catalog_rpc::client::{CatalogClient, CatalogClientError, PolicyStateReceipt};
    use crate::catalog_rpc::mailbox::CatalogMailbox;
    use crate::catalog_rpc::{
        BoundaryCrossingRecord, BridgeAssignmentRecord, BridgeCrossingRecord, CatalogCandidate,
        CatalogFrontierPost, CatalogLedgerEvent, CatalogMutation, CatalogRelation,
        CatalogRideOutcome, CatalogRideReport, CatalogRideWork, CatalogSnapshot,
        DescriptorHoleProposal, INCUMBENT_SAMPLE_DRAW, PolicyState, PopulationEpochState,
        PopulationPlan, PopulationSelection, ProtocolRejection, SPARSE_SAMPLE_DRAW,
        TransitionDestination,
    };
    use crate::compatibility::EngineDescriptor;
    use crate::discovery_roster::DiscoveryRole;
    use crate::methods::feynman_kac::population_family_position;
    use crate::pes_exploration::RideMethod;
    use crate::ride_ledger::{RideCredit, RideDirection, RideFailure, RideWorkOrder};

    use super::ledger::{ChargeKind, CooperativeLedger, LedgerError, ReplicaLedgerEvent};

    /// Stable event classifications written by a cooperative run.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum TraceKind {
        /// Charged local engine work.
        LocalWork,
        /// A validated candidate entered or improved the active catalog.
        Admission,
        /// Candidate validation or admission was rejected.
        Rejection,
        /// A versioned coordinator snapshot was received.
        SnapshotRefresh,
        /// Policy retained independent local search.
        PolicyLocal,
        /// Policy selected a validated remote anchor.
        PolicyExploit,
        /// Policy selected a catalog-space exploration proposal.
        PolicyExplore,
        /// Policy selected a descriptor-hole leave proposal.
        PolicyLeave,
        /// A synchronous population epoch is waiting for other replicas.
        PopulationPending,
        /// A synchronous population epoch returned an immutable parent plan.
        PopulationReady,
        /// Coordinator communication failed and local execution remained active.
        RpcFallback,
        /// The run has no sharing transport by construction.
        SharingDisabled,
        /// One complete local-slice and transition diagnostic.
        Slice,
        /// One explicit action-conditioned perturb--quench observation.
        Transition,
        /// One validated perturb--quench execution before coordinator registration.
        TransitionExecution,
        /// One exclusive transition-search arm claimed from the coordinator.
        RideClaim,
        /// One charged transition-search result accepted by the coordinator.
        RideReport,
    }

    impl TraceKind {
        const fn code(self) -> &'static str {
            match self {
                Self::LocalWork => "local_work",
                Self::Admission => "admission",
                Self::Rejection => "rejection",
                Self::SnapshotRefresh => "snapshot_refresh",
                Self::PolicyLocal => "policy_local",
                Self::PolicyExploit => "policy_exploit",
                Self::PolicyExplore => "policy_explore",
                Self::PolicyLeave => "policy_leave",
                Self::PopulationPending => "population_pending",
                Self::PopulationReady => "population_ready",
                Self::RpcFallback => "rpc_fallback",
                Self::SharingDisabled => "sharing_disabled",
                Self::Slice => "slice",
                Self::Transition => "transition",
                Self::TransitionExecution => "transition_execution",
                Self::RideClaim => "ride_claim",
                Self::RideReport => "ride_report",
            }
        }
    }

    /// One deterministic newline-delimited run event.
    #[derive(Debug, Clone, PartialEq)]
    pub struct TraceEvent {
        /// Replica identity within the isolated ensemble.
        pub replica: u32,
        /// Monotone trace sequence for this replica.
        pub sequence: u64,
        /// Aggregate charged work after the event.
        pub aggregate_charged: u64,
        /// Coordinator version observed by the event, when available.
        pub catalog_version: Option<u64>,
        /// Stable event classification.
        pub kind: TraceKind,
        /// Stable policy or rejection reason.
        pub reason: Option<&'static str>,
        /// Genealogy evidence attached to a completed population epoch.
        pub population: Option<PopulationTrace>,
        /// Exact coordinator evidence attached to a policy-state refresh.
        pub policy: Option<PolicyTrace>,
        /// Complete diagnostic attached to a local slice boundary.
        pub slice: Option<SliceTrace>,
        /// Exact active-catalog mutation attached to an offer event.
        pub catalog: Option<CatalogMutation>,
        /// Action and outcome attached to a transition event.
        pub transition: Option<TransitionTrace>,
        /// Work identity and producer/receiver evidence for a transition ride.
        pub ride: Option<RideTrace>,
    }

    /// Auditable identity, cost, and verdict of one transition-search experiment.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct RideTrace {
        /// Coordinator-local work identifier.
        pub work: u64,
        /// Exact source basin named by the commissioned arm.
        pub source_basin: Option<u64>,
        /// Source-local invariant environment class.
        pub environment_class: Option<u32>,
        /// Ranked localized mode seed.
        pub mode_rank: Option<u16>,
        /// Sign of the initial displacement.
        pub direction: Option<RideDirection>,
        /// Minimum-mode solver assigned by the coordinator.
        pub method: Option<RideMethod>,
        /// Atom representing the commissioned environment class.
        pub representative_atom: Option<u32>,
        /// One-based attempt number for the arm.
        pub attempt: Option<u64>,
        /// Deterministic experiment seed.
        pub seed: Option<u64>,
        /// PES evaluations charged by the producer.
        pub producer_charged_evaluations: Option<u64>,
        /// Whether the producer supplied a certified connection.
        pub producer_certified_connection: Option<bool>,
        /// Producer-side classified failure.
        pub producer_failure: Option<RideFailure>,
        /// PES evaluations charged during receiving certification.
        pub receiver_charged_evaluations: Option<u64>,
        /// Whether the receiver certified the physical connection.
        pub receiver_certified_connection: Option<bool>,
        /// Receiver-side classified failure.
        pub receiver_failure: Option<RideFailure>,
        /// Whether the receiver added a previously unseen exact saddle.
        pub novel_saddle: Option<bool>,
        /// Whether the receiver certified a quotient-space self-loop.
        pub degenerate_rearrangement: Option<bool>,
        /// Whether the receiver added a previously unseen endpoint pair.
        pub novel_edge: Option<bool>,
        /// Producer plus receiving-side charged evaluations.
        pub total_charged_evaluations: Option<u64>,
    }

    impl RideTrace {
        fn claim(order: &RideWorkOrder) -> Self {
            Self::with_order(order.id, Some(order))
        }

        fn report(order: Option<&RideWorkOrder>, report: &CatalogRideReport) -> Self {
            let mut trace = Self::with_order(report.work, order);
            trace.producer_charged_evaluations = Some(report.charged_evaluations);
            match &report.outcome {
                CatalogRideOutcome::Certified(_) => {
                    trace.producer_certified_connection = Some(true);
                }
                CatalogRideOutcome::Unresolved(evidence) => {
                    trace.producer_certified_connection = Some(false);
                    trace.producer_failure = Some(evidence.failure);
                }
                CatalogRideOutcome::Failed(failure) => {
                    trace.producer_certified_connection = Some(false);
                    trace.producer_failure = Some(*failure);
                }
            }
            trace
        }

        fn with_credit(mut self, credit: RideCredit) -> Self {
            self.receiver_charged_evaluations = self
                .producer_charged_evaluations
                .and_then(|producer| credit.total_charged_evaluations.checked_sub(producer));
            self.receiver_certified_connection = Some(credit.certified_connection);
            self.receiver_failure = credit.failure;
            self.novel_saddle = Some(credit.novel_saddle);
            self.degenerate_rearrangement = Some(credit.degenerate_rearrangement);
            self.novel_edge = Some(credit.novel_edge);
            self.total_charged_evaluations = Some(credit.total_charged_evaluations);
            self
        }

        fn with_order(work: u64, order: Option<&RideWorkOrder>) -> Self {
            Self {
                work,
                source_basin: order.map(|order| order.arm.source_basin),
                environment_class: order.map(|order| order.arm.environment_class),
                mode_rank: order.map(|order| order.arm.mode_rank),
                direction: order.map(|order| order.arm.direction),
                method: order.map(|order| order.arm.method),
                representative_atom: order.map(|order| order.representative_atom),
                attempt: order.map(|order| order.attempt),
                seed: order.map(|order| order.seed),
                producer_charged_evaluations: None,
                producer_certified_connection: None,
                producer_failure: None,
                receiver_charged_evaluations: None,
                receiver_certified_connection: None,
                receiver_failure: None,
                novel_saddle: None,
                degenerate_rearrangement: None,
                novel_edge: None,
                total_charged_evaluations: None,
            }
        }
    }

    /// Replayable action-conditioned transition diagnostic.
    #[derive(Debug, Clone, PartialEq)]
    pub struct TransitionTrace {
        /// Stable target-blind proposal action.
        pub action: String,
        /// Live-chain hop at which the execution completed, when locally observed.
        pub hop: Option<u64>,
        /// Source energy of a locally observed execution.
        pub from_energy: Option<f64>,
        /// Destination energy of a locally observed execution.
        pub to_energy: Option<f64>,
        /// Whether the perturb--quench produced a classified destination.
        pub resolved: bool,
        /// Whether the destination became the replica's live state.
        pub adopted: bool,
    }

    /// Catalog, policy, and latent-field evidence for one policy query.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct PolicyTrace {
        /// Stable census-basin identifier, or `None` for an unassigned descriptor.
        pub local_basin: Option<u64>,
        /// Relation between the query and the active catalog.
        pub relation: CatalogRelation,
        /// Exact number of fixed-census observations.
        pub total_visits: u64,
        /// Exact number of singleton basins in the fixed census.
        pub singleton_basins: u64,
        /// Exact visits assigned to the query basin.
        pub local_basin_visits: u64,
        /// Whether the fixed census meets its declared saturation rule.
        pub globally_saturated: bool,
        /// Distance from the query descriptor to its immutable census medoid.
        pub local_basin_distance: f64,
        /// Universal descriptor acquisition from novelty and model uncertainty.
        pub novelty: f64,
        /// Posterior uncertainty of the latent Gaussian transition field.
        pub transition_uncertainty: f64,
        /// Joint-minimum-information role for this same-system replica.
        pub discovery_role: DiscoveryRole,
        /// Coordinator evidence epoch behind the stable shared-batch assignment.
        pub discovery_epoch: u64,
        /// One-sided upper bound on unseen exact-basin mass.
        pub basin_unseen_mass_upper: f64,
        /// One-sided upper bound on unseen exact-saddle mass.
        pub saddle_unseen_mass_upper: f64,
        /// Same-system basin-escape attempts retained by the shared ledger.
        pub basin_discovery_attempts: u64,
        /// Potential calls charged to same-system basin-escape attempts.
        pub basin_discovery_charged: u64,
        /// Same-system saddle-ride attempts retained by the shared ledger.
        pub saddle_discovery_attempts: u64,
        /// Potential calls charged to same-system saddle-ride attempts.
        pub saddle_discovery_charged: u64,
        /// Whether exact-saddle reobservations meet the coverage rule.
        pub saddle_coverage_saturated: bool,
        /// Energy used to classify the query against the active catalog.
        pub query_energy: f64,
        /// Successive-halving retire decision for this replica.
        pub retired: bool,
    }

    /// Cooperative role selected for one local slice.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum PolicyRole {
        /// Continue the independent local trajectory.
        Local,
        /// Adopt a validated active-catalog anchor subject to policy conditions.
        Exploit,
        /// Explore a target-free descriptor-space direction.
        Explore,
        /// Leave a locally exhausted basin through a target-free proposal.
        Leave,
        /// No remote policy evidence was available.
        Unavailable,
    }

    /// Proposal family dispatched during one slice.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ProposalFamily {
        /// Independent local quenched search.
        Local,
        /// Active-catalog anchor sampling.
        CatalogSample,
        /// Farthest-hole descriptor proposal and Cartesian pullback.
        DescriptorHole,
        /// Aligned displacement from an observed attraction-region crossing.
        BoundaryTransport,
        /// Minimum--saddle--minimum endpoint certified by the coordinator.
        TransitionRide,
        /// Synchronous fixed-population reconfiguration.
        PopulationReconfiguration,
        /// New random start after successive-halving prune.
        HyperbandReseed,
    }

    /// Receiving validation result for a slice transition.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SliceValidation {
        /// The transition did not require receiving validation.
        NotAttempted,
        /// Receiving validation accepted the transition state.
        Accepted,
        /// Receiving validation rejected the transition state.
        Rejected,
    }

    /// Quench result associated with a slice transition.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SliceQuench {
        /// No transition quench was attempted.
        NotAttempted,
        /// The transition quench converged and met its validity contract.
        Converged,
        /// The transition quench or its validity contract failed.
        Rejected,
    }

    /// Adoption result for a proposed slice transition.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SliceAdoption {
        /// No nonlocal adoption was attempted.
        NotAttempted,
        /// The validated transition became the local state.
        Adopted,
        /// A valid transition did not meet the policy's adoption condition.
        NotImproved,
        /// Proposal construction or receiving validation rejected the transition.
        Rejected,
    }

    /// Complete evidence recorded exactly once for one local-search slice.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct SliceTrace {
        /// One-based slice index within the replica trajectory.
        pub slice: u64,
        /// Current fixed-census basin identifier, when assigned.
        pub current_basin: Option<u64>,
        /// Relation between the current state and active catalog.
        pub active_relation: Option<CatalogRelation>,
        /// Cooperative role selected for the slice.
        pub policy_role: PolicyRole,
        /// Stable policy or availability reason.
        pub policy_reason: &'static str,
        /// Proposal family dispatched by the slice.
        pub proposal_family: ProposalFamily,
        /// Sampled census basin, when an active-catalog anchor was requested.
        pub sampled_basin: Option<u64>,
        /// Norm of the requested target-free descriptor increment.
        pub descriptor_step_norm: Option<f64>,
        /// Norm of the realized Cartesian increment.
        pub cartesian_step_norm: Option<f64>,
        /// Receiving validation result.
        pub validation: SliceValidation,
        /// Transition quench result.
        pub quench: SliceQuench,
        /// Whether the proposed transition entered the local trajectory.
        pub adoption: SliceAdoption,
        /// Target-independent novelty supplied to the policy.
        pub novelty: Option<f64>,
        /// Best local energy at the slice boundary, absent when no valid state exists.
        pub energy: Option<f64>,
        /// Exact charged work consumed by the slice.
        pub charged_work: u64,
    }

    /// Genealogy evidence for one destination at a completed epoch.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct PopulationTrace {
        /// Immutable synchronization epoch.
        pub epoch: u64,
        /// Source replica assigned to this destination.
        pub parent: u32,
        /// Zero-based position among destinations sharing this parent.
        pub family_ordinal: u32,
        /// Realized number of offspring sharing this parent.
        pub family_size: u32,
        /// Kish effective sample size of source selection weights.
        pub effective_sample_size: f64,
        /// Branch that produced this barrier's parent map.
        pub selection: PopulationSelection,
    }

    /// Required manifest identity emitted before run events.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct RunManifest {
        /// Campaign identity.
        pub campaign: String,
        /// Independent ensemble identity.
        pub ensemble: String,
        /// Whether a catalog transport is part of the run arm.
        pub sharing: bool,
        /// Objective-engine protocol and native bridge descriptor.
        pub engine: EngineDescriptor,
    }

    /// Result of offering a candidate without making RPC availability fatal.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum CatalogOfferOutcome {
        /// Active catalog size increased or its state changed from an empty snapshot.
        Admitted,
        /// Coordinator rejected validation or left active size unchanged.
        Rejected,
        /// Communication failed and the local trajectory remains authoritative.
        LocalFallback,
        /// Sharing is disabled for the independent control arm.
        SharingDisabled,
    }

    /// Result of registering current state or one transition observation.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum TransitionRecordOutcome {
        /// The coordinator accepted the replay-safe record.
        Recorded,
        /// Scientific validation or transition ordering rejected the record.
        Rejected,
        /// Communication failed and local execution remains authoritative.
        LocalFallback,
        /// Sharing is disabled for the independent control arm.
        SharingDisabled,
    }

    /// Result of one explicit synchronization boundary.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SynchronizationOutcome {
        /// A remote snapshot was received.
        Refreshed(CatalogSnapshot),
        /// Communication failed and local search continues.
        LocalFallback,
        /// No coordinator exists for this run arm.
        SharingDisabled,
    }

    /// Availability and validation result for exact remote policy evidence.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum PolicyEvidenceOutcome {
        /// Exact coordinator evidence mapped into the pure policy input.
        Remote(CatalogPolicyInput),
        /// The coordinator rejected the evidence request.
        Rejected,
        /// Communication failed and local search remains authoritative.
        LocalFallback,
        /// No coordinator exists for this run arm.
        SharingDisabled,
    }

    /// Result of requesting one active-catalog candidate.
    #[derive(Debug, Clone, PartialEq)]
    pub enum CatalogSampleOutcome {
        /// One validated candidate was sampled.
        Candidate(CatalogCandidate),
        /// The active catalog is empty.
        Empty,
        /// The coordinator rejected the request.
        Rejected,
        /// Communication failed and local search remains authoritative.
        LocalFallback,
        /// No coordinator exists for this run arm.
        SharingDisabled,
    }

    /// Result of requesting one batch of active-catalog candidates.
    #[derive(Debug, Clone, PartialEq)]
    pub enum CatalogSamplesOutcome {
        /// Validated candidates returned for the completed draw batch.
        Candidates(Vec<CatalogCandidate>),
        /// The active catalog returned no candidate for any draw.
        Empty,
        /// The coordinator rejected the request.
        Rejected,
        /// Communication failed and local search remains authoritative.
        LocalFallback,
        /// No coordinator exists for this run arm.
        SharingDisabled,
    }

    /// Availability and validation result of claiming transition-search work.
    #[derive(Debug, Clone, PartialEq)]
    pub enum RideClaimOutcome {
        /// Exclusive same-system work and its validated source minimum.
        Work(CatalogRideWork),
        /// No unclaimed portfolio arm is available.
        Empty,
        /// The coordinator rejected the claim.
        Rejected,
        /// Communication failed and the local trajectory remains authoritative.
        LocalFallback,
        /// Sharing is disabled for the independent control arm.
        SharingDisabled,
    }

    /// Availability and validation result of reporting transition-search work.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum RideReportOutcome {
        /// The coordinator accepted the evidence and computed edge novelty.
        Credited(RideCredit),
        /// The coordinator rejected the report.
        Rejected,
        /// Communication failed and the local trajectory remains authoritative.
        LocalFallback,
        /// Sharing is disabled for the independent control arm.
        SharingDisabled,
    }

    /// Result of requesting one descriptor-hole proposal.
    #[derive(Debug, Clone, PartialEq)]
    pub enum CatalogHoleOutcome {
        /// One seeded target-free proposal was returned.
        Proposal(DescriptorHoleProposal),
        /// The coordinator rejected the request.
        Rejected,
        /// Communication failed and local search remains authoritative.
        LocalFallback,
        /// No coordinator exists for this run arm.
        SharingDisabled,
    }

    /// Result of requesting an observed attraction-region boundary crossing.
    #[derive(Debug, Clone, PartialEq)]
    pub enum CatalogBoundaryOutcome {
        /// A validated adopted inter-basin crossing was returned.
        Crossing(BoundaryCrossingRecord),
        /// No crossing leaves the query's inferred attraction region.
        Empty,
        /// The coordinator rejected the request.
        Rejected,
        /// Communication failed and local search remains authoritative.
        LocalFallback,
        /// No coordinator exists for this run arm.
        SharingDisabled,
    }

    /// Result of one bridge assignment poll.
    #[derive(Debug, Clone, PartialEq)]
    pub enum CatalogBridgeOutcome {
        /// A commissioned bridge segment for this replica.
        Assignment(BridgeAssignmentRecord),
        /// No bridge is commissioned.
        Empty,
        /// The coordinator rejected the request.
        Rejected,
        /// Communication failed and local search remains authoritative.
        LocalFallback,
        /// No coordinator exists for this run arm.
        SharingDisabled,
    }

    /// Result of one synchronous population submission or poll.
    #[derive(Debug, Clone, PartialEq)]
    pub enum PopulationSynchronizationOutcome {
        /// The coordinator has not received every replica representative.
        Pending {
            /// Unique representatives received for this epoch.
            submitted: u32,
            /// Complete population size required to close the epoch.
            required: u32,
        },
        /// The epoch is closed and this destination has a validated parent.
        Ready {
            /// Parent candidate assigned to the requesting replica.
            parent: CatalogCandidate,
            /// Immutable full-population plan and genealogy diagnostics.
            plan: PopulationPlan,
        },
        /// The epoch closed without this replica: it abstained, so the plan
        /// addresses only the replicas that joined, and there is nothing for
        /// this one to adopt. Not an error; the replica rejoins next epoch.
        Unaddressed,
        /// The coordinator rejected the representative or epoch request.
        Rejected,
        /// Communication failed and the local trajectory remains authoritative.
        LocalFallback,
        /// No coordinator exists for this run arm.
        SharingDisabled,
    }

    /// Invalid cooperative-run operation.
    #[derive(Debug, thiserror::Error)]
    pub enum CooperativeRunError {
        /// Aggregate ledger validation failed.
        #[error("cooperative ledger failed: {0}")]
        Ledger(#[from] LedgerError),
        /// Exact coordinator evidence is internally inconsistent.
        #[error("cooperative policy input failed: {0}")]
        PolicyInput(#[from] PolicyInputError),
        /// The coordinator rejected a charged-work ledger boundary.
        #[error("coordinator rejected cooperative ledger event: {0:?}")]
        CoordinatorLedgerRejected(ProtocolRejection),
        /// An operation names a replica outside the run manifest.
        #[error("unknown cooperative replica {replica}")]
        UnknownReplica {
            /// Foreign replica identity.
            replica: u32,
        },
        /// A replica counter cannot be represented.
        #[error("cooperative replica counter overflow")]
        CounterOverflow,
        /// A completed population plan cannot be addressed safely.
        #[error("invalid population plan for epoch {epoch}: {reason}")]
        InvalidPopulationPlan {
            /// Epoch carrying the malformed plan.
            epoch: u64,
            /// Stable invariant violated by the plan.
            reason: &'static str,
        },
        /// A per-slice diagnostic is incomplete, non-finite, or out of order.
        #[error("invalid diagnostic for cooperative slice {slice}: {reason}")]
        InvalidSliceDiagnostic {
            /// Slice carrying invalid evidence.
            slice: u64,
            /// Stable invariant violated by the diagnostic.
            reason: &'static str,
        },
    }

    struct PolicyRequest {
        descriptor: Vec<f64>,
        energy: f64,
        leftover_lambda: f64,
    }

    impl PolicyRequest {
        fn matches(&self, descriptor: &[f64], energy: f64, leftover_lambda: f64) -> bool {
            self.descriptor == descriptor
                && self.energy.to_bits() == energy.to_bits()
                && self.leftover_lambda.to_bits() == leftover_lambda.to_bits()
        }
    }

    struct DescriptorHoleRequest {
        current: Vec<f64>,
        samples: u32,
        draw: u64,
    }

    impl DescriptorHoleRequest {
        fn matches(&self, current: &[f64], samples: u32, draw: u64) -> bool {
            self.current == current && self.samples == samples && self.draw == draw
        }
    }

    struct BoundaryCrossingRequest {
        current: Vec<f64>,
        draw: u64,
    }

    impl BoundaryCrossingRequest {
        fn matches(&self, current: &[f64], draw: u64) -> bool {
            self.current == current && self.draw == draw
        }
    }

    type SampleResult = Result<Option<CatalogCandidate>, CatalogClientError>;
    type SampleBatchResult = Result<Vec<Option<CatalogCandidate>>, CatalogClientError>;

    struct SampleChannel {
        slot: Arc<Mutex<Option<SampleResult>>>,
        pending: bool,
    }

    impl SampleChannel {
        fn new() -> Self {
            Self {
                slot: Arc::new(Mutex::new(None)),
                pending: false,
            }
        }

        fn reset(&mut self) {
            self.pending = false;
            *self.slot.lock().expect("sample slot") = None;
        }
    }

    #[derive(Clone, Copy)]
    enum SampleRole {
        Reference,
        Incumbent,
        Sparse,
    }

    impl SampleRole {
        fn for_draw(draw: u64) -> Self {
            match draw {
                INCUMBENT_SAMPLE_DRAW => Self::Incumbent,
                SPARSE_SAMPLE_DRAW => Self::Sparse,
                _ => Self::Reference,
            }
        }

        const fn index(self) -> usize {
            match self {
                Self::Reference => 0,
                Self::Incumbent => 1,
                Self::Sparse => 2,
            }
        }
    }

    struct ReplicaState {
        trace_sequence: u64,
        ledger_sequence: u64,
        rpc_sequence: u64,
        cumulative_charged: u64,
        client: Option<CatalogMailbox>,
        snapshot: Option<CatalogSnapshot>,
        last_slice: u64,
        policy_slot: Arc<Mutex<Option<Result<PolicyStateReceipt, CatalogClientError>>>>,
        policy_pending: bool,
        policy_request: Option<PolicyRequest>,
        hole_slot: Arc<Mutex<Option<Result<DescriptorHoleProposal, CatalogClientError>>>>,
        hole_pending: bool,
        hole_request: Option<DescriptorHoleRequest>,
        sample_channels: [SampleChannel; 3],
        basin_sample_slot: Arc<Mutex<Option<SampleResult>>>,
        basin_sample_pending: bool,
        basin_sample_request: Option<u64>,
        sample_batch_slot: Arc<Mutex<Option<SampleBatchResult>>>,
        sample_batch_pending: bool,
        crossing_slot:
            Arc<Mutex<Option<Result<Option<BoundaryCrossingRecord>, CatalogClientError>>>>,
        crossing_pending: bool,
        crossing_request: Option<BoundaryCrossingRequest>,
        ride_claims: BTreeMap<u64, RideWorkOrder>,
    }

    /// Four-replica-compatible driver for accounting, policy, RPC, and event output.
    pub struct CooperativeRun {
        ledger: CooperativeLedger,
        replicas: BTreeMap<u32, ReplicaState>,
        events: Vec<TraceEvent>,
        on_published_prize: bool,
    }

    impl CooperativeRun {
        /// Construct a server-free run; clients can be attached per replica.
        pub fn new(
            replicas: impl IntoIterator<Item = u32>,
            per_replica_budget: u64,
        ) -> Result<Self, CooperativeRunError> {
            let replicas = replicas.into_iter().collect::<Vec<_>>();
            let ledger = CooperativeLedger::new(replicas.iter().copied(), per_replica_budget)?;
            let replicas = replicas
                .into_iter()
                .map(|replica| {
                    (
                        replica,
                        ReplicaState {
                            trace_sequence: 0,
                            ledger_sequence: 0,
                            rpc_sequence: 0,
                            cumulative_charged: 0,
                            client: None,
                            snapshot: None,
                            last_slice: 0,
                            policy_slot: Arc::new(Mutex::new(None)),
                            policy_pending: false,
                            policy_request: None,
                            hole_slot: Arc::new(Mutex::new(None)),
                            hole_pending: false,
                            hole_request: None,
                            sample_channels: [
                                SampleChannel::new(),
                                SampleChannel::new(),
                                SampleChannel::new(),
                            ],
                            basin_sample_slot: Arc::new(Mutex::new(None)),
                            basin_sample_pending: false,
                            basin_sample_request: None,
                            sample_batch_slot: Arc::new(Mutex::new(None)),
                            sample_batch_pending: false,
                            crossing_slot: Arc::new(Mutex::new(None)),
                            crossing_pending: false,
                            crossing_request: None,
                            ride_claims: BTreeMap::new(),
                        },
                    )
                })
                .collect();
            Ok(Self {
                ledger,
                replicas,
                events: Vec::new(),
                on_published_prize: false,
            })
        }

        /// Whether the run has reached a published reference energy.
        ///
        /// The coordinator holds no published reference and the policy
        /// state carries no best energy, so this is the caller's to
        /// report; every policy input built afterwards carries it.
        pub fn set_on_published_prize(&mut self, reached: bool) {
            self.on_published_prize = reached;
        }

        /// Attach or replace the coordinator connection for one replica.
        pub fn attach_client(
            &mut self,
            replica: u32,
            client: CatalogClient,
        ) -> Result<(), CooperativeRunError> {
            let used_sequence = client.last_event_sequence();
            let state = self.replica_mut(replica)?;
            state.rpc_sequence = state.rpc_sequence.max(used_sequence);
            state.client = Some(CatalogMailbox::spawn(client));
            state.policy_pending = false;
            state.policy_request = None;
            *state.policy_slot.lock().expect("policy slot") = None;
            state.hole_pending = false;
            state.hole_request = None;
            *state.hole_slot.lock().expect("hole slot") = None;
            for channel in &mut state.sample_channels {
                channel.reset();
            }
            state.basin_sample_pending = false;
            state.basin_sample_request = None;
            *state.basin_sample_slot.lock().expect("basin sample slot") = None;
            state.sample_batch_pending = false;
            *state.sample_batch_slot.lock().expect("sample batch slot") = None;
            state.crossing_pending = false;
            state.crossing_request = None;
            *state.crossing_slot.lock().expect("crossing slot") = None;
            Ok(())
        }

        /// Record one exact local work boundary in the aggregate ledger.
        pub fn record_work(
            &mut self,
            replica: u32,
            kind: ChargeKind,
            charged_calls: u64,
        ) -> Result<(), CooperativeRunError> {
            self.record_work_batch(replica, [(kind, charged_calls)])
        }

        /// Record exact local work boundaries and send them in one request.
        pub fn record_work_batch(
            &mut self,
            replica: u32,
            work: impl IntoIterator<Item = (ChargeKind, u64)>,
        ) -> Result<(), CooperativeRunError> {
            let work = work.into_iter().collect::<Vec<_>>();
            if work.is_empty() {
                return Ok(());
            }
            let (mut ledger_sequence, mut rpc_sequence, mut cumulative_charged, version) = {
                let state = self.replica_mut(replica)?;
                (
                    state.ledger_sequence,
                    state.rpc_sequence,
                    state.cumulative_charged,
                    state.snapshot.map(|snapshot| snapshot.version),
                )
            };
            let mut staged_ledger = self.ledger.clone();
            let mut remote_events = Vec::with_capacity(work.len());
            for (kind, charged_calls) in work {
                ledger_sequence = ledger_sequence
                    .checked_add(1)
                    .ok_or(CooperativeRunError::CounterOverflow)?;
                rpc_sequence = rpc_sequence
                    .checked_add(1)
                    .ok_or(CooperativeRunError::CounterOverflow)?;
                cumulative_charged = cumulative_charged
                    .checked_add(charged_calls)
                    .ok_or(CooperativeRunError::CounterOverflow)?;
                staged_ledger.record(ReplicaLedgerEvent {
                    replica,
                    sequence: ledger_sequence,
                    kind,
                    charged_calls,
                    cumulative_charged,
                })?;
                remote_events.push(CatalogLedgerEvent {
                    sequence: rpc_sequence,
                    kind: kind.wire_code(),
                    charged_calls,
                    cumulative_charged,
                });
            }
            self.ledger = staged_ledger;
            let event_count = remote_events.len();
            {
                let state = self.replica_mut(replica)?;
                state.ledger_sequence = ledger_sequence;
                state.rpc_sequence = rpc_sequence;
                state.cumulative_charged = cumulative_charged;
                if let Some(mailbox) = state.client.as_ref() {
                    let event_sequence = rpc_sequence;
                    mailbox.post(move |client| {
                        let _ = client.record_ledger_batch(event_sequence, remote_events);
                    });
                }
            }
            for _ in 0..event_count {
                self.push_event(replica, TraceKind::LocalWork, version, None)?;
            }
            Ok(())
        }

        /// Wait for this replica's posted work to reach the coordinator.
        ///
        /// [`CooperativeRun::record_work`] posts and returns, which is
        /// what keeps the hop loop off the socket. A caller that then
        /// reads an ensemble aggregate is reading a counter the
        /// coordinator may not have been told about yet.
        pub fn flush(&mut self, replica: u32) -> Result<(), CooperativeRunError> {
            if let Some(mailbox) = self.replica_mut(replica)?.client.as_ref() {
                mailbox.drain();
            }
            Ok(())
        }

        /// Offer one candidate and convert rejection or transport loss into trace state.
        pub fn offer_candidate(
            &mut self,
            replica: u32,
            candidate: CatalogCandidate,
        ) -> Result<CatalogOfferOutcome, CooperativeRunError> {
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let result = {
                let state = self.replica_mut(replica)?;
                state.client.as_ref().map(|mailbox| {
                    mailbox.exec(move |client| client.offer_candidate(rpc_sequence, candidate))
                })
            };
            match result {
                None => {
                    self.push_event(replica, TraceKind::SharingDisabled, None, None)?;
                    Ok(CatalogOfferOutcome::SharingDisabled)
                }
                Some(Ok(receipt)) => {
                    let admitted = receipt.catalog.as_ref().map_or_else(
                        || {
                            self.replicas
                                .get(&replica)
                                .and_then(|state| state.snapshot)
                                .is_none_or(|snapshot| {
                                    receipt.snapshot.active_entries > snapshot.active_entries
                                })
                        },
                        |mutation| mutation.kind.admitted(),
                    );
                    self.replica_mut(replica)?.snapshot = Some(receipt.snapshot);
                    self.push_event(
                        replica,
                        if admitted {
                            TraceKind::Admission
                        } else {
                            TraceKind::Rejection
                        },
                        Some(receipt.version),
                        None,
                    )?;
                    self.events
                        .last_mut()
                        .expect("catalog-offer push appends one trace event")
                        .catalog = receipt.catalog;
                    Ok(if admitted {
                        CatalogOfferOutcome::Admitted
                    } else {
                        CatalogOfferOutcome::Rejected
                    })
                }
                Some(Err(CatalogClientError::Rejected(reason))) => {
                    self.push_event(
                        replica,
                        TraceKind::Rejection,
                        None,
                        Some(rejection_code(reason)),
                    )?;
                    Ok(CatalogOfferOutcome::Rejected)
                }
                Some(Err(_)) => {
                    self.push_event(replica, TraceKind::RpcFallback, None, None)?;
                    Ok(CatalogOfferOutcome::LocalFallback)
                }
            }
        }

        /// Register the validated basin occupied by a replica without adding an edge.
        pub fn record_current(
            &mut self,
            replica: u32,
            candidate: CatalogCandidate,
        ) -> Result<TransitionRecordOutcome, CooperativeRunError> {
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let result = {
                let state = self.replica_mut(replica)?;
                state.client.as_ref().map(|mailbox| {
                    mailbox.exec(move |client| client.record_visit(rpc_sequence, candidate))
                })
            };
            self.handle_transition_record(
                replica,
                "register_current".to_owned(),
                true,
                true,
                result,
            )
        }

        /// Record one action-conditioned perturb--quench result.
        pub fn record_transition(
            &mut self,
            replica: u32,
            action: impl Into<String>,
            destination: TransitionDestination,
            adopted: bool,
        ) -> Result<TransitionRecordOutcome, CooperativeRunError> {
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let action = action.into();
            let action_rpc = action.clone();
            let resolved = matches!(destination, TransitionDestination::Resolved(_));
            let result = {
                let state = self.replica_mut(replica)?;
                state.client.as_ref().map(|mailbox| {
                    mailbox.exec(move |client| {
                        client.record_transition(rpc_sequence, action_rpc, destination, adopted)
                    })
                })
            };
            self.handle_transition_record(replica, action, resolved, adopted, result)
        }

        /// Queue a current-state visit. The hop does not wait.
        pub fn post_record_current(
            &mut self,
            replica: u32,
            candidate: CatalogCandidate,
        ) -> Result<TransitionRecordOutcome, CooperativeRunError> {
            if self.replica_mut(replica)?.client.is_none() {
                return Ok(TransitionRecordOutcome::SharingDisabled);
            }
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            if let Some(mailbox) = self.replica_mut(replica)?.client.as_ref() {
                mailbox.post(move |client| {
                    let _ = client.record_visit(rpc_sequence, candidate);
                });
            }
            Ok(TransitionRecordOutcome::LocalFallback)
        }

        /// Queue a transition record. The hop does not wait.
        pub fn post_record_transition(
            &mut self,
            replica: u32,
            action: impl Into<String>,
            destination: TransitionDestination,
            adopted: bool,
        ) -> Result<TransitionRecordOutcome, CooperativeRunError> {
            if self.replica_mut(replica)?.client.is_none() {
                return Ok(TransitionRecordOutcome::SharingDisabled);
            }
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let action = action.into();
            if let Some(mailbox) = self.replica_mut(replica)?.client.as_ref() {
                mailbox.post(move |client| {
                    let _ = client.record_transition(rpc_sequence, action, destination, adopted);
                });
            }
            Ok(TransitionRecordOutcome::LocalFallback)
        }

        /// Record one validated local perturb--quench independently of RPC registration.
        pub fn record_executed_transition(
            &mut self,
            replica: u32,
            hop: u64,
            action: impl Into<String>,
            from_energy: f64,
            to_energy: f64,
            adopted: bool,
        ) -> Result<(), CooperativeRunError> {
            self.push_event(replica, TraceKind::TransitionExecution, None, None)?;
            self.events
                .last_mut()
                .expect("transition-execution push appends one trace event")
                .transition = Some(TransitionTrace {
                action: action.into(),
                hop: Some(hop),
                from_energy: Some(from_energy),
                to_energy: Some(to_energy),
                resolved: true,
                adopted,
            });
            Ok(())
        }

        /// Poll the coordinator or retain independent local execution.
        pub fn synchronize(
            &mut self,
            replica: u32,
        ) -> Result<SynchronizationOutcome, CooperativeRunError> {
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let result = {
                let state = self.replica_mut(replica)?;
                state
                    .client
                    .as_ref()
                    .map(|mailbox| mailbox.exec(move |client| client.snapshot(rpc_sequence)))
            };
            match result {
                None => {
                    self.push_event(replica, TraceKind::SharingDisabled, None, None)?;
                    Ok(SynchronizationOutcome::SharingDisabled)
                }
                Some(Ok(snapshot)) => {
                    self.replica_mut(replica)?.snapshot = Some(snapshot);
                    self.push_event(
                        replica,
                        TraceKind::SnapshotRefresh,
                        Some(snapshot.version),
                        None,
                    )?;
                    Ok(SynchronizationOutcome::Refreshed(snapshot))
                }
                Some(Err(_)) => {
                    self.push_event(replica, TraceKind::RpcFallback, None, None)?;
                    Ok(SynchronizationOutcome::LocalFallback)
                }
            }
        }

        /// Fetch exact coordinator evidence and map it into the pure policy input.
        pub fn policy_input(
            &mut self,
            replica: u32,
            descriptor: Vec<f64>,
            energy: f64,
            local_stall_slices: u32,
            local_deepened: bool,
        ) -> Result<PolicyEvidenceOutcome, CooperativeRunError> {
            self.policy_input_with_lambda(
                replica,
                descriptor,
                energy,
                0.0,
                local_stall_slices,
                local_deepened,
            )
        }

        /// Policy evidence with the replica's leftover-SOAP \(\lambda\).
        pub fn policy_input_with_lambda(
            &mut self,
            replica: u32,
            descriptor: Vec<f64>,
            energy: f64,
            leftover_lambda: f64,
            local_stall_slices: u32,
            local_deepened: bool,
        ) -> Result<PolicyEvidenceOutcome, CooperativeRunError> {
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let result = {
                let state = self.replica_mut(replica)?;
                state.client.as_ref().map(|mailbox| {
                    mailbox.exec(move |client| {
                        client.policy_state_with_lambda(
                            rpc_sequence,
                            descriptor,
                            energy,
                            leftover_lambda,
                        )
                    })
                })
            };
            match result {
                None => {
                    self.push_event(replica, TraceKind::SharingDisabled, None, None)?;
                    Ok(PolicyEvidenceOutcome::SharingDisabled)
                }
                Some(Ok(receipt)) => {
                    let input = policy_input_from_state(
                        receipt.state,
                        local_stall_slices,
                        local_deepened,
                        self.on_published_prize,
                        leftover_lambda,
                    )?;
                    self.replica_mut(replica)?.snapshot = Some(receipt.snapshot);
                    self.push_event(
                        replica,
                        TraceKind::SnapshotRefresh,
                        Some(receipt.snapshot.version),
                        None,
                    )?;
                    self.events
                        .last_mut()
                        .expect("snapshot-refresh push appends one trace event")
                        .policy = Some(policy_trace(receipt.state, energy));
                    Ok(PolicyEvidenceOutcome::Remote(input))
                }
                Some(Err(CatalogClientError::Rejected(reason))) => {
                    self.push_event(
                        replica,
                        TraceKind::Rejection,
                        None,
                        Some(rejection_code(reason)),
                    )?;
                    Ok(PolicyEvidenceOutcome::Rejected)
                }
                Some(Err(_)) => {
                    self.push_event(replica, TraceKind::RpcFallback, None, None)?;
                    Ok(PolicyEvidenceOutcome::LocalFallback)
                }
            }
        }

        /// Post policy to the mailbox and apply the last answer.
        /// The hop thread never waits on the socket. A late policy
        /// from the coordinator, a leader, or another chain redirects
        /// the next slice; until then this replica is the single chain.
        pub fn try_policy_input(
            &mut self,
            replica: u32,
            descriptor: Vec<f64>,
            energy: f64,
            local_stall_slices: u32,
            local_deepened: bool,
        ) -> Result<PolicyEvidenceOutcome, CooperativeRunError> {
            self.try_policy_input_with_lambda(
                replica,
                descriptor,
                energy,
                0.0,
                local_stall_slices,
                local_deepened,
            )
        }

        /// Poll one mailbox policy request with the replica's leftover-SOAP \(\lambda\).
        /// Each completed receipt is consumed exactly once.
        pub fn try_policy_input_with_lambda(
            &mut self,
            replica: u32,
            descriptor: Vec<f64>,
            energy: f64,
            leftover_lambda: f64,
            local_stall_slices: u32,
            local_deepened: bool,
        ) -> Result<PolicyEvidenceOutcome, CooperativeRunError> {
            if self.replica_mut(replica)?.client.is_none() {
                self.push_event(replica, TraceKind::SharingDisabled, None, None)?;
                return Ok(PolicyEvidenceOutcome::SharingDisabled);
            }
            let finished = {
                let state = self.replica_mut(replica)?;
                if state.policy_pending {
                    let result = state.policy_slot.lock().expect("policy slot").take();
                    result.map(|result| (result, state.policy_request.take()))
                } else {
                    None
                }
            };
            if let Some((result, request)) = finished {
                self.replica_mut(replica)?.policy_pending = false;
                let request_matches = request
                    .as_ref()
                    .is_some_and(|request| request.matches(&descriptor, energy, leftover_lambda));
                if request_matches {
                    match result {
                        Ok(receipt) => {
                            let input = policy_input_from_state(
                                receipt.state,
                                local_stall_slices,
                                local_deepened,
                                self.on_published_prize,
                                leftover_lambda,
                            )?;
                            self.replica_mut(replica)?.snapshot = Some(receipt.snapshot);
                            self.push_event(
                                replica,
                                TraceKind::SnapshotRefresh,
                                Some(receipt.snapshot.version),
                                None,
                            )?;
                            self.events
                                .last_mut()
                                .expect("snapshot-refresh push appends one trace event")
                                .policy = Some(policy_trace(receipt.state, energy));
                            return Ok(PolicyEvidenceOutcome::Remote(input));
                        }
                        Err(CatalogClientError::Rejected(reason)) => {
                            self.push_event(
                                replica,
                                TraceKind::Rejection,
                                None,
                                Some(rejection_code(reason)),
                            )?;
                            return Ok(PolicyEvidenceOutcome::Rejected);
                        }
                        Err(_) => {
                            self.push_event(replica, TraceKind::RpcFallback, None, None)?;
                            return Ok(PolicyEvidenceOutcome::LocalFallback);
                        }
                    }
                }
            }
            let should_post = !self.replica_mut(replica)?.policy_pending;
            if should_post {
                let rpc_sequence = self.next_rpc_sequence(replica)?;
                let state = self.replica_mut(replica)?;
                let slot = Arc::clone(&state.policy_slot);
                if let Some(mailbox) = state.client.as_ref() {
                    state.policy_request = Some(PolicyRequest {
                        descriptor: descriptor.clone(),
                        energy,
                        leftover_lambda,
                    });
                    mailbox.post(move |client| {
                        let answer = client.policy_state_with_lambda(
                            rpc_sequence,
                            descriptor,
                            energy,
                            leftover_lambda,
                        );
                        *slot.lock().expect("policy slot") = Some(answer);
                    });
                    state.policy_pending = true;
                }
            }
            Ok(PolicyEvidenceOutcome::LocalFallback)
        }

        /// Queue an offer. The hop continues without the admission result.
        pub fn post_offer_candidate(
            &mut self,
            replica: u32,
            candidate: CatalogCandidate,
        ) -> Result<CatalogOfferOutcome, CooperativeRunError> {
            if self.replica_mut(replica)?.client.is_none() {
                self.push_event(replica, TraceKind::SharingDisabled, None, None)?;
                return Ok(CatalogOfferOutcome::SharingDisabled);
            }
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            if let Some(mailbox) = self.replica_mut(replica)?.client.as_ref() {
                mailbox.post(move |client| {
                    let _ = client.offer_candidate(rpc_sequence, candidate);
                });
            }
            Ok(CatalogOfferOutcome::LocalFallback)
        }

        /// Register one validated live-chain state and query policy evidence
        /// for that exact coordinator-assigned census observation.
        pub fn registered_policy_input(
            &mut self,
            replica: u32,
            candidate: CatalogCandidate,
            local_stall_slices: u32,
            local_deepened: bool,
        ) -> Result<PolicyEvidenceOutcome, CooperativeRunError> {
            let descriptor = candidate.descriptor.clone();
            let energy = candidate.energy;
            match self.record_current(replica, candidate)? {
                TransitionRecordOutcome::Recorded => self.policy_input(
                    replica,
                    descriptor,
                    energy,
                    local_stall_slices,
                    local_deepened,
                ),
                TransitionRecordOutcome::Rejected => Ok(PolicyEvidenceOutcome::Rejected),
                TransitionRecordOutcome::LocalFallback => Ok(PolicyEvidenceOutcome::LocalFallback),
                TransitionRecordOutcome::SharingDisabled => {
                    Ok(PolicyEvidenceOutcome::SharingDisabled)
                }
            }
        }

        /// Sample one validated active-catalog candidate.
        pub fn sample_candidate(
            &mut self,
            replica: u32,
            draw: u64,
        ) -> Result<CatalogSampleOutcome, CooperativeRunError> {
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let result = {
                let state = self.replica_mut(replica)?;
                state.client.as_ref().map(|mailbox| {
                    mailbox.exec(move |client| client.sample_candidate(rpc_sequence, draw))
                })
            };
            match result {
                None => {
                    self.push_event(replica, TraceKind::SharingDisabled, None, None)?;
                    Ok(CatalogSampleOutcome::SharingDisabled)
                }
                Some(Ok(Some(candidate))) => Ok(CatalogSampleOutcome::Candidate(candidate)),
                Some(Ok(None)) => Ok(CatalogSampleOutcome::Empty),
                Some(Err(CatalogClientError::Rejected(reason))) => {
                    self.push_event(
                        replica,
                        TraceKind::Rejection,
                        None,
                        Some(rejection_code(reason)),
                    )?;
                    Ok(CatalogSampleOutcome::Rejected)
                }
                Some(Err(_)) => {
                    self.push_event(replica, TraceKind::RpcFallback, None, None)?;
                    Ok(CatalogSampleOutcome::LocalFallback)
                }
            }
        }

        /// Claim one exclusive same-system transition-search experiment.
        pub fn claim_ride(
            &mut self,
            replica: u32,
            seed: u64,
        ) -> Result<RideClaimOutcome, CooperativeRunError> {
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let result = {
                let state = self.replica_mut(replica)?;
                state.client.as_ref().map(|mailbox| {
                    mailbox.exec(move |client| client.claim_ride(rpc_sequence, seed))
                })
            };
            match result {
                None => {
                    self.push_event(replica, TraceKind::SharingDisabled, None, None)?;
                    Ok(RideClaimOutcome::SharingDisabled)
                }
                Some(Ok(Some(work))) => {
                    let trace = RideTrace::claim(&work.order);
                    self.replica_mut(replica)?
                        .ride_claims
                        .insert(work.order.id, work.order.clone());
                    self.push_event(replica, TraceKind::RideClaim, None, None)?;
                    self.events
                        .last_mut()
                        .expect("ride-claim push appends one trace event")
                        .ride = Some(trace);
                    Ok(RideClaimOutcome::Work(work))
                }
                Some(Ok(None)) => {
                    self.push_event(replica, TraceKind::RideClaim, None, Some("ride_empty"))?;
                    Ok(RideClaimOutcome::Empty)
                }
                Some(Err(CatalogClientError::Rejected(reason))) => {
                    self.push_event(
                        replica,
                        TraceKind::Rejection,
                        None,
                        Some(rejection_code(reason)),
                    )?;
                    Ok(RideClaimOutcome::Rejected)
                }
                Some(Err(_)) => {
                    self.push_event(replica, TraceKind::RpcFallback, None, None)?;
                    Ok(RideClaimOutcome::LocalFallback)
                }
            }
        }

        /// Report charged transition-search evidence for receiving-side certification.
        pub fn report_ride(
            &mut self,
            replica: u32,
            report: CatalogRideReport,
        ) -> Result<RideReportOutcome, CooperativeRunError> {
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let order = self
                .replica_mut(replica)?
                .ride_claims
                .get(&report.work)
                .cloned();
            let ride_trace = RideTrace::report(order.as_ref(), &report);
            let work = report.work;
            let result = {
                let state = self.replica_mut(replica)?;
                state.client.as_ref().map(|mailbox| {
                    mailbox.exec(move |client| client.report_ride(rpc_sequence, report))
                })
            };
            match result {
                None => {
                    self.push_event(replica, TraceKind::SharingDisabled, None, None)?;
                    self.events
                        .last_mut()
                        .expect("sharing-disabled push appends one trace event")
                        .ride = Some(ride_trace);
                    Ok(RideReportOutcome::SharingDisabled)
                }
                Some(Ok(credit)) => {
                    self.push_event(replica, TraceKind::RideReport, None, None)?;
                    self.events
                        .last_mut()
                        .expect("ride-report push appends one trace event")
                        .ride = Some(ride_trace.with_credit(credit));
                    self.replica_mut(replica)?.ride_claims.remove(&work);
                    Ok(RideReportOutcome::Credited(credit))
                }
                Some(Err(CatalogClientError::Rejected(reason))) => {
                    self.push_event(
                        replica,
                        TraceKind::Rejection,
                        None,
                        Some(rejection_code(reason)),
                    )?;
                    self.events
                        .last_mut()
                        .expect("ride rejection push appends one trace event")
                        .ride = Some(ride_trace);
                    Ok(RideReportOutcome::Rejected)
                }
                Some(Err(_)) => {
                    self.push_event(replica, TraceKind::RpcFallback, None, None)?;
                    self.events
                        .last_mut()
                        .expect("ride fallback push appends one trace event")
                        .ride = Some(ride_trace);
                    Ok(RideReportOutcome::LocalFallback)
                }
            }
        }

        /// Post a sample request and return the last candidate. Hop never waits.
        /// Post one raw frontier excursion state to the shared ladder.
        /// Returns whether a coordinator accepted the post.
        pub fn offer_frontier(
            &mut self,
            replica: u32,
            post: CatalogFrontierPost,
        ) -> Result<bool, CooperativeRunError> {
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let result = {
                let state = self.replica_mut(replica)?;
                state.client.as_ref().map(|mailbox| {
                    let post = post.clone();
                    mailbox.exec(move |client| client.post_frontier(rpc_sequence, post))
                })
            };
            match result {
                None => Ok(false),
                Some(Ok(())) => Ok(true),
                Some(Err(_)) => Ok(false),
            }
        }

        /// Draw one shared frontier post, if the ladder holds any.
        pub fn draw_frontier(
            &mut self,
            replica: u32,
            draw: u64,
        ) -> Result<Option<CatalogFrontierPost>, CooperativeRunError> {
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let result = {
                let state = self.replica_mut(replica)?;
                state.client.as_ref().map(|mailbox| {
                    mailbox.exec(move |client| client.draw_frontier(rpc_sequence, draw))
                })
            };
            match result {
                None => Ok(None),
                Some(Ok(post)) => Ok(post),
                Some(Err(_)) => Ok(None),
            }
        }

        /// Poll one asynchronous sample request without blocking.
        /// Each completed mailbox result is consumed exactly once.
        pub fn try_sample_candidate(
            &mut self,
            replica: u32,
            draw: u64,
        ) -> Result<CatalogSampleOutcome, CooperativeRunError> {
            if self.replica_mut(replica)?.client.is_none() {
                return Ok(CatalogSampleOutcome::SharingDisabled);
            }
            let role = SampleRole::for_draw(draw);
            let finished = {
                let state = self.replica_mut(replica)?;
                let channel = &mut state.sample_channels[role.index()];
                if channel.pending {
                    channel.slot.lock().expect("sample slot").take()
                } else {
                    None
                }
            };
            if let Some(result) = finished {
                self.replica_mut(replica)?.sample_channels[role.index()].pending = false;
                match result {
                    Ok(Some(candidate)) => return Ok(CatalogSampleOutcome::Candidate(candidate)),
                    Ok(None) => return Ok(CatalogSampleOutcome::Empty),
                    Err(CatalogClientError::Rejected(_)) => {
                        return Ok(CatalogSampleOutcome::Rejected);
                    }
                    Err(_) => return Ok(CatalogSampleOutcome::LocalFallback),
                }
            }
            if !self.replica_mut(replica)?.sample_channels[role.index()].pending {
                let rpc_sequence = self.next_rpc_sequence(replica)?;
                let state = self.replica_mut(replica)?;
                let slot = Arc::clone(&state.sample_channels[role.index()].slot);
                if let Some(mailbox) = state.client.as_ref() {
                    mailbox.post(move |client| {
                        let answer = client.sample_candidate(rpc_sequence, draw);
                        *slot.lock().expect("sample slot") = Some(answer);
                    });
                    state.sample_channels[role.index()].pending = true;
                }
            }
            Ok(CatalogSampleOutcome::LocalFallback)
        }

        /// Poll one immutable census-basin representative without blocking.
        ///
        /// A completed response is returned only when it addresses the basin
        /// requested by the current exploration decree. A superseded response
        /// is consumed and the current basin request is posted in its place.
        pub fn try_sample_basin(
            &mut self,
            replica: u32,
            basin: u64,
        ) -> Result<CatalogSampleOutcome, CooperativeRunError> {
            if self.replica_mut(replica)?.client.is_none() {
                return Ok(CatalogSampleOutcome::SharingDisabled);
            }
            let finished = {
                let state = self.replica_mut(replica)?;
                if state.basin_sample_pending {
                    let result = state
                        .basin_sample_slot
                        .lock()
                        .expect("basin sample slot")
                        .take();
                    result.map(|result| (result, state.basin_sample_request.take()))
                } else {
                    None
                }
            };
            if let Some((result, request)) = finished {
                self.replica_mut(replica)?.basin_sample_pending = false;
                if request == Some(basin) {
                    return Ok(match result {
                        Ok(Some(candidate)) => CatalogSampleOutcome::Candidate(candidate),
                        Ok(None) => CatalogSampleOutcome::Empty,
                        Err(CatalogClientError::Rejected(_)) => CatalogSampleOutcome::Rejected,
                        Err(_) => CatalogSampleOutcome::LocalFallback,
                    });
                }
            }
            if !self.replica_mut(replica)?.basin_sample_pending {
                let rpc_sequence = self.next_rpc_sequence(replica)?;
                let state = self.replica_mut(replica)?;
                let slot = Arc::clone(&state.basin_sample_slot);
                if let Some(mailbox) = state.client.as_ref() {
                    state.basin_sample_request = Some(basin);
                    mailbox.post(move |client| {
                        let answer = client.sample_basin(rpc_sequence, basin);
                        *slot.lock().expect("basin sample slot") = Some(answer);
                    });
                    state.basin_sample_pending = true;
                }
            }
            Ok(CatalogSampleOutcome::LocalFallback)
        }

        /// Poll one asynchronous batch of reference-cloud draws without blocking.
        /// A completed batch is consumed once while the supplied draws start the
        /// succeeding batch on the mailbox thread.
        pub fn try_sample_candidates(
            &mut self,
            replica: u32,
            draws: impl IntoIterator<Item = u64>,
        ) -> Result<CatalogSamplesOutcome, CooperativeRunError> {
            if self.replica_mut(replica)?.client.is_none() {
                return Ok(CatalogSamplesOutcome::SharingDisabled);
            }
            let draws = draws.into_iter().collect::<Vec<_>>();
            let finished = {
                let state = self.replica_mut(replica)?;
                if state.sample_batch_pending {
                    state
                        .sample_batch_slot
                        .lock()
                        .expect("sample batch slot")
                        .take()
                } else {
                    None
                }
            };
            if finished.is_some() {
                self.replica_mut(replica)?.sample_batch_pending = false;
            }
            if !draws.is_empty() && !self.replica_mut(replica)?.sample_batch_pending {
                let mut requests = Vec::with_capacity(draws.len());
                for draw in draws {
                    requests.push((self.next_rpc_sequence(replica)?, draw));
                }
                let state = self.replica_mut(replica)?;
                let slot = Arc::clone(&state.sample_batch_slot);
                if let Some(mailbox) = state.client.as_ref() {
                    mailbox.post(move |client| {
                        let answer = requests
                            .into_iter()
                            .map(|(rpc_sequence, draw)| client.sample_candidate(rpc_sequence, draw))
                            .collect::<Result<Vec<_>, _>>();
                        *slot.lock().expect("sample batch slot") = Some(answer);
                    });
                    state.sample_batch_pending = true;
                }
            }
            match finished {
                Some(Ok(samples)) => {
                    let candidates = samples.into_iter().flatten().collect::<Vec<_>>();
                    if candidates.is_empty() {
                        Ok(CatalogSamplesOutcome::Empty)
                    } else {
                        Ok(CatalogSamplesOutcome::Candidates(candidates))
                    }
                }
                Some(Err(CatalogClientError::Rejected(_))) => Ok(CatalogSamplesOutcome::Rejected),
                Some(Err(_)) => Ok(CatalogSamplesOutcome::LocalFallback),
                None => Ok(CatalogSamplesOutcome::LocalFallback),
            }
        }

        /// Request one seeded target-free descriptor-hole proposal.
        pub fn descriptor_hole(
            &mut self,
            replica: u32,
            current: Vec<f64>,
            samples: u32,
            draw: u64,
        ) -> Result<CatalogHoleOutcome, CooperativeRunError> {
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let result = {
                let state = self.replica_mut(replica)?;
                state.client.as_ref().map(|mailbox| {
                    mailbox.exec(move |client| {
                        client.descriptor_hole(rpc_sequence, current, samples, draw)
                    })
                })
            };
            match result {
                None => {
                    self.push_event(replica, TraceKind::SharingDisabled, None, None)?;
                    Ok(CatalogHoleOutcome::SharingDisabled)
                }
                Some(Ok(proposal)) => Ok(CatalogHoleOutcome::Proposal(proposal)),
                Some(Err(CatalogClientError::Rejected(reason))) => {
                    self.push_event(
                        replica,
                        TraceKind::Rejection,
                        None,
                        Some(rejection_code(reason)),
                    )?;
                    Ok(CatalogHoleOutcome::Rejected)
                }
                Some(Err(_)) => {
                    self.push_event(replica, TraceKind::RpcFallback, None, None)?;
                    Ok(CatalogHoleOutcome::LocalFallback)
                }
            }
        }

        /// Poll one shared-cloud hole request without blocking.
        /// Each completed mailbox result is consumed exactly once.
        /// The hop thread never waits. Leave applies this hole, not a
        /// private well list, so extras bias off known superbasins.
        pub fn try_descriptor_hole(
            &mut self,
            replica: u32,
            current: Vec<f64>,
            samples: u32,
            draw: u64,
        ) -> Result<CatalogHoleOutcome, CooperativeRunError> {
            if self.replica_mut(replica)?.client.is_none() {
                self.push_event(replica, TraceKind::SharingDisabled, None, None)?;
                return Ok(CatalogHoleOutcome::SharingDisabled);
            }
            let finished = {
                let state = self.replica_mut(replica)?;
                if state.hole_pending {
                    let result = state.hole_slot.lock().expect("hole slot").take();
                    result.map(|result| (result, state.hole_request.take()))
                } else {
                    None
                }
            };
            if let Some((result, request)) = finished {
                self.replica_mut(replica)?.hole_pending = false;
                let request_matches = request
                    .as_ref()
                    .is_some_and(|request| request.matches(&current, samples, draw));
                if request_matches {
                    match result {
                        Ok(proposal) => return Ok(CatalogHoleOutcome::Proposal(proposal)),
                        Err(CatalogClientError::Rejected(reason)) => {
                            self.push_event(
                                replica,
                                TraceKind::Rejection,
                                None,
                                Some(rejection_code(reason)),
                            )?;
                            return Ok(CatalogHoleOutcome::Rejected);
                        }
                        Err(_) => {
                            self.push_event(replica, TraceKind::RpcFallback, None, None)?;
                            return Ok(CatalogHoleOutcome::LocalFallback);
                        }
                    }
                }
            }
            let should_post = !self.replica_mut(replica)?.hole_pending;
            if should_post {
                let rpc_sequence = self.next_rpc_sequence(replica)?;
                let state = self.replica_mut(replica)?;
                let slot = Arc::clone(&state.hole_slot);
                if let Some(mailbox) = state.client.as_ref() {
                    state.hole_request = Some(DescriptorHoleRequest {
                        current: current.clone(),
                        samples,
                        draw,
                    });
                    mailbox.post(move |client| {
                        let answer = client.descriptor_hole(rpc_sequence, current, samples, draw);
                        *slot.lock().expect("hole slot") = Some(answer);
                    });
                    state.hole_pending = true;
                }
            }
            Ok(CatalogHoleOutcome::LocalFallback)
        }

        /// Request one observed crossing leaving the current attraction region.
        pub fn boundary_crossing(
            &mut self,
            replica: u32,
            current: Vec<f64>,
            draw: u64,
        ) -> Result<CatalogBoundaryOutcome, CooperativeRunError> {
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let result = {
                let state = self.replica_mut(replica)?;
                state.client.as_ref().map(|mailbox| {
                    mailbox
                        .exec(move |client| client.boundary_crossing(rpc_sequence, current, draw))
                })
            };
            match result {
                None => {
                    self.push_event(replica, TraceKind::SharingDisabled, None, None)?;
                    Ok(CatalogBoundaryOutcome::SharingDisabled)
                }
                Some(Ok(Some(crossing))) => Ok(CatalogBoundaryOutcome::Crossing(crossing)),
                Some(Ok(None)) => Ok(CatalogBoundaryOutcome::Empty),
                Some(Err(CatalogClientError::Rejected(reason))) => {
                    self.push_event(
                        replica,
                        TraceKind::Rejection,
                        None,
                        Some(rejection_code(reason)),
                    )?;
                    Ok(CatalogBoundaryOutcome::Rejected)
                }
                Some(Err(_)) => {
                    self.push_event(replica, TraceKind::RpcFallback, None, None)?;
                    Ok(CatalogBoundaryOutcome::LocalFallback)
                }
            }
        }

        /// Poll one crossing request without blocking.
        /// Each completed mailbox result is consumed exactly once.
        pub fn try_boundary_crossing(
            &mut self,
            replica: u32,
            current: Vec<f64>,
            draw: u64,
        ) -> Result<CatalogBoundaryOutcome, CooperativeRunError> {
            if self.replica_mut(replica)?.client.is_none() {
                return Ok(CatalogBoundaryOutcome::SharingDisabled);
            }
            let finished = {
                let state = self.replica_mut(replica)?;
                if state.crossing_pending {
                    let result = state.crossing_slot.lock().expect("crossing slot").take();
                    result.map(|result| (result, state.crossing_request.take()))
                } else {
                    None
                }
            };
            if let Some((result, request)) = finished {
                self.replica_mut(replica)?.crossing_pending = false;
                let request_matches = request
                    .as_ref()
                    .is_some_and(|request| request.matches(&current, draw));
                if request_matches {
                    match result {
                        Ok(Some(crossing)) => {
                            return Ok(CatalogBoundaryOutcome::Crossing(crossing));
                        }
                        Ok(None) => return Ok(CatalogBoundaryOutcome::Empty),
                        Err(CatalogClientError::Rejected(_)) => {
                            return Ok(CatalogBoundaryOutcome::Rejected);
                        }
                        Err(_) => return Ok(CatalogBoundaryOutcome::LocalFallback),
                    }
                }
            }
            if !self.replica_mut(replica)?.crossing_pending {
                let rpc_sequence = self.next_rpc_sequence(replica)?;
                let state = self.replica_mut(replica)?;
                let slot = Arc::clone(&state.crossing_slot);
                if let Some(mailbox) = state.client.as_ref() {
                    state.crossing_request = Some(BoundaryCrossingRequest {
                        current: current.clone(),
                        draw,
                    });
                    mailbox.post(move |client| {
                        let answer = client.boundary_crossing(rpc_sequence, current, draw);
                        *slot.lock().expect("crossing slot") = Some(answer);
                    });
                    state.crossing_pending = true;
                }
            }
            Ok(CatalogBoundaryOutcome::LocalFallback)
        }

        /// Poll for a bridge segment assignment.
        pub fn bridge_assignment(
            &mut self,
            replica: u32,
            draw: u64,
        ) -> Result<CatalogBridgeOutcome, CooperativeRunError> {
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let result = {
                let state = self.replica_mut(replica)?;
                state.client.as_ref().map(|mailbox| {
                    mailbox.exec(move |client| client.bridge_assignment(rpc_sequence, draw))
                })
            };
            match result {
                None => Ok(CatalogBridgeOutcome::SharingDisabled),
                Some(Ok(Some(assignment))) => Ok(CatalogBridgeOutcome::Assignment(assignment)),
                Some(Ok(None)) => Ok(CatalogBridgeOutcome::Empty),
                Some(Err(CatalogClientError::Rejected(reason))) => {
                    self.push_event(
                        replica,
                        TraceKind::Rejection,
                        None,
                        Some(rejection_code(reason)),
                    )?;
                    Ok(CatalogBridgeOutcome::Rejected)
                }
                Some(Err(_)) => {
                    self.push_event(replica, TraceKind::RpcFallback, None, None)?;
                    Ok(CatalogBridgeOutcome::LocalFallback)
                }
            }
        }

        /// Report one attempted exit from a bridge region.
        pub fn bridge_crossing(
            &mut self,
            replica: u32,
            crossing: BridgeCrossingRecord,
        ) -> Result<(), CooperativeRunError> {
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let result = {
                let state = self.replica_mut(replica)?;
                state.client.as_ref().map(|mailbox| {
                    mailbox.exec(move |client| client.bridge_crossing(rpc_sequence, crossing))
                })
            };
            match result {
                None | Some(Ok(())) => Ok(()),
                Some(Err(CatalogClientError::Rejected(reason))) => {
                    self.push_event(
                        replica,
                        TraceKind::Rejection,
                        None,
                        Some(rejection_code(reason)),
                    )?;
                    Ok(())
                }
                Some(Err(_)) => {
                    self.push_event(replica, TraceKind::RpcFallback, None, None)?;
                    Ok(())
                }
            }
        }

        /// Submit one validated chain representative to a population epoch.
        pub fn submit_population(
            &mut self,
            replica: u32,
            epoch: u64,
            candidate: CatalogCandidate,
        ) -> Result<PopulationSynchronizationOutcome, CooperativeRunError> {
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let result = {
                let state = self.replica_mut(replica)?;
                state.client.as_ref().map(|mailbox| {
                    mailbox.exec(move |client| {
                        client.submit_population_with_snapshot(rpc_sequence, epoch, candidate)
                    })
                })
            };
            self.handle_population_result(replica, epoch, result)
        }

        /// Poll an existing population epoch without changing its evidence.
        /// Join this epoch by reference to the best validated candidate the
        /// coordinator already holds for this replica.
        pub fn join_population(
            &mut self,
            replica: u32,
            epoch: u64,
        ) -> Result<PopulationSynchronizationOutcome, CooperativeRunError> {
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let result = {
                let state = self.replica_mut(replica)?;
                state.client.as_ref().map(|mailbox| {
                    mailbox.exec(move |client| {
                        client.population_join_with_snapshot(rpc_sequence, epoch)
                    })
                })
            };
            self.handle_population_result(replica, epoch, result)
        }

        /// Decline this epoch so the replicas waiting on it are released.
        pub fn abstain_population(
            &mut self,
            replica: u32,
            epoch: u64,
        ) -> Result<PopulationSynchronizationOutcome, CooperativeRunError> {
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let result = {
                let state = self.replica_mut(replica)?;
                state.client.as_ref().map(|mailbox| {
                    mailbox.exec(move |client| {
                        client.population_abstain_with_snapshot(rpc_sequence, epoch)
                    })
                })
            };
            self.handle_population_result(replica, epoch, result)
        }

        /// Ask once whether the epoch has closed, without waiting.
        pub fn poll_population(
            &mut self,
            replica: u32,
            epoch: u64,
        ) -> Result<PopulationSynchronizationOutcome, CooperativeRunError> {
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let result = {
                let state = self.replica_mut(replica)?;
                state.client.as_ref().map(|mailbox| {
                    mailbox.exec(move |client| {
                        client.population_plan_with_snapshot(rpc_sequence, epoch)
                    })
                })
            };
            self.handle_population_result(replica, epoch, result)
        }

        /// Evaluate and record one pure catalog policy decision.
        pub fn decide(
            &mut self,
            replica: u32,
            input: CatalogPolicyInput,
        ) -> Result<PolicyDecision, CooperativeRunError> {
            let decision = CatalogPolicy::decide(input);
            let kind = match decision.action {
                PolicyAction::ContinueLocal => TraceKind::PolicyLocal,
                PolicyAction::Exploit { .. } => TraceKind::PolicyExploit,
                PolicyAction::Explore => TraceKind::PolicyExplore,
                PolicyAction::Leave => TraceKind::PolicyLeave,
            };
            self.push_event(replica, kind, None, Some(decision.reason.code()))?;
            Ok(decision)
        }

        /// Exact aggregate ledger.
        pub fn ledger(&self) -> &CooperativeLedger {
            &self.ledger
        }

        /// Deterministic trace in append order.
        pub fn events(&self) -> &[TraceEvent] {
            &self.events
        }

        /// Record one complete, ordered local-slice diagnostic.
        pub fn record_slice(
            &mut self,
            replica: u32,
            diagnostic: SliceTrace,
        ) -> Result<(), CooperativeRunError> {
            validate_slice_trace(diagnostic)?;
            let (expected, cumulative_charged, catalog_version) = {
                let state = self.replica_mut(replica)?;
                (
                    state
                        .last_slice
                        .checked_add(1)
                        .ok_or(CooperativeRunError::CounterOverflow)?,
                    state.cumulative_charged,
                    state.snapshot.map(|snapshot| snapshot.version),
                )
            };
            if diagnostic.slice != expected {
                return Err(CooperativeRunError::InvalidSliceDiagnostic {
                    slice: diagnostic.slice,
                    reason: "slice indices must be contiguous and unique per replica",
                });
            }
            if diagnostic.charged_work > cumulative_charged {
                return Err(CooperativeRunError::InvalidSliceDiagnostic {
                    slice: diagnostic.slice,
                    reason: "slice work exceeds the replica cumulative charged work",
                });
            }
            self.push_event(replica, TraceKind::Slice, catalog_version, None)?;
            self.replica_mut(replica)?.last_slice = diagnostic.slice;
            self.events
                .last_mut()
                .expect("slice push appends one trace event")
                .slice = Some(diagnostic);
            Ok(())
        }

        /// Encode a manifest header followed by one JSON object per trace event.
        pub fn json_lines(&self, manifest: &RunManifest) -> String {
            let engine = serde_json::to_string(&manifest.engine)
                .expect("engine compatibility descriptor must serialize");
            let mut output = format!(
                "{{\"kind\":\"manifest_header\",\"campaign\":\"{}\",\"ensemble\":\"{}\",\"sharing\":{},\"engine\":{}}}\n",
                json_escape(&manifest.campaign),
                json_escape(&manifest.ensemble),
                manifest.sharing,
                engine
            );
            for event in &self.events {
                let version = event
                    .catalog_version
                    .map_or_else(|| "null".to_owned(), |value| value.to_string());
                let reason = event.reason.map_or_else(
                    || "null".to_owned(),
                    |value| format!("\"{}\"", json_escape(value)),
                );
                let (
                    population_epoch,
                    population_parent,
                    family_ordinal,
                    family_size,
                    ess,
                    population_selection,
                ) = event.population.map_or_else(
                    || {
                        (
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                        )
                    },
                    |population| {
                        (
                            population.epoch.to_string(),
                            population.parent.to_string(),
                            population.family_ordinal.to_string(),
                            population.family_size.to_string(),
                            population.effective_sample_size.to_string(),
                            format!("\"{}\"", population.selection.as_trace_str()),
                        )
                    },
                );
                let (
                    policy_local_basin,
                    policy_relation,
                    policy_total_visits,
                    policy_singleton_basins,
                    policy_local_basin_visits,
                    policy_globally_saturated,
                    policy_local_basin_distance,
                    policy_novelty,
                    policy_transition_uncertainty,
                    policy_discovery_role,
                    policy_discovery_epoch,
                    policy_basin_unseen_mass_upper,
                    policy_saddle_unseen_mass_upper,
                    policy_basin_discovery_attempts,
                    policy_basin_discovery_charged,
                    policy_saddle_discovery_attempts,
                    policy_saddle_discovery_charged,
                    policy_saddle_coverage_saturated,
                    policy_query_energy,
                ) = event.policy.map_or_else(
                    || {
                        (
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                        )
                    },
                    |policy| {
                        (
                            policy
                                .local_basin
                                .map_or_else(|| "null".to_owned(), |value| value.to_string()),
                            format!("\"{}\"", catalog_relation_code(policy.relation)),
                            policy.total_visits.to_string(),
                            policy.singleton_basins.to_string(),
                            policy.local_basin_visits.to_string(),
                            policy.globally_saturated.to_string(),
                            policy.local_basin_distance.to_string(),
                            policy.novelty.to_string(),
                            policy.transition_uncertainty.to_string(),
                            format!("\"{}\"", discovery_role_code(policy.discovery_role)),
                            policy.discovery_epoch.to_string(),
                            policy.basin_unseen_mass_upper.to_string(),
                            policy.saddle_unseen_mass_upper.to_string(),
                            policy.basin_discovery_attempts.to_string(),
                            policy.basin_discovery_charged.to_string(),
                            policy.saddle_discovery_attempts.to_string(),
                            policy.saddle_discovery_charged.to_string(),
                            policy.saddle_coverage_saturated.to_string(),
                            policy.query_energy.to_string(),
                        )
                    },
                );
                let [
                    slice,
                    slice_current_basin,
                    slice_active_relation,
                    slice_policy_role,
                    slice_policy_reason,
                    slice_proposal_family,
                    slice_sampled_basin,
                    slice_descriptor_step_norm,
                    slice_cartesian_step_norm,
                    slice_validation,
                    slice_quench,
                    slice_adoption,
                    slice_novelty,
                    slice_energy,
                    slice_charged_work,
                ] = event
                    .slice
                    .map_or_else(
                        || vec!["null".to_owned(); 15],
                        |slice| {
                            vec![
                                slice.slice.to_string(),
                                optional_u64(slice.current_basin),
                                optional_catalog_relation(slice.active_relation),
                                format!("\"{}\"", policy_role_code(slice.policy_role)),
                                format!("\"{}\"", json_escape(slice.policy_reason)),
                                format!("\"{}\"", proposal_family_code(slice.proposal_family)),
                                optional_u64(slice.sampled_basin),
                                optional_f64(slice.descriptor_step_norm),
                                optional_f64(slice.cartesian_step_norm),
                                format!("\"{}\"", slice_validation_code(slice.validation)),
                                format!("\"{}\"", slice_quench_code(slice.quench)),
                                format!("\"{}\"", slice_adoption_code(slice.adoption)),
                                optional_f64(slice.novelty),
                                optional_f64(slice.energy),
                                slice.charged_work.to_string(),
                            ]
                        },
                    )
                    .try_into()
                    .expect("slice JSON field count is fixed");
                let (
                    catalog_basin,
                    catalog_new_basin,
                    catalog_mutation,
                    catalog_evicted,
                    catalog_incumbent,
                ) = event.catalog.as_ref().map_or_else(
                    || {
                        (
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                        )
                    },
                    |catalog| {
                        (
                            catalog.basin_id.to_string(),
                            catalog.new_basin.to_string(),
                            format!("\"{}\"", catalog.kind.code()),
                            format!(
                                "[{}]",
                                catalog
                                    .evicted
                                    .iter()
                                    .map(u64::to_string)
                                    .collect::<Vec<_>>()
                                    .join(",")
                            ),
                            optional_u64(catalog.incumbent_basin),
                        )
                    },
                );
                let (
                    transition_action,
                    transition_hop,
                    transition_from_energy,
                    transition_to_energy,
                    transition_resolved,
                    transition_adopted,
                ) = event.transition.as_ref().map_or_else(
                    || {
                        (
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                            "null".to_owned(),
                        )
                    },
                    |transition| {
                        (
                            format!("\"{}\"", json_escape(&transition.action)),
                            optional_u64(transition.hop),
                            optional_f64(transition.from_energy),
                            optional_f64(transition.to_energy),
                            transition.resolved.to_string(),
                            transition.adopted.to_string(),
                        )
                    },
                );
                let [
                    ride_work,
                    ride_source_basin,
                    ride_environment_class,
                    ride_mode_rank,
                    ride_direction,
                    ride_method,
                    ride_representative_atom,
                    ride_attempt,
                    ride_seed,
                    ride_producer_charged,
                    ride_producer_certified,
                    ride_producer_failure,
                    ride_receiver_charged,
                    ride_receiver_certified,
                    ride_receiver_failure,
                    ride_novel_saddle,
                    ride_degenerate_rearrangement,
                    ride_novel_edge,
                    ride_total_charged,
                ] = event
                    .ride
                    .as_ref()
                    .map_or_else(
                        || vec!["null".to_owned(); 19],
                        |ride| {
                            vec![
                                ride.work.to_string(),
                                optional_u64(ride.source_basin),
                                optional_u64(ride.environment_class.map(u64::from)),
                                optional_u64(ride.mode_rank.map(u64::from)),
                                optional_ride_direction(ride.direction),
                                optional_ride_method(ride.method),
                                optional_u64(ride.representative_atom.map(u64::from)),
                                optional_u64(ride.attempt),
                                optional_u64(ride.seed),
                                optional_u64(ride.producer_charged_evaluations),
                                optional_bool(ride.producer_certified_connection),
                                optional_ride_failure(ride.producer_failure),
                                optional_u64(ride.receiver_charged_evaluations),
                                optional_bool(ride.receiver_certified_connection),
                                optional_ride_failure(ride.receiver_failure),
                                optional_bool(ride.novel_saddle),
                                optional_bool(ride.degenerate_rearrangement),
                                optional_bool(ride.novel_edge),
                                optional_u64(ride.total_charged_evaluations),
                            ]
                        },
                    )
                    .try_into()
                    .expect("ride JSON field count is fixed");
                output.push_str(&format!(
                "{{\"kind\":\"{}\",\"replica\":{},\"sequence\":{},\"aggregate_charged\":{},\"catalog_version\":{},\"reason\":{},\"population_epoch\":{},\"population_parent\":{},\"population_family_ordinal\":{},\"population_family_size\":{},\"population_effective_sample_size\":{},\"population_selection\":{},\"policy_local_basin\":{},\"policy_relation\":{},\"policy_total_visits\":{},\"policy_singleton_basins\":{},\"policy_local_basin_visits\":{},\"policy_globally_saturated\":{},\"policy_local_basin_distance\":{},\"policy_novelty\":{},\"policy_transition_uncertainty\":{},\"policy_discovery_role\":{},\"policy_discovery_epoch\":{},\"policy_basin_unseen_mass_upper\":{},\"policy_saddle_unseen_mass_upper\":{},\"policy_basin_discovery_attempts\":{},\"policy_basin_discovery_charged\":{},\"policy_saddle_discovery_attempts\":{},\"policy_saddle_discovery_charged\":{},\"policy_saddle_coverage_saturated\":{},\"policy_query_energy\":{},\"slice\":{},\"slice_current_basin\":{},\"slice_active_relation\":{},\"slice_policy_role\":{},\"slice_policy_reason\":{},\"slice_proposal_family\":{},\"slice_sampled_basin\":{},\"slice_descriptor_step_norm\":{},\"slice_cartesian_step_norm\":{},\"slice_validation\":{},\"slice_quench\":{},\"slice_adoption\":{},\"slice_novelty\":{},\"slice_energy\":{},\"slice_charged_work\":{},\"catalog_basin\":{},\"catalog_new_basin\":{},\"catalog_mutation\":{},\"catalog_evicted\":{},\"catalog_incumbent\":{},\"transition_action\":{},\"transition_hop\":{},\"transition_from_energy\":{},\"transition_to_energy\":{},\"transition_resolved\":{},\"transition_adopted\":{},\"ride_work\":{},\"ride_source_basin\":{},\"ride_environment_class\":{},\"ride_mode_rank\":{},\"ride_direction\":{},\"ride_method\":{},\"ride_representative_atom\":{},\"ride_attempt\":{},\"ride_seed\":{},\"ride_producer_charged\":{},\"ride_producer_certified\":{},\"ride_producer_failure\":{},\"ride_receiver_charged\":{},\"ride_receiver_certified\":{},\"ride_receiver_failure\":{},\"ride_novel_saddle\":{},\"ride_degenerate_rearrangement\":{},\"ride_novel_edge\":{},\"ride_total_charged\":{}}}\n",
                event.kind.code(),
                event.replica,
                event.sequence,
                event.aggregate_charged,
                version,
                reason,
                population_epoch,
                population_parent,
                family_ordinal,
                family_size,
                ess,
                population_selection,
                policy_local_basin,
                policy_relation,
                policy_total_visits,
                policy_singleton_basins,
                policy_local_basin_visits,
                policy_globally_saturated,
                policy_local_basin_distance,
                policy_novelty,
                policy_transition_uncertainty,
                policy_discovery_role,
                policy_discovery_epoch,
                policy_basin_unseen_mass_upper,
                policy_saddle_unseen_mass_upper,
                policy_basin_discovery_attempts,
                policy_basin_discovery_charged,
                policy_saddle_discovery_attempts,
                policy_saddle_discovery_charged,
                policy_saddle_coverage_saturated,
                policy_query_energy,
                slice,
                slice_current_basin,
                slice_active_relation,
                slice_policy_role,
                slice_policy_reason,
                slice_proposal_family,
                slice_sampled_basin,
                slice_descriptor_step_norm,
                slice_cartesian_step_norm,
                slice_validation,
                slice_quench,
                slice_adoption,
                slice_novelty,
                slice_energy,
                slice_charged_work,
                catalog_basin,
                catalog_new_basin,
                catalog_mutation,
                catalog_evicted,
                catalog_incumbent,
                transition_action,
                transition_hop,
                transition_from_energy,
                transition_to_energy,
                transition_resolved,
                transition_adopted,
                ride_work,
                ride_source_basin,
                ride_environment_class,
                ride_mode_rank,
                ride_direction,
                ride_method,
                ride_representative_atom,
                ride_attempt,
                ride_seed,
                ride_producer_charged,
                ride_producer_certified,
                ride_producer_failure,
                ride_receiver_charged,
                ride_receiver_certified,
                ride_receiver_failure,
                ride_novel_saddle,
                ride_degenerate_rearrangement,
                ride_novel_edge,
                ride_total_charged,
            ));
            }
            output
        }

        fn next_rpc_sequence(&mut self, replica: u32) -> Result<u64, CooperativeRunError> {
            let state = self.replica_mut(replica)?;
            state.rpc_sequence = state
                .rpc_sequence
                .checked_add(1)
                .ok_or(CooperativeRunError::CounterOverflow)?;
            Ok(state.rpc_sequence)
        }

        fn handle_population_result(
            &mut self,
            replica: u32,
            epoch: u64,
            result: Option<
                Result<crate::catalog_rpc::client::PopulationEpochReceipt, CatalogClientError>,
            >,
        ) -> Result<PopulationSynchronizationOutcome, CooperativeRunError> {
            match result {
                None => {
                    self.push_event(replica, TraceKind::SharingDisabled, None, None)?;
                    Ok(PopulationSynchronizationOutcome::SharingDisabled)
                }
                Some(Ok(receipt)) => {
                    self.replica_mut(replica)?.snapshot = Some(receipt.snapshot);
                    self.population_outcome(replica, epoch, receipt.state, receipt.snapshot.version)
                }
                Some(Err(CatalogClientError::Rejected(reason))) => {
                    self.push_event(
                        replica,
                        TraceKind::Rejection,
                        None,
                        Some(rejection_code(reason)),
                    )?;
                    Ok(PopulationSynchronizationOutcome::Rejected)
                }
                Some(Err(_)) => {
                    self.push_event(replica, TraceKind::RpcFallback, None, None)?;
                    Ok(PopulationSynchronizationOutcome::LocalFallback)
                }
            }
        }

        fn handle_transition_record(
            &mut self,
            replica: u32,
            action: String,
            resolved: bool,
            adopted: bool,
            result: Option<Result<crate::catalog_rpc::client::MutationReceipt, CatalogClientError>>,
        ) -> Result<TransitionRecordOutcome, CooperativeRunError> {
            match result {
                None => {
                    self.push_event(replica, TraceKind::SharingDisabled, None, None)?;
                    Ok(TransitionRecordOutcome::SharingDisabled)
                }
                Some(Ok(receipt)) => {
                    self.replica_mut(replica)?.snapshot = Some(receipt.snapshot);
                    self.push_event(
                        replica,
                        TraceKind::Transition,
                        Some(receipt.snapshot.version),
                        None,
                    )?;
                    self.events
                        .last_mut()
                        .expect("transition push appends one trace event")
                        .transition = Some(TransitionTrace {
                        action,
                        hop: None,
                        from_energy: None,
                        to_energy: None,
                        resolved,
                        adopted,
                    });
                    Ok(TransitionRecordOutcome::Recorded)
                }
                Some(Err(CatalogClientError::Rejected(reason))) => {
                    self.push_event(
                        replica,
                        TraceKind::Rejection,
                        None,
                        Some(rejection_code(reason)),
                    )?;
                    Ok(TransitionRecordOutcome::Rejected)
                }
                Some(Err(_)) => {
                    self.push_event(replica, TraceKind::RpcFallback, None, None)?;
                    Ok(TransitionRecordOutcome::LocalFallback)
                }
            }
        }

        fn population_outcome(
            &mut self,
            replica: u32,
            epoch: u64,
            state: PopulationEpochState,
            catalog_version: u64,
        ) -> Result<PopulationSynchronizationOutcome, CooperativeRunError> {
            if state.epoch != epoch {
                // The coordinator already left this epoch (vacant close or
                // retire) or has not reached it. Killing the walk here is
                // how LJ75 died at epoch 2. Skip a stale request; wait on
                // a request that is still in the future.
                if state.epoch > epoch {
                    self.push_event(
                        replica,
                        TraceKind::PopulationReady,
                        Some(catalog_version),
                        None,
                    )?;
                    return Ok(PopulationSynchronizationOutcome::Unaddressed);
                }
                self.push_event(
                    replica,
                    TraceKind::PopulationPending,
                    Some(catalog_version),
                    None,
                )?;
                return Ok(PopulationSynchronizationOutcome::Pending {
                    submitted: state.submitted,
                    required: state.required,
                });
            }
            let Some(plan) = state.plan else {
                if state.required == 0 && state.submitted == 0 {
                    // The epoch closed with every replica abstaining, so
                    // there is no plan and nothing to adopt; the counter
                    // still advances past it.
                    self.push_event(
                        replica,
                        TraceKind::PopulationReady,
                        Some(catalog_version),
                        None,
                    )?;
                    return Ok(PopulationSynchronizationOutcome::Unaddressed);
                }
                if state.required == 0 || state.submitted >= state.required {
                    // Vacant close or a retire that advanced the epoch
                    // without a plan. The replica skips it and stays
                    // on its walk instead of dying.
                    self.push_event(
                        replica,
                        TraceKind::PopulationReady,
                        Some(catalog_version),
                        None,
                    )?;
                    return Ok(PopulationSynchronizationOutcome::Unaddressed);
                }
                self.push_event(
                    replica,
                    TraceKind::PopulationPending,
                    Some(catalog_version),
                    None,
                )?;
                return Ok(PopulationSynchronizationOutcome::Pending {
                    submitted: state.submitted,
                    required: state.required,
                });
            };
            let population_size = plan.destinations.len();
            if plan.epoch != epoch
                || population_size != state.required as usize
                || state.submitted != state.required
                || plan.parents.len() != population_size
                || plan.parent_candidates.len() != population_size
                || plan.weights.len() != population_size
            {
                return Err(CooperativeRunError::InvalidPopulationPlan {
                    epoch,
                    reason: "completed barrier vectors or counts are inconsistent",
                });
            }
            let mut destinations = plan
                .destinations
                .iter()
                .enumerate()
                .filter(|(_, destination)| **destination == replica);
            let Some((index, _)) = destinations.next() else {
                // The plan excludes replicas that abstained from the epoch,
                // so a completed plan without this replica means it declined
                // rather than that the coordinator misaddressed anything.
                self.push_event(
                    replica,
                    TraceKind::PopulationReady,
                    Some(catalog_version),
                    None,
                )?;
                return Ok(PopulationSynchronizationOutcome::Unaddressed);
            };
            if destinations.next().is_some() {
                return Err(CooperativeRunError::InvalidPopulationPlan {
                    epoch,
                    reason: "requesting replica has multiple destinations",
                });
            }
            let parent = plan.parent_candidates[index].clone();
            if parent.producer_replica != plan.parents[index] {
                return Err(CooperativeRunError::InvalidPopulationPlan {
                    epoch,
                    reason: "parent candidate identity does not match genealogy",
                });
            }
            let Some(family) =
                population_family_position(&plan.destinations, &plan.parents, replica)
            else {
                return Err(CooperativeRunError::InvalidPopulationPlan {
                    epoch,
                    reason: "destination family cannot be resolved",
                });
            };
            if !plan.effective_sample_size.is_finite()
                || plan.effective_sample_size < 1.0
                || plan.effective_sample_size > population_size as f64
            {
                return Err(CooperativeRunError::InvalidPopulationPlan {
                    epoch,
                    reason: "effective sample size is outside population bounds",
                });
            }
            let family_ordinal = u32::try_from(family.ordinal()).map_err(|_| {
                CooperativeRunError::InvalidPopulationPlan {
                    epoch,
                    reason: "family ordinal exceeds protocol range",
                }
            })?;
            let family_size = u32::try_from(family.family_size()).map_err(|_| {
                CooperativeRunError::InvalidPopulationPlan {
                    epoch,
                    reason: "family size exceeds protocol range",
                }
            })?;
            self.push_event(
                replica,
                TraceKind::PopulationReady,
                Some(catalog_version),
                None,
            )?;
            self.events
                .last_mut()
                .expect("population-ready push appends one trace event")
                .population = Some(PopulationTrace {
                epoch,
                parent: family.parent(),
                family_ordinal,
                family_size,
                effective_sample_size: plan.effective_sample_size,
                selection: plan.selection,
            });
            Ok(PopulationSynchronizationOutcome::Ready { parent, plan })
        }

        fn push_event(
            &mut self,
            replica: u32,
            kind: TraceKind,
            catalog_version: Option<u64>,
            reason: Option<&'static str>,
        ) -> Result<(), CooperativeRunError> {
            let (sequence, remote_aggregate) = {
                let state = self.replica_mut(replica)?;
                state.trace_sequence = state
                    .trace_sequence
                    .checked_add(1)
                    .ok_or(CooperativeRunError::CounterOverflow)?;
                (
                    state.trace_sequence,
                    state.snapshot.map(|snapshot| snapshot.aggregate_charged),
                )
            };
            self.events.push(TraceEvent {
                replica,
                sequence,
                aggregate_charged: remote_aggregate.unwrap_or_else(|| self.ledger.ensemble_total()),
                catalog_version,
                kind,
                reason,
                population: None,
                policy: None,
                slice: None,
                catalog: None,
                transition: None,
                ride: None,
            });
            Ok(())
        }

        fn replica_mut(&mut self, replica: u32) -> Result<&mut ReplicaState, CooperativeRunError> {
            self.replicas
                .get_mut(&replica)
                .ok_or(CooperativeRunError::UnknownReplica { replica })
        }
    }

    fn rejection_code(reason: ProtocolRejection) -> &'static str {
        match reason {
            ProtocolRejection::Malformed => "malformed",
            ProtocolRejection::UnsupportedVersion => "unsupported_version",
            ProtocolRejection::CampaignMismatch => "campaign_mismatch",
            ProtocolRejection::EnsembleMismatch => "ensemble_mismatch",
            ProtocolRejection::ReplicaMismatch => "replica_mismatch",
            ProtocolRejection::SignatureMismatch => "signature_mismatch",
            ProtocolRejection::SequenceReplay => "sequence_replay",
            ProtocolRejection::SequenceRegression => "sequence_regression",
            ProtocolRejection::SnapshotRegression => "snapshot_regression",
            ProtocolRejection::ValidationRejected => "validation_rejected",
        }
    }

    fn catalog_relation_code(relation: CatalogRelation) -> &'static str {
        match relation {
            CatalogRelation::Empty => "empty",
            CatalogRelation::Incumbent => "incumbent",
            CatalogRelation::SameBasin => "same_basin",
            CatalogRelation::UnrelatedNoAnchor => "unrelated_no_anchor",
            CatalogRelation::UnrelatedLowerAnchor => "unrelated_lower_anchor",
        }
    }

    fn discovery_role_code(role: DiscoveryRole) -> &'static str {
        match role {
            DiscoveryRole::BasinEscape => "basin_escape",
            DiscoveryRole::SaddleRide => "saddle_ride",
        }
    }

    fn policy_role_code(role: PolicyRole) -> &'static str {
        match role {
            PolicyRole::Local => "local",
            PolicyRole::Exploit => "exploit",
            PolicyRole::Explore => "explore",
            PolicyRole::Leave => "leave",
            PolicyRole::Unavailable => "unavailable",
        }
    }

    fn proposal_family_code(family: ProposalFamily) -> &'static str {
        match family {
            ProposalFamily::Local => "local",
            ProposalFamily::CatalogSample => "catalog_sample",
            ProposalFamily::DescriptorHole => "descriptor_hole",
            ProposalFamily::BoundaryTransport => "boundary_transport",
            ProposalFamily::TransitionRide => "transition_ride",
            ProposalFamily::PopulationReconfiguration => "population_reconfiguration",
            ProposalFamily::HyperbandReseed => "hyperband_reseed",
        }
    }

    fn slice_validation_code(result: SliceValidation) -> &'static str {
        match result {
            SliceValidation::NotAttempted => "not_attempted",
            SliceValidation::Accepted => "accepted",
            SliceValidation::Rejected => "rejected",
        }
    }

    fn slice_quench_code(result: SliceQuench) -> &'static str {
        match result {
            SliceQuench::NotAttempted => "not_attempted",
            SliceQuench::Converged => "converged",
            SliceQuench::Rejected => "rejected",
        }
    }

    fn slice_adoption_code(result: SliceAdoption) -> &'static str {
        match result {
            SliceAdoption::NotAttempted => "not_attempted",
            SliceAdoption::Adopted => "adopted",
            SliceAdoption::NotImproved => "not_improved",
            SliceAdoption::Rejected => "rejected",
        }
    }

    fn optional_u64(value: Option<u64>) -> String {
        value.map_or_else(|| "null".to_owned(), |value| value.to_string())
    }

    fn optional_f64(value: Option<f64>) -> String {
        value.map_or_else(|| "null".to_owned(), |value| value.to_string())
    }

    fn optional_bool(value: Option<bool>) -> String {
        value.map_or_else(|| "null".to_owned(), |value| value.to_string())
    }

    fn optional_ride_direction(value: Option<RideDirection>) -> String {
        value.map_or_else(
            || "null".to_owned(),
            |value| {
                format!(
                    "\"{}\"",
                    match value {
                        RideDirection::Negative => "negative",
                        RideDirection::Positive => "positive",
                    }
                )
            },
        )
    }

    fn optional_ride_method(value: Option<RideMethod>) -> String {
        value.map_or_else(
            || "null".to_owned(),
            |value| {
                format!(
                    "\"{}\"",
                    match value {
                        RideMethod::Dimer => "dimer",
                        RideMethod::Lanczos => "lanczos",
                    }
                )
            },
        )
    }

    fn optional_ride_failure(value: Option<RideFailure>) -> String {
        value.map_or_else(
            || "null".to_owned(),
            |value| format!("\"{}\"", ride_failure_code(value)),
        )
    }

    fn ride_failure_code(value: RideFailure) -> &'static str {
        match value {
            RideFailure::QuenchNotConverged => "quench_not_converged",
            RideFailure::SaddleNotConverged => "saddle_not_converged",
            RideFailure::MinimumModeNotConverged => "minimum_mode_not_converged",
            RideFailure::PrfoNotConverged => "prfo_not_converged",
            RideFailure::SaddleForceNotConverged => "saddle_force_not_converged",
            RideFailure::ActivationNotEscaped => "activation_not_escaped",
            RideFailure::MinimumModeLostCurvature => "minimum_mode_lost_curvature",
            RideFailure::NoNegativeMode => "no_negative_mode",
            RideFailure::HigherIndex => "higher_index",
            RideFailure::IrcNotConverged => "irc_not_converged",
            RideFailure::CollapsedConnection => "collapsed_connection",
            RideFailure::Surface => "surface",
            RideFailure::BudgetExhausted => "budget_exhausted",
            RideFailure::DisconnectedConnection => "disconnected_connection",
        }
    }

    fn optional_catalog_relation(value: Option<CatalogRelation>) -> String {
        value.map_or_else(
            || "null".to_owned(),
            |value| format!("\"{}\"", catalog_relation_code(value)),
        )
    }

    fn validate_slice_trace(diagnostic: SliceTrace) -> Result<(), CooperativeRunError> {
        let finite_nonnegative =
            |value: Option<f64>| value.is_none_or(|value| value.is_finite() && value >= 0.0);
        if diagnostic.slice == 0 {
            return Err(CooperativeRunError::InvalidSliceDiagnostic {
                slice: diagnostic.slice,
                reason: "slice indices are one-based",
            });
        }
        if diagnostic.energy.is_some_and(|energy| !energy.is_finite())
            || !finite_nonnegative(diagnostic.descriptor_step_norm)
            || !finite_nonnegative(diagnostic.cartesian_step_norm)
            || !finite_nonnegative(diagnostic.novelty)
        {
            return Err(CooperativeRunError::InvalidSliceDiagnostic {
                slice: diagnostic.slice,
                reason: "slice energy must be finite and norms and novelty must be nonnegative",
            });
        }
        Ok(())
    }

    fn policy_trace(state: PolicyState, query_energy: f64) -> PolicyTrace {
        PolicyTrace {
            local_basin: state.local_basin,
            relation: state.relation,
            total_visits: state.total_visits,
            singleton_basins: state.singleton_basins,
            local_basin_visits: state.local_basin_visits,
            globally_saturated: state.globally_saturated,
            local_basin_distance: state.local_basin_distance,
            novelty: state.novelty,
            transition_uncertainty: state.transition_uncertainty,
            discovery_role: state.discovery_role,
            discovery_epoch: state.discovery_epoch,
            basin_unseen_mass_upper: state.basin_unseen_mass_upper,
            saddle_unseen_mass_upper: state.saddle_unseen_mass_upper,
            basin_discovery_attempts: state.basin_discovery_attempts,
            basin_discovery_charged: state.basin_discovery_charged,
            saddle_discovery_attempts: state.saddle_discovery_attempts,
            saddle_discovery_charged: state.saddle_discovery_charged,
            saddle_coverage_saturated: state.saddle_coverage_saturated,
            query_energy,
            retired: state.retired,
        }
    }

    fn policy_input_from_state(
        state: PolicyState,
        local_stall_slices: u32,
        local_deepened: bool,
        on_published_prize: bool,
        path_lambda: f64,
    ) -> Result<CatalogPolicyInput, PolicyInputError> {
        let relation = match state.relation {
            CatalogRelation::Empty => ActiveCatalogRelation::Empty,
            CatalogRelation::Incumbent => ActiveCatalogRelation::Incumbent,
            CatalogRelation::SameBasin => ActiveCatalogRelation::SameBasin,
            CatalogRelation::UnrelatedNoAnchor => ActiveCatalogRelation::Unrelated {
                lower_energy_anchor: false,
            },
            CatalogRelation::UnrelatedLowerAnchor => ActiveCatalogRelation::Unrelated {
                lower_energy_anchor: true,
            },
        };
        Ok(CatalogPolicyInput {
            validation: ValidationState::Validated,
            relation,
            census: CensusEvidence::from_exact_counts(
                state.total_visits,
                state.singleton_basins,
                state.local_basin_visits,
                state.globally_saturated,
            )?,
            progress: AggregateProgress::new(state.aggregate_charged, state.aggregate_budget)?,
            local_stall_slices,
            local_deepened,
            mixing: MixingEvidence {
                explore_collapsed: state.explore_collapsed,
                certified_attractor: state.certified_attractor,
                pruned: state.pruned,
            },
            // The TIS order parameter is the maximum over the path, and
            // the state reports one frame. The client posted the maximum
            // of the frames before this one, so the two together are it.
            leftover_lambda: if path_lambda.is_finite() {
                path_lambda.max(state.leftover_lambda)
            } else {
                state.leftover_lambda
            },
            interface_rank: state.interface_rank,
            interface_threshold: state.interface_threshold,
            occupied_family_count: state.occupied_family_count as usize,
            packing_saturated: state.packing_saturated,
            leftover_dwell: state.leftover_dwell,
            ei_exhausted: state.ei_exhausted,
            min_families: state.min_families as usize,
            on_published_prize,
        })
    }

    fn json_escape(value: &str) -> String {
        value
            .chars()
            .flat_map(|character| match character {
                '\\' => "\\\\".chars().collect::<Vec<_>>(),
                '"' => "\\\"".chars().collect(),
                '\n' => "\\n".chars().collect(),
                '\r' => "\\r".chars().collect(),
                '\t' => "\\t".chars().collect(),
                character => vec![character],
            })
            .collect()
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::compatibility::{AbiStamp, EngineDescriptor, ProtocolVersion};

        #[test]
        fn json_manifest_header_contains_engine_compatibility_descriptor() {
            let run = CooperativeRun::new([0], 1).expect("valid cooperative run");
            let manifest = RunManifest {
                campaign: "campaign".to_owned(),
                ensemble: "ensemble".to_owned(),
                sharing: false,
                engine: EngineDescriptor::new(
                    "rgpot",
                    ProtocolVersion::new(1, 2),
                    AbiStamp::anneal_default(),
                ),
            };

            let trace = run.json_lines(&manifest);
            let header = trace.lines().next().expect("manifest header");
            let value: serde_json::Value =
                serde_json::from_str(header).expect("manifest header is JSON");
            assert_eq!(value["engine"]["engine_id"], "rgpot");
            assert_eq!(value["engine"]["protocol"]["major"], 1);
            assert_eq!(value["engine"]["protocol"]["minor"], 2);
            let native = eindir_core::ffi::eindir_core_abi_stamp();
            assert_eq!(
                value["engine"]["abi"]["abi_major"],
                serde_json::Value::from(native.abi_major)
            );
            assert_eq!(
                value["engine"]["abi"]["abi_minor"],
                serde_json::Value::from(native.abi_minor)
            );
            assert_eq!(
                value["engine"]["abi"]["layout_revision"],
                serde_json::Value::from(native.objective_layout)
            );
            assert_eq!(
                value["engine"]["abi"]["objective_size"],
                serde_json::Value::from(native.objective_size)
            );
            assert_eq!(
                value["engine"]["abi"]["objective_align"],
                serde_json::Value::from(native.objective_align)
            );
        }
    }
}

#[cfg(feature = "bank-rpc")]
pub use run::*;
