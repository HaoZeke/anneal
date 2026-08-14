//! Cooperative search composition and exact aggregate work accounting.

pub mod ledger;

#[cfg(feature = "bank-rpc")]
mod run {
    use std::collections::BTreeMap;

    use crate::catalog_policy::{
        ActiveCatalogRelation, AggregateProgress, CatalogPolicy, CatalogPolicyInput,
        CensusEvidence, PolicyAction, PolicyDecision, PolicyInputError, ValidationState,
    };
    use crate::catalog_rpc::client::{CatalogClient, CatalogClientError};
    use crate::catalog_rpc::{
        BoundaryCrossingRecord, CatalogCandidate, CatalogMutation, CatalogRelation,
        CatalogSnapshot, DescriptorHoleProposal, PolicyState, PopulationEpochState,
        PopulationPlan, ProtocolRejection, TransitionDestination,
    };
    use crate::methods::feynman_kac::population_family_position;

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
        /// Distance from the query to the nearest distinct census medoid.
        pub novelty: f64,
        /// Posterior uncertainty of the latent Gaussian transition field.
        pub transition_uncertainty: f64,
        /// Energy used to classify the query against the active catalog.
        pub query_energy: f64,
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
        /// Synchronous fixed-population reconfiguration.
        PopulationReconfiguration,
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

    struct ReplicaState {
        trace_sequence: u64,
        ledger_sequence: u64,
        rpc_sequence: u64,
        cumulative_charged: u64,
        client: Option<CatalogClient>,
        snapshot: Option<CatalogSnapshot>,
        last_slice: u64,
    }

    /// Four-replica-compatible driver for accounting, policy, RPC, and event output.
    pub struct CooperativeRun {
        ledger: CooperativeLedger,
        replicas: BTreeMap<u32, ReplicaState>,
        events: Vec<TraceEvent>,
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
                        },
                    )
                })
                .collect();
            Ok(Self {
                ledger,
                replicas,
                events: Vec::new(),
            })
        }

        /// Attach or replace the coordinator connection for one replica.
        pub fn attach_client(
            &mut self,
            replica: u32,
            client: CatalogClient,
        ) -> Result<(), CooperativeRunError> {
            self.replica_mut(replica)?.client = Some(client);
            Ok(())
        }

        /// Record one exact local work boundary in the aggregate ledger.
        pub fn record_work(
            &mut self,
            replica: u32,
            kind: ChargeKind,
            charged_calls: u64,
        ) -> Result<(), CooperativeRunError> {
            let (sequence, cumulative_charged) = {
                let state = self.replica_mut(replica)?;
                state.ledger_sequence = state
                    .ledger_sequence
                    .checked_add(1)
                    .ok_or(CooperativeRunError::CounterOverflow)?;
                state.cumulative_charged = state
                    .cumulative_charged
                    .checked_add(charged_calls)
                    .ok_or(CooperativeRunError::CounterOverflow)?;
                (state.ledger_sequence, state.cumulative_charged)
            };
            self.ledger.record(ReplicaLedgerEvent {
                replica,
                sequence,
                kind,
                charged_calls,
                cumulative_charged,
            })?;
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let result = {
                let state = self.replica_mut(replica)?;
                state.client.as_mut().map(|client| {
                    client.record_ledger_event(
                        rpc_sequence,
                        kind,
                        charged_calls,
                        cumulative_charged,
                    )
                })
            };
            match result {
                None => self.push_event(replica, TraceKind::LocalWork, None, None)?,
                Some(Ok(receipt)) => {
                    self.replica_mut(replica)?.snapshot = Some(receipt.snapshot);
                    self.push_event(
                        replica,
                        TraceKind::LocalWork,
                        Some(receipt.snapshot.version),
                        None,
                    )?;
                }
                Some(Err(CatalogClientError::Rejected(reason))) => {
                    self.push_event(
                        replica,
                        TraceKind::Rejection,
                        None,
                        Some(rejection_code(reason)),
                    )?;
                    return Err(CooperativeRunError::CoordinatorLedgerRejected(reason));
                }
                Some(Err(_)) => {
                    self.push_event(replica, TraceKind::LocalWork, None, None)?;
                    self.push_event(replica, TraceKind::RpcFallback, None, None)?;
                }
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
                state
                    .client
                    .as_mut()
                    .map(|client| client.offer_candidate(rpc_sequence, candidate))
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
                state
                    .client
                    .as_mut()
                    .map(|client| client.record_visit(rpc_sequence, candidate))
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
            let resolved = matches!(destination, TransitionDestination::Resolved(_));
            let result = {
                let state = self.replica_mut(replica)?;
                state.client.as_mut().map(|client| {
                    client.record_transition(rpc_sequence, action.clone(), destination, adopted)
                })
            };
            self.handle_transition_record(replica, action, resolved, adopted, result)
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
                    .as_mut()
                    .map(|client| client.snapshot(rpc_sequence))
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
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let result = {
                let state = self.replica_mut(replica)?;
                state.client.as_mut().map(|client| {
                    client.policy_state_with_snapshot(rpc_sequence, descriptor, energy)
                })
            };
            match result {
                None => {
                    self.push_event(replica, TraceKind::SharingDisabled, None, None)?;
                    Ok(PolicyEvidenceOutcome::SharingDisabled)
                }
                Some(Ok(receipt)) => {
                    let input =
                        policy_input_from_state(receipt.state, local_stall_slices, local_deepened)?;
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

        /// Sample one validated active-catalog candidate.
        pub fn sample_candidate(
            &mut self,
            replica: u32,
            draw: u64,
        ) -> Result<CatalogSampleOutcome, CooperativeRunError> {
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let result = {
                let state = self.replica_mut(replica)?;
                state
                    .client
                    .as_mut()
                    .map(|client| client.sample_candidate(rpc_sequence, draw))
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
                state
                    .client
                    .as_mut()
                    .map(|client| client.descriptor_hole(rpc_sequence, current, samples, draw))
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
                state
                    .client
                    .as_mut()
                    .map(|client| client.boundary_crossing(rpc_sequence, current, draw))
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
                state.client.as_mut().map(|client| {
                    client.submit_population_with_snapshot(rpc_sequence, epoch, candidate)
                })
            };
            self.handle_population_result(replica, epoch, result)
        }

        /// Poll an existing population epoch without changing its evidence.
        pub fn poll_population(
            &mut self,
            replica: u32,
            epoch: u64,
        ) -> Result<PopulationSynchronizationOutcome, CooperativeRunError> {
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let result = {
                let state = self.replica_mut(replica)?;
                state
                    .client
                    .as_mut()
                    .map(|client| client.population_plan_with_snapshot(rpc_sequence, epoch))
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
            let mut output = format!(
                "{{\"kind\":\"manifest_header\",\"campaign\":\"{}\",\"ensemble\":\"{}\",\"sharing\":{}}}\n",
                json_escape(&manifest.campaign),
                json_escape(&manifest.ensemble),
                manifest.sharing
            );
            for event in &self.events {
                let version = event
                    .catalog_version
                    .map_or_else(|| "null".to_owned(), |value| value.to_string());
                let reason = event.reason.map_or_else(
                    || "null".to_owned(),
                    |value| format!("\"{}\"", json_escape(value)),
                );
                let (population_epoch, population_parent, family_ordinal, family_size, ess) =
                    event.population.map_or_else(
                        || {
                            (
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
                let (catalog_basin, catalog_mutation, catalog_evicted, catalog_incumbent) =
                    event.catalog.as_ref().map_or_else(
                        || {
                            (
                                "null".to_owned(),
                                "null".to_owned(),
                                "null".to_owned(),
                                "null".to_owned(),
                            )
                        },
                        |catalog| {
                            (
                                catalog.basin_id.to_string(),
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
                ) =
                    event.transition.as_ref().map_or_else(
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
                output.push_str(&format!(
                "{{\"kind\":\"{}\",\"replica\":{},\"sequence\":{},\"aggregate_charged\":{},\"catalog_version\":{},\"reason\":{},\"population_epoch\":{},\"population_parent\":{},\"population_family_ordinal\":{},\"population_family_size\":{},\"population_effective_sample_size\":{},\"policy_local_basin\":{},\"policy_relation\":{},\"policy_total_visits\":{},\"policy_singleton_basins\":{},\"policy_local_basin_visits\":{},\"policy_globally_saturated\":{},\"policy_local_basin_distance\":{},\"policy_novelty\":{},\"policy_transition_uncertainty\":{},\"policy_query_energy\":{},\"slice\":{},\"slice_current_basin\":{},\"slice_active_relation\":{},\"slice_policy_role\":{},\"slice_policy_reason\":{},\"slice_proposal_family\":{},\"slice_sampled_basin\":{},\"slice_descriptor_step_norm\":{},\"slice_cartesian_step_norm\":{},\"slice_validation\":{},\"slice_quench\":{},\"slice_adoption\":{},\"slice_novelty\":{},\"slice_energy\":{},\"slice_charged_work\":{},\"catalog_basin\":{},\"catalog_mutation\":{},\"catalog_evicted\":{},\"catalog_incumbent\":{},\"transition_action\":{},\"transition_hop\":{},\"transition_from_energy\":{},\"transition_to_energy\":{},\"transition_resolved\":{},\"transition_adopted\":{}}}\n",
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
                policy_local_basin,
                policy_relation,
                policy_total_visits,
                policy_singleton_basins,
                policy_local_basin_visits,
                policy_globally_saturated,
                policy_local_basin_distance,
                policy_novelty,
                policy_transition_uncertainty,
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
                catalog_mutation,
                catalog_evicted,
                catalog_incumbent,
                transition_action,
                transition_hop,
                transition_from_energy,
                transition_to_energy,
                transition_resolved,
                transition_adopted,
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
                return Err(CooperativeRunError::InvalidPopulationPlan {
                    epoch,
                    reason: "barrier epoch does not match the request",
                });
            }
            let Some(plan) = state.plan else {
                if state.required == 0 || state.submitted >= state.required {
                    return Err(CooperativeRunError::InvalidPopulationPlan {
                        epoch,
                        reason: "pending barrier has invalid submission counts",
                    });
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
                return Err(CooperativeRunError::InvalidPopulationPlan {
                    epoch,
                    reason: "requesting replica has no destination",
                });
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
            ProposalFamily::PopulationReconfiguration => "population_reconfiguration",
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
            query_energy,
        }
    }

    fn policy_input_from_state(
        state: PolicyState,
        local_stall_slices: u32,
        local_deepened: bool,
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
}

#[cfg(feature = "bank-rpc")]
pub use run::*;
