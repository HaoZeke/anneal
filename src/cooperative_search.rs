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
        CatalogCandidate, CatalogRelation, CatalogSnapshot, PolicyState, ProtocolRejection,
    };

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
        /// Coordinator communication failed and local execution remained active.
        RpcFallback,
        /// The run has no sharing transport by construction.
        SharingDisabled,
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
                Self::RpcFallback => "rpc_fallback",
                Self::SharingDisabled => "sharing_disabled",
            }
        }
    }

    /// One deterministic newline-delimited run event.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

    /// Invalid cooperative-run operation.
    #[derive(Debug, thiserror::Error)]
    pub enum CooperativeRunError {
        /// Aggregate ledger validation failed.
        #[error("cooperative ledger failed: {0}")]
        Ledger(#[from] LedgerError),
        /// Exact coordinator evidence is internally inconsistent.
        #[error("cooperative policy input failed: {0}")]
        PolicyInput(#[from] PolicyInputError),
        /// An operation names a replica outside the run manifest.
        #[error("unknown cooperative replica {replica}")]
        UnknownReplica {
            /// Foreign replica identity.
            replica: u32,
        },
        /// A replica counter cannot be represented.
        #[error("cooperative replica counter overflow")]
        CounterOverflow,
    }

    struct ReplicaState {
        trace_sequence: u64,
        ledger_sequence: u64,
        rpc_sequence: u64,
        cumulative_charged: u64,
        client: Option<CatalogClient>,
        snapshot: Option<CatalogSnapshot>,
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
            self.push_event(replica, TraceKind::LocalWork, None, None)?;
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
                match state.client.as_mut() {
                    Some(client) => Some(client.offer_candidate(rpc_sequence, candidate)),
                    None => None,
                }
            };
            match result {
                None => {
                    self.push_event(replica, TraceKind::SharingDisabled, None, None)?;
                    Ok(CatalogOfferOutcome::SharingDisabled)
                }
                Some(Ok(receipt)) => {
                    let admitted = self
                        .replicas
                        .get(&replica)
                        .and_then(|state| state.snapshot)
                        .is_none_or(|snapshot| {
                            receipt.snapshot.active_entries > snapshot.active_entries
                        });
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

        /// Poll the coordinator or retain independent local execution.
        pub fn synchronize(
            &mut self,
            replica: u32,
        ) -> Result<SynchronizationOutcome, CooperativeRunError> {
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let result = {
                let state = self.replica_mut(replica)?;
                match state.client.as_mut() {
                    Some(client) => Some(client.snapshot(rpc_sequence)),
                    None => None,
                }
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
            progress: AggregateProgress,
            local_stall_slices: u32,
            local_deepened: bool,
        ) -> Result<PolicyEvidenceOutcome, CooperativeRunError> {
            let rpc_sequence = self.next_rpc_sequence(replica)?;
            let result = {
                let state = self.replica_mut(replica)?;
                match state.client.as_mut() {
                    Some(client) => {
                        Some(client.policy_state_with_snapshot(rpc_sequence, descriptor, energy))
                    }
                    None => None,
                }
            };
            match result {
                None => {
                    self.push_event(replica, TraceKind::SharingDisabled, None, None)?;
                    Ok(PolicyEvidenceOutcome::SharingDisabled)
                }
                Some(Ok(receipt)) => {
                    let input = policy_input_from_state(
                        receipt.state,
                        progress,
                        local_stall_slices,
                        local_deepened,
                    )?;
                    self.replica_mut(replica)?.snapshot = Some(receipt.snapshot);
                    self.push_event(
                        replica,
                        TraceKind::SnapshotRefresh,
                        Some(receipt.snapshot.version),
                        None,
                    )?;
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
                output.push_str(&format!(
                "{{\"kind\":\"{}\",\"replica\":{},\"sequence\":{},\"aggregate_charged\":{},\"catalog_version\":{},\"reason\":{}}}\n",
                event.kind.code(),
                event.replica,
                event.sequence,
                event.aggregate_charged,
                version,
                reason
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

        fn push_event(
            &mut self,
            replica: u32,
            kind: TraceKind,
            catalog_version: Option<u64>,
            reason: Option<&'static str>,
        ) -> Result<(), CooperativeRunError> {
            let sequence = {
                let state = self.replica_mut(replica)?;
                state.trace_sequence = state
                    .trace_sequence
                    .checked_add(1)
                    .ok_or(CooperativeRunError::CounterOverflow)?;
                state.trace_sequence
            };
            self.events.push(TraceEvent {
                replica,
                sequence,
                aggregate_charged: self.ledger.ensemble_total(),
                catalog_version,
                kind,
                reason,
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

    fn policy_input_from_state(
        state: PolicyState,
        progress: AggregateProgress,
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
            progress,
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
