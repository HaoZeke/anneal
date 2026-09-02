//! Timeout-bounded client for one isolated catalog coordinator.

use std::collections::BTreeMap;
use std::io::Write;
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

use capnp::message::ReaderOptions;
use capnp::serialize;

use super::{
    AcceptedPayload, AcceptedReply, BoundaryCrossingRecord, CatalogCandidate, CatalogFrontierPost,
    CatalogIdentity, CatalogLedgerEvent, CatalogMutation, CatalogOperation, CatalogReply,
    CatalogRequest, CatalogRideReport, CatalogRideWork, CatalogSnapshot, DescriptorHoleProposal,
    PROTOCOL_VERSION, PolicyState, PopulationEpochState, ProtocolError, ProtocolRejection,
    RosterReply, TransitionDestination, decode_reply_reader, encode_request,
};
use crate::Catalog_capnp::catalog_reply;
use crate::cooperative_search::ledger::ChargeKind;

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MutationReceipt {
    /// Coordinator snapshot version after the mutation.
    pub version: u64,
    /// Whether the coordinator recognized an identical replay.
    pub duplicate: bool,
    /// Coordinator counters after the mutation or replay.
    pub snapshot: CatalogSnapshot,
    /// Exact active-catalog result for an offer operation.
    pub catalog: Option<CatalogMutation>,
}

/// Exact policy evidence and the coordinator snapshot that carried it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolicyStateReceipt {
    /// Exact census and active-catalog evidence.
    pub state: PolicyState,
    /// Coordinator snapshot observed with the evidence.
    pub snapshot: CatalogSnapshot,
}

/// Synchronous population state and the coordinator snapshot that carried it.
#[derive(Debug, Clone, PartialEq)]
pub struct PopulationEpochReceipt {
    /// Pending barrier evidence or a complete parent plan.
    pub state: PopulationEpochState,
    /// Coordinator snapshot observed with the evidence.
    pub snapshot: CatalogSnapshot,
}

/// Result of a coordinator read when local execution remains available.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogAccess {
    /// The coordinator returned a current snapshot.
    Remote(CatalogSnapshot),
    /// The coordinator was unavailable within the configured deadline.
    LocalFallback,
}

/// Observable client-side events that affect cooperative execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogClientEvent {
    /// A coordinator operation failed and execution continued locally.
    LocalFallback {
        /// Replica event sequence associated with the failed operation.
        event_sequence: u64,
    },
}

/// Invalid cooperative synchronization schedule or slice charge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SyncScheduleError {
    /// Synchronization bounds must be nonzero.
    #[error("synchronization bounds must be nonzero")]
    ZeroBound,
    /// One slice exceeded the declared maximum catalog calls.
    #[error("slice charged {charged} catalog calls, maximum is {maximum}")]
    SliceChargeExceeded {
        /// Catalog calls charged to the completed slice.
        charged: u64,
        /// Declared maximum catalog calls per slice.
        maximum: u64,
    },
    /// The staleness bound cannot be represented as a `u64`.
    #[error("synchronization staleness bound overflowed")]
    BoundOverflow,
}

/// Counter-based synchronization schedule with a declared staleness bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyncSchedule {
    interval_slices: u64,
    maximum_calls_per_slice: u64,
    slices_since_sync: u64,
}

impl SyncSchedule {
    /// Construct a schedule from a slice interval and per-slice call bound.
    pub fn new(
        interval_slices: u64,
        maximum_calls_per_slice: u64,
    ) -> Result<Self, SyncScheduleError> {
        if interval_slices == 0 || maximum_calls_per_slice == 0 {
            return Err(SyncScheduleError::ZeroBound);
        }
        interval_slices
            .checked_mul(maximum_calls_per_slice)
            .ok_or(SyncScheduleError::BoundOverflow)?;
        Ok(Self {
            interval_slices,
            maximum_calls_per_slice,
            slices_since_sync: 0,
        })
    }

    /// Maximum catalog calls a replica can make between synchronizations.
    pub fn maximum_staleness_calls(&self) -> u64 {
        self.interval_slices * self.maximum_calls_per_slice
    }

    /// Charge one completed slice and report whether synchronization is due.
    pub fn record_slice(&mut self, charged_calls: u64) -> Result<bool, SyncScheduleError> {
        if charged_calls > self.maximum_calls_per_slice {
            return Err(SyncScheduleError::SliceChargeExceeded {
                charged: charged_calls,
                maximum: self.maximum_calls_per_slice,
            });
        }
        self.slices_since_sync = self.slices_since_sync.saturating_add(1);
        Ok(self.slices_since_sync >= self.interval_slices)
    }

    /// Reset the staleness counter after a successful synchronization.
    pub fn synchronized(&mut self) {
        self.slices_since_sync = 0;
    }
}

/// Persistent client bound to one replica identity.
pub struct CatalogClient {
    stream: TcpStream,
    addr: SocketAddr,
    config: ClientConfig,
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
        let stream = Self::open_stream(addr, config)?;
        Ok(Self {
            stream,
            addr,
            config,
            identity,
            snapshot_version: 0,
            requests: BTreeMap::new(),
        })
    }

    fn open_stream(
        addr: SocketAddr,
        config: ClientConfig,
    ) -> Result<TcpStream, CatalogClientError> {
        let stream = TcpStream::connect_timeout(&addr, config.connect_timeout)?;
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(config.io_timeout))?;
        stream.set_write_timeout(Some(config.io_timeout))?;
        Ok(stream)
    }

    fn reconnect(&mut self) -> Result<(), CatalogClientError> {
        self.stream = Self::open_stream(self.addr, self.config)?;
        Ok(())
    }

    /// Highest event sequence this client has already sent.
    pub fn last_event_sequence(&self) -> u64 {
        self.requests.keys().copied().max().unwrap_or(0)
    }

    /// Read the current coordinator snapshot.
    pub fn snapshot(&mut self, event_sequence: u64) -> Result<CatalogSnapshot, CatalogClientError> {
        Ok(self
            .call(event_sequence, CatalogOperation::Snapshot)?
            .snapshot)
    }

    /// Read a snapshot or record an explicit local-fallback event.
    pub fn snapshot_or_fallback(
        &mut self,
        event_sequence: u64,
        events: &mut Vec<CatalogClientEvent>,
    ) -> CatalogAccess {
        match self.snapshot(event_sequence) {
            Ok(snapshot) => CatalogAccess::Remote(snapshot),
            Err(_) => {
                events.push(CatalogClientEvent::LocalFallback { event_sequence });
                CatalogAccess::LocalFallback
            }
        }
    }

    /// Record one exact census observation.
    pub fn record_visit(
        &mut self,
        event_sequence: u64,
        candidate: CatalogCandidate,
    ) -> Result<MutationReceipt, CatalogClientError> {
        let reply = self.call(event_sequence, CatalogOperation::RecordVisit { candidate })?;
        Ok(MutationReceipt {
            version: reply.snapshot.version,
            duplicate: reply.duplicate,
            snapshot: reply.snapshot,
            catalog: None,
        })
    }

    /// Validate, observe, and offer one candidate to the active catalog.
    pub fn offer_candidate(
        &mut self,
        event_sequence: u64,
        candidate: CatalogCandidate,
    ) -> Result<MutationReceipt, CatalogClientError> {
        let reply = self.call(
            event_sequence,
            CatalogOperation::OfferCandidate { candidate },
        )?;
        let catalog = match reply.payload {
            AcceptedPayload::CatalogMutation(mutation) => Some(mutation),
            AcceptedPayload::None => None,
            AcceptedPayload::BoundaryCrossing(_)
            | AcceptedPayload::Candidate(_)
            | AcceptedPayload::DescriptorHole(_)
            | AcceptedPayload::CoordinatorStatus(_)
            | AcceptedPayload::BridgeAssignment(_)
            | AcceptedPayload::PolicyState(_)
            | AcceptedPayload::PopulationEpoch(_)
            | AcceptedPayload::FrontierPost(_)
            | AcceptedPayload::RideWork(_)
            | AcceptedPayload::RideCredit(_)
            | AcceptedPayload::Roster(_) => {
                return Err(ProtocolError::Malformed(
                    "catalog offer returned an incompatible payload".into(),
                )
                .into());
            }
        };
        Ok(MutationReceipt {
            version: reply.snapshot.version,
            duplicate: reply.duplicate,
            snapshot: reply.snapshot,
            catalog,
        })
    }

    /// Record one action-conditioned transition from the registered live basin.
    pub fn record_transition(
        &mut self,
        event_sequence: u64,
        action: impl Into<String>,
        destination: TransitionDestination,
        adopted: bool,
    ) -> Result<MutationReceipt, CatalogClientError> {
        let reply = self.call(
            event_sequence,
            CatalogOperation::RecordTransition {
                action: action.into(),
                destination,
                adopted,
            },
        )?;
        Ok(MutationReceipt {
            version: reply.snapshot.version,
            duplicate: reply.duplicate,
            snapshot: reply.snapshot,
            catalog: None,
        })
    }

    /// Claim one exclusive same-system transition experiment.
    pub fn claim_ride(
        &mut self,
        event_sequence: u64,
        seed: u64,
    ) -> Result<Option<CatalogRideWork>, CatalogClientError> {
        let reply = self.call(event_sequence, CatalogOperation::ClaimRide { seed })?;
        match reply.payload {
            AcceptedPayload::RideWork(work) => Ok(Some(work)),
            AcceptedPayload::None => Ok(None),
            _ => Err(ProtocolError::Malformed(
                "ride claim returned an incompatible payload".into(),
            )
            .into()),
        }
    }

    /// Share the charged result of one transition experiment.
    pub fn report_ride(
        &mut self,
        event_sequence: u64,
        report: CatalogRideReport,
    ) -> Result<crate::ride_ledger::RideCredit, CatalogClientError> {
        let reply = self.call(event_sequence, CatalogOperation::ReportRide { report })?;
        match reply.payload {
            AcceptedPayload::RideCredit(credit) => Ok(credit),
            _ => Err(ProtocolError::Malformed(
                "ride report returned an incompatible payload".into(),
            )
            .into()),
        }
    }

    /// Submit one exact replay-safe charged-work boundary.
    pub fn record_ledger_event(
        &mut self,
        event_sequence: u64,
        kind: ChargeKind,
        charged_calls: u64,
        cumulative_charged: u64,
    ) -> Result<MutationReceipt, CatalogClientError> {
        let reply = self.call(
            event_sequence,
            CatalogOperation::LedgerEvent {
                kind: kind.wire_code(),
                charged_calls,
                cumulative_charged,
            },
        )?;
        Ok(MutationReceipt {
            version: reply.snapshot.version,
            duplicate: reply.duplicate,
            snapshot: reply.snapshot,
            catalog: None,
        })
    }

    /// Submit consecutive exact charged-work boundaries in one request.
    pub fn record_ledger_batch(
        &mut self,
        event_sequence: u64,
        events: Vec<CatalogLedgerEvent>,
    ) -> Result<MutationReceipt, CatalogClientError> {
        let reply = self.call(event_sequence, CatalogOperation::LedgerBatch { events })?;
        Ok(MutationReceipt {
            version: reply.snapshot.version,
            duplicate: reply.duplicate,
            snapshot: reply.snapshot,
            catalog: None,
        })
    }

    /// Draw one validated active-catalog candidate with an explicit seed.
    pub fn sample_candidate(
        &mut self,
        event_sequence: u64,
        draw: u64,
    ) -> Result<Option<CatalogCandidate>, CatalogClientError> {
        match self
            .call(event_sequence, CatalogOperation::Sample { draw })?
            .payload
        {
            AcceptedPayload::Candidate(candidate) => Ok(Some(candidate)),
            AcceptedPayload::None => Ok(None),
            AcceptedPayload::DescriptorHole(_)
            | AcceptedPayload::BoundaryCrossing(_)
            | AcceptedPayload::CoordinatorStatus(_)
            | AcceptedPayload::BridgeAssignment(_)
            | AcceptedPayload::PolicyState(_)
            | AcceptedPayload::PopulationEpoch(_)
            | AcceptedPayload::CatalogMutation(_)
            | AcceptedPayload::FrontierPost(_)
            | AcceptedPayload::RideWork(_)
            | AcceptedPayload::RideCredit(_)
            | AcceptedPayload::Roster(_) => Err(ProtocolError::Malformed(
                "sample returned an incompatible payload".into(),
            )
            .into()),
        }
    }

    /// Retrieve the validated representative of one immutable census basin.
    pub fn sample_basin(
        &mut self,
        event_sequence: u64,
        basin: u64,
    ) -> Result<Option<CatalogCandidate>, CatalogClientError> {
        match self
            .call(event_sequence, CatalogOperation::SampleBasin { basin })?
            .payload
        {
            AcceptedPayload::Candidate(candidate) => Ok(Some(candidate)),
            AcceptedPayload::None => Ok(None),
            AcceptedPayload::DescriptorHole(_)
            | AcceptedPayload::BoundaryCrossing(_)
            | AcceptedPayload::CoordinatorStatus(_)
            | AcceptedPayload::BridgeAssignment(_)
            | AcceptedPayload::PolicyState(_)
            | AcceptedPayload::PopulationEpoch(_)
            | AcceptedPayload::CatalogMutation(_)
            | AcceptedPayload::FrontierPost(_)
            | AcceptedPayload::RideWork(_)
            | AcceptedPayload::RideCredit(_)
            | AcceptedPayload::Roster(_) => Err(ProtocolError::Malformed(
                "basin sample returned an incompatible payload".into(),
            )
            .into()),
        }
    }

    /// Post one raw frontier excursion state to the shared ladder.
    pub fn post_frontier(
        &mut self,
        event_sequence: u64,
        post: CatalogFrontierPost,
    ) -> Result<(), CatalogClientError> {
        match self
            .call(event_sequence, CatalogOperation::PostFrontier { post })?
            .payload
        {
            AcceptedPayload::None => Ok(()),
            _ => Err(ProtocolError::Malformed(
                "frontier post returned an incompatible payload".into(),
            )
            .into()),
        }
    }

    /// Draw one shared frontier post, if the ladder holds any.
    pub fn draw_frontier(
        &mut self,
        event_sequence: u64,
        draw: u64,
    ) -> Result<Option<CatalogFrontierPost>, CatalogClientError> {
        match self
            .call(event_sequence, CatalogOperation::DrawFrontier { draw })?
            .payload
        {
            AcceptedPayload::FrontierPost(post) => Ok(Some(post)),
            AcceptedPayload::None => Ok(None),
            _ => Err(ProtocolError::Malformed(
                "frontier draw returned an incompatible payload".into(),
            )
            .into()),
        }
    }

    /// Request one seeded target-free descriptor-hole proposal.
    pub fn descriptor_hole(
        &mut self,
        event_sequence: u64,
        current: Vec<f64>,
        samples: u32,
        draw: u64,
    ) -> Result<DescriptorHoleProposal, CatalogClientError> {
        match self
            .call(
                event_sequence,
                CatalogOperation::DescriptorHole {
                    current,
                    samples,
                    draw,
                },
            )?
            .payload
        {
            AcceptedPayload::DescriptorHole(hole) => Ok(hole),
            AcceptedPayload::None
            | AcceptedPayload::Candidate(_)
            | AcceptedPayload::BoundaryCrossing(_)
            | AcceptedPayload::CoordinatorStatus(_)
            | AcceptedPayload::BridgeAssignment(_)
            | AcceptedPayload::PolicyState(_)
            | AcceptedPayload::PopulationEpoch(_)
            | AcceptedPayload::CatalogMutation(_)
            | AcceptedPayload::FrontierPost(_)
            | AcceptedPayload::RideWork(_)
            | AcceptedPayload::RideCredit(_)
            | AcceptedPayload::Roster(_) => Err(ProtocolError::Malformed(
                "descriptor-hole request returned an incompatible payload".into(),
            )
            .into()),
        }
    }

    /// Sample one observed adopted crossing from the query attraction region.
    pub fn boundary_crossing(
        &mut self,
        event_sequence: u64,
        current: Vec<f64>,
        draw: u64,
    ) -> Result<Option<BoundaryCrossingRecord>, CatalogClientError> {
        match self
            .call(
                event_sequence,
                CatalogOperation::BoundaryCrossing { current, draw },
            )?
            .payload
        {
            AcceptedPayload::BoundaryCrossing(crossing) => Ok(Some(crossing)),
            AcceptedPayload::None => Ok(None),
            AcceptedPayload::Candidate(_)
            | AcceptedPayload::DescriptorHole(_)
            | AcceptedPayload::CoordinatorStatus(_)
            | AcceptedPayload::BridgeAssignment(_)
            | AcceptedPayload::PolicyState(_)
            | AcceptedPayload::PopulationEpoch(_)
            | AcceptedPayload::CatalogMutation(_)
            | AcceptedPayload::FrontierPost(_)
            | AcceptedPayload::RideWork(_)
            | AcceptedPayload::RideCredit(_)
            | AcceptedPayload::Roster(_) => Err(ProtocolError::Malformed(
                "boundary-crossing request returned an incompatible payload".into(),
            )
            .into()),
        }
    }

    /// Read exact census and active-catalog evidence for one candidate.
    pub fn policy_state(
        &mut self,
        event_sequence: u64,
        descriptor: Vec<f64>,
        energy: f64,
    ) -> Result<PolicyState, CatalogClientError> {
        Ok(self
            .policy_state_with_snapshot(event_sequence, descriptor, energy)?
            .state)
    }

    /// Read exact policy evidence together with its coordinator snapshot.
    pub fn policy_state_with_snapshot(
        &mut self,
        event_sequence: u64,
        descriptor: Vec<f64>,
        energy: f64,
    ) -> Result<PolicyStateReceipt, CatalogClientError> {
        self.policy_state_with_lambda(event_sequence, descriptor, energy, 0.0)
    }

    /// Policy evidence with the replica's leftover-SOAP \(\lambda\).
    pub fn policy_state_with_lambda(
        &mut self,
        event_sequence: u64,
        descriptor: Vec<f64>,
        energy: f64,
        leftover_lambda: f64,
    ) -> Result<PolicyStateReceipt, CatalogClientError> {
        let reply = self.call(
            event_sequence,
            CatalogOperation::PolicyState {
                descriptor,
                energy,
                leftover_lambda,
            },
        )?;
        match reply.payload {
            AcceptedPayload::PolicyState(state) => Ok(PolicyStateReceipt {
                state,
                snapshot: reply.snapshot,
            }),
            AcceptedPayload::None
            | AcceptedPayload::Candidate(_)
            | AcceptedPayload::DescriptorHole(_)
            | AcceptedPayload::BoundaryCrossing(_)
            | AcceptedPayload::CoordinatorStatus(_)
            | AcceptedPayload::BridgeAssignment(_)
            | AcceptedPayload::PopulationEpoch(_)
            | AcceptedPayload::CatalogMutation(_)
            | AcceptedPayload::FrontierPost(_)
            | AcceptedPayload::RideWork(_)
            | AcceptedPayload::RideCredit(_)
            | AcceptedPayload::Roster(_) => Err(ProtocolError::Malformed(
                "policy-state request returned an incompatible payload".into(),
            )
            .into()),
        }
    }

    /// Submit one validated representative to a synchronous population epoch.
    pub fn submit_population(
        &mut self,
        event_sequence: u64,
        epoch: u64,
        candidate: CatalogCandidate,
    ) -> Result<PopulationEpochState, CatalogClientError> {
        Ok(self
            .submit_population_with_snapshot(event_sequence, epoch, candidate)?
            .state)
    }

    /// Submit population evidence and retain its coordinator snapshot.
    pub fn submit_population_with_snapshot(
        &mut self,
        event_sequence: u64,
        epoch: u64,
        candidate: CatalogCandidate,
    ) -> Result<PopulationEpochReceipt, CatalogClientError> {
        let reply = self.call(
            event_sequence,
            CatalogOperation::PopulationSubmit { epoch, candidate },
        )?;
        Ok(PopulationEpochReceipt {
            state: population_epoch_payload(reply.payload, "population submission")?,
            snapshot: reply.snapshot,
        })
    }

    /// Poll a synchronous population epoch without resubmitting evidence.
    pub fn population_plan(
        &mut self,
        event_sequence: u64,
        epoch: u64,
    ) -> Result<PopulationEpochState, CatalogClientError> {
        Ok(self
            .population_plan_with_snapshot(event_sequence, epoch)?
            .state)
    }

    /// Decline to submit to one epoch and retain its coordinator snapshot.
    ///
    /// Called when the barrier arrives and the replica's own state yields no
    /// validated representative, so that the replicas already waiting are
    /// released instead of polling until their budgets drain.
    pub fn population_abstain_with_snapshot(
        &mut self,
        event_sequence: u64,
        epoch: u64,
    ) -> Result<PopulationEpochReceipt, CatalogClientError> {
        let reply = self.call(
            event_sequence,
            CatalogOperation::PopulationAbstain { epoch },
        )?;
        Ok(PopulationEpochReceipt {
            state: population_epoch_payload(reply.payload, "population abstain")?,
            snapshot: reply.snapshot,
        })
    }

    /// Join an epoch by reference and retain its coordinator snapshot.
    ///
    /// The coordinator forms the member from the replica's best candidate it
    /// has already validated, so no state crosses the wire at barrier time
    /// and no re-validation is charged. Rejected when nothing is on file,
    /// which the caller answers by abstaining.
    pub fn population_join_with_snapshot(
        &mut self,
        event_sequence: u64,
        epoch: u64,
    ) -> Result<PopulationEpochReceipt, CatalogClientError> {
        let reply = self.call(event_sequence, CatalogOperation::PopulationJoin { epoch })?;
        Ok(PopulationEpochReceipt {
            state: population_epoch_payload(reply.payload, "population join")?,
            snapshot: reply.snapshot,
        })
    }

    /// Poll a population plan and retain its coordinator snapshot.
    pub fn population_plan_with_snapshot(
        &mut self,
        event_sequence: u64,
        epoch: u64,
    ) -> Result<PopulationEpochReceipt, CatalogClientError> {
        let reply = self.call(event_sequence, CatalogOperation::PopulationPlan { epoch })?;
        Ok(PopulationEpochReceipt {
            state: population_epoch_payload(reply.payload, "population plan")?,
            snapshot: reply.snapshot,
        })
    }

    /// Admit this client's replica onto the live roster.
    pub fn attach(&mut self, event_sequence: u64) -> Result<RosterReply, CatalogClientError> {
        roster_payload(
            self.call(event_sequence, CatalogOperation::Attach)?.payload,
            "attach",
        )
    }

    /// Retire this client's replica from the live roster.
    pub fn detach(
        &mut self,
        event_sequence: u64,
        reason: impl Into<String>,
    ) -> Result<RosterReply, CatalogClientError> {
        roster_payload(
            self.call(
                event_sequence,
                CatalogOperation::Detach {
                    reason: reason.into(),
                },
            )?
            .payload,
            "detach",
        )
    }

    /// Advance the coordinator clock by one tick of `millis` milliseconds.
    pub fn tick(
        &mut self,
        event_sequence: u64,
        millis: u64,
    ) -> Result<AcceptedPayload, CatalogClientError> {
        Ok(self
            .call(event_sequence, CatalogOperation::Tick { millis })?
            .payload)
    }

    /// Request a manual live-population target.
    pub fn scale(
        &mut self,
        event_sequence: u64,
        live_target: u32,
    ) -> Result<RosterReply, CatalogClientError> {
        roster_payload(
            self.call(
                event_sequence,
                CatalogOperation::Scale { live_target },
            )?
            .payload,
            "scale",
        )
    }

    /// Read-only aggregate status for an observer bound to the coordinator's
    /// campaign, ensemble, and system signature. The replica id is ignored.
    pub fn observer_status(
        &mut self,
        event_sequence: u64,
    ) -> Result<crate::catalog_rpc::CoordinatorStatus, CatalogClientError> {
        let reply = self.call(event_sequence, CatalogOperation::ObserverStatus)?;
        match reply.payload {
            AcceptedPayload::CoordinatorStatus(status) => Ok(status),
            _ => Err(CatalogClientError::Protocol(ProtocolError::Malformed(
                "observer status reply carried the wrong payload".to_owned(),
            ))),
        }
    }

    /// Poll for a bridge segment assignment. `None` when no bridge is
    /// commissioned; the draw selects among the region's stored entries.
    pub fn bridge_assignment(
        &mut self,
        event_sequence: u64,
        draw: u64,
    ) -> Result<Option<crate::catalog_rpc::BridgeAssignmentRecord>, CatalogClientError> {
        let reply = self.call(event_sequence, CatalogOperation::BridgeAssignment { draw })?;
        match reply.payload {
            AcceptedPayload::BridgeAssignment(assignment) => Ok(Some(assignment)),
            AcceptedPayload::None => Ok(None),
            _ => Err(CatalogClientError::Protocol(ProtocolError::Malformed(
                "bridge assignment reply carried the wrong payload".to_owned(),
            ))),
        }
    }

    /// Report one attempted exit from a bridge region.
    pub fn bridge_crossing(
        &mut self,
        event_sequence: u64,
        crossing: crate::catalog_rpc::BridgeCrossingRecord,
    ) -> Result<(), CatalogClientError> {
        self.call(
            event_sequence,
            CatalogOperation::BridgeCrossing { crossing },
        )?;
        Ok(())
    }

    /// The framed Cap'n Proto reply to a status query, byte-exact as the
    /// coordinator sent it, validated as a status before it is handed on.
    pub fn observer_status_frame(
        &mut self,
        event_sequence: u64,
    ) -> Result<Vec<u8>, CatalogClientError> {
        let request = CatalogRequest {
            protocol_version: PROTOCOL_VERSION,
            identity: self.identity.clone(),
            event_sequence,
            snapshot_version: self.snapshot_version,
            operation: CatalogOperation::ObserverStatus,
        };
        self.stream.write_all(&encode_request(&request)?)?;
        self.stream.flush()?;
        let message = serialize::read_message(&mut self.stream, ReaderOptions::new())
            .map_err(|error| ProtocolError::Malformed(error.to_string()))?;
        {
            let root = message
                .get_root::<catalog_reply::Reader>()
                .map_err(|error| ProtocolError::Malformed(error.to_string()))?;
            match decode_reply_reader(root)? {
                CatalogReply::Accepted(reply)
                    if matches!(reply.payload, AcceptedPayload::CoordinatorStatus(_)) => {}
                CatalogReply::Accepted(_) => {
                    return Err(CatalogClientError::Protocol(ProtocolError::Malformed(
                        "observer status reply carried the wrong payload".to_owned(),
                    )));
                }
                CatalogReply::Rejected { reason, .. } => {
                    return Err(CatalogClientError::Rejected(reason));
                }
            }
        }
        let mut frame = Vec::new();
        capnp::serialize::write_message_segments(&mut frame, &message.into_segments())
            .map_err(|error| ProtocolError::Malformed(error.to_string()))?;
        Ok(frame)
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
        match self.exchange(&request) {
            Ok(reply) => return Ok(reply),
            Err(error @ CatalogClientError::Rejected(_)) => return Err(error),
            Err(_) => self.reconnect()?,
        }
        self.exchange(&request)
    }

    fn exchange(&mut self, request: &CatalogRequest) -> Result<AcceptedReply, CatalogClientError> {
        self.stream.write_all(&encode_request(request)?)?;
        self.stream.flush()?;
        let message = serialize::read_message(&mut self.stream, ReaderOptions::new())
            .map_err(|error| ProtocolError::Malformed(error.to_string()))?;
        let root = message
            .get_root::<catalog_reply::Reader>()
            .map_err(|error| ProtocolError::Malformed(error.to_string()))?;
        match decode_reply_reader(root)? {
            CatalogReply::Accepted(reply) if reply.event_sequence == request.event_sequence => {
                self.snapshot_version = self.snapshot_version.max(reply.snapshot.version);
                Ok(reply)
            }
            CatalogReply::Rejected {
                event_sequence,
                reason,
                ..
            } if event_sequence == request.event_sequence => {
                Err(CatalogClientError::Rejected(reason))
            }
            CatalogReply::Accepted(_) | CatalogReply::Rejected { .. } => {
                Err(ProtocolError::Malformed(
                    "catalog reply sequence does not match the request".into(),
                )
                .into())
            }
        }
    }
}

fn population_epoch_payload(
    payload: AcceptedPayload,
    operation: &str,
) -> Result<PopulationEpochState, CatalogClientError> {
    match payload {
        AcceptedPayload::PopulationEpoch(state) => Ok(state),
        AcceptedPayload::None
        | AcceptedPayload::Candidate(_)
        | AcceptedPayload::DescriptorHole(_)
        | AcceptedPayload::BoundaryCrossing(_)
        | AcceptedPayload::CoordinatorStatus(_)
        | AcceptedPayload::BridgeAssignment(_)
        | AcceptedPayload::PolicyState(_)
        | AcceptedPayload::CatalogMutation(_)
        | AcceptedPayload::FrontierPost(_)
        | AcceptedPayload::RideWork(_)
        | AcceptedPayload::RideCredit(_)
            | AcceptedPayload::Roster(_) => Err(ProtocolError::Malformed(format!(
            "{operation} returned an incompatible payload"
        ))
        .into()),
    }
}

fn roster_payload(
    payload: AcceptedPayload,
    operation: &str,
) -> Result<RosterReply, CatalogClientError> {
    match payload {
        AcceptedPayload::Roster(roster) => Ok(roster),
        _ => Err(ProtocolError::Malformed(format!(
            "{operation} returned an incompatible payload"
        ))
        .into()),
    }
}
