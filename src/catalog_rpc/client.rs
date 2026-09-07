//! Timeout-bounded client for one isolated catalog coordinator.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use capnp::capability::Promise;
use capnp_rpc::RpcSystem;
use capnp_rpc::pry;
use capnp_rpc::rpc_twoparty_capnp::Side;
use capnp_rpc::twoparty::VatNetwork;
use futures::AsyncReadExt;
use tokio_util::compat::TokioAsyncReadCompatExt;

use super::{
    AcceptedPayload, AcceptedReply, BoundaryCrossingRecord, CatalogCandidate, CatalogFrontierPost,
    CatalogIdentity, CatalogLedgerEvent, CatalogMutation, CatalogOperation, CatalogReply,
    CatalogRequest, CatalogRideReport, CatalogRideWork, CatalogSnapshot, CoordinatorEvent,
    CoordinatorStatus, DescriptorHoleProposal, PROTOCOL_VERSION, PolicyState, PopulationEpochState,
    ProtocolError, ProtocolRejection, RosterReply, TransitionDestination, decode_reply_reader,
    encode_reply, fill_identity, fill_request, read_coordinator_status, read_event, read_roster,
};
use crate::Catalog_capnp::{coordinator, session, subscriber};
use crate::cooperative_search::ledger::ChargeKind;
use crate::coreclass::CoreVerdict;

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

enum ClientJob {
    Observe {
        reply: std::sync::mpsc::Sender<Result<CoordinatorStatus, CatalogClientError>>,
    },
    Call {
        request: CatalogRequest,
        reply: std::sync::mpsc::Sender<Result<AcceptedReply, CatalogClientError>>,
    },
    CallRaw {
        version: u16,
        digest: Vec<u8>,
        sequence: u64,
        identity: CatalogIdentity,
        reply: std::sync::mpsc::Sender<Result<CatalogReply, CatalogClientError>>,
    },
    Shutdown,
}

struct EventInbox {
    events: Arc<Mutex<Vec<CoordinatorEvent>>>,
}

impl subscriber::Server for EventInbox {
    fn event(
        &mut self,
        params: subscriber::EventParams,
        _results: subscriber::EventResults,
    ) -> Promise<(), capnp::Error> {
        let event = pry!(
            read_event(pry!(pry!(params.get()).get_event()))
                .map_err(|error| capnp::Error::failed(error.to_string()))
        );
        self.events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(event);
        Promise::ok(())
    }
}

/// Persistent client bound to one replica identity.
pub struct CatalogClient {
    jobs: Option<tokio::sync::mpsc::UnboundedSender<ClientJob>>,
    thread: Option<JoinHandle<()>>,
    identity: CatalogIdentity,
    snapshot_version: Arc<Mutex<u64>>,
    requests: BTreeMap<u64, CatalogRequest>,
    events: Arc<Mutex<Vec<CoordinatorEvent>>>,
}

impl CatalogClient {
    /// Connect to a coordinator with explicit deadlines.
    pub fn connect(
        addr: SocketAddr,
        identity: CatalogIdentity,
        config: ClientConfig,
    ) -> Result<Self, CatalogClientError> {
        let (jobs, rx) = tokio::sync::mpsc::unbounded_channel();
        let events = Arc::new(Mutex::new(Vec::new()));
        let snapshot_version = Arc::new(Mutex::new(0));
        let thread_events = Arc::clone(&events);
        let thread_identity = identity.clone();
        let thread_snapshots = Arc::clone(&snapshot_version);
        let thread = thread::Builder::new()
            .name("catalog-rpc-client".to_owned())
            .spawn(move || {
                run_client_executor(
                    addr,
                    config,
                    thread_identity,
                    thread_events,
                    thread_snapshots,
                    rx,
                );
            })
            .expect("catalog RPC client thread starts");
        Ok(Self {
            jobs: Some(jobs),
            thread: Some(thread),
            identity,
            snapshot_version,
            requests: BTreeMap::new(),
            events,
        })
    }

    /// Read coordinator status without presenting an identity.
    pub fn observe(&mut self) -> Result<CoordinatorStatus, CatalogClientError> {
        self.post(|reply| ClientJob::Observe { reply })
    }

    /// Drain queued coordinator events.
    pub fn events(&mut self) -> Vec<CoordinatorEvent> {
        self.events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .drain(..)
            .collect()
    }

    /// Send one catalog request on the bound session.
    pub fn session_call(
        &mut self,
        request: CatalogRequest,
    ) -> Result<CatalogReply, CatalogClientError> {
        match self.dispatch(request.clone()) {
            Ok(accepted) => Ok(CatalogReply::Accepted(accepted)),
            Err(CatalogClientError::Rejected(reason)) => Ok(CatalogReply::Rejected {
                event_sequence: request.event_sequence,
                snapshot_version: *self
                    .snapshot_version
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
                reason,
            }),
            Err(error) => Err(error),
        }
    }

    /// Send a session call whose identity digest is not necessarily 32 bytes.
    pub fn session_call_digest(
        &mut self,
        version: u16,
        digest: &[u8],
        sequence: u64,
    ) -> Result<CatalogReply, CatalogClientError> {
        self.post(|reply| ClientJob::CallRaw {
            version,
            digest: digest.to_vec(),
            sequence,
            identity: self.identity.clone(),
            reply,
        })
    }

    fn post<T, F>(&self, job: F) -> Result<T, CatalogClientError>
    where
        T: Send + 'static,
        F: FnOnce(std::sync::mpsc::Sender<Result<T, CatalogClientError>>) -> ClientJob,
    {
        let (tx, rx) = std::sync::mpsc::channel();
        let Some(jobs) = self.jobs.as_ref() else {
            return Err(CatalogClientError::Transport(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "catalog client is shut down",
            )));
        };
        jobs.send(job(tx)).map_err(|_| {
            CatalogClientError::Transport(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "catalog RPC executor stopped",
            ))
        })?;
        rx.recv().map_err(|_| {
            CatalogClientError::Transport(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "catalog RPC executor stopped",
            ))
        })?
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
            | AcceptedPayload::Roster(_)
            | AcceptedPayload::SurfaceEvidence(_)
            | AcceptedPayload::CoreVerdict(_) => {
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

    /// Exchange cumulative local surface rewards for peer-only evidence.
    pub fn exchange_surface_evidence(
        &mut self,
        event_sequence: u64,
        report: crate::surface_evidence::SurfaceReport,
    ) -> Result<crate::surface_evidence::SurfaceReport, CatalogClientError> {
        match self.call(event_sequence, CatalogOperation::ExchangeSurfaceEvidence { report })?.payload {
            AcceptedPayload::SurfaceEvidence(report) => Ok(report),
            _ => Err(ProtocolError::Malformed("surface exchange returned an incompatible payload".into()).into()),
        }
    }

    /// Report one chain's motif class and energy to the shared table.
    pub fn report_core_class(
        &mut self,
        event_sequence: u64,
        class: u8,
        energy: f64,
        charged: u64,
    ) -> Result<CoreVerdict, CatalogClientError> {
        match self
            .call(
                event_sequence,
                CatalogOperation::ReportCoreClass {
                    class,
                    energy,
                    charged,
                },
            )?
            .payload
        {
            AcceptedPayload::CoreVerdict(verdict) => Ok(verdict),
            _ => Err(ProtocolError::Malformed(
                "core-class report returned an incompatible payload".into(),
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
            | AcceptedPayload::Roster(_)
            | AcceptedPayload::SurfaceEvidence(_)
            | AcceptedPayload::CoreVerdict(_) => Err(ProtocolError::Malformed(
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
            | AcceptedPayload::Roster(_)
            | AcceptedPayload::SurfaceEvidence(_)
            | AcceptedPayload::CoreVerdict(_) => Err(ProtocolError::Malformed(
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
            | AcceptedPayload::Roster(_)
            | AcceptedPayload::SurfaceEvidence(_)
            | AcceptedPayload::CoreVerdict(_) => Err(ProtocolError::Malformed(
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
            | AcceptedPayload::Roster(_)
            | AcceptedPayload::SurfaceEvidence(_)
            | AcceptedPayload::CoreVerdict(_) => Err(ProtocolError::Malformed(
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
            | AcceptedPayload::Roster(_)
            | AcceptedPayload::SurfaceEvidence(_)
            | AcceptedPayload::CoreVerdict(_) => Err(ProtocolError::Malformed(
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
            self.call(event_sequence, CatalogOperation::Scale { live_target })?
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
        let status = self.observer_status(event_sequence)?;
        let reply = CatalogReply::Accepted(AcceptedReply {
            event_sequence,
            duplicate: true,
            snapshot: CatalogSnapshot {
                version: status.snapshot_version,
                census_visits: status.census_visits,
                active_entries: status.active_entries,
                aggregate_charged: status.aggregate_charged,
                aggregate_budget: status.aggregate_budget,
            },
            payload: AcceptedPayload::CoordinatorStatus(status),
        });
        encode_reply(reply).map_err(CatalogClientError::from)
    }

    fn call(
        &mut self,
        event_sequence: u64,
        operation: CatalogOperation,
    ) -> Result<AcceptedReply, CatalogClientError> {
        let snapshot_version = *self
            .snapshot_version
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let request = self
            .requests
            .entry(event_sequence)
            .or_insert_with(|| CatalogRequest {
                protocol_version: PROTOCOL_VERSION,
                identity: self.identity.clone(),
                event_sequence,
                snapshot_version,
                operation,
            })
            .clone();
        self.dispatch(request)
    }

    fn dispatch(&mut self, request: CatalogRequest) -> Result<AcceptedReply, CatalogClientError> {
        self.post(|reply| ClientJob::Call { request, reply })
    }
}

impl Drop for CatalogClient {
    fn drop(&mut self) {
        if let Some(jobs) = self.jobs.take() {
            let _ = jobs.send(ClientJob::Shutdown);
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct ClientSession {
    coordinator: coordinator::Client,
    session: Option<session::Client>,
    attached: Option<CatalogIdentity>,
    connected: bool,
    events: Arc<Mutex<Vec<CoordinatorEvent>>>,
    snapshot_version: Arc<Mutex<u64>>,
    addr: SocketAddr,
    config: ClientConfig,
}

fn run_client_executor(
    addr: SocketAddr,
    config: ClientConfig,
    _identity: CatalogIdentity,
    events: Arc<Mutex<Vec<CoordinatorEvent>>>,
    snapshot_version: Arc<Mutex<u64>>,
    mut jobs: tokio::sync::mpsc::UnboundedReceiver<ClientJob>,
) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("catalog client runtime starts");
    let local = tokio::task::LocalSet::new();
    local.block_on(&runtime, async move {
        let mut session = match open_rpc(addr, config, Arc::clone(&events)).await {
            Ok(mut opened) => {
                opened.snapshot_version = snapshot_version;
                opened
            }
            Err(_) => ClientSession {
                coordinator: capnp_rpc::new_client(UnreachableCoordinator),
                session: None,
                attached: None,
                connected: false,
                events,
                snapshot_version,
                addr,
                config,
            },
        };
        while let Some(job) = jobs.recv().await {
            if matches!(job, ClientJob::Shutdown) {
                break;
            }
            handle_job(&mut session, job).await;
        }
    });
}

struct UnreachableCoordinator;

impl coordinator::Server for UnreachableCoordinator {}

async fn handle_job(session: &mut ClientSession, job: ClientJob) {
    match job {
        ClientJob::Observe { reply } => {
            let _ = reply.send(observe_status(session).await);
        }
        ClientJob::Call { request, reply } => {
            let _ = reply.send(call_session(session, request).await);
        }
        ClientJob::CallRaw {
            version,
            digest,
            sequence,
            identity,
            reply,
        } => {
            let _ =
                reply.send(call_session_raw(session, version, digest, sequence, identity).await);
        }
        ClientJob::Shutdown => {}
    }
}

async fn open_rpc(
    addr: SocketAddr,
    config: ClientConfig,
    events: Arc<Mutex<Vec<CoordinatorEvent>>>,
) -> Result<ClientSession, CatalogClientError> {
    let stream = tokio::time::timeout(config.connect_timeout, tokio::net::TcpStream::connect(addr))
        .await
        .map_err(|_| {
            CatalogClientError::Transport(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "catalog connect timed out",
            ))
        })?
        .map_err(CatalogClientError::Transport)?;
    stream
        .set_nodelay(true)
        .map_err(CatalogClientError::Transport)?;
    let (reader, writer) = TokioAsyncReadCompatExt::compat(stream).split();
    let network = VatNetwork::new(
        futures::io::BufReader::new(reader),
        futures::io::BufWriter::new(writer),
        Side::Client,
        Default::default(),
    );
    let mut rpc = RpcSystem::new(Box::new(network), None);
    let coordinator: coordinator::Client = rpc.bootstrap(Side::Server);
    tokio::task::spawn_local(rpc);
    Ok(ClientSession {
        coordinator,
        session: None,
        attached: None,
        connected: true,
        events,
        snapshot_version: Arc::new(Mutex::new(0)),
        addr,
        config,
    })
}

async fn reconnect(session: &mut ClientSession) -> Result<(), CatalogClientError> {
    let events = Arc::clone(&session.events);
    let snapshot_version = Arc::clone(&session.snapshot_version);
    let addr = session.addr;
    let config = session.config;
    let attached = session.attached.clone();
    *session = open_rpc(addr, config, events).await?;
    session.snapshot_version = snapshot_version;
    session.attached = attached;
    session.connected = true;
    Ok(())
}

async fn attach_session(
    session: &mut ClientSession,
    identity: CatalogIdentity,
) -> Result<RosterReply, CatalogClientError> {
    if session.session.is_none()
        && let Err(error) = ensure_coordinator(session).await
    {
        reconnect(session).await?;
        ensure_coordinator(session).await.map_err(|_| error)?;
    }
    let subscriber: subscriber::Client = capnp_rpc::new_client(EventInbox {
        events: Arc::clone(&session.events),
    });
    let mut request = session.coordinator.attach_request();
    {
        let mut params = request.get();
        fill_identity(params.reborrow().init_identity(), &identity);
        params.set_subscriber(subscriber);
    }
    let response = tokio::time::timeout(session.config.io_timeout, request.send().promise)
        .await
        .map_err(|_| {
            CatalogClientError::Transport(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "catalog attach timed out",
            ))
        })?
        .map_err(|error| ProtocolError::Malformed(error.to_string()))?;
    let roster = read_roster(
        response
            .get()
            .map_err(|error| ProtocolError::Malformed(error.to_string()))?
            .get_roster()
            .map_err(|error| ProtocolError::Malformed(error.to_string()))?,
    )?;
    session.session = Some(
        response
            .get()
            .map_err(|error| ProtocolError::Malformed(error.to_string()))?
            .get_session()
            .map_err(|error| ProtocolError::Malformed(error.to_string()))?,
    );
    session.attached = Some(identity);
    Ok(roster)
}

async fn ensure_coordinator(session: &mut ClientSession) -> Result<(), CatalogClientError> {
    if session.connected {
        return Ok(());
    }
    let snapshot_version = Arc::clone(&session.snapshot_version);
    let attached = session.attached.clone();
    let opened = open_rpc(session.addr, session.config, Arc::clone(&session.events)).await?;
    session.coordinator = opened.coordinator;
    session.connected = true;
    session.snapshot_version = snapshot_version;
    session.attached = attached;
    Ok(())
}

async fn observe_status(
    session: &mut ClientSession,
) -> Result<CoordinatorStatus, CatalogClientError> {
    if let Err(error) = ensure_coordinator(session).await {
        reconnect(session).await?;
        ensure_coordinator(session).await.map_err(|_| error)?;
    }
    let request = session.coordinator.observe_request();
    let response = tokio::time::timeout(session.config.io_timeout, request.send().promise)
        .await
        .map_err(|_| {
            CatalogClientError::Transport(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "catalog observe timed out",
            ))
        })?
        .map_err(|error| ProtocolError::Malformed(error.to_string()))?;
    read_coordinator_status(
        response
            .get()
            .map_err(|error| ProtocolError::Malformed(error.to_string()))?
            .get_status()
            .map_err(|error| ProtocolError::Malformed(error.to_string()))?,
    )
    .map_err(CatalogClientError::from)
}

async fn call_session(
    session: &mut ClientSession,
    request: CatalogRequest,
) -> Result<AcceptedReply, CatalogClientError> {
    match call_session_once(session, &request).await {
        Ok(reply) => Ok(reply),
        Err(error @ CatalogClientError::Rejected(_)) => Err(error),
        Err(_) => {
            reconnect(session).await?;
            if let Some(identity) = session.attached.clone() {
                attach_session(session, identity).await?;
            } else {
                attach_session(session, request.identity.clone()).await?;
            }
            call_session_once(session, &request).await
        }
    }
}

async fn call_session_once(
    session: &mut ClientSession,
    request: &CatalogRequest,
) -> Result<AcceptedReply, CatalogClientError> {
    if session.session.is_none() {
        attach_session(session, request.identity.clone()).await?;
    }
    let bound = session.session.clone().ok_or_else(|| {
        CatalogClientError::Transport(std::io::Error::new(
            std::io::ErrorKind::NotConnected,
            "catalog session is not attached",
        ))
    })?;
    let mut rpc = bound.call_request();
    fill_request(rpc.get().init_request(), request).map_err(CatalogClientError::from)?;
    let response = tokio::time::timeout(session.config.io_timeout, rpc.send().promise)
        .await
        .map_err(|_| {
            CatalogClientError::Transport(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "catalog call timed out",
            ))
        })?
        .map_err(|error| ProtocolError::Malformed(error.to_string()))?;
    let reply = decode_reply_reader(
        response
            .get()
            .map_err(|error| ProtocolError::Malformed(error.to_string()))?
            .get_reply()
            .map_err(|error| ProtocolError::Malformed(error.to_string()))?,
    )?;
    match reply {
        CatalogReply::Accepted(accepted) if accepted.event_sequence == request.event_sequence => {
            let mut version = session
                .snapshot_version
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            *version = (*version).max(accepted.snapshot.version);
            Ok(accepted)
        }
        CatalogReply::Rejected {
            event_sequence,
            reason,
            ..
        } if event_sequence == request.event_sequence => Err(CatalogClientError::Rejected(reason)),
        CatalogReply::Accepted(_) | CatalogReply::Rejected { .. } => Err(ProtocolError::Malformed(
            "catalog reply sequence does not match the request".into(),
        )
        .into()),
    }
}

async fn call_session_raw(
    session: &mut ClientSession,
    version: u16,
    digest: Vec<u8>,
    sequence: u64,
    identity: CatalogIdentity,
) -> Result<CatalogReply, CatalogClientError> {
    if session.session.is_none() {
        attach_session(session, identity.clone()).await?;
    }
    let bound = session.session.clone().ok_or_else(|| {
        CatalogClientError::Transport(std::io::Error::new(
            std::io::ErrorKind::NotConnected,
            "catalog session is not attached",
        ))
    })?;
    let mut rpc = bound.call_request();
    {
        let mut root = rpc.get().init_request();
        root.set_protocol_version(version);
        root.set_event_sequence(sequence);
        root.set_snapshot_version(0);
        {
            let mut wire_identity = root.reborrow().init_identity();
            wire_identity.set_campaign(identity.campaign.as_str());
            wire_identity.set_ensemble(identity.ensemble.as_str());
            wire_identity.set_replica(identity.replica);
            wire_identity.set_signature_digest(&digest);
        }
        root.init_operation().set_snapshot(());
    }
    let response = tokio::time::timeout(session.config.io_timeout, rpc.send().promise)
        .await
        .map_err(|_| {
            CatalogClientError::Transport(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "catalog call timed out",
            ))
        })?
        .map_err(|error| ProtocolError::Malformed(error.to_string()))?;
    decode_reply_reader(
        response
            .get()
            .map_err(|error| ProtocolError::Malformed(error.to_string()))?
            .get_reply()
            .map_err(|error| ProtocolError::Malformed(error.to_string()))?,
    )
    .map_err(CatalogClientError::from)
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
        | AcceptedPayload::Roster(_)
        | AcceptedPayload::SurfaceEvidence(_)
        | AcceptedPayload::CoreVerdict(_) => Err(ProtocolError::Malformed(format!(
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
        _ => Err(
            ProtocolError::Malformed(format!("{operation} returned an incompatible payload"))
                .into(),
        ),
    }
}
