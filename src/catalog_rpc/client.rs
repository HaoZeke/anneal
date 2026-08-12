//! Timeout-bounded client for one isolated catalog coordinator.

use std::collections::BTreeMap;
use std::io::Write;
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

use capnp::message::ReaderOptions;
use capnp::serialize;

use super::{
    AcceptedPayload, AcceptedReply, CatalogCandidate, CatalogIdentity, CatalogOperation,
    CatalogReply, CatalogRequest, CatalogSnapshot, DescriptorHoleProposal, PROTOCOL_VERSION,
    PolicyState, PopulationEpochState, ProtocolError, ProtocolRejection, decode_reply_reader,
    encode_request,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MutationReceipt {
    /// Coordinator snapshot version after the mutation.
    pub version: u64,
    /// Whether the coordinator recognized an identical replay.
    pub duplicate: bool,
    /// Coordinator counters after the mutation or replay.
    pub snapshot: CatalogSnapshot,
}

/// Exact policy evidence and the coordinator snapshot that carried it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        let stream = TcpStream::connect_timeout(&addr, config.connect_timeout)?;
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(config.io_timeout))?;
        stream.set_write_timeout(Some(config.io_timeout))?;
        Ok(Self {
            stream,
            identity,
            snapshot_version: 0,
            requests: BTreeMap::new(),
        })
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
        Ok(MutationReceipt {
            version: reply.snapshot.version,
            duplicate: reply.duplicate,
            snapshot: reply.snapshot,
        })
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
            | AcceptedPayload::PolicyState(_)
            | AcceptedPayload::PopulationEpoch(_) => Err(ProtocolError::Malformed(
                "sample returned an incompatible payload".into(),
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
            | AcceptedPayload::PolicyState(_)
            | AcceptedPayload::PopulationEpoch(_) => Err(ProtocolError::Malformed(
                "descriptor-hole request returned an incompatible payload".into(),
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
        let reply = self.call(
            event_sequence,
            CatalogOperation::PolicyState { descriptor, energy },
        )?;
        match reply.payload {
            AcceptedPayload::PolicyState(state) => Ok(PolicyStateReceipt {
                state,
                snapshot: reply.snapshot,
            }),
            AcceptedPayload::None
            | AcceptedPayload::Candidate(_)
            | AcceptedPayload::DescriptorHole(_)
            | AcceptedPayload::PopulationEpoch(_) => Err(ProtocolError::Malformed(
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
        self.stream.write_all(&encode_request(&request)?)?;
        self.stream.flush()?;
        let message = serialize::read_message(&mut self.stream, ReaderOptions::new())
            .map_err(|error| ProtocolError::Malformed(error.to_string()))?;
        let root = message
            .get_root::<catalog_reply::Reader>()
            .map_err(|error| ProtocolError::Malformed(error.to_string()))?;
        match decode_reply_reader(root)? {
            CatalogReply::Accepted(reply) => {
                self.snapshot_version = self.snapshot_version.max(reply.snapshot.version);
                Ok(reply)
            }
            CatalogReply::Rejected { reason, .. } => Err(CatalogClientError::Rejected(reason)),
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
        | AcceptedPayload::PolicyState(_) => Err(ProtocolError::Malformed(format!(
            "{operation} returned an incompatible payload"
        ))
        .into()),
    }
}
