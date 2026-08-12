//! Versioned, ensemble-isolated cooperative catalog wire types.

use std::io::Cursor;

use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;

use crate::Catalog_capnp::{
    QuenchStatus as WireQuenchStatus, RejectionKind, accepted_reply, candidate_record,
    catalog_reply, catalog_request,
};

pub mod client;
pub mod server;

/// Wire protocol version accepted by this release.
pub const PROTOCOL_VERSION: u16 = 2;

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
    /// Submit one replay-safe charged-work event.
    LedgerEvent {
        /// Stable charge-kind discriminant.
        kind: u16,
        /// Potential calls charged by this event.
        charged_calls: u64,
        /// Replica counter including this event.
        cumulative_charged: u64,
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

/// Optional scientific payload returned by an accepted operation.
#[derive(Debug, Clone, PartialEq)]
pub enum AcceptedPayload {
    /// Snapshot or mutation response without an additional scientific record.
    None,
    /// Sampled validated catalog candidate.
    Candidate(CatalogCandidate),
    /// Seeded target-free descriptor-hole proposal.
    DescriptorHole(DescriptorHoleProposal),
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
        catalog_request::operation::LedgerEvent(ledger) => {
            let ledger = ledger.map_err(wire_error)?;
            CatalogOperation::LedgerEvent {
                kind: ledger.get_kind(),
                charged_calls: ledger.get_charged_calls(),
                cumulative_charged: ledger.get_cumulative_charged(),
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
            };
            Ok(CatalogReply::Accepted(AcceptedReply {
                event_sequence,
                duplicate: body.get_duplicate(),
                snapshot: CatalogSnapshot {
                    version: snapshot_version,
                    census_visits: body.get_census_visits(),
                    active_entries: body.get_active_entries(),
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
    })
}

fn list_f64(input: capnp::primitive_list::Reader<'_, f64>) -> Vec<f64> {
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
