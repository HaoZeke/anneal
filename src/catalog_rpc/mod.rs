//! Versioned, ensemble-isolated cooperative catalog wire types.

use std::io::Cursor;

use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;

use crate::Catalog_capnp::catalog_request;

/// Wire protocol version accepted by this release.
pub const PROTOCOL_VERSION: u16 = 1;

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

/// Mutation or read operation carried by a catalog request.
#[derive(Debug, Clone, PartialEq)]
pub enum CatalogOperation {
    /// Read the current versioned snapshot.
    Snapshot,
    /// Record one exact fixed-census observation.
    RecordVisit {
        /// Assigned basin identity.
        basin_id: u64,
        /// Whether the observation opened the basin.
        created: bool,
        /// Normalized descriptor.
        descriptor: Vec<f64>,
    },
    /// Offer one validated candidate to the active catalog.
    OfferCandidate {
        /// Fixed-census basin identity.
        basin_id: u64,
        /// Validated energy.
        energy: f64,
        /// Cartesian coordinates.
        coordinates: Vec<f64>,
        /// Normalized descriptor.
        descriptor: Vec<f64>,
        /// Stable producer provenance.
        provenance: String,
    },
    /// Sample an admissible active representative.
    Sample {
        /// Explicit deterministic random draw.
        draw: u64,
    },
    /// Request a sampled farthest-hole proposal.
    DescriptorHole {
        /// Number of unit-sphere samples.
        samples: u32,
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
        CatalogOperation::RecordVisit {
            basin_id,
            created,
            descriptor,
        } => {
            let mut record = operation.init_record_visit();
            record.set_basin_id(*basin_id);
            record.set_created(*created);
            fill_f64(record.init_descriptor(descriptor.len() as u32), descriptor);
        }
        CatalogOperation::OfferCandidate {
            basin_id,
            energy,
            coordinates,
            descriptor,
            provenance,
        } => {
            let mut candidate = operation.init_offer_candidate();
            candidate.set_basin_id(*basin_id);
            candidate.set_energy(*energy);
            fill_f64(
                candidate
                    .reborrow()
                    .init_coordinates(coordinates.len() as u32),
                coordinates,
            );
            fill_f64(
                candidate
                    .reborrow()
                    .init_descriptor(descriptor.len() as u32),
                descriptor,
            );
            candidate.set_provenance(provenance.as_str());
        }
        CatalogOperation::Sample { draw } => operation.set_sample(*draw),
        CatalogOperation::DescriptorHole { samples } => operation.set_descriptor_hole(*samples),
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
        catalog_request::operation::RecordVisit(record) => {
            let record = record.map_err(wire_error)?;
            CatalogOperation::RecordVisit {
                basin_id: record.get_basin_id(),
                created: record.get_created(),
                descriptor: list_f64(record.get_descriptor().map_err(wire_error)?),
            }
        }
        catalog_request::operation::OfferCandidate(candidate) => {
            let candidate = candidate.map_err(wire_error)?;
            CatalogOperation::OfferCandidate {
                basin_id: candidate.get_basin_id(),
                energy: candidate.get_energy(),
                coordinates: list_f64(candidate.get_coordinates().map_err(wire_error)?),
                descriptor: list_f64(candidate.get_descriptor().map_err(wire_error)?),
                provenance: text_value(candidate.get_provenance().map_err(wire_error)?)?,
            }
        }
        catalog_request::operation::Sample(draw) => CatalogOperation::Sample { draw },
        catalog_request::operation::DescriptorHole(samples) => {
            CatalogOperation::DescriptorHole { samples }
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
