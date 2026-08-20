//! Versioned compatibility descriptors for engine-backed objective bridges.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use eindir_core::ffi::{EINDIR_ABI_FEATURE_BATCH, EINDIR_ABI_FEATURE_GRADIENT, eindir_objective_t};

/// Protocol family used by the anneal objective boundary.
pub const PROTOCOL_FAMILY: &str = "anneal.objective";

/// A major/minor protocol version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolVersion {
    /// Wire-incompatible revision.
    pub major: u16,
    /// Additive revision.
    pub minor: u16,
}

impl ProtocolVersion {
    /// Construct a protocol version.
    pub const fn new(major: u16, minor: u16) -> Self {
        Self { major, minor }
    }
}

/// Native objective bridge metadata shared by eindir-compatible consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbiStamp {
    /// Major ABI revision for the embedded eindir objective handle.
    pub abi_major: u16,
    /// Additive ABI revision for the embedded eindir objective handle.
    pub abi_minor: u16,
    /// ABI layout revision for the embedded objective handle.
    pub layout_revision: u32,
    /// Major DLPack callback revision.
    pub dlpack_major: u16,
    /// Minor DLPack callback revision.
    pub dlpack_minor: u16,
    /// Feature bits required by the consumer.
    pub features: u64,
}

impl AbiStamp {
    /// The stamp required by the anneal-side bridge.
    pub const fn anneal_default() -> Self {
        Self {
            abi_major: 1,
            abi_minor: 1,
            layout_revision: 3,
            dlpack_major: 1,
            dlpack_minor: 0,
            features: EINDIR_ABI_FEATURE_GRADIENT | EINDIR_ABI_FEATURE_BATCH,
        }
    }
}

/// Self-description returned by an objective engine or RPC endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineDescriptor {
    /// Stable producer name.
    pub engine_id: String,
    /// Stable protocol family name.
    pub protocol_family: String,
    /// Protocol spoken by the producer.
    pub protocol: ProtocolVersion,
    /// Native bridge metadata.
    pub abi: AbiStamp,
    /// Optional source or build identity supplied by the producer.
    pub build_identity: Option<String>,
}

impl EngineDescriptor {
    /// Construct a descriptor for a producer.
    pub fn new(engine_id: impl Into<String>, protocol: ProtocolVersion, abi: AbiStamp) -> Self {
        Self {
            engine_id: engine_id.into(),
            protocol_family: PROTOCOL_FAMILY.to_owned(),
            protocol,
            abi,
            build_identity: None,
        }
    }

    /// Construct a descriptor with an explicit protocol family.
    pub fn with_family(
        engine_id: impl Into<String>,
        protocol_family: impl Into<String>,
        protocol: ProtocolVersion,
        abi: AbiStamp,
    ) -> Self {
        Self {
            engine_id: engine_id.into(),
            protocol_family: protocol_family.into(),
            protocol,
            abi,
            build_identity: None,
        }
    }

    /// Validate a producer against the requested protocol and bridge stamp.
    pub fn validate(
        &self,
        expected_family: &str,
        expected_protocol: ProtocolVersion,
        expected_abi: AbiStamp,
    ) -> Result<(), CompatibilityError> {
        if self.protocol_family != expected_family {
            return Err(CompatibilityError::ProtocolFamily {
                expected: expected_family.to_owned(),
                received: self.protocol_family.clone(),
            });
        }
        if self.protocol.major != expected_protocol.major {
            return Err(CompatibilityError::ProtocolMajor {
                expected: expected_protocol.major,
                received: self.protocol.major,
            });
        }
        if self.protocol.minor < expected_protocol.minor {
            return Err(CompatibilityError::ProtocolMinor {
                expected: expected_protocol.minor,
                received: self.protocol.minor,
            });
        }
        if self.abi.layout_revision != expected_abi.layout_revision {
            return Err(CompatibilityError::AbiLayout {
                expected: expected_abi.layout_revision,
                received: self.abi.layout_revision,
            });
        }
        if self.abi.abi_major != expected_abi.abi_major {
            return Err(CompatibilityError::AbiMajor {
                expected: expected_abi.abi_major,
                received: self.abi.abi_major,
            });
        }
        if self.abi.abi_minor < expected_abi.abi_minor {
            return Err(CompatibilityError::AbiMinor {
                expected: expected_abi.abi_minor,
                received: self.abi.abi_minor,
            });
        }
        if self.abi.dlpack_major != expected_abi.dlpack_major {
            return Err(CompatibilityError::DlpackMajor {
                expected: expected_abi.dlpack_major,
                received: self.abi.dlpack_major,
            });
        }
        if self.abi.dlpack_minor < expected_abi.dlpack_minor {
            return Err(CompatibilityError::DlpackMinor {
                expected: expected_abi.dlpack_minor,
                received: self.abi.dlpack_minor,
            });
        }
        let missing = expected_abi.features & !self.abi.features;
        if missing != 0 {
            return Err(CompatibilityError::MissingFeatures { missing });
        }
        Ok(())
    }
}

impl Default for EngineDescriptor {
    fn default() -> Self {
        Self::new(
            PROTOCOL_FAMILY,
            ProtocolVersion::new(1, 0),
            AbiStamp::anneal_default(),
        )
    }
}

/// Validate the memory and numerical shape invariants required by an eindir
/// objective before adapting it to an [`eindir_core::Objective`].
///
/// The handle is borrowed and no callback is invoked. Semantic metadata such
/// as units and force sign requires the versioned objective descriptor tracked
/// separately from the embedded binary handle.
///
/// # Safety
///
/// `objective` must point to a readable `eindir_objective_t` whose bound
/// pointers reference `objective.dim` readable `f64` values.
pub unsafe fn validate_eindir_objective(
    objective: *const eindir_objective_t,
    expected_dim: usize,
) -> Result<(), CompatibilityError> {
    if objective.is_null() {
        return Err(CompatibilityError::ObjectiveNull);
    }
    if expected_dim == 0 {
        return Err(CompatibilityError::ObjectiveDimension {
            expected: 1,
            received: 0,
        });
    }

    let objective = unsafe { &*objective };
    if objective.dim != expected_dim {
        return Err(CompatibilityError::ObjectiveDimension {
            expected: expected_dim,
            received: objective.dim,
        });
    }
    if objective.low.is_null() {
        return Err(CompatibilityError::ObjectiveBoundsNull { bound: "low" });
    }
    if objective.high.is_null() {
        return Err(CompatibilityError::ObjectiveBoundsNull { bound: "high" });
    }

    let low = unsafe { std::slice::from_raw_parts(objective.low, objective.dim) };
    let high = unsafe { std::slice::from_raw_parts(objective.high, objective.dim) };
    for (index, (&lower, &upper)) in low.iter().zip(high.iter()).enumerate() {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(CompatibilityError::ObjectiveBoundsInvalid {
                index,
                lower: lower.to_string(),
                upper: upper.to_string(),
            });
        }
    }
    Ok(())
}

/// Reason an engine descriptor cannot be consumed.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CompatibilityError {
    /// The supplied objective handle is null.
    #[error("eindir objective handle is NULL")]
    ObjectiveNull,
    /// The objective dimension does not match the consumer's request.
    #[error("objective dimension mismatch: expected {expected}, received {received}")]
    ObjectiveDimension {
        /// Dimension requested by the consumer.
        expected: usize,
        /// Dimension reported by the handle.
        received: usize,
    },
    /// A required objective bound pointer is null.
    #[error("eindir objective {bound} bounds are NULL")]
    ObjectiveBoundsNull {
        /// Bound pointer that is absent.
        bound: &'static str,
    },
    /// An objective bound is non-finite or inverted.
    #[error("invalid objective bounds at index {index}: lower={lower}, upper={upper}")]
    ObjectiveBoundsInvalid {
        /// Coordinate containing the invalid interval.
        index: usize,
        /// Lower bound rendered for a stable diagnostic.
        lower: String,
        /// Upper bound rendered for a stable diagnostic.
        upper: String,
    },
    /// The embedded objective ABI major differs.
    #[error("ABI major mismatch: expected {expected}, received {received}")]
    AbiMajor {
        /// Consumer-required ABI major revision.
        expected: u16,
        /// Producer-reported ABI major revision.
        received: u16,
    },
    /// The producer is older than the requested additive ABI revision.
    #[error("ABI minor too old: expected at least {expected}, received {received}")]
    AbiMinor {
        /// Consumer-required ABI minor revision.
        expected: u16,
        /// Producer-reported ABI minor revision.
        received: u16,
    },
    /// The producer belongs to another protocol family.
    #[error("protocol family mismatch: expected {expected}, received {received}")]
    ProtocolFamily {
        /// Consumer-required family name.
        expected: String,
        /// Producer-reported family name.
        received: String,
    },
    /// The wire protocol major differs.
    #[error("protocol major mismatch: expected {expected}, received {received}")]
    ProtocolMajor {
        /// Consumer-required major revision.
        expected: u16,
        /// Producer-reported major revision.
        received: u16,
    },
    /// The producer is older than the requested additive protocol revision.
    #[error("protocol minor too old: expected at least {expected}, received {received}")]
    ProtocolMinor {
        /// Consumer-required minor revision.
        expected: u16,
        /// Producer-reported minor revision.
        received: u16,
    },
    /// The embedded objective layout differs.
    #[error("ABI layout mismatch: expected {expected}, received {received}")]
    AbiLayout {
        /// Consumer-required layout revision.
        expected: u32,
        /// Producer-reported layout revision.
        received: u32,
    },
    /// The DLPack major differs.
    #[error("DLPack major mismatch: expected {expected}, received {received}")]
    DlpackMajor {
        /// Consumer-required DLPack major revision.
        expected: u16,
        /// Producer-reported DLPack major revision.
        received: u16,
    },
    /// The producer is older than the requested DLPack minor revision.
    #[error("DLPack minor too old: expected at least {expected}, received {received}")]
    DlpackMinor {
        /// Consumer-required DLPack minor revision.
        expected: u16,
        /// Producer-reported DLPack minor revision.
        received: u16,
    },
    /// Required bridge capabilities are absent.
    #[error("required ABI features missing: 0x{missing:016x}")]
    MissingFeatures {
        /// Feature bits required by the consumer but absent from the producer.
        missing: u64,
    },
}
