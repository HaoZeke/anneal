//! Versioned compatibility descriptors for engine-backed objective bridges.

use serde::{Deserialize, Serialize};
use thiserror::Error;

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
            layout_revision: 1,
            dlpack_major: 1,
            dlpack_minor: 0,
            features: 0,
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
}

impl EngineDescriptor {
    /// Construct a descriptor for a producer.
    pub fn new(engine_id: impl Into<String>, protocol: ProtocolVersion, abi: AbiStamp) -> Self {
        Self {
            engine_id: engine_id.into(),
            protocol_family: PROTOCOL_FAMILY.to_owned(),
            protocol,
            abi,
        }
    }

    /// Construct a descriptor with an explicit protocol family.
    pub fn with_family(
        engine_id: impl Into<String>,
        protocol_family: impl Into<String>,
        protocol: ProtocolVersion,
        abi: AbiStamp,
    ) -> Self {
        Self { engine_id: engine_id.into(), protocol_family: protocol_family.into(), protocol, abi }
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
        Self::new(PROTOCOL_FAMILY, ProtocolVersion::new(1, 0), AbiStamp::anneal_default())
    }
}

/// Reason an engine descriptor cannot be consumed.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CompatibilityError {
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
