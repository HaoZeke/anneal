//! Deterministic provenance records for reproducible engine-backed runs.

use crate::compatibility::EngineDescriptor;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

/// SHA-256 digest for an artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ArtifactDigest {
    /// Relative artifact name.
    pub name: String,
    /// Lowercase hexadecimal SHA-256 digest.
    pub sha256: String,
    /// Artifact size in bytes.
    pub bytes: usize,
}

impl ArtifactDigest {
    /// Hash an artifact with its stable name.
    pub fn of(bytes: &[u8]) -> Self {
        Self::named("", bytes)
    }

    fn named(name: impl Into<String>, bytes: &[u8]) -> Self {
        let digest = Sha256::digest(bytes);
        Self { name: name.into(), sha256: format!("{digest:x}"), bytes: bytes.len() }
    }
}

/// Minimal machine-readable provenance manifest.
#[derive(Debug, Clone, Serialize)]
pub struct RunManifest {
    /// Stable manifest schema version.
    pub manifest_version: u16,
    /// Caller-provided run identity.
    pub run_id: String,
    /// Random seed used by the search.
    pub seed: u64,
    /// Evaluation budget.
    pub evaluation_budget: u64,
    /// Objective engine compatibility descriptor.
    pub engine: EngineDescriptor,
    /// Content-addressed run artifacts sorted by name.
    pub artifacts: Vec<ArtifactDigest>,
}

impl RunManifest {
    /// Create a manifest with the current public schema version.
    pub fn new(run_id: impl Into<String>, seed: u64, evaluation_budget: u64) -> Self {
        Self {
            manifest_version: 1,
            run_id: run_id.into(),
            seed,
            evaluation_budget,
            engine: EngineDescriptor::default(),
            artifacts: Vec::new(),
        }
    }

    /// Add or replace an artifact digest.
    pub fn add_artifact(&mut self, name: impl Into<String>, bytes: &[u8]) {
        let digest = ArtifactDigest::named(name, bytes);
        let mut by_name: BTreeMap<String, ArtifactDigest> = self
            .artifacts
            .drain(..)
            .map(|item| (item.name.clone(), item))
            .collect();
        by_name.insert(digest.name.clone(), digest);
        self.artifacts = by_name.into_values().collect();
    }

    /// Serialize the manifest deterministically as pretty JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Verify bytes against a recorded artifact digest.
    pub fn verify_artifact(&self, name: &str, bytes: &[u8]) -> Result<(), ManifestError> {
        let expected = self
            .artifacts
            .iter()
            .find(|item| item.name == name)
            .ok_or_else(|| ManifestError::MissingArtifact(name.to_owned()))?;
        let actual = ArtifactDigest::named(name, bytes);
        if actual.sha256 != expected.sha256 || actual.bytes != expected.bytes {
            return Err(ManifestError::DigestMismatch { name: name.to_owned() });
        }
        Ok(())
    }
}

/// Manifest verification failure.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ManifestError {
    /// No digest exists for the requested artifact.
    #[error("manifest has no artifact named {0}")]
    MissingArtifact(String),
    /// The supplied bytes do not match the recorded digest.
    #[error("artifact digest mismatch for {name}")]
    DigestMismatch {
        /// Artifact whose bytes differ from the recorded digest.
        name: String,
    },
}
