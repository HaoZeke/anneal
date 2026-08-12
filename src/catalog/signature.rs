//! Canonical identity for systems that may share a basin catalog.

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const SIGNATURE_PREFIX: &[u8] = b"ANNEAL\0SYSTEM_SIGNATURE\0";
const SIGNATURE_ENCODING_VERSION: u32 = 1;

/// SHA-256 digest of a canonical system signature.
pub type SignatureDigest = [u8; 32];

/// Potential engine and immutable external inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineSignature {
    /// Stable engine family name.
    pub kind: String,
    /// Digest of the complete engine configuration.
    pub config_digest: SignatureDigest,
    /// Named immutable files or parameter sets consumed by the engine.
    pub external_inputs: BTreeMap<String, SignatureDigest>,
}

/// Descriptor schema used for basin identity and distance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescriptorSignature {
    /// Stable schema name.
    pub schema: String,
    /// Schema version.
    pub version: u32,
    /// Canonical textual hyperparameters with physical units where required.
    pub hyperparameters: BTreeMap<String, String>,
    /// Ordered species channels used by the descriptor.
    pub species_channels: Vec<u32>,
}

/// Complete identity of a search system and its catalog semantics.
#[derive(Debug, Clone)]
pub struct SystemSignature {
    /// Atomic numbers in coordinate order.
    pub atomic_numbers: Vec<u32>,
    /// Declared length of the Cartesian coordinate vector.
    pub coordinate_dim: u64,
    /// Group label for each atom.
    pub group_labels: Vec<u32>,
    /// Stable group-constraint schema name and version.
    pub group_schema: String,
    /// Frozen-atom mask in coordinate order.
    pub frozen_mask: Vec<bool>,
    /// Row-major cell matrix, or no cell for a nonperiodic finite system.
    pub cell: Option<[f64; 9]>,
    /// Periodicity on each cell axis.
    pub periodic: [bool; 3],
    /// Coordinate value represented by one canonical length unit.
    pub length_scale: f64,
    /// Energy value represented by one canonical energy unit.
    pub energy_scale: f64,
    /// Engine identity.
    pub engine: EngineSignature,
    /// Descriptor identity.
    pub descriptor: DescriptorSignature,
    /// Candidate-validation contract version.
    pub validation_schema_version: u32,
}

impl SystemSignature {
    /// Encode every identity field into the versioned canonical wire form.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(SIGNATURE_PREFIX);
        push_u32(&mut out, SIGNATURE_ENCODING_VERSION);
        push_u32_slice(&mut out, &self.atomic_numbers);
        push_u64(&mut out, self.coordinate_dim);
        push_u32_slice(&mut out, &self.group_labels);
        push_str(&mut out, &self.group_schema);
        push_bools(&mut out, &self.frozen_mask);
        match self.cell {
            Some(cell) => {
                out.push(1);
                for value in cell {
                    push_f64(&mut out, value);
                }
            }
            None => out.push(0),
        }
        for value in self.periodic {
            out.push(u8::from(value));
        }
        push_f64(&mut out, self.length_scale);
        push_f64(&mut out, self.energy_scale);
        push_str(&mut out, &self.engine.kind);
        out.extend_from_slice(&self.engine.config_digest);
        push_digest_map(&mut out, &self.engine.external_inputs);
        push_str(&mut out, &self.descriptor.schema);
        push_u32(&mut out, self.descriptor.version);
        push_string_map(&mut out, &self.descriptor.hyperparameters);
        push_u32_slice(&mut out, &self.descriptor.species_channels);
        push_u32(&mut out, self.validation_schema_version);
        out
    }

    /// Hash the canonical wire form with SHA-256.
    pub fn digest(&self) -> SignatureDigest {
        let mut hasher = Sha256::new();
        hasher.update(self.canonical_bytes());
        hasher.finalize().into()
    }
}

impl PartialEq for SystemSignature {
    fn eq(&self, other: &Self) -> bool {
        self.canonical_bytes() == other.canonical_bytes()
    }
}

impl Eq for SystemSignature {}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_len(out: &mut Vec<u8>, len: usize) {
    let value = u64::try_from(len).expect("signature field length exceeds u64");
    push_u64(out, value);
}

fn push_str(out: &mut Vec<u8>, value: &str) {
    push_len(out, value.len());
    out.extend_from_slice(value.as_bytes());
}

fn push_u32_slice(out: &mut Vec<u8>, values: &[u32]) {
    push_len(out, values.len());
    for &value in values {
        push_u32(out, value);
    }
}

fn push_bools(out: &mut Vec<u8>, values: &[bool]) {
    push_len(out, values.len());
    out.extend(values.iter().map(|&value| u8::from(value)));
}

fn push_f64(out: &mut Vec<u8>, value: f64) {
    let bits = if value == 0.0 {
        0
    } else if value.is_nan() {
        f64::NAN.to_bits()
    } else {
        value.to_bits()
    };
    push_u64(out, bits);
}

fn push_digest_map(out: &mut Vec<u8>, values: &BTreeMap<String, SignatureDigest>) {
    push_len(out, values.len());
    for (name, digest) in values {
        push_str(out, name);
        out.extend_from_slice(digest);
    }
}

fn push_string_map(out: &mut Vec<u8>, values: &BTreeMap<String, String>) {
    push_len(out, values.len());
    for (name, value) in values {
        push_str(out, name);
        push_str(out, value);
    }
}
