//! Stable boundary for externally implemented invariant descriptors.

use super::{DescriptorError, DescriptorGeometry};
use ndarray::{Array2, ArrayView1};
use sha2::{Digest, Sha256};

const PROVIDER_IDENTITY_PREFIX: &[u8] = b"ANNEAL\0DESCRIPTOR_PROVIDER\0";

/// One fixed-dimensional output exposed by a descriptor provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescriptorOutputSpec {
    name: String,
    dimension: usize,
}

impl DescriptorOutputSpec {
    /// Define a named output with a fixed number of scalar channels.
    pub fn new(name: impl Into<String>, dimension: usize) -> Result<Self, DescriptorError> {
        let name = name.into();
        if name.is_empty() {
            return Err(DescriptorError::EmptyProviderOutput);
        }
        if dimension == 0 {
            return Err(DescriptorError::ZeroProviderDimension);
        }
        Ok(Self { name, dimension })
    }

    /// Metatomic or provider-specific output name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Fixed number of scalar channels in the output.
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Versioned identity and metric semantics of an invariant descriptor provider.
#[derive(Debug, Clone, PartialEq)]
pub struct DescriptorProviderContract {
    schema_name: String,
    schema_version: u32,
    model_digest: [u8; 32],
    system_output: DescriptorOutputSpec,
    atomic_output: Option<DescriptorOutputSpec>,
    interaction_range: f64,
    normalization: String,
}

impl DescriptorProviderContract {
    /// Define an invariant provider whose scalar outputs use Euclidean distance.
    ///
    /// `system_output` represents one row per system. `atomic_output`, when
    /// present, represents one row per atom. The provider implementation
    /// guarantees translation, rotation, and like-species permutation
    /// invariance for the system output and the corresponding equivariance of
    /// atomic rows. `model_digest` pins the complete model or analytic
    /// calculator configuration that gives these outputs their meaning.
    pub fn new(
        schema_name: impl Into<String>,
        schema_version: u32,
        model_digest: [u8; 32],
        system_output: DescriptorOutputSpec,
        atomic_output: Option<DescriptorOutputSpec>,
        interaction_range: f64,
        normalization: impl Into<String>,
    ) -> Result<Self, DescriptorError> {
        let schema_name = schema_name.into();
        if schema_name.is_empty() {
            return Err(DescriptorError::EmptySchemaName);
        }
        if schema_version == 0 {
            return Err(DescriptorError::ZeroSchemaVersion);
        }
        if model_digest.iter().all(|&byte| byte == 0) {
            return Err(DescriptorError::MissingProviderDigest);
        }
        if !interaction_range.is_finite() || interaction_range <= 0.0 {
            return Err(DescriptorError::InvalidProviderInteractionRange);
        }
        let normalization = normalization.into();
        if normalization.is_empty() {
            return Err(DescriptorError::EmptyProviderNormalization);
        }
        Ok(Self {
            schema_name,
            schema_version,
            model_digest,
            system_output,
            atomic_output,
            interaction_range,
            normalization,
        })
    }

    /// Stable descriptor-family name.
    pub fn schema_name(&self) -> &str {
        &self.schema_name
    }

    /// Positive descriptor-family version.
    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// SHA-256 digest of the complete model or calculator configuration.
    pub fn model_digest(&self) -> [u8; 32] {
        self.model_digest
    }

    /// Required one-row-per-system output.
    pub fn system_output(&self) -> &DescriptorOutputSpec {
        &self.system_output
    }

    /// Optional one-row-per-atom output used for local ride targets.
    pub fn atomic_output(&self) -> Option<&DescriptorOutputSpec> {
        self.atomic_output.as_ref()
    }

    /// Largest physical interaction range needed by the provider.
    pub fn interaction_range(&self) -> f64 {
        self.interaction_range
    }

    /// Versioned preprocessing and normalization convention.
    pub fn normalization(&self) -> &str {
        &self.normalization
    }

    pub(super) fn identity_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(PROVIDER_IDENTITY_PREFIX);
        update_string(&mut hasher, &self.schema_name);
        hasher.update(self.schema_version.to_be_bytes());
        hasher.update(self.model_digest);
        update_output(&mut hasher, &self.system_output);
        match &self.atomic_output {
            Some(output) => {
                hasher.update([1]);
                update_output(&mut hasher, output);
            }
            None => hasher.update([0]),
        }
        hasher.update(self.interaction_range.to_bits().to_be_bytes());
        update_string(&mut hasher, &self.normalization);
        hasher.finalize().into()
    }
}

fn update_output(hasher: &mut Sha256, output: &DescriptorOutputSpec) {
    update_string(hasher, &output.name);
    hasher.update((output.dimension as u64).to_be_bytes());
}

fn update_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

/// Validated geometry and species passed to an external descriptor provider.
#[derive(Debug, Clone, Copy)]
pub struct DescriptorProviderInput<'a> {
    geometry: DescriptorGeometry,
    coordinates: ArrayView1<'a, f64>,
    species: Option<&'a [u32]>,
}

impl<'a> DescriptorProviderInput<'a> {
    pub(super) fn new(
        geometry: DescriptorGeometry,
        coordinates: ArrayView1<'a, f64>,
        species: Option<&'a [u32]>,
    ) -> Self {
        Self {
            geometry,
            coordinates,
            species,
        }
    }

    /// Length scale, cell, and periodic axes for this evaluation.
    pub fn geometry(self) -> DescriptorGeometry {
        self.geometry
    }

    /// Flat Cartesian coordinates with three entries per atom.
    pub fn coordinates(self) -> ArrayView1<'a, f64> {
        self.coordinates
    }

    /// Atomic numbers in coordinate order, when supplied.
    pub fn species(self) -> Option<&'a [u32]> {
        self.species
    }
}

/// Evaluation failure reported by an external descriptor implementation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct DescriptorProviderError {
    message: String,
}

impl DescriptorProviderError {
    /// Construct an evaluation error without exposing provider internals.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Provider-supplied diagnostic.
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// External source of fixed-dimensional invariant atomistic features.
///
/// Implementations typically wrap a metatomic `feature` or
/// `mtt::feature::<layer>` output, or a featomic invariant calculator. Anneal
/// owns neither the feature construction nor its fitted parameters.
pub trait InvariantDescriptorProvider: std::fmt::Debug + Send + Sync {
    /// Immutable identity and shape contract.
    fn contract(&self) -> &DescriptorProviderContract;

    /// Evaluate the one-row-per-system invariant feature.
    fn describe_system(
        &self,
        input: DescriptorProviderInput<'_>,
    ) -> Result<Vec<f64>, DescriptorProviderError>;

    /// Evaluate the optional one-row-per-atom invariant feature.
    fn describe_atoms(
        &self,
        _input: DescriptorProviderInput<'_>,
    ) -> Result<Option<Array2<f64>>, DescriptorProviderError> {
        Ok(None)
    }
}
