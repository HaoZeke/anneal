//! Versioned multiscale invariant descriptor spaces for catalog geometry.

pub mod pullback;

use crate::soap::{SoapSpec, local_nu3_z, local_spectra_z};
use ndarray::{Array2, ArrayView1};

const NORMALIZATION_SCHEMA: &str = "l2-v1";

/// Invariant aggregation used by one descriptor block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorBlockKind {
    /// Mean of species-aware local SOAP power spectra.
    SoapMean,
    /// Diagonal second central moment of local SOAP spectra.
    SoapVariance,
    /// Mean ACE nu=3 contraction of local spherical expansions.
    AceNu3Mean,
}

/// Resolution and aggregation contract for one normalized block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DescriptorBlockSpec {
    kind: DescriptorBlockKind,
    soap: SoapSpec,
}

impl DescriptorBlockSpec {
    /// Construct a validated block specification.
    pub fn new(
        kind: DescriptorBlockKind,
        n_max: usize,
        l_max: usize,
        cutoff: f64,
    ) -> Result<Self, DescriptorError> {
        if n_max == 0 {
            return Err(DescriptorError::ZeroRadialFunctions);
        }
        if !cutoff.is_finite() || cutoff <= 0.0 {
            return Err(DescriptorError::InvalidCutoff);
        }
        Ok(Self {
            kind,
            soap: SoapSpec {
                n_max,
                l_max,
                rcut_nn: cutoff,
            },
        })
    }

    /// Aggregation used by this block.
    pub fn kind(self) -> DescriptorBlockKind {
        self.kind
    }

    /// Number of radial basis functions.
    pub fn n_max(self) -> usize {
        self.soap.n_max
    }

    /// Largest angular momentum channel.
    pub fn l_max(self) -> usize {
        self.soap.l_max
    }

    /// Fixed radial cutoff in coordinate units.
    pub fn cutoff(self) -> f64 {
        self.soap.rcut_nn
    }
}

/// Stable name, version, and ordered normalized-block definition.
#[derive(Debug, Clone, PartialEq)]
pub struct DescriptorSchema {
    name: String,
    version: u32,
    blocks: Vec<DescriptorBlockSpec>,
}

impl DescriptorSchema {
    /// Construct a versioned descriptor schema.
    pub fn new(
        name: impl Into<String>,
        version: u32,
        blocks: Vec<DescriptorBlockSpec>,
    ) -> Result<Self, DescriptorError> {
        let name = name.into();
        if name.is_empty() {
            return Err(DescriptorError::EmptySchemaName);
        }
        if version == 0 {
            return Err(DescriptorError::ZeroSchemaVersion);
        }
        if blocks.is_empty() {
            return Err(DescriptorError::EmptySchema);
        }
        Ok(Self {
            name,
            version,
            blocks,
        })
    }

    /// Stable schema name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Schema version.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Ordered block definitions.
    pub fn blocks(&self) -> &[DescriptorBlockSpec] {
        &self.blocks
    }
}

/// Metadata locating one normalized block in a descriptor vector.
#[derive(Debug, Clone, PartialEq)]
pub struct DescriptorBlockMetadata {
    kind: DescriptorBlockKind,
    n_max: usize,
    l_max: usize,
    cutoff: f64,
    offset: usize,
    len: usize,
    raw_norm: f64,
}

impl DescriptorBlockMetadata {
    /// Aggregation represented by this block.
    pub fn kind(&self) -> DescriptorBlockKind {
        self.kind
    }

    /// Number of radial basis functions.
    pub fn n_max(&self) -> usize {
        self.n_max
    }

    /// Largest angular momentum channel.
    pub fn l_max(&self) -> usize {
        self.l_max
    }

    /// Fixed radial cutoff in coordinate units.
    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }

    /// First value belonging to this block.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Number of values in this block.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this block contains no values.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Norm measured before block normalization.
    pub fn raw_norm(&self) -> f64 {
        self.raw_norm
    }

    /// Versioned normalization convention.
    pub fn normalization(&self) -> &'static str {
        NORMALIZATION_SCHEMA
    }
}

/// One descriptor vector and the metadata needed to interpret every value.
#[derive(Debug, Clone, PartialEq)]
pub struct DescriptorVector {
    schema_name: String,
    schema_version: u32,
    values: Vec<f64>,
    blocks: Vec<DescriptorBlockMetadata>,
}

impl DescriptorVector {
    /// Stable schema name.
    pub fn schema_name(&self) -> &str {
        &self.schema_name
    }

    /// Schema version.
    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Concatenated normalized block values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Ordered block metadata.
    pub fn blocks(&self) -> &[DescriptorBlockMetadata] {
        &self.blocks
    }
}

/// Input or schema failure that prevents descriptor evaluation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DescriptorError {
    /// Schema name must identify its semantics.
    #[error("descriptor schema name must be nonempty")]
    EmptySchemaName,
    /// Schema version zero is reserved.
    #[error("descriptor schema version must be positive")]
    ZeroSchemaVersion,
    /// A descriptor requires at least one block.
    #[error("descriptor schema must contain at least one block")]
    EmptySchema,
    /// SOAP requires at least one radial function.
    #[error("SOAP radial function count must be positive")]
    ZeroRadialFunctions,
    /// SOAP cutoff must be finite and positive.
    #[error("SOAP cutoff must be finite and positive")]
    InvalidCutoff,
    /// Cartesian coordinates must have exactly three components per atom.
    #[error("coordinate dimension {actual} is not a positive multiple of three")]
    CoordinateDimension {
        /// Supplied coordinate length.
        actual: usize,
    },
    /// Species count must equal atom count.
    #[error("species count is {actual}, expected {expected}")]
    SpeciesDimension {
        /// Number of Cartesian atoms.
        expected: usize,
        /// Supplied species count.
        actual: usize,
    },
    /// Cartesian input contains NaN or infinity.
    #[error("nonfinite coordinate at index {index}")]
    NonFiniteCoordinate {
        /// Index of the first invalid coordinate.
        index: usize,
    },
    /// A SOAP or ACE primitive produced NaN or infinity.
    #[error("nonfinite descriptor value in block {block} at index {index}")]
    NonFiniteDescriptor {
        /// Ordered block index.
        block: usize,
        /// Index within that block.
        index: usize,
    },
}

/// Evaluator bound to one immutable descriptor schema.
#[derive(Debug, Clone)]
pub struct DescriptorSpace {
    schema: DescriptorSchema,
}

impl DescriptorSpace {
    /// Bind an evaluator to a validated schema.
    pub fn new(schema: DescriptorSchema) -> Self {
        Self { schema }
    }

    /// Immutable schema interpreted by this evaluator.
    pub fn schema(&self) -> &DescriptorSchema {
        &self.schema
    }

    /// Evaluate all invariant blocks and concatenate their normalized values.
    pub fn describe(
        &self,
        coordinates: ArrayView1<f64>,
        species: Option<&[u32]>,
    ) -> Result<DescriptorVector, DescriptorError> {
        if coordinates.is_empty() || coordinates.len() % 3 != 0 {
            return Err(DescriptorError::CoordinateDimension {
                actual: coordinates.len(),
            });
        }
        if let Some(index) = coordinates.iter().position(|value| !value.is_finite()) {
            return Err(DescriptorError::NonFiniteCoordinate { index });
        }
        let atoms = coordinates.len() / 3;
        if let Some(species) = species {
            if species.len() != atoms {
                return Err(DescriptorError::SpeciesDimension {
                    expected: atoms,
                    actual: species.len(),
                });
            }
        }

        let mut values = Vec::new();
        let mut metadata = Vec::with_capacity(self.schema.blocks.len());
        for (block_index, block) in self.schema.blocks.iter().copied().enumerate() {
            let local = match block.kind {
                DescriptorBlockKind::SoapMean | DescriptorBlockKind::SoapVariance => {
                    local_spectra_z(coordinates, block.soap, species)
                }
                DescriptorBlockKind::AceNu3Mean => local_nu3_z(coordinates, block.soap, species),
            };
            let mut aggregated = match block.kind {
                DescriptorBlockKind::SoapMean => column_mean(&local, 0),
                DescriptorBlockKind::SoapVariance => column_variance(&local),
                DescriptorBlockKind::AceNu3Mean => {
                    column_mean(&local, block.soap.feat_dim(species))
                }
            };
            if let Some(index) = aggregated.iter().position(|value| !value.is_finite()) {
                return Err(DescriptorError::NonFiniteDescriptor {
                    block: block_index,
                    index,
                });
            }
            let raw_norm = aggregated
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            if raw_norm > 0.0 {
                for value in &mut aggregated {
                    *value /= raw_norm;
                }
            }
            let offset = values.len();
            let len = aggregated.len();
            values.extend(aggregated);
            metadata.push(DescriptorBlockMetadata {
                kind: block.kind,
                n_max: block.soap.n_max,
                l_max: block.soap.l_max,
                cutoff: block.soap.rcut_nn,
                offset,
                len,
                raw_norm,
            });
        }
        Ok(DescriptorVector {
            schema_name: self.schema.name.clone(),
            schema_version: self.schema.version,
            values,
            blocks: metadata,
        })
    }
}

fn column_mean(local: &Array2<f64>, start: usize) -> Vec<f64> {
    let rows = local.nrows();
    let mut mean = vec![0.0; local.ncols().saturating_sub(start)];
    for row in 0..rows {
        for (index, value) in mean.iter_mut().enumerate() {
            *value += local[[row, start + index]];
        }
    }
    if rows != 0 {
        for value in &mut mean {
            *value /= rows as f64;
        }
    }
    mean
}

fn column_variance(local: &Array2<f64>) -> Vec<f64> {
    let mean = column_mean(local, 0);
    let rows = local.nrows();
    let mut variance = vec![0.0; local.ncols()];
    for row in 0..rows {
        for column in 0..local.ncols() {
            let residual = local[[row, column]] - mean[column];
            variance[column] += residual * residual;
        }
    }
    if rows != 0 {
        for value in &mut variance {
            *value /= rows as f64;
        }
    }
    variance
}
