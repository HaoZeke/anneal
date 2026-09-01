//! Versioned multiscale invariant descriptor spaces for catalog geometry.

pub mod pullback;
mod universal;

pub use universal::{
    DescriptorGeometry, UNIVERSAL_DESCRIPTOR_SCHEMA, UNIVERSAL_DESCRIPTOR_VERSION,
    UNIVERSAL_LOCAL_ENVIRONMENT_RADIUS, universal_descriptor_space,
};

use crate::soap::{SoapSpec, jacobian_ace, jacobian_z, local_nu3_z, local_spectra_z};
use ndarray::{Array2, ArrayView1};

const L2_NORMALIZATION_SCHEMA: &str = "l2-v1";
pub(super) const SOFT_L2_NORMALIZATION_SCHEMA: &str = "soft-l2-eps-v1";

/// Invariant aggregation used by one descriptor block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorBlockKind {
    /// Mean of species-aware local SOAP power spectra.
    SoapMean,
    /// Diagonal second central moment of local SOAP spectra.
    SoapVariance,
    /// Mean ACE nu=3 contraction of local spherical expansions.
    AceNu3Mean,
    /// Stacked species-conditioned leftover \(p_i-\mu_{z(i)}\).
    SoapLeftover,
    /// Fixed-channel radial spectrum of minimum-image pair distances.
    PairRadial,
    /// Fixed-channel radial and angular spectrum of centred triples.
    ThreeBodyAngular,
    /// Multiscale graph moments and connectivity statistics.
    GraphTopology,
    /// Fixed-channel rotation-invariant SOAP mean.
    InvariantSoapMean,
    /// Fixed-channel rotation-invariant ACE nu=3 mean.
    InvariantAceNu3Mean,
    /// Permutation-invariant pseudoscalar moments that distinguish mirror minima.
    ChiralMoment,
}

impl DescriptorBlockKind {
    fn requires_geometry(self) -> bool {
        matches!(
            self,
            Self::PairRadial
                | Self::ThreeBodyAngular
                | Self::GraphTopology
                | Self::InvariantSoapMean
                | Self::InvariantAceNu3Mean
                | Self::ChiralMoment
        )
    }
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
    normalization: &'static str,
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
        self.normalization
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

    /// Euclidean pseudometric between vectors from the same descriptor schema.
    pub fn distance(&self, other: &Self) -> Result<f64, DescriptorError> {
        let compatible = self.schema_name == other.schema_name
            && self.schema_version == other.schema_version
            && self.values.len() == other.values.len()
            && self.blocks.len() == other.blocks.len()
            && self.blocks.iter().zip(&other.blocks).all(|(left, right)| {
                left.kind == right.kind
                    && left.n_max == right.n_max
                    && left.l_max == right.l_max
                    && left.cutoff == right.cutoff
                    && left.offset == right.offset
                    && left.len == right.len
                    && left.normalization == right.normalization
            });
        if !compatible {
            return Err(DescriptorError::IncompatibleDescriptorVectors);
        }
        Ok(self
            .values
            .iter()
            .zip(&other.values)
            .map(|(left, right)| {
                let difference = left - right;
                difference * difference
            })
            .sum::<f64>()
            .sqrt())
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
    /// Universal coordinates require a finite positive reference length.
    #[error("descriptor length scale must be finite and positive")]
    InvalidLengthScale,
    /// Periodic axes require an explicit simulation cell.
    #[error("periodic descriptor geometry requires a simulation cell")]
    PeriodicCellRequired,
    /// A simulation cell must be finite and nonsingular.
    #[error("descriptor simulation cell must be finite and nonsingular")]
    InvalidCell,
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
    /// Central differences require a finite, strictly positive Cartesian step.
    #[error("finite-difference step must be finite and positive")]
    InvalidFiniteDifferenceStep,
    /// Distances require vectors from one exact descriptor schema.
    #[error("descriptor vectors use incompatible schemas")]
    IncompatibleDescriptorVectors,
    /// Universal blocks need the length scale and boundary conditions they encode.
    #[error("universal descriptor blocks require descriptor geometry")]
    UniversalGeometryRequired,
    /// Periodic linked-cell construction or search rejected the geometry.
    #[error("universal descriptor neighbour search failed")]
    NeighborSearch,
}

/// Evaluator bound to one immutable descriptor schema.
#[derive(Debug, Clone)]
pub struct DescriptorSpace {
    schema: DescriptorSchema,
    geometry: Option<DescriptorGeometry>,
}

impl DescriptorSpace {
    /// Bind an evaluator to a validated schema.
    pub fn new(schema: DescriptorSchema) -> Self {
        Self {
            schema,
            geometry: None,
        }
    }

    pub(crate) fn with_geometry(schema: DescriptorSchema, geometry: DescriptorGeometry) -> Self {
        Self {
            schema,
            geometry: Some(geometry),
        }
    }

    /// Immutable schema interpreted by this evaluator.
    pub fn schema(&self) -> &DescriptorSchema {
        &self.schema
    }

    /// Geometry contract carried by universal descriptor spaces.
    pub fn geometry(&self) -> Option<DescriptorGeometry> {
        self.geometry
    }

    /// Evaluate all invariant blocks and concatenate their normalized values.
    pub fn describe(
        &self,
        coordinates: ArrayView1<f64>,
        species: Option<&[u32]>,
    ) -> Result<DescriptorVector, DescriptorError> {
        validate_coordinates(coordinates, species)?;

        if let Some(geometry) = self.geometry {
            return universal::describe(&self.schema, geometry, coordinates, species);
        }
        if self
            .schema
            .blocks
            .iter()
            .any(|block| block.kind.requires_geometry())
        {
            return Err(DescriptorError::UniversalGeometryRequired);
        }

        let mut values = Vec::new();
        let mut metadata = Vec::with_capacity(self.schema.blocks.len());
        for (block_index, block) in self.schema.blocks.iter().copied().enumerate() {
            let local = match block.kind {
                DescriptorBlockKind::SoapMean
                | DescriptorBlockKind::SoapVariance
                | DescriptorBlockKind::SoapLeftover => {
                    local_spectra_z(coordinates, block.soap, species)
                }
                DescriptorBlockKind::AceNu3Mean => local_nu3_z(coordinates, block.soap, species),
                DescriptorBlockKind::PairRadial
                | DescriptorBlockKind::ThreeBodyAngular
                | DescriptorBlockKind::GraphTopology
                | DescriptorBlockKind::InvariantSoapMean
                | DescriptorBlockKind::InvariantAceNu3Mean
                | DescriptorBlockKind::ChiralMoment => unreachable!(),
            };
            let mut aggregated = match block.kind {
                DescriptorBlockKind::SoapMean => column_mean(&local, 0),
                DescriptorBlockKind::SoapVariance => column_variance(&local),
                DescriptorBlockKind::AceNu3Mean => {
                    column_mean(&local, block.soap.feat_dim(species))
                }
                DescriptorBlockKind::SoapLeftover => leftover_stack(&local, species),
                DescriptorBlockKind::PairRadial
                | DescriptorBlockKind::ThreeBodyAngular
                | DescriptorBlockKind::GraphTopology
                | DescriptorBlockKind::InvariantSoapMean
                | DescriptorBlockKind::InvariantAceNu3Mean
                | DescriptorBlockKind::ChiralMoment => unreachable!(),
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
                normalization: L2_NORMALIZATION_SCHEMA,
            });
        }
        Ok(DescriptorVector {
            schema_name: self.schema.name.clone(),
            schema_version: self.schema.version,
            values,
            blocks: metadata,
        })
    }

    /// Evaluate one fixed-dimensional invariant row for every atomic centre.
    ///
    /// Universal local rows use the same geometry, species sketch, ordered
    /// blocks, and soft normalization as the global descriptor. Their final
    /// norm is at most one, so one schema-bound Euclidean radius has stable
    /// meaning for clusters, molecules, and surfaces.
    pub fn describe_local(
        &self,
        coordinates: ArrayView1<f64>,
        species: Option<&[u32]>,
    ) -> Result<Array2<f64>, DescriptorError> {
        validate_coordinates(coordinates, species)?;
        let geometry = self
            .geometry
            .ok_or(DescriptorError::UniversalGeometryRequired)?;
        universal::describe_local(&self.schema, geometry, coordinates, species)
    }

    /// Evaluate the Cartesian Jacobian of the normalized descriptor by central differences.
    pub fn jacobian_fd(
        &self,
        coordinates: ArrayView1<f64>,
        species: Option<&[u32]>,
        step: f64,
    ) -> Result<Array2<f64>, DescriptorError> {
        if !step.is_finite() || step <= 0.0 {
            return Err(DescriptorError::InvalidFiniteDifferenceStep);
        }
        let descriptor_dimension = self.describe(coordinates, species)?.values.len();
        let coordinate_dimension = coordinates.len();
        let mut jacobian = Array2::zeros((descriptor_dimension, coordinate_dimension));
        let mut plus = coordinates.to_owned();
        let mut minus = plus.clone();
        for column in 0..coordinate_dimension {
            plus[column] += step;
            minus[column] -= step;
            let plus_descriptor = self.describe(plus.view(), species)?;
            let minus_descriptor = self.describe(minus.view(), species)?;
            for row in 0..descriptor_dimension {
                jacobian[[row, column]] =
                    (plus_descriptor.values[row] - minus_descriptor.values[row]) / (2.0 * step);
            }
            plus[column] = coordinates[column];
            minus[column] = coordinates[column];
        }
        Ok(jacobian)
    }

    /// Evaluate the analytic Cartesian Jacobian of every normalized block.
    pub fn jacobian_analytic(
        &self,
        coordinates: ArrayView1<f64>,
        species: Option<&[u32]>,
    ) -> Result<Array2<f64>, DescriptorError> {
        if self.geometry.is_some() {
            return self.jacobian_fd(coordinates, species, 1e-6);
        }
        let descriptor = self.describe(coordinates, species)?;
        let coordinate_dimension = coordinates.len();
        let mut jacobian = Array2::zeros((descriptor.values.len(), coordinate_dimension));
        let mut output_offset = 0;
        for block in self.schema.blocks.iter().copied() {
            let (raw, raw_jacobian) = match block.kind {
                DescriptorBlockKind::SoapMean => {
                    let local = local_spectra_z(coordinates, block.soap, species);
                    let local_jacobian = jacobian_z(coordinates, block.soap, species);
                    mean_with_jacobian(&local, 0, &local_jacobian, local.ncols())
                }
                DescriptorBlockKind::SoapVariance => {
                    let local = local_spectra_z(coordinates, block.soap, species);
                    let local_jacobian = jacobian_z(coordinates, block.soap, species);
                    variance_with_jacobian(&local, &local_jacobian)
                }
                DescriptorBlockKind::AceNu3Mean => {
                    let local = local_nu3_z(coordinates, block.soap, species);
                    let soap_dimension = block.soap.feat_dim(species);
                    let local_jacobian = jacobian_ace(coordinates, block.soap, species);
                    mean_with_jacobian(
                        &local,
                        soap_dimension,
                        &local_jacobian,
                        local.ncols() - soap_dimension,
                    )
                }
                DescriptorBlockKind::SoapLeftover => {
                    return self.jacobian_fd(coordinates, species, 1e-6);
                }
                DescriptorBlockKind::PairRadial
                | DescriptorBlockKind::ThreeBodyAngular
                | DescriptorBlockKind::GraphTopology
                | DescriptorBlockKind::InvariantSoapMean
                | DescriptorBlockKind::InvariantAceNu3Mean
                | DescriptorBlockKind::ChiralMoment => {
                    return Err(DescriptorError::UniversalGeometryRequired);
                }
            };
            debug_assert_eq!(raw_jacobian.ncols(), coordinate_dimension);
            debug_assert_eq!(raw_jacobian.nrows(), raw.len());
            write_normalized_jacobian(&raw, &raw_jacobian, &mut jacobian, output_offset);
            output_offset += raw.len();
        }
        Ok(jacobian)
    }
}

fn validate_coordinates(
    coordinates: ArrayView1<'_, f64>,
    species: Option<&[u32]>,
) -> Result<(), DescriptorError> {
    if coordinates.is_empty() || !coordinates.len().is_multiple_of(3) {
        return Err(DescriptorError::CoordinateDimension {
            actual: coordinates.len(),
        });
    }
    if let Some(index) = coordinates.iter().position(|value| !value.is_finite()) {
        return Err(DescriptorError::NonFiniteCoordinate { index });
    }
    let atoms = coordinates.len() / 3;
    if let Some(species) = species
        && species.len() != atoms
    {
        return Err(DescriptorError::SpeciesDimension {
            expected: atoms,
            actual: species.len(),
        });
    }
    Ok(())
}

fn mean_with_jacobian(
    local: &Array2<f64>,
    local_start: usize,
    local_jacobian: &Array2<f64>,
    jacobian_stride: usize,
) -> (Vec<f64>, Array2<f64>) {
    let atoms = local.nrows();
    let dimension = local.ncols() - local_start;
    let coordinate_dimension = local_jacobian.ncols();
    let raw = column_mean(local, local_start);
    let mut jacobian = Array2::zeros((dimension, coordinate_dimension));
    for atom in 0..atoms {
        for row in 0..dimension {
            for column in 0..coordinate_dimension {
                jacobian[[row, column]] +=
                    local_jacobian[[atom * jacobian_stride + row, column]] / atoms as f64;
            }
        }
    }
    (raw, jacobian)
}

fn variance_with_jacobian(
    local: &Array2<f64>,
    local_jacobian: &Array2<f64>,
) -> (Vec<f64>, Array2<f64>) {
    let atoms = local.nrows();
    let dimension = local.ncols();
    let coordinate_dimension = local_jacobian.ncols();
    let mean = column_mean(local, 0);
    let raw = column_variance(local);
    let mut jacobian = Array2::zeros((dimension, coordinate_dimension));
    for atom in 0..atoms {
        for row in 0..dimension {
            let scale = 2.0 * (local[[atom, row]] - mean[row]) / atoms as f64;
            for column in 0..coordinate_dimension {
                jacobian[[row, column]] += scale * local_jacobian[[atom * dimension + row, column]];
            }
        }
    }
    (raw, jacobian)
}

fn write_normalized_jacobian(
    raw: &[f64],
    raw_jacobian: &Array2<f64>,
    output: &mut Array2<f64>,
    offset: usize,
) {
    let norm_squared = raw.iter().map(|value| value * value).sum::<f64>();
    if norm_squared == 0.0 {
        return;
    }
    let norm = norm_squared.sqrt();
    for column in 0..raw_jacobian.ncols() {
        let radial_derivative = raw
            .iter()
            .enumerate()
            .map(|(row, value)| value * raw_jacobian[[row, column]])
            .sum::<f64>();
        for row in 0..raw.len() {
            output[[offset + row, column]] = raw_jacobian[[row, column]] / norm
                - raw[row] * radial_derivative / (norm_squared * norm);
        }
    }
}

fn leftover_stack(local: &Array2<f64>, species: Option<&[u32]>) -> Vec<f64> {
    let rows = local.nrows();
    let cols = local.ncols();
    let mut leftover = vec![0.0; rows * cols];
    if rows == 0 || cols == 0 {
        return leftover;
    }
    match species {
        None => {
            let mean = column_mean(local, 0);
            for row in 0..rows {
                for column in 0..cols {
                    leftover[row * cols + column] = local[[row, column]] - mean[column];
                }
            }
        }
        Some(labels) => {
            let mut channels = Vec::new();
            for &atomic_number in labels.iter().take(rows) {
                if !channels.contains(&atomic_number) {
                    channels.push(atomic_number);
                }
            }
            let mut mean = vec![vec![0.0; cols]; channels.len()];
            let mut count = vec![0.0; channels.len()];
            for row in 0..rows {
                let Some(channel) = labels.get(row).and_then(|atomic_number| {
                    channels.iter().position(|value| value == atomic_number)
                }) else {
                    continue;
                };
                count[channel] += 1.0;
                for column in 0..cols {
                    mean[channel][column] += local[[row, column]];
                }
            }
            for (channel, occupancy) in count.iter().copied().enumerate() {
                if occupancy > 0.0 {
                    for column in 0..cols {
                        mean[channel][column] /= occupancy;
                    }
                }
            }
            for row in 0..rows {
                let Some(channel) = labels.get(row).and_then(|atomic_number| {
                    channels.iter().position(|value| value == atomic_number)
                }) else {
                    continue;
                };
                for column in 0..cols {
                    leftover[row * cols + column] = local[[row, column]] - mean[channel][column];
                }
            }
        }
    }
    leftover
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
