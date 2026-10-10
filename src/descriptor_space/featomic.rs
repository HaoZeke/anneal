//! Multiscale SOAP provider backed by featomic calculators.

use super::{
    DescriptorError, DescriptorOutputSpec, DescriptorProviderContract, DescriptorProviderError,
    DescriptorProviderInput, InvariantDescriptorProvider,
};
use featomic::systems::UnitCell;
use featomic::{CalculationOptions, Calculator, Matrix3, SimpleSystem, System, Vector3D};
use metatensor::Labels;
use ndarray::Array2;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::HashMap;

const CALCULATOR: &str = "soap_power_spectrum";
/// Stable schema name for featomic-backed multiscale SOAP features.
pub const FEATOMIC_SOAP_SCHEMA: &str = "featomic-soap-invariant";
/// Schema version of [`FEATOMIC_SOAP_SCHEMA`].
pub const FEATOMIC_SOAP_VERSION: u32 = 1;
/// Metric preprocessing applied to each featomic SOAP scale.
pub const FEATOMIC_SOAP_NORMALIZATION: &str = "per-scale-local-contractive-l2-mean-v1";
const CONFIG_PREFIX: &[u8] = b"ANNEAL\0FEATOMIC_SOAP_PROVIDER\0featomic=0.6.0\0";

thread_local! {
    static CALCULATORS: RefCell<HashMap<String, Calculator>> = RefCell::new(HashMap::new());
}

/// One radial and angular resolution in a multiscale SOAP provider.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatomicSoapScale {
    cutoff: f64,
    max_radial: usize,
    max_angular: usize,
    density_width: f64,
    smoothing_width: f64,
}

impl FeatomicSoapScale {
    /// Construct a scale with widths tied to its cutoff.
    pub fn new(
        cutoff: f64,
        max_radial: usize,
        max_angular: usize,
    ) -> Result<Self, FeatomicSoapProviderError> {
        Self::with_widths(
            cutoff,
            max_radial,
            max_angular,
            (0.15 * cutoff).min(0.35),
            (0.20 * cutoff).min(0.50),
        )
    }

    /// Construct a scale with explicit Gaussian-density and cutoff widths.
    pub fn with_widths(
        cutoff: f64,
        max_radial: usize,
        max_angular: usize,
        density_width: f64,
        smoothing_width: f64,
    ) -> Result<Self, FeatomicSoapProviderError> {
        if !cutoff.is_finite()
            || cutoff <= 0.0
            || !density_width.is_finite()
            || density_width <= 0.0
            || !smoothing_width.is_finite()
            || smoothing_width <= 0.0
            || smoothing_width >= cutoff
        {
            return Err(FeatomicSoapProviderError::InvalidScale);
        }
        Ok(Self {
            cutoff,
            max_radial,
            max_angular,
            density_width,
            smoothing_width,
        })
    }

    /// Spherical cutoff in reduced descriptor coordinates.
    pub fn cutoff(self) -> f64 {
        self.cutoff
    }

    /// Largest radial basis index; the basis contains `max_radial + 1` rows.
    pub fn max_radial(self) -> usize {
        self.max_radial
    }

    /// Largest angular channel.
    pub fn max_angular(self) -> usize {
        self.max_angular
    }

    /// Gaussian atomic-density width in reduced descriptor coordinates.
    pub fn density_width(self) -> f64 {
        self.density_width
    }

    /// Shifted-cosine smoothing width in reduced descriptor coordinates.
    pub fn smoothing_width(self) -> f64 {
        self.smoothing_width
    }

    fn properties(self) -> Option<usize> {
        let radial = self.max_radial.checked_add(1)?;
        self.max_angular
            .checked_add(1)?
            .checked_mul(radial.checked_mul(radial)?)
    }

    fn hyperparameters(self) -> String {
        format!(
            "{{\"cutoff\":{{\"radius\":{},\"smoothing\":{{\"type\":\"ShiftedCosine\",\"width\":{}}}}},\
             \"density\":{{\"type\":\"Gaussian\",\"width\":{},\"center_atom_weight\":1.0}},\
             \"basis\":{{\"type\":\"TensorProduct\",\"max_angular\":{},\"radial\":{{\"type\":\"Gto\",\"max_radial\":{}}}}}}}",
            self.cutoff,
            self.smoothing_width,
            self.density_width,
            self.max_angular,
            self.max_radial,
        )
    }
}

/// Construction failure for a featomic SOAP provider.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum FeatomicSoapProviderError {
    /// At least one chemical species channel is required.
    #[error("featomic SOAP provider requires at least one species channel")]
    EmptySpecies,
    /// Atomic numbers must be positive and fit featomic's signed label type.
    #[error("atomic number {0} is outside the supported featomic label range")]
    InvalidAtomicNumber(u32),
    /// At least one radial scale is required.
    #[error("featomic SOAP provider requires at least one scale")]
    EmptyScales,
    /// Cutoff and density widths must define a finite positive smooth cutoff.
    #[error("invalid featomic SOAP scale")]
    InvalidScale,
    /// The selected basis and species channels exceed addressable memory.
    #[error("featomic SOAP provider dimension overflow")]
    DimensionOverflow,
    /// Generic invariant-provider contract rejected the derived configuration.
    #[error(transparent)]
    Contract(#[from] DescriptorError),
}

/// Fixed-channel multiscale SOAP features computed by featomic.
///
/// Species channels and all calculator hyperparameters are fixed at
/// construction. Missing `(center, neighbor_1, neighbor_2)` blocks are zero,
/// so the feature dimension does not depend on which species or coordination
/// shells happen to be present in one structure.
#[derive(Debug)]
pub struct FeatomicSoapProvider {
    species_channels: Vec<u32>,
    scales: Vec<FeatomicSoapScale>,
    keys: Vec<[i32; 3]>,
    contract: DescriptorProviderContract,
}

impl FeatomicSoapProvider {
    /// Bind fixed species channels and ordered SOAP resolutions.
    pub fn new(
        mut species_channels: Vec<u32>,
        scales: Vec<FeatomicSoapScale>,
    ) -> Result<Self, FeatomicSoapProviderError> {
        species_channels.sort_unstable();
        species_channels.dedup();
        if species_channels.is_empty() {
            return Err(FeatomicSoapProviderError::EmptySpecies);
        }
        for &atomic_number in &species_channels {
            if atomic_number == 0 || i32::try_from(atomic_number).is_err() {
                return Err(FeatomicSoapProviderError::InvalidAtomicNumber(
                    atomic_number,
                ));
            }
        }
        if scales.is_empty() {
            return Err(FeatomicSoapProviderError::EmptyScales);
        }

        let keys = selected_keys(&species_channels);
        let dimension = scales.iter().try_fold(0usize, |dimension, scale| {
            let scale_dimension = keys
                .len()
                .checked_mul(
                    scale
                        .properties()
                        .ok_or(FeatomicSoapProviderError::DimensionOverflow)?,
                )
                .ok_or(FeatomicSoapProviderError::DimensionOverflow)?;
            dimension
                .checked_add(scale_dimension)
                .ok_or(FeatomicSoapProviderError::DimensionOverflow)
        })?;
        let model_digest = configuration_digest(&species_channels, &scales);
        let output = DescriptorOutputSpec::new("feature", dimension)?;
        let interaction_range = scales.iter().map(|scale| scale.cutoff).fold(0.0, f64::max);
        let contract = DescriptorProviderContract::new(
            FEATOMIC_SOAP_SCHEMA,
            FEATOMIC_SOAP_VERSION,
            model_digest,
            output.clone(),
            Some(output),
            interaction_range,
            FEATOMIC_SOAP_NORMALIZATION,
        )?;
        Ok(Self {
            species_channels,
            scales,
            keys,
            contract,
        })
    }

    /// Ordered chemical species channels encoded by every output row.
    pub fn species_channels(&self) -> &[u32] {
        &self.species_channels
    }

    /// Ordered short- and medium-range SOAP resolutions.
    pub fn scales(&self) -> &[FeatomicSoapScale] {
        &self.scales
    }

    fn atomic_features(
        &self,
        input: DescriptorProviderInput<'_>,
    ) -> Result<Array2<f64>, DescriptorProviderError> {
        let atoms = input.coordinates().len() / 3;
        let mut systems = self.system(input)?;
        let labels = Labels::new(
            ["center_type", "neighbor_1_type", "neighbor_2_type"],
            &self.keys,
        );
        let dimension = self.contract.system_output().dimension();
        let mut features = Array2::zeros((atoms, dimension));
        let scale_weight = 1.0 / (self.scales.len() as f64).sqrt();
        let mut scale_offset = 0;

        for &scale in &self.scales {
            let properties = scale.properties().ok_or_else(|| {
                DescriptorProviderError::new("featomic SOAP property dimension overflow")
            })?;
            let hyperparameters = scale.hyperparameters();
            // Take the calculator out of the thread-local cache for the
            // duration of the call. featomic computes on its own rayon pool,
            // and a work-stealing thread can re-enter this function on the
            // same OS thread while the borrow is live: measured as a
            // "RefCell already borrowed" abort of the coordinator on the
            // first LJ38 cooperative smoke. Nothing borrows across `compute`.
            let mut calculator = CALCULATORS
                .with(|calculators| calculators.borrow_mut().remove(&hyperparameters))
                .unwrap_or_else(|| {
                    Calculator::new(CALCULATOR, hyperparameters.clone())
                        .expect("validated featomic SOAP hyperparameters")
                });
            let descriptor = calculator.compute(
                &mut systems,
                CalculationOptions {
                    selected_keys: Some(&labels),
                    ..Default::default()
                },
            );
            CALCULATORS.with(|calculators| {
                calculators
                    .borrow_mut()
                    .insert(hyperparameters.clone(), calculator);
            });
            let descriptor = descriptor.map_err(|error| {
                DescriptorProviderError::new(format!("featomic SOAP evaluation failed: {error}"))
            })?;
            if descriptor.keys().count() != self.keys.len() {
                return Err(DescriptorProviderError::new(
                    "featomic SOAP returned an unexpected key count",
                ));
            }

            for (block_index, expected_key) in self.keys.iter().enumerate() {
                let actual_key = &descriptor.keys()[block_index];
                if actual_key.len() != 3
                    || actual_key[0].i32() != expected_key[0]
                    || actual_key[1].i32() != expected_key[1]
                    || actual_key[2].i32() != expected_key[2]
                {
                    return Err(DescriptorProviderError::new(
                        "featomic SOAP returned keys in an incompatible layout",
                    ));
                }
                let block = descriptor.block_by_id(block_index);
                let values = block.values().to_array();
                if values.ndim() != 2 || values.shape()[1] != properties {
                    return Err(DescriptorProviderError::new(
                        "featomic SOAP returned an incompatible property layout",
                    ));
                }
                let samples = block.samples();
                let atom_column = samples
                    .names()
                    .iter()
                    .position(|&name| name == "atom")
                    .ok_or_else(|| {
                        DescriptorProviderError::new(
                            "featomic SOAP samples do not identify their centre atom",
                        )
                    })?;
                let flat = values.iter().copied().collect::<Vec<_>>();
                let property_offset = scale_offset + block_index * properties;
                for sample in 0..values.shape()[0] {
                    let atom = samples[sample][atom_column].usize();
                    if atom >= atoms {
                        return Err(DescriptorProviderError::new(
                            "featomic SOAP returned an out-of-range centre atom",
                        ));
                    }
                    for property in 0..properties {
                        features[[atom, property_offset + property]] =
                            flat[sample * properties + property];
                    }
                }
            }

            let scale_dimension = self.keys.len() * properties;
            for atom in 0..atoms {
                let norm_squared = (scale_offset..scale_offset + scale_dimension)
                    .map(|column| features[[atom, column]].powi(2))
                    .sum::<f64>();
                let factor = scale_weight / (1.0 + norm_squared).sqrt();
                for column in scale_offset..scale_offset + scale_dimension {
                    features[[atom, column]] *= factor;
                }
            }
            scale_offset += scale_dimension;
        }
        Ok(features)
    }

    fn system(
        &self,
        input: DescriptorProviderInput<'_>,
    ) -> Result<Vec<Box<dyn System>>, DescriptorProviderError> {
        let geometry = input.geometry();
        let periodic = geometry.periodic();
        let cell = if periodic.iter().all(|&value| !value) {
            UnitCell::infinite()
        } else if periodic.iter().all(|&value| value) {
            let matrix = geometry.cell().ok_or_else(|| {
                DescriptorProviderError::new("periodic featomic SOAP input has no cell")
            })?;
            let scale = geometry.length_scale();
            UnitCell::from(Matrix3::new([
                [matrix[0] / scale, matrix[1] / scale, matrix[2] / scale],
                [matrix[3] / scale, matrix[4] / scale, matrix[5] / scale],
                [matrix[6] / scale, matrix[7] / scale, matrix[8] / scale],
            ]))
        } else {
            return Err(DescriptorProviderError::new(
                "featomic SOAP requires either finite or fully periodic geometry",
            ));
        };

        let atoms = input.coordinates().len() / 3;
        let species = match input.species() {
            Some(species) => species.to_vec(),
            None if self.species_channels.len() == 1 => vec![self.species_channels[0]; atoms],
            None => {
                return Err(DescriptorProviderError::new(
                    "multi-species featomic SOAP requires atomic numbers",
                ));
            }
        };
        let length_scale = geometry.length_scale();
        let coordinates = input.coordinates();
        let mut system = SimpleSystem::new(cell);
        for atom in 0..atoms {
            let atomic_number = species[atom];
            if !self.species_channels.contains(&atomic_number) {
                return Err(DescriptorProviderError::new(format!(
                    "atomic number {atomic_number} is absent from the provider channels"
                )));
            }
            system.add_atom(
                atomic_number as i32,
                Vector3D::new(
                    coordinates[3 * atom] / length_scale,
                    coordinates[3 * atom + 1] / length_scale,
                    coordinates[3 * atom + 2] / length_scale,
                ),
            );
        }
        Ok(vec![Box::new(system)])
    }
}

impl InvariantDescriptorProvider for FeatomicSoapProvider {
    fn contract(&self) -> &DescriptorProviderContract {
        &self.contract
    }

    fn describe_system(
        &self,
        input: DescriptorProviderInput<'_>,
    ) -> Result<Vec<f64>, DescriptorProviderError> {
        let atomic = self.atomic_features(input)?;
        let atoms = atomic.nrows();
        let mut system = vec![0.0; atomic.ncols()];
        for atom in 0..atoms {
            for column in 0..atomic.ncols() {
                system[column] += atomic[[atom, column]] / atoms as f64;
            }
        }
        Ok(system)
    }

    fn describe_atoms(
        &self,
        input: DescriptorProviderInput<'_>,
    ) -> Result<Option<Array2<f64>>, DescriptorProviderError> {
        self.atomic_features(input).map(Some)
    }
}

fn selected_keys(species_channels: &[u32]) -> Vec<[i32; 3]> {
    let mut keys = Vec::new();
    for &center in species_channels {
        for (left_index, &left) in species_channels.iter().enumerate() {
            for &right in &species_channels[left_index..] {
                keys.push([center as i32, left as i32, right as i32]);
            }
        }
    }
    keys
}

fn configuration_digest(species_channels: &[u32], scales: &[FeatomicSoapScale]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(CONFIG_PREFIX);
    hasher.update((species_channels.len() as u64).to_be_bytes());
    for &atomic_number in species_channels {
        hasher.update(atomic_number.to_be_bytes());
    }
    hasher.update((scales.len() as u64).to_be_bytes());
    for scale in scales {
        hasher.update(scale.cutoff.to_bits().to_be_bytes());
        hasher.update((scale.max_radial as u64).to_be_bytes());
        hasher.update((scale.max_angular as u64).to_be_bytes());
        hasher.update(scale.density_width.to_bits().to_be_bytes());
        hasher.update(scale.smoothing_width.to_bits().to_be_bytes());
    }
    hasher.update(FEATOMIC_SOAP_NORMALIZATION.as_bytes());
    hasher.finalize().into()
}
