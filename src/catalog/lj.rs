//! Canonical reduced-unit Lennard-Jones catalog configuration.

use std::collections::BTreeMap;

use ndarray::ArrayView1;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use sha2::{Digest, Sha256};

use super::{
    DescriptorSignature, EngineSignature, FreshEvaluation, SystemSignature, ValidatorConfig,
};
use crate::descriptor_space::{
    DescriptorBlockKind, DescriptorBlockSpec, DescriptorSchema, DescriptorSpace,
};
use crate::potentials::PairPotential;

/// Invalid Lennard-Jones catalog preset input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LjCatalogPresetError {
    /// At least two sites are required for a Lennard-Jones cluster.
    #[error("Lennard-Jones catalogs require at least two sites")]
    InvalidSiteCount,
    /// Cartesian length must be exactly three times the site count.
    #[error("Lennard-Jones Cartesian dimension is inconsistent")]
    CoordinateDimension,
    /// Descriptor dimension must be positive.
    #[error("Lennard-Jones descriptor dimension must be positive")]
    DescriptorDimension,
    /// A development reference does not contain exactly one finite XYZ row per site.
    #[error("invalid Lennard-Jones reference coordinates")]
    ReferenceCoordinates,
    /// A development perturbation has invalid dimensions or scale.
    #[error("invalid Lennard-Jones calibration perturbation")]
    CalibrationPerturbation,
}

/// Maximum energy difference used by the development-only exact-minimum check.
pub const CALIBRATION_ENERGY_TOLERANCE: f64 = 1e-7;

/// Maximum gradient norm used by the development-only exact-minimum check.
pub const CALIBRATION_GRADIENT_TOLERANCE: f64 = 1e-5;

/// Maximum IRA Hausdorff distance used by the development-only identity check.
pub const CALIBRATION_IRA_TOLERANCE: f64 = 1e-4;

/// Parse the whitespace-separated three-column coordinate format used by the
/// Cambridge Cluster Database Lennard-Jones reference files.
pub fn parse_reference_coordinates(
    text: &str,
    n_points: usize,
) -> Result<Vec<f64>, LjCatalogPresetError> {
    if n_points < 2 {
        return Err(LjCatalogPresetError::InvalidSiteCount);
    }
    let mut coordinates = Vec::with_capacity(3 * n_points);
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let values = line
            .split_whitespace()
            .map(str::parse::<f64>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| LjCatalogPresetError::ReferenceCoordinates)?;
        if values.len() != 3 || values.iter().any(|value| !value.is_finite()) {
            return Err(LjCatalogPresetError::ReferenceCoordinates);
        }
        coordinates.extend(values);
    }
    if coordinates.len() != 3 * n_points {
        return Err(LjCatalogPresetError::ReferenceCoordinates);
    }
    Ok(coordinates)
}

/// Apply a deterministic independent Gaussian perturbation while preserving
/// the reference centroid exactly on each Cartesian axis.
pub fn perturb_reference(
    reference: &[f64],
    n_points: usize,
    seed: u64,
    sigma: f64,
) -> Result<Vec<f64>, LjCatalogPresetError> {
    if n_points < 2
        || reference.len() != 3 * n_points
        || reference.iter().any(|value| !value.is_finite())
        || !sigma.is_finite()
        || sigma <= 0.0
    {
        return Err(LjCatalogPresetError::CalibrationPerturbation);
    }
    let distribution =
        Normal::new(0.0, sigma).map_err(|_| LjCatalogPresetError::CalibrationPerturbation)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut displacement = (0..reference.len())
        .map(|_| distribution.sample(&mut rng))
        .collect::<Vec<f64>>();
    for axis in 0..3 {
        let mean = (0..n_points)
            .map(|atom| displacement[3 * atom + axis])
            .sum::<f64>()
            / n_points as f64;
        for atom in 0..n_points {
            displacement[3 * atom + axis] -= mean;
        }
    }
    Ok(reference
        .iter()
        .zip(displacement)
        .map(|(coordinate, delta)| coordinate + delta)
        .collect())
}

/// Whether a development quench has independent evidence for the declared
/// Lennard-Jones minimum used to calibrate a descriptor census radius.
pub fn accepts_calibration_minimum(
    reference_energy: f64,
    candidate_energy: f64,
    gradient_norm: f64,
    ira_distance: f64,
) -> bool {
    [
        reference_energy,
        candidate_energy,
        gradient_norm,
        ira_distance,
    ]
    .iter()
    .all(|value| value.is_finite())
        && (candidate_energy - reference_energy).abs() <= CALIBRATION_ENERGY_TOLERANCE
        && gradient_norm <= CALIBRATION_GRADIENT_TOLERANCE
        && ira_distance <= CALIBRATION_IRA_TOLERANCE
}

/// Versioned multiscale SOAP/ACE space used by every LJ catalog size.
pub fn descriptor_space() -> DescriptorSpace {
    DescriptorSpace::new(
        DescriptorSchema::new(
            "jcc-lj-multiscale-soap-ace",
            1,
            vec![
                DescriptorBlockSpec::new(DescriptorBlockKind::SoapMean, 3, 6, 3.5)
                    .expect("fixed LJ SOAP mean block is valid"),
                DescriptorBlockSpec::new(DescriptorBlockKind::SoapVariance, 3, 6, 3.5)
                    .expect("fixed LJ SOAP variance block is valid"),
                DescriptorBlockSpec::new(DescriptorBlockKind::AceNu3Mean, 2, 3, 2.5)
                    .expect("fixed LJ ACE block is valid"),
            ],
        )
        .expect("fixed LJ descriptor schema is valid"),
    )
}

/// Canonical system signature for one reduced-unit LJ cluster size.
pub fn system_signature(n_points: usize) -> Result<SystemSignature, LjCatalogPresetError> {
    if n_points < 2 {
        return Err(LjCatalogPresetError::InvalidSiteCount);
    }
    let coordinate_dim = n_points
        .checked_mul(3)
        .and_then(|dimension| u64::try_from(dimension).ok())
        .ok_or(LjCatalogPresetError::CoordinateDimension)?;
    let descriptor = descriptor_space();
    let mut hyperparameters = BTreeMap::new();
    hyperparameters.insert(
        "blocks".into(),
        "soap-mean,soap-variance,ace-nu3-mean".into(),
    );
    hyperparameters.insert("normalization".into(), "block-l2-v1".into());
    hyperparameters.insert("soap".into(), "nmax=3,lmax=6,cutoff=3.5".into());
    hyperparameters.insert("ace".into(), "nu=3,nmax=2,lmax=3,cutoff=2.5".into());
    let mut engine_hasher = Sha256::new();
    engine_hasher.update(b"lennard-jones-reduced-v1;epsilon=1;sigma=1;cutoff=none");
    Ok(SystemSignature {
        atomic_numbers: vec![18; n_points],
        coordinate_dim,
        group_labels: (0..n_points)
            .map(|index| {
                u32::try_from(index).map_err(|_| LjCatalogPresetError::CoordinateDimension)
            })
            .collect::<Result<Vec<_>, _>>()?,
        group_schema: "independent-lj-sites-v1".into(),
        frozen_mask: vec![false; n_points],
        cell: None,
        periodic: [false; 3],
        length_scale: 1.0,
        energy_scale: 1.0,
        engine: EngineSignature {
            kind: "lennard-jones-reduced-v1".into(),
            config_digest: engine_hasher.finalize().into(),
            external_inputs: BTreeMap::new(),
        },
        descriptor: DescriptorSignature {
            schema: descriptor.schema().name().into(),
            version: descriptor.schema().version(),
            hyperparameters,
            species_channels: vec![18],
        },
        validation_schema_version: 1,
    })
}

/// Receiving-side validation settings for one LJ coordinate reference.
pub fn validator_config(
    reference_coordinates: &[f64],
    descriptor_dim: usize,
) -> Result<ValidatorConfig, LjCatalogPresetError> {
    if reference_coordinates.len() < 6 || !reference_coordinates.len().is_multiple_of(3) {
        return Err(LjCatalogPresetError::CoordinateDimension);
    }
    if descriptor_dim == 0 {
        return Err(LjCatalogPresetError::DescriptorDimension);
    }
    Ok(ValidatorConfig {
        reference_coordinates: reference_coordinates.to_vec(),
        descriptor_dim,
        min_separation: 0.5,
        coordinate_tolerance: 1e-10,
        max_gradient_norm: 1e-5,
        energy_abs_tolerance: 1e-9,
        energy_rel_tolerance: 1e-10,
    })
}

/// Deterministic nonoverlapping reference used to configure a coordinator.
pub fn reference_coordinates(n_points: usize) -> Result<Vec<f64>, LjCatalogPresetError> {
    if n_points < 2 {
        return Err(LjCatalogPresetError::InvalidSiteCount);
    }
    let mut coordinates = Vec::with_capacity(3 * n_points);
    for index in 0..n_points {
        coordinates.extend_from_slice(&[1.2 * index as f64, 0.0, 0.0]);
    }
    Ok(coordinates)
}

/// Fresh reduced-unit LJ energy and forces for receiving-side validation.
pub fn fresh_evaluation(n_points: usize, coordinates: &[f64]) -> Result<FreshEvaluation, String> {
    if n_points < 2 || coordinates.len() != 3 * n_points {
        return Err("Lennard-Jones coordinate dimension mismatch".into());
    }
    let potential = PairPotential::lennard_jones(n_points);
    let (energy, gradient) = potential.value_and_gradient(ArrayView1::from(coordinates));
    Ok(FreshEvaluation {
        energy,
        forces: gradient.iter().map(|value| -*value).collect(),
    })
}
