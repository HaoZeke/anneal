//! Canonical reduced-unit Lennard-Jones catalog configuration.

use std::collections::BTreeMap;

use ndarray::ArrayView1;
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
