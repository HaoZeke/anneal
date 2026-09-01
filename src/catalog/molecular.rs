//! GFN2-xTB water-cluster catalog identity.
//!
//! The catalog uses the same fixed-dimensional multiscale invariant schema as
//! clusters and surfaces through [`descriptor_space`]. The GFN2 engine,
//! atomic species, flexible Cartesian degrees of freedom, and coordinate
//! dimension remain in the system signature, so sharing a descriptor schema
//! never merges PES state. Water groups remain proposal metadata and do not
//! impose rigid distances on GFN2 minima.
//! [`leftover_space`] remains the molecular proposal feature: its stacked
//! species-conditioned residual `p_i - mu_z(i)` is not basin identity.
//!
//! A complete visit still needs a loaded rgpot engine. Required engine
//! fields:
//! - `kind`: [`ENGINE_KIND`]
//! - `config_digest`: SHA-256 of the declared GFN2 handle
//!   (`method=3`, `accuracy=0.01`, `etemp=300`, `maxiter=500`,
//!   `net_electron_count=0`, `uhf=0`, `vacuum_box=60`)
//! - `external_inputs["libxtb_engine"]`: SHA-256 of the loaded
//!   `libxtb_engine.so`
//!
//! [`fresh_evaluation`] is a compile-time stub under both
//! `rgpot-ex` and the default feature set. anneal-core does not vendor
//! the GFN2 Hamiltonian, so this module does not invent an energy.

use std::collections::BTreeMap;

use ndarray::ArrayView1;
use sha2::{Digest, Sha256};

use super::{
    DescriptorSignature, EngineSignature, FreshEvaluation, SystemSignature, ValidatorConfig,
};
use crate::descriptor_space::{
    DescriptorBlockKind, DescriptorBlockSpec, DescriptorGeometry, DescriptorSchema,
    DescriptorSpace, UNIVERSAL_DESCRIPTOR_SCHEMA, UNIVERSAL_DESCRIPTOR_VERSION,
    universal_descriptor_space,
};
use crate::methods::cluster_hopping::covalent_radius;
use crate::soap::{SoapSpec, local_spectra_z};

/// Invalid GFN2-xTB water catalog preset input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum MolecularCatalogPresetError {
    /// At least one water molecule is required.
    #[error("GFN2-xTB water catalogs require at least one molecule")]
    InvalidMoleculeCount,
    /// Cartesian length must be exactly nine times the molecule count.
    #[error("water Cartesian dimension is inconsistent")]
    CoordinateDimension,
    /// Descriptor dimension must be positive.
    #[error("water descriptor dimension must be positive")]
    DescriptorDimension,
    /// Species vector must be `[8, 1, 1]` repeated once per molecule.
    #[error("water species vector is inconsistent")]
    SpeciesMismatch,
}

/// Stable engine family for the in-process rgpot GFN2-xTB handle.
pub const ENGINE_KIND: &str = "gfn2-xtb-rgpot-v1";

/// External-input name for the loaded GFN2 engine shared object.
pub const ENGINE_BINARY_INPUT: &str = "libxtb_engine";

/// xtb method code for GFN2 in the rgpot handle (`XtbConfig::method`).
pub const GFN2_METHOD: i32 = 3;

/// SCF accuracy resolving cold-start force noise below the catalog threshold.
pub const GFN2_ACCURACY: f64 = 0.01;

/// SCF iteration ceiling paired with [`GFN2_ACCURACY`].
pub const GFN2_MAX_ITERATIONS: i32 = 500;

/// Vacuum box edge used by `molecular_cluster` (A).
pub const VACUUM_BOX: f64 = 60.0;

/// Paper water-hexamer molecule count.
pub const WATER_HEXAMER_MOLECULES: usize = 6;

/// Isolated-water geometry used to seed `(H2O)m` (A).
const WATER: [[f64; 3]; 3] = [
    [0.0, 0.0, 0.0],
    [0.7572, 0.5865, 0.0],
    [-0.7572, 0.5865, 0.0],
];

/// Radial / angular leftover hop resolution (`soap_arm` / `step_away_cloud`).
const LEFTOVER_N_MAX: usize = 3;
const LEFTOVER_L_MAX: usize = 6;
const LEFTOVER_CUTOFF_PER_LENGTH: f64 = 3.5;

/// Covalent diameter used by `Config::for_molecular`.
const COVALENT_DIAMETER: f64 = 2.0;

/// Declared GFN2 handle configuration, excluding the engine binary.
const ENGINE_CONFIG: &[u8] = b"gfn2-xtb-rgpot-v1;method=3;accuracy=0.01;etemp=300;maxiter=500;net_electron_count=0;uhf=0;vacuum_box=60";

/// Flexible atomic degrees of freedom of the GFN2 water PES.
pub const GROUP_SCHEMA: &str = "flexible-water-atoms-v1";

/// Universal catalog descriptor schema shared across system families.
pub const DESCRIPTOR_SCHEMA: &str = UNIVERSAL_DESCRIPTOR_SCHEMA;

/// Schema version of [`DESCRIPTOR_SCHEMA`].
pub const DESCRIPTOR_VERSION: u32 = UNIVERSAL_DESCRIPTOR_VERSION;

/// Molecular proposal-only leftover SOAP schema.
pub const LEFTOVER_DESCRIPTOR_SCHEMA: &str = "jcc-water-soap-leftover";

/// Schema version of [`LEFTOVER_DESCRIPTOR_SCHEMA`].
pub const LEFTOVER_DESCRIPTOR_VERSION: u32 = 1;

/// Atomic numbers of `(H2O)m` in coordinate order.
pub fn water_species(n_molecules: usize) -> Result<Vec<u32>, MolecularCatalogPresetError> {
    if n_molecules == 0 {
        return Err(MolecularCatalogPresetError::InvalidMoleculeCount);
    }
    Ok((0..n_molecules).flat_map(|_| [8, 1, 1]).collect())
}

/// Molecular proposal groups of `(H2O)m`: one water per three consecutive atoms.
pub fn water_groups(n_molecules: usize) -> Result<Vec<Vec<usize>>, MolecularCatalogPresetError> {
    if n_molecules == 0 {
        return Err(MolecularCatalogPresetError::InvalidMoleculeCount);
    }
    Ok((0..n_molecules)
        .map(|molecule| (3 * molecule..3 * molecule + 3).collect())
        .collect())
}

/// Length scale of `Config::for_molecular` on the given species.
pub fn length_scale(species: &[u32]) -> Result<f64, MolecularCatalogPresetError> {
    if species.is_empty() {
        return Err(MolecularCatalogPresetError::InvalidMoleculeCount);
    }
    let radius = species
        .iter()
        .copied()
        .map(covalent_radius)
        .fold(0.0_f64, f64::max);
    let scale = COVALENT_DIAMETER * radius;
    if !(scale > 0.0) {
        return Err(MolecularCatalogPresetError::SpeciesMismatch);
    }
    Ok(scale)
}

/// Leftover SOAP spec used by the molecular hop on this species set.
pub fn leftover_spec(species: &[u32]) -> Result<SoapSpec, MolecularCatalogPresetError> {
    Ok(SoapSpec {
        n_max: LEFTOVER_N_MAX,
        l_max: LEFTOVER_L_MAX,
        rcut_nn: LEFTOVER_CUTOFF_PER_LENGTH * length_scale(species)?,
    })
}

/// Stacked leftover dimension `n_atoms * feat_dim(species)`.
pub fn leftover_descriptor_dim(species: &[u32]) -> Result<usize, MolecularCatalogPresetError> {
    let spec = leftover_spec(species)?;
    spec.feat_dim(Some(species))
        .checked_mul(species.len())
        .filter(|dimension| *dimension > 0)
        .ok_or(MolecularCatalogPresetError::DescriptorDimension)
}

/// Species-conditioned leftover `p_i - mu_z(i)` stacked over atoms.
pub fn leftover_values(
    coordinates: &[f64],
    species: &[u32],
) -> Result<Vec<f64>, MolecularCatalogPresetError> {
    if species.is_empty() {
        return Err(MolecularCatalogPresetError::InvalidMoleculeCount);
    }
    if coordinates.len() != 3 * species.len() {
        return Err(MolecularCatalogPresetError::CoordinateDimension);
    }
    if coordinates.iter().any(|value| !value.is_finite()) {
        return Err(MolecularCatalogPresetError::CoordinateDimension);
    }
    let spec = leftover_spec(species)?;
    let local = local_spectra_z(ArrayView1::from(coordinates), spec, Some(species));
    let n_atoms = local.nrows();
    let feat = local.ncols();
    if n_atoms != species.len() || feat == 0 {
        return Err(MolecularCatalogPresetError::DescriptorDimension);
    }
    let mut labels = Vec::new();
    for &atomic_number in species {
        if !labels.contains(&atomic_number) {
            labels.push(atomic_number);
        }
    }
    let mut mean = vec![vec![0.0; feat]; labels.len()];
    let mut count = vec![0.0; labels.len()];
    for atom in 0..n_atoms {
        let channel = labels
            .iter()
            .position(|&atomic_number| atomic_number == species[atom])
            .ok_or(MolecularCatalogPresetError::SpeciesMismatch)?;
        count[channel] += 1.0;
        for feature in 0..feat {
            mean[channel][feature] += local[[atom, feature]];
        }
    }
    for (channel, occupancy) in count.iter().copied().enumerate() {
        if occupancy > 0.0 {
            for feature in 0..feat {
                mean[channel][feature] /= occupancy;
            }
        }
    }
    let mut leftover = vec![0.0; n_atoms * feat];
    for atom in 0..n_atoms {
        let channel = labels
            .iter()
            .position(|&atomic_number| atomic_number == species[atom])
            .ok_or(MolecularCatalogPresetError::SpeciesMismatch)?;
        for feature in 0..feat {
            leftover[atom * feat + feature] = local[[atom, feature]] - mean[channel][feature];
        }
    }
    Ok(leftover)
}

/// Leftover SOAP used by the molecular proposal mechanism.
pub fn leftover_space(species: &[u32]) -> Result<DescriptorSpace, MolecularCatalogPresetError> {
    let spec = leftover_spec(species)?;
    let block = DescriptorBlockSpec::new(
        DescriptorBlockKind::SoapLeftover,
        spec.n_max,
        spec.l_max,
        spec.rcut_nn,
    )
    .map_err(|_| MolecularCatalogPresetError::DescriptorDimension)?;
    let schema = DescriptorSchema::new(
        LEFTOVER_DESCRIPTOR_SCHEMA,
        LEFTOVER_DESCRIPTOR_VERSION,
        vec![block],
    )
    .map_err(|_| MolecularCatalogPresetError::DescriptorDimension)?;
    Ok(DescriptorSpace::new(schema))
}

/// Fixed-dimensional universal invariant space for a water catalog.
pub fn descriptor_space(species: &[u32]) -> Result<DescriptorSpace, MolecularCatalogPresetError> {
    let geometry = DescriptorGeometry::finite(length_scale(species)?)
        .map_err(|_| MolecularCatalogPresetError::DescriptorDimension)?;
    Ok(universal_descriptor_space(geometry))
}

/// SHA-256 of the declared GFN2 handle. This is not the engine binary.
pub fn engine_config_digest() -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(ENGINE_CONFIG);
    hasher.finalize().into()
}

/// SHA-256 of a loaded `libxtb_engine.so`. Callers pass the file bytes.
pub fn engine_binary_digest(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

/// Canonical system signature for one `(H2O)m` GFN2-xTB cluster.
///
/// `engine_binary_digest` is the SHA-256 of the loaded `libxtb_engine.so`.
/// This function does not invent that digest.
pub fn system_signature(
    n_molecules: usize,
    engine_binary_digest: [u8; 32],
) -> Result<SystemSignature, MolecularCatalogPresetError> {
    let species = water_species(n_molecules)?;
    let n_atoms = species.len();
    let coordinate_dim = n_atoms
        .checked_mul(3)
        .and_then(|dimension| u64::try_from(dimension).ok())
        .ok_or(MolecularCatalogPresetError::CoordinateDimension)?;
    let scale = length_scale(&species)?;
    let descriptor = descriptor_space(&species)?;
    let reference = reference_coordinates(n_molecules)?;
    let descriptor_dim = descriptor
        .describe(ArrayView1::from(reference.as_slice()), Some(&species))
        .map_err(|_| MolecularCatalogPresetError::DescriptorDimension)?
        .values()
        .len();
    let mut hyperparameters = BTreeMap::new();
    hyperparameters.insert(
        "blocks".into(),
        "pair-radial@2.5,6;three-body-angular@3,6;graph-topology@6;\
         invariant-soap@3,6;invariant-ace-nu3@3,6;chiral-moment@3,6"
            .into(),
    );
    hyperparameters.insert("normalization".into(), "soft-l2-eps-v1".into());
    hyperparameters.insert("geometry".into(), format!("finite;length-scale={scale}"));
    hyperparameters.insert("descriptor_dim".into(), descriptor_dim.to_string());
    let mut external_inputs = BTreeMap::new();
    external_inputs.insert(ENGINE_BINARY_INPUT.into(), engine_binary_digest);
    let mut species_channels = species.clone();
    species_channels.sort_unstable();
    species_channels.dedup();
    Ok(SystemSignature {
        atomic_numbers: species,
        coordinate_dim,
        group_labels: (0..n_atoms)
            .map(|index| {
                u32::try_from(index).map_err(|_| MolecularCatalogPresetError::CoordinateDimension)
            })
            .collect::<Result<Vec<_>, _>>()?,
        group_schema: GROUP_SCHEMA.into(),
        frozen_mask: vec![false; n_atoms],
        cell: None,
        periodic: [false; 3],
        length_scale: scale,
        energy_scale: 1.0,
        engine: EngineSignature {
            kind: ENGINE_KIND.into(),
            config_digest: engine_config_digest(),
            external_inputs,
        },
        descriptor: DescriptorSignature {
            schema: descriptor.schema().name().into(),
            version: descriptor.schema().version(),
            hyperparameters,
            species_channels,
        },
        validation_schema_version: 1,
    })
}

/// Receiving-side validation settings for one water coordinate reference.
///
/// The numeric floors are the catalog validator contract, not a
/// measured GFN2 gradient or energy tolerance.
pub fn validator_config(
    reference_coordinates: &[f64],
    descriptor_dim: usize,
) -> Result<ValidatorConfig, MolecularCatalogPresetError> {
    if reference_coordinates.len() < 9 || !reference_coordinates.len().is_multiple_of(3) {
        return Err(MolecularCatalogPresetError::CoordinateDimension);
    }
    if descriptor_dim == 0 {
        return Err(MolecularCatalogPresetError::DescriptorDimension);
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

/// Nonoverlapping development reference from the isolated-water template.
pub fn reference_coordinates(n_molecules: usize) -> Result<Vec<f64>, MolecularCatalogPresetError> {
    if n_molecules == 0 {
        return Err(MolecularCatalogPresetError::InvalidMoleculeCount);
    }
    let mut coordinates = Vec::with_capacity(9 * n_molecules);
    for molecule in 0..n_molecules {
        let shift = 3.0 * molecule as f64;
        for atom in &WATER {
            coordinates.extend_from_slice(&[atom[0] + shift, atom[1], atom[2]]);
        }
    }
    Ok(coordinates)
}

/// Fresh GFN2-xTB energy and forces for receiving-side validation.
///
/// The Hamiltonian is not in this crate. Both feature configurations
/// refuse so a coordinator cannot mint an energy without the engine.
pub fn fresh_evaluation(
    n_molecules: usize,
    coordinates: &[f64],
) -> Result<FreshEvaluation, String> {
    if n_molecules == 0 || coordinates.len() != 9 * n_molecules {
        return Err("water coordinate dimension mismatch".into());
    }
    if coordinates.iter().any(|value| !value.is_finite()) {
        return Err("water coordinates must be finite".into());
    }
    #[cfg(feature = "rgpot-ex")]
    {
        Err("GFN2-xTB catalog evaluation requires a loaded rgpot engine handle".into())
    }
    #[cfg(not(feature = "rgpot-ex"))]
    {
        Err(
            "GFN2-xTB catalog evaluation requires feature rgpot-ex and a loaded rgpot engine handle"
                .into(),
        )
    }
}
