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

use super::census::BasinCensus;
use super::{
    DescriptorSignature, EngineSignature, FreshEvaluation, SystemSignature, ValidatorConfig,
};
use crate::descriptor_space::{
    DescriptorBlockKind, DescriptorBlockSpec, DescriptorGeometry, DescriptorSchema, DescriptorSpace,
};
#[cfg(feature = "featomic")]
use crate::descriptor_space::{
    FEATOMIC_SOAP_SCHEMA, FEATOMIC_SOAP_VERSION, FeatomicSoapProvider, FeatomicSoapScale,
};
#[cfg(not(feature = "featomic"))]
use crate::descriptor_space::{
    UNIVERSAL_DESCRIPTOR_SCHEMA, UNIVERSAL_DESCRIPTOR_VERSION, universal_descriptor_space,
};
use crate::methods::cluster_hopping::Config;
use crate::methods::cluster_hopping::covalent_radius;
use crate::methods::minima_hopping::SharedVisitPolicy;
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
    /// The isolated-water template is a start, not a deposited isomer.
    #[error("the isolated-water template is not an isomer reference")]
    IsolatedTemplate,
    /// Species-aware IRA is the assignment. This build did not link it.
    #[error("isomer assignment needs the ira feature")]
    AssignmentNeedsIra,
    /// IRA refused the two point sets.
    #[error("species-aware IRA match failed")]
    IraMatch,
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

/// Receiving-side Euclidean gradient-norm gate for GFN2 stationary points.
pub const MAX_GRADIENT_NORM: f64 = 1e-5;

/// Vacuum box edge used by `molecular_cluster` (A).
pub const VACUUM_BOX: f64 = 60.0;

/// Paper water-hexamer molecule count.
pub const WATER_HEXAMER_MOLECULES: usize = 6;

/// Componentwise solver tolerance that guarantees [`MAX_GRADIENT_NORM`].
pub fn component_gradient_tolerance(
    coordinate_dim: usize,
) -> Result<f64, MolecularCatalogPresetError> {
    if coordinate_dim == 0 {
        return Err(MolecularCatalogPresetError::CoordinateDimension);
    }
    Ok(MAX_GRADIENT_NORM / (coordinate_dim as f64).sqrt())
}

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

/// Catalog descriptor schema selected by this build.
#[cfg(not(feature = "featomic"))]
pub const DESCRIPTOR_SCHEMA: &str = UNIVERSAL_DESCRIPTOR_SCHEMA;

/// Featomic-backed catalog descriptor schema used by production builds.
#[cfg(feature = "featomic")]
pub const DESCRIPTOR_SCHEMA: &str = FEATOMIC_SOAP_SCHEMA;

/// Schema version of [`DESCRIPTOR_SCHEMA`].
#[cfg(not(feature = "featomic"))]
pub const DESCRIPTOR_VERSION: u32 = UNIVERSAL_DESCRIPTOR_VERSION;

/// Schema version of [`DESCRIPTOR_SCHEMA`].
#[cfg(feature = "featomic")]
pub const DESCRIPTOR_VERSION: u32 = FEATOMIC_SOAP_VERSION;

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
    #[cfg(feature = "featomic")]
    {
        let mut channels = species.to_vec();
        channels.sort_unstable();
        channels.dedup();
        let provider = FeatomicSoapProvider::new(
            channels,
            vec![
                FeatomicSoapScale::new(3.0, 3, 4)
                    .map_err(|_| MolecularCatalogPresetError::DescriptorDimension)?,
                FeatomicSoapScale::new(6.0, 3, 6)
                    .map_err(|_| MolecularCatalogPresetError::DescriptorDimension)?,
            ],
        )
        .map_err(|_| MolecularCatalogPresetError::DescriptorDimension)?;
        DescriptorSpace::from_provider(geometry, provider)
            .map_err(|_| MolecularCatalogPresetError::DescriptorDimension)
    }
    #[cfg(not(feature = "featomic"))]
    {
        Ok(universal_descriptor_space(geometry))
    }
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
    hyperparameters.insert("geometry".into(), format!("finite;length-scale={scale}"));
    hyperparameters.insert("descriptor_dim".into(), descriptor_dim.to_string());
    if let Some(contract) = descriptor.provider_contract() {
        hyperparameters.insert("provider".into(), contract.schema_name().into());
        hyperparameters.insert("model_sha256".into(), contract.model_digest_hex());
        hyperparameters.insert(
            "system_output".into(),
            format!(
                "{};dimension={}",
                contract.system_output().name(),
                contract.system_output().dimension()
            ),
        );
        if let Some(output) = contract.atomic_output() {
            hyperparameters.insert(
                "atomic_output".into(),
                format!("{};dimension={}", output.name(), output.dimension()),
            );
        }
        hyperparameters.insert(
            "interaction_range".into(),
            contract.interaction_range().to_string(),
        );
        hyperparameters.insert("normalization".into(), contract.normalization().into());
    } else {
        hyperparameters.insert(
            "blocks".into(),
            "pair-radial@2.5,6;three-body-angular@3,6;graph-topology@6;\
             invariant-soap@3,6;invariant-ace-nu3@3,6;chiral-moment@3,6"
                .into(),
        );
        hyperparameters.insert("normalization".into(), "contractive-l2-unit-v2".into());
    }
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
        max_gradient_norm: MAX_GRADIENT_NORM,
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

/// Maximum O-H separation that still counts as a bond in the integrity check (A).
///
/// A quenched hexamer fails when an oxygen is left with one hydrogen or three.
pub const WATER_OH_BOND_MAX_A: f64 = 1.2;

/// First seed of the water-hexamer block. Inclusive end is 31.
///
/// IDEAS H5 names 0-31. An earlier plan used 5100-5131 because the
/// registry had been keeping paper seeds in high blocks. IDEAS wins.
pub const WATER_HEXAMER_SEED_START: u64 = 0;

/// Seeds per engine. The block is 0-31.
pub const WATER_HEXAMER_SEED_COUNT: u64 = 32;

/// Force budget per seed, the same on every engine.
///
/// FIXES R6 logged 1200 PBE evaluations and 4000 GFN2 evaluations.
/// The outline sets 4000 for every engine, and that is the budget here.
pub const WATER_HEXAMER_BUDGET: usize = 4000;

/// GFN2-xTB force budget. Same as [`WATER_HEXAMER_BUDGET`].
pub const GFN2_HEXAMER_BUDGET: usize = WATER_HEXAMER_BUDGET;

/// PBE/6-31G* force budget. Same as [`WATER_HEXAMER_BUDGET`].
pub const PBE_HEXAMER_BUDGET: usize = WATER_HEXAMER_BUDGET;

/// PET-MAD force budget. Same as [`WATER_HEXAMER_BUDGET`].
pub const PETMAD_HEXAMER_BUDGET: usize = WATER_HEXAMER_BUDGET;

/// NWChem embed engine id used by the rgpot profile.
pub const PBE_ENGINE_KIND: &str = "nwchemc";

/// Gaussian basis named for the PBE hexamer rerun.
pub const PBE_BASIS: &str = "6-31g*";

/// NWChem theory arm for the PBE hexamer rerun.
pub const PBE_THEORY: &str = "dft";

/// Exchange-correlation functional. `scfType` when `theory` is `dft`.
pub const PBE_XC: &str = "pbe";

/// No dispersion correction is named for the PBE/6-31G* rerun.
pub const PBE_DISPERSION: &str = "none named";

/// Molecular charge of neutral (H2O)6.
pub const PBE_CHARGE: i32 = 0;

/// Spin multiplicity 2S+1 of singlet water.
pub const PBE_MULTIPLICITY: i32 = 1;

/// NWChem task. The frontend calls gradient.
pub const PBE_TASK: &str = "gradient";

/// Metatomic engine family for the PET-MAD hexamer rerun.
pub const PETMAD_ENGINE_KIND: &str = "metatomic-rgpot-v1";

/// Model file named by the potlib metatomic dlopen test.
pub const PETMAD_MODEL_NAME: &str = "pet-mad-xs-v1.5.0.pt";

/// Device recorded for the PET-MAD force call.
pub const PETMAD_DEVICE: &str = "cpu";

/// rgpot-core checksum in `Cargo.lock`. The crate is registry 3.5.1.
pub const RGPOT_REV: &str = "df34626c5c9d888c234e6a0921e7e6c0335561ec50460fcf74e4b77772c89ede";

/// rgpot-core version this tree links.
pub const RGPOT_VERSION: &str = "3.5.1";

/// eindir-core checksum in `Cargo.lock`. The crate is registry 0.6.0.
pub const EINDIR_REV: &str = "e330e3abad9c1a597b144d40eb4b07ab3511d79f7d2cdbbcb37a44662f974db0";

/// eindir-core version this tree links.
pub const EINDIR_VERSION: &str = "0.6.0";

/// Isomer names the hexamer assignment knows.
pub const WATER_ISOMER_NAMES: [&str; 6] = [
    "prism",
    "cage",
    "book",
    "bag",
    "cyclic-chair",
    "cyclic-boat",
];

/// FIXES R6 wrote an RMSD gate of 0.1 A in a proposed sentence.
///
/// The outline replaces that gate. The assignment cutoff is half the
/// minimum IRA RMSD between distinct same-level references, computed
/// after those references are relaxed and before any search seed.
/// No deposited references are in this tree, so no cutoff is stored.
pub const ISOMER_RMSD_GATE_NAMED_BY_FIXES_A: f64 = 0.1;

/// Census radius on the sorted oxygen-oxygen distances (A).
///
/// This is the Good-Turing descriptor radius for the water arm. It is
/// not the isomer assignment cutoff.
pub const OXYGEN_DISTANCE_CENSUS_RADIUS_A: f64 = 0.1;

/// Charged-evaluation interval of the Good-Turing checkpoint.
pub const GOOD_TURING_CHECKPOINT_INTERVAL: usize = 1;

/// Search arm on the molecular preset.
///
/// Communicating and derived keep the molecular groups. They do not
/// install [`Config::communicating`] or [`Config::derived`] as atomic
/// presets. Good-Turing uses the recommended flags and a census stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaterSearchArm {
    /// [`Config::recommended_molecular`].
    Recommended,
    /// Orbit, Recognition, and the return screen, on the molecular library.
    Communicating,
    /// Budget-window temperature and the cost-asymmetric Bayes screen.
    Derived,
    /// Recommended flags, plus the production Good-Turing stop.
    GoodTuring,
}

/// Overlay one water arm on a molecular configuration.
///
/// The move library is left as the caller set it. The atomic two-phase
/// surface is not copied onto these arms.
pub fn configure_water_arm(cfg: &mut Config, arm: WaterSearchArm) {
    match arm {
        WaterSearchArm::Recommended | WaterSearchArm::GoodTuring => {}
        WaterSearchArm::Communicating => {
            cfg.depth_reward = false;
            cfg.allocate_moves = true;
            cfg.return_screen = true;
            cfg.orbit_complete_on_new = true;
            cfg.shared_deposits = 0;
            cfg.shared_visit_policy = SharedVisitPolicy::Recognition;
            cfg.jump_on_stall = false;
        }
        WaterSearchArm::Derived => {
            cfg.budget_window = true;
            cfg.bayes_screen = true;
            cfg.bayes_threshold =
                crate::screen::cost_asymmetric_threshold(cfg.screen_steps, cfg.relax_steps);
        }
    }
}

/// Whether every oxygen has exactly two hydrogens within [`WATER_OH_BOND_MAX_A`].
pub fn monomers_intact(
    coordinates: &[f64],
    n_molecules: usize,
) -> Result<bool, MolecularCatalogPresetError> {
    let points = water_points(coordinates, n_molecules)?;
    let limit2 = WATER_OH_BOND_MAX_A * WATER_OH_BOND_MAX_A;
    for molecule in &points {
        let oxygen = molecule[0];
        let mut near = 0usize;
        for other in &points {
            for hydrogen in [other[1], other[2]] {
                if dist2(oxygen, hydrogen) <= limit2 {
                    near += 1;
                }
            }
        }
        if near != 2 {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Sorted oxygen-oxygen distances of `(H2O)m`. The Good-Turing descriptor.
pub fn sorted_oxygen_distances(
    coordinates: &[f64],
    n_molecules: usize,
) -> Result<Vec<f64>, MolecularCatalogPresetError> {
    let points = water_points(coordinates, n_molecules)?;
    let mut distances = Vec::with_capacity(n_molecules * n_molecules.saturating_sub(1) / 2);
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            distances.push(dist2(points[i][0], points[j][0]).sqrt());
        }
    }
    distances.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    Ok(distances)
}

/// `Some("saturated")` when the production Good-Turing rule is met.
pub fn good_turing_stop_reason(census: &BasinCensus) -> Option<&'static str> {
    census.is_saturated().then_some("saturated")
}

/// One isomer match, or none when no reference is within the gate.
#[derive(Debug, Clone, PartialEq)]
pub struct IsomerMatch {
    /// Name from [`WATER_ISOMER_NAMES`], when the RMSD is inside the gate.
    pub name: Option<String>,
    /// Lowest species-aware IRA RMSD among the references, in angstroms.
    pub rmsd: Option<f64>,
}

/// Same label for a search geometry and a deposited GOFEE geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassifiedGeometry {
    /// Every oxygen has exactly two hydrogens within [`WATER_OH_BOND_MAX_A`].
    pub intact: bool,
    /// Isomer name when the geometry is intact and inside the gate.
    pub name: Option<String>,
    /// Species-aware IRA RMSD to the nearest reference, when one was computed.
    pub rmsd: Option<f64>,
}

/// Half the smallest positive gap.
///
/// A zero gap is the same geometry twice, not a separation between
/// distinct references. No positive gap leaves the gate unset.
pub fn assignment_cutoff(gaps: &[f64]) -> Option<f64> {
    let mut min = f64::INFINITY;
    for gap in gaps {
        if gap.is_finite() && *gap > 0.0 && *gap < min {
            min = *gap;
        }
    }
    min.is_finite().then_some(0.5 * min)
}

/// Assignment gate from deposited references, before any search seed.
///
/// Fewer than two references cannot form a gap, so the gate is unset.
/// Two or more references need species-aware IRA. A build without that
/// feature returns [`MolecularCatalogPresetError::AssignmentNeedsIra`]
/// and does not fall back to the Kabsch helper.
pub fn isomer_reference_gate(
    references: &[(&str, &[f64])],
    n_molecules: usize,
) -> Result<Option<f64>, MolecularCatalogPresetError> {
    if references.len() < 2 {
        return Ok(None);
    }
    reject_template_references(references, n_molecules)?;
    #[cfg(not(feature = "ira"))]
    {
        Err(MolecularCatalogPresetError::AssignmentNeedsIra)
    }
    #[cfg(feature = "ira")]
    {
        let species = water_species(n_molecules)?;
        let mut gaps = Vec::new();
        for left in 0..references.len() {
            for right in (left + 1)..references.len() {
                gaps.push(ira_water_rmsd(
                    references[left].1,
                    references[right].1,
                    &species,
                )?);
            }
        }
        Ok(assignment_cutoff(&gaps))
    }
}

/// Assign a hexamer by species-aware IRA against same-level references.
///
/// An empty reference list is unassigned. A dissociated monomer is
/// unassigned and does not receive an RMSD. `cutoff` is the engine's
/// gate from [`isomer_reference_gate`]. `None` reports the nearest RMSD
/// and does not name an isomer. The isolated-water template is not a
/// valid entry of `references`. Primitive-ring counts are not this label.
pub fn assign_isomer(
    candidate: &[f64],
    n_molecules: usize,
    references: &[(&str, &[f64])],
    cutoff: Option<f64>,
) -> Result<IsomerMatch, MolecularCatalogPresetError> {
    if references.is_empty() {
        return Ok(IsomerMatch {
            name: None,
            rmsd: None,
        });
    }
    if !monomers_intact(candidate, n_molecules)? {
        return Ok(IsomerMatch {
            name: None,
            rmsd: None,
        });
    }
    reject_template_references(references, n_molecules)?;
    for (name, _) in references {
        if !WATER_ISOMER_NAMES.contains(name) {
            return Err(MolecularCatalogPresetError::SpeciesMismatch);
        }
    }
    #[cfg(not(feature = "ira"))]
    {
        let _ = cutoff;
        Err(MolecularCatalogPresetError::AssignmentNeedsIra)
    }
    #[cfg(feature = "ira")]
    {
        let species = water_species(n_molecules)?;
        let mut best_name: Option<&str> = None;
        let mut best = f64::INFINITY;
        for (name, reference) in references {
            let rmsd = ira_water_rmsd(reference, candidate, &species)?;
            if rmsd < best {
                best = rmsd;
                best_name = Some(*name);
            }
        }
        let inside = cutoff.is_some_and(|gate| best.is_finite() && best <= gate);
        Ok(IsomerMatch {
            name: inside.then(|| best_name.unwrap_or("").to_owned()),
            rmsd: best.is_finite().then_some(best),
        })
    }
}

/// Classify one geometry the way a search seed and a GOFEE dump are both scored.
pub fn classify_geometry(
    candidate: &[f64],
    n_molecules: usize,
    references: &[(&str, &[f64])],
    cutoff: Option<f64>,
) -> Result<ClassifiedGeometry, MolecularCatalogPresetError> {
    let intact = monomers_intact(candidate, n_molecules)?;
    if !intact {
        return Ok(ClassifiedGeometry {
            intact: false,
            name: None,
            rmsd: None,
        });
    }
    let matched = assign_isomer(candidate, n_molecules, references, cutoff)?;
    Ok(ClassifiedGeometry {
        intact: true,
        name: matched.name,
        rmsd: matched.rmsd,
    })
}

fn reject_template_references(
    references: &[(&str, &[f64])],
    n_molecules: usize,
) -> Result<(), MolecularCatalogPresetError> {
    let template = reference_coordinates(n_molecules)?;
    if references
        .iter()
        .any(|(_, coordinates)| *coordinates == template.as_slice())
    {
        return Err(MolecularCatalogPresetError::IsolatedTemplate);
    }
    Ok(())
}

#[cfg(feature = "ira")]
fn ira_water_rmsd(
    reference: &[f64],
    candidate: &[f64],
    species: &[u32],
) -> Result<f64, MolecularCatalogPresetError> {
    if reference.len() != candidate.len() || reference.len() != 3 * species.len() {
        return Err(MolecularCatalogPresetError::CoordinateDimension);
    }
    crate::shape::match_shapes_typed(
        ArrayView1::from(reference),
        species,
        ArrayView1::from(candidate),
        species,
        1.8,
    )
    .map(|matched| matched.distance)
    .map_err(|_| MolecularCatalogPresetError::IraMatch)
}

/// Kabsch RMSD of two `(H2O)m` geometries (A).
///
/// This is not the isomer assignment. Molecules may be reordered. The
/// two hydrogens of a molecule may swap. Alignment is the Kabsch
/// rotation, inversion included, after both clouds are centred.
pub fn water_rmsd(
    reference: &[f64],
    candidate: &[f64],
    n_molecules: usize,
) -> Result<f64, MolecularCatalogPresetError> {
    let left = water_points(reference, n_molecules)?;
    let right = water_points(candidate, n_molecules)?;
    let mut order = vec![0usize; n_molecules];
    let mut used = vec![false; n_molecules];
    let mut flips = vec![false; n_molecules];
    let mut best = f64::INFINITY;
    assign_molecules(
        &left, &right, 0, &mut order, &mut used, &mut flips, &mut best,
    );
    Ok(best)
}

fn water_points(
    coordinates: &[f64],
    n_molecules: usize,
) -> Result<Vec<[[f64; 3]; 3]>, MolecularCatalogPresetError> {
    if n_molecules == 0 {
        return Err(MolecularCatalogPresetError::InvalidMoleculeCount);
    }
    if coordinates.len() != 9 * n_molecules || coordinates.iter().any(|value| !value.is_finite()) {
        return Err(MolecularCatalogPresetError::CoordinateDimension);
    }
    let mut molecules = Vec::with_capacity(n_molecules);
    for molecule in 0..n_molecules {
        let base = 9 * molecule;
        let mut points = [[0.0; 3]; 3];
        for atom in 0..3 {
            for axis in 0..3 {
                points[atom][axis] = coordinates[base + 3 * atom + axis];
            }
        }
        molecules.push(points);
    }
    Ok(molecules)
}

fn assign_molecules(
    reference: &[[[f64; 3]; 3]],
    candidate: &[[[f64; 3]; 3]],
    placed: usize,
    order: &mut [usize],
    used: &mut [bool],
    flips: &mut [bool],
    best: &mut f64,
) {
    let n = reference.len();
    if placed == n {
        choose_flips(reference, candidate, order, flips, 0, best);
        return;
    }
    for index in 0..n {
        if used[index] {
            continue;
        }
        used[index] = true;
        order[placed] = index;
        assign_molecules(reference, candidate, placed + 1, order, used, flips, best);
        used[index] = false;
    }
}

fn choose_flips(
    reference: &[[[f64; 3]; 3]],
    candidate: &[[[f64; 3]; 3]],
    order: &[usize],
    flips: &mut [bool],
    molecule: usize,
    best: &mut f64,
) {
    if molecule == reference.len() {
        let rmsd = aligned_rmsd(reference, candidate, order, flips);
        if rmsd < *best {
            *best = rmsd;
        }
        return;
    }
    flips[molecule] = false;
    choose_flips(reference, candidate, order, flips, molecule + 1, best);
    flips[molecule] = true;
    choose_flips(reference, candidate, order, flips, molecule + 1, best);
}

fn aligned_rmsd(
    reference: &[[[f64; 3]; 3]],
    candidate: &[[[f64; 3]; 3]],
    order: &[usize],
    flips: &[bool],
) -> f64 {
    let mut left = Vec::with_capacity(3 * reference.len());
    let mut right = Vec::with_capacity(3 * reference.len());
    for (molecule, (&source, &flip)) in order.iter().zip(flips.iter()).enumerate() {
        left.extend_from_slice(&reference[molecule]);
        let group = candidate[source];
        if flip {
            right.extend_from_slice(&[group[0], group[2], group[1]]);
        } else {
            right.extend_from_slice(&group);
        }
    }
    let left = centre(&left);
    let right = centre(&right);
    let mut covariance = [[0.0; 3]; 3];
    for (from, to) in right.iter().zip(left.iter()) {
        for row in 0..3 {
            for col in 0..3 {
                covariance[row][col] += from[row] * to[col];
            }
        }
    }
    let rotation = kabsch_rotation(covariance);
    let mut flipped = rotation;
    for row in 0..3 {
        flipped[row][2] = -flipped[row][2];
    }
    [
        identity(),
        rotation,
        transpose(rotation),
        flipped,
        transpose(flipped),
    ]
    .into_iter()
    .map(|candidate| point_rmsd(&left, &right, candidate))
    .fold(f64::INFINITY, f64::min)
}

fn point_rmsd(left: &[[f64; 3]], right: &[[f64; 3]], rotation: [[f64; 3]; 3]) -> f64 {
    let mut error = 0.0;
    for (to, from) in left.iter().zip(right.iter()) {
        let mapped = rotate(&rotation, *from);
        for axis in 0..3 {
            let delta = mapped[axis] - to[axis];
            error += delta * delta;
        }
    }
    (error / left.len() as f64).sqrt()
}

fn identity() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn transpose(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = matrix[col][row];
        }
    }
    out
}

/// Proper rotation from the covariance, by a 3x3 Jacobi SVD.
///
/// A planar water template makes one singular value zero. The missing
/// axis is the cross product of the other two, with the sign that keeps
/// the determinant positive.
fn kabsch_rotation(covariance: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut gram = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            for axis in 0..3 {
                gram[row][col] += covariance[axis][row] * covariance[axis][col];
            }
        }
    }
    let (vectors, values) = jacobi_eigen(gram);
    let mut left = [[0.0; 3]; 3];
    let mut filled = [false; 3];
    for col in 0..3 {
        let sigma = values[col].max(0.0).sqrt();
        if sigma <= 1e-10 {
            continue;
        }
        for row in 0..3 {
            for axis in 0..3 {
                left[row][col] += covariance[row][axis] * vectors[axis][col];
            }
            left[row][col] /= sigma;
        }
        filled[col] = true;
    }
    complete_columns(&mut left, &filled);
    let mut rotation = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            for axis in 0..3 {
                rotation[row][col] += left[row][axis] * vectors[col][axis];
            }
        }
    }
    rotation
}

fn complete_columns(columns: &mut [[f64; 3]; 3], filled: &[bool; 3]) {
    let present: Vec<usize> = (0..3).filter(|index| filled[*index]).collect();
    let missing: Vec<usize> = (0..3).filter(|index| !filled[*index]).collect();
    if missing.is_empty() {
        return;
    }
    if present.len() == 2 {
        let a = column(columns, present[0]);
        let b = column(columns, present[1]);
        let mut cross = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
        let norm = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        if norm > 1e-12 {
            for axis in 0..3 {
                cross[axis] /= norm;
                columns[axis][missing[0]] = cross[axis];
            }
        }
        return;
    }
    for (slot, axis) in missing.iter().zip(0..3) {
        columns[axis][*slot] = 1.0;
    }
}

fn column(matrix: &[[f64; 3]; 3], index: usize) -> [f64; 3] {
    [matrix[0][index], matrix[1][index], matrix[2][index]]
}

fn jacobi_eigen(mut matrix: [[f64; 3]; 3]) -> ([[f64; 3]; 3], [f64; 3]) {
    let mut vectors = identity();
    for _ in 0..16 {
        let pairs = [(0, 1), (0, 2), (1, 2)];
        let mut pivot = (0, 1);
        let mut biggest = 0.0;
        for pair in pairs {
            let value = matrix[pair.0][pair.1].abs();
            if value > biggest {
                biggest = value;
                pivot = pair;
            }
        }
        if biggest < 1e-15 {
            break;
        }
        jacobi_rotate(&mut matrix, &mut vectors, pivot.0, pivot.1);
    }
    (vectors, [matrix[0][0], matrix[1][1], matrix[2][2]])
}

fn jacobi_rotate(matrix: &mut [[f64; 3]; 3], vectors: &mut [[f64; 3]; 3], p: usize, q: usize) {
    let off = matrix[p][q];
    let diag_p = matrix[p][p];
    let diag_q = matrix[q][q];
    let tau = (diag_q - diag_p) / (2.0 * off);
    let step = if tau >= 0.0 {
        1.0 / (tau + (1.0 + tau * tau).sqrt())
    } else {
        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
    };
    let cosine = 1.0 / (1.0 + step * step).sqrt();
    let sine = step * cosine;
    matrix[p][p] = diag_p - step * off;
    matrix[q][q] = diag_q + step * off;
    matrix[p][q] = 0.0;
    matrix[q][p] = 0.0;
    for row in 0..3 {
        if row == p || row == q {
            continue;
        }
        let left = matrix[row][p];
        let right = matrix[row][q];
        matrix[row][p] = cosine * left - sine * right;
        matrix[p][row] = matrix[row][p];
        matrix[row][q] = sine * left + cosine * right;
        matrix[q][row] = matrix[row][q];
    }
    for row in 0..3 {
        let left = vectors[row][p];
        let right = vectors[row][q];
        vectors[row][p] = cosine * left - sine * right;
        vectors[row][q] = sine * left + cosine * right;
    }
}

fn centre(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let scale = points.len() as f64;
    let mut shift = [0.0; 3];
    for point in points {
        for axis in 0..3 {
            shift[axis] += point[axis];
        }
    }
    for axis in 0..3 {
        shift[axis] /= scale;
    }
    points
        .iter()
        .map(|point| {
            [
                point[0] - shift[0],
                point[1] - shift[1],
                point[2] - shift[2],
            ]
        })
        .collect()
}

fn rotate(matrix: &[[f64; 3]; 3], point: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * point[0] + matrix[1][0] * point[1] + matrix[2][0] * point[2],
        matrix[0][1] * point[0] + matrix[1][1] * point[1] + matrix[2][1] * point[2],
        matrix[0][2] * point[0] + matrix[1][2] * point[1] + matrix[2][2] * point[2],
    ]
}

fn dist2(left: [f64; 3], right: [f64; 3]) -> f64 {
    let mut total = 0.0;
    for axis in 0..3 {
        let delta = left[axis] - right[axis];
        total += delta * delta;
    }
    total
}

#[cfg(test)]
mod water_hexamer_tests {
    use super::*;
    use crate::methods::cluster_hopping::{
        CheckpointAction, Ledger, MoveLibrary, SoapProposalMode,
    };
    use crate::methods::cluster_search::search_from_with_checkpoint;
    use crate::potentials::{PairKind, PairPotential};

    #[test]
    fn the_template_hexamer_is_intact_and_a_loose_hydrogen_is_not() {
        let coordinates = reference_coordinates(WATER_HEXAMER_MOLECULES).unwrap();
        assert!(monomers_intact(&coordinates, WATER_HEXAMER_MOLECULES).unwrap());
        let mut broken = coordinates.clone();
        broken[3] += 5.0;
        assert!(!monomers_intact(&broken, WATER_HEXAMER_MOLECULES).unwrap());
        let mut crowded = coordinates.clone();
        crowded[12] = 0.2;
        crowded[13] = 0.2;
        crowded[14] = 0.0;
        assert!(!monomers_intact(&crowded, WATER_HEXAMER_MOLECULES).unwrap());
    }

    #[test]
    fn oxygen_distances_are_sorted_and_the_stop_follows_the_production_rule() {
        let coordinates = reference_coordinates(WATER_HEXAMER_MOLECULES).unwrap();
        let distances = sorted_oxygen_distances(&coordinates, WATER_HEXAMER_MOLECULES).unwrap();
        assert_eq!(distances.len(), 15);
        assert!(distances.windows(2).all(|pair| pair[0] <= pair[1]));
        let mut census = BasinCensus::new(1, OXYGEN_DISTANCE_CENSUS_RADIUS_A).unwrap();
        assert!(good_turing_stop_reason(&census).is_none());
        for index in 0..20 {
            census.observe(&[index as f64]).unwrap();
        }
        assert!(good_turing_stop_reason(&census).is_none());
        let mut repeated = BasinCensus::new(1, OXYGEN_DISTANCE_CENSUS_RADIUS_A).unwrap();
        for _ in 0..20 {
            repeated.observe(&[0.0]).unwrap();
        }
        assert_eq!(good_turing_stop_reason(&repeated), Some("saturated"));
    }

    #[test]
    fn rmsd_is_zero_after_a_rigid_motion_or_a_molecule_swap() {
        let coordinates = reference_coordinates(2).unwrap();
        assert!(water_rmsd(&coordinates, &coordinates, 2).unwrap() < 1e-8);
        let mut moved = coordinates.clone();
        for value in moved.iter_mut() {
            *value += 3.0;
        }
        for atom in 0..6 {
            let x = moved[3 * atom];
            let y = moved[3 * atom + 1];
            moved[3 * atom] = -y;
            moved[3 * atom + 1] = x;
        }
        assert!(
            water_rmsd(&coordinates, &moved, 2).unwrap() < 1e-6,
            "{}",
            water_rmsd(&coordinates, &moved, 2).unwrap()
        );
        let mut swapped = coordinates.clone();
        let (head, tail) = swapped.split_at_mut(9);
        head.swap_with_slice(tail);
        assert!(water_rmsd(&coordinates, &swapped, 2).unwrap() < 1e-6);
        let mut flipped = coordinates.clone();
        flipped.swap(3, 6);
        flipped.swap(4, 7);
        flipped.swap(5, 8);
        assert!(water_rmsd(&coordinates, &flipped, 2).unwrap() < 1e-6);
        let mut shifted = coordinates.clone();
        shifted[3] += 0.4;
        assert!(water_rmsd(&coordinates, &shifted, 2).unwrap() > ISOMER_RMSD_GATE_NAMED_BY_FIXES_A);
        let mut book = coordinates.clone();
        for value in book.iter_mut().take(9) {
            *value += 0.4;
        }
        assert!(monomers_intact(&book, 2).unwrap());
        let named = assign_isomer(&book, 2, &[("book", book.as_slice())], Some(1.0));
        #[cfg(not(feature = "ira"))]
        assert_eq!(named, Err(MolecularCatalogPresetError::AssignmentNeedsIra));
        #[cfg(feature = "ira")]
        {
            let named = named.unwrap();
            assert_eq!(named.name.as_deref(), Some("book"));
            assert!(named.rmsd.unwrap() < 1e-6);
        }
        let waiting = assign_isomer(&book, 2, &[("book", book.as_slice())], None);
        #[cfg(not(feature = "ira"))]
        assert_eq!(
            waiting,
            Err(MolecularCatalogPresetError::AssignmentNeedsIra)
        );
        #[cfg(feature = "ira")]
        {
            let waiting = waiting.unwrap();
            assert!(waiting.name.is_none());
            assert!(waiting.rmsd.unwrap() < 1e-6);
        }
        assert!(assign_isomer(&book, 2, &[], None).unwrap().name.is_none());
        assert_eq!(
            assign_isomer(
                &coordinates,
                2,
                &[("book", coordinates.as_slice())],
                Some(1.0)
            ),
            Err(MolecularCatalogPresetError::IsolatedTemplate)
        );
    }

    #[test]
    fn a_dissociated_monomer_is_unassigned_for_search_and_gofee() {
        let mut book = reference_coordinates(2).unwrap();
        for value in book.iter_mut().take(9) {
            *value += 0.4;
        }
        let mut broken = book.clone();
        broken[3] += 5.0;
        assert!(!monomers_intact(&broken, 2).unwrap());
        let references = [("book", book.as_slice())];
        let search = classify_geometry(&broken, 2, &references, Some(1.0)).unwrap();
        let gofee = classify_geometry(&broken, 2, &references, Some(1.0)).unwrap();
        assert_eq!(search, gofee);
        assert!(!search.intact);
        assert!(search.name.is_none());
        assert!(search.rmsd.is_none());
        assert_eq!(assignment_cutoff(&[]), None);
        assert_eq!(assignment_cutoff(&[0.0, 0.0]), None);
        assert_eq!(assignment_cutoff(&[1.0, 0.5, 0.0]), Some(0.25));
        assert!(isomer_reference_gate(&[], 2).unwrap().is_none());
    }

    #[test]
    fn water_arms_keep_the_molecular_library() {
        let species = water_species(WATER_HEXAMER_MOLECULES).unwrap();
        let groups = water_groups(WATER_HEXAMER_MOLECULES).unwrap();
        let recommended = Config::recommended_molecular(species.clone(), groups.clone(), 1.0);
        let library = recommended.move_library.clone();
        for arm in [
            WaterSearchArm::Recommended,
            WaterSearchArm::Communicating,
            WaterSearchArm::Derived,
            WaterSearchArm::GoodTuring,
        ] {
            let mut cfg = Config::recommended_molecular(species.clone(), groups.clone(), 1.0);
            configure_water_arm(&mut cfg, arm);
            assert_eq!(cfg.move_library, library);
            assert!(cfg.surfaces.is_empty());
            match &cfg.move_library {
                MoveLibrary::Molecular { groups, reactive } => {
                    assert_eq!(groups.len(), WATER_HEXAMER_MOLECULES);
                    assert!(!reactive);
                }
                other => panic!("molecular library replaced by {other:?}"),
            }
        }
        let mut communicating = Config::recommended_molecular(species.clone(), groups.clone(), 1.0);
        configure_water_arm(&mut communicating, WaterSearchArm::Communicating);
        assert!(!communicating.depth_reward);
        assert!(communicating.return_screen);
        assert!(communicating.orbit_complete_on_new);
        assert_eq!(communicating.shared_deposits, 0);
        assert!(matches!(
            communicating.shared_visit_policy,
            SharedVisitPolicy::Recognition
        ));
        assert!(!communicating.jump_on_stall);
        let mut derived = Config::recommended_molecular(species, groups, 1.0);
        configure_water_arm(&mut derived, WaterSearchArm::Derived);
        assert!(derived.budget_window);
        assert!(derived.bayes_screen);
        assert!(derived.depth_reward);
        let threshold =
            crate::screen::cost_asymmetric_threshold(derived.screen_steps, derived.relax_steps);
        assert!((derived.bayes_threshold - threshold).abs() < 1e-12);
    }

    #[test]
    fn a_saturated_census_retires_the_search_before_the_budget() {
        let n_molecules = 2;
        let species = water_species(n_molecules).unwrap();
        let groups = water_groups(n_molecules).unwrap();
        let mut cfg = Config::recommended_molecular(species, groups, 1.0);
        cfg.relax_steps = 0;
        cfg.screen_steps = 0;
        cfg.soap_mode = SoapProposalMode::Off;
        cfg.adaptive_height = false;
        cfg.allocate_moves = false;
        cfg.depth_reward = false;
        let start = reference_coordinates(n_molecules).unwrap();
        let objective = PairPotential::new(3 * n_molecules, PairKind::LennardJones, 8.0);
        let budget = 2_000;
        let mut ledger = Ledger::new(budget);
        let mut census = BasinCensus::new(1, OXYGEN_DISTANCE_CENSUS_RADIUS_A).unwrap();
        for _ in 0..20 {
            census.observe(&[0.0]).unwrap();
        }
        let mut checkpoint =
            |_: crate::methods::cluster_hopping::ChainCheckpoint<'_>| CheckpointAction::Retire {
                reason: good_turing_stop_reason(&census)
                    .unwrap_or("continuing")
                    .to_owned(),
            };
        let (_out, _stats) = search_from_with_checkpoint(
            &objective,
            &cfg,
            &mut ledger,
            ndarray::ArrayView1::from(&start),
            5100,
            GOOD_TURING_CHECKPOINT_INTERVAL,
            &mut checkpoint,
        )
        .expect("finite start");
        assert!(
            ledger.spent() < budget / 2,
            "retire spent {} of {budget}",
            ledger.spent()
        );
        assert!(ledger.spent() > 0);
    }

    #[test]
    fn engine_pins_match_the_lockfile() {
        let lock = include_str!("../../Cargo.lock");
        assert!(lock.contains(RGPOT_REV));
        assert!(lock.contains(EINDIR_REV));
        assert_eq!(PBE_DISPERSION, "none named");
        assert_eq!(PBE_BASIS, "6-31g*");
        assert_eq!(PBE_XC, "pbe");
        assert_eq!(GFN2_METHOD, 3);
        assert_eq!(WATER_HEXAMER_SEED_START + WATER_HEXAMER_SEED_COUNT - 1, 31);
        assert_eq!(PBE_HEXAMER_BUDGET, 4000);
        assert_eq!(GFN2_HEXAMER_BUDGET, 4000);
    }
}
