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

use super::census::{BasinCensus, WitnessId};
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

/// rgpot-core git revision in `Cargo.lock`. This develop pin is 3.0.2.
pub const RGPOT_REV: &str = "386c0342e008903472925ad78f8f32a91db51e14";

/// rgpot-core version this tree links.
pub const RGPOT_VERSION: &str = "3.0.2";

/// eindir-core git revision in `Cargo.lock`.
pub const EINDIR_REV: &str = "f3c42130bb389ba6cd6e4cfdc8b2e182f4a764e9";

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
#[cfg(test)]
mod water_hexamer_tests {
    use super::*;
    use crate::methods::cluster_hopping::{
        CheckpointAction, Ledger, MoveLibrary, SoapProposalMode,
    };
    use crate::methods::cluster_search::search_from_with_checkpoint;
    use crate::potentials::{PairKind, PairPotential};

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
