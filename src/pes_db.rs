//! Durable, content-addressed snapshots of exact PES networks.

use std::collections::BTreeMap;
use std::path::Path;

use ndarray::Array1;
use readcon_core::helpers::atomic_number_to_symbol;
use readcon_core::types::{ConFrame, ConFrameBuilder, meta};
use readcon_db::{ConCorpus, FrameKey, hash_frame_bytes};
use serde::{Deserialize, Serialize};

use crate::descriptor_space::{
    DescriptorBlockKind, DescriptorGeometry, DescriptorSpace, DescriptorVector,
};
use crate::pes_exploration::{
    MinimumRecord, PesNetwork, RideMethod, SaddleConnection, StructureContext,
};

const METADATA_KEY: &str = "anneal_pes";
const STORE_SCHEMA: &str = "anneal-pes-network";
const STORE_VERSION: u32 = 3;

/// Conversion from a surface's native units to CON v3 storage units.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PesUnits {
    /// Multiply native coordinates and cells by this value to obtain Å.
    pub length_to_angstrom: f64,
    /// Multiply native energies by this value to obtain eV.
    pub energy_to_ev: f64,
    /// Multiply native masses by this value to obtain atomic mass units.
    pub mass_to_amu: f64,
    /// Human-readable native length unit, such as `sigma` or `angstrom`.
    pub native_length: String,
    /// Human-readable native energy unit, such as `epsilon` or `eV`.
    pub native_energy: String,
    /// Human-readable native mass unit.
    pub native_mass: String,
}

impl PesUnits {
    /// Construct a finite, positive conversion contract.
    pub fn new(
        length_to_angstrom: f64,
        energy_to_ev: f64,
        mass_to_amu: f64,
        native_length: impl Into<String>,
        native_energy: impl Into<String>,
        native_mass: impl Into<String>,
    ) -> Result<Self, PesDbError> {
        let units = Self {
            length_to_angstrom,
            energy_to_ev,
            mass_to_amu,
            native_length: native_length.into(),
            native_energy: native_energy.into(),
            native_mass: native_mass.into(),
        };
        units.validate()?;
        Ok(units)
    }

    fn validate(&self) -> Result<(), PesDbError> {
        for (name, value) in [
            ("length-to-angstrom", self.length_to_angstrom),
            ("energy-to-eV", self.energy_to_ev),
            ("mass-to-amu", self.mass_to_amu),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(PesDbError::InvalidRecord(format!(
                    "{name} conversion must be finite and positive"
                )));
            }
        }
        if self.native_length.is_empty()
            || self.native_energy.is_empty()
            || self.native_mass.is_empty()
        {
            return Err(PesDbError::InvalidRecord(
                "native unit labels must be nonempty".into(),
            ));
        }
        Ok(())
    }
}

/// Reproducibility metadata repeated in every frame of one snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PesProvenance {
    /// Campaign identifier.
    pub campaign: String,
    /// Producer replica identifier.
    pub replica: u32,
    /// Last producer-global event sequence included in the snapshot.
    pub event_sequence: u64,
    /// Potential identity and parameters.
    pub potential: serde_json::Value,
    /// Exploration configuration, including ride and quench controls.
    pub exploration: serde_json::Value,
    /// Exact software versions or source revisions by component name.
    pub software: BTreeMap<String, String>,
    /// Native-to-CON unit conversion.
    pub units: PesUnits,
}

/// Content hashes produced by one committed database snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PesSnapshotReceipt {
    /// User-assigned readcon-db trajectory identifier.
    pub snapshot_id: u64,
    /// xxHash3 fingerprints of the exact stored CON frame blobs.
    pub frame_hashes: Vec<String>,
}

/// Reloaded network, provenance, and storage-integrity hashes.
#[derive(Debug, Clone)]
pub struct StoredPesNetwork {
    /// Exact stationary-point network.
    pub network: PesNetwork,
    /// Snapshot-wide producer provenance.
    pub provenance: PesProvenance,
    /// Exact stored-frame fingerprints in frame order.
    pub frame_hashes: Vec<String>,
}

/// Error preventing a network snapshot from being stored or trusted.
#[derive(Debug, thiserror::Error)]
pub enum PesDbError {
    /// readcon-db rejected an LMDB or frame operation.
    #[error(transparent)]
    Database(#[from] readcon_db::Error),
    /// JSON metadata could not be encoded or decoded.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Universal descriptor evaluation failed.
    #[error(transparent)]
    Descriptor(#[from] crate::descriptor_space::DescriptorError),
    /// CON frame construction failed.
    #[error("CON frame construction failed: {0}")]
    Frame(String),
    /// Stored metadata or graph references violate the versioned contract.
    #[error("invalid PES database record: {0}")]
    InvalidRecord(String),
}

/// One LMDB-backed corpus containing immutable PES-network snapshots.
pub struct PesNetworkDatabase {
    corpus: ConCorpus,
}

impl PesNetworkDatabase {
    /// Open or create a readcon-db corpus at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, PesDbError> {
        Ok(Self {
            corpus: ConCorpus::open(path)?,
        })
    }

    /// Append one immutable network snapshot as a readcon-db trajectory.
    pub fn write_snapshot(
        &self,
        snapshot_id: u64,
        network: &PesNetwork,
        provenance: &PesProvenance,
    ) -> Result<PesSnapshotReceipt, PesDbError> {
        provenance.units.validate()?;
        if network.minimum_count() == 0 {
            return Err(PesDbError::InvalidRecord(
                "a snapshot needs at least one minimum".into(),
            ));
        }
        validate_network(network)?;

        let mut frames = Vec::with_capacity(network.minimum_count() + network.saddle_count());
        for minimum in network.minima() {
            frames.push(frame_for_minimum(snapshot_id, minimum, provenance)?);
        }
        for saddle in network.saddles() {
            frames.push(frame_for_saddle(snapshot_id, saddle, provenance)?);
        }
        self.corpus.append_trajectory_frames_with_precision(
            snapshot_id,
            &frames,
            format!("{STORE_SCHEMA} snapshot {snapshot_id}"),
            17,
        )?;
        let frame_hashes = (0..frames.len())
            .map(|frame_idx| {
                let frame_idx = u32::try_from(frame_idx).map_err(|_| {
                    PesDbError::InvalidRecord("snapshot has more than u32::MAX frames".into())
                })?;
                Ok(self
                    .corpus
                    .frame_hash(FrameKey {
                        traj_id: snapshot_id,
                        frame_idx,
                    })?
                    .to_hex())
            })
            .collect::<Result<Vec<_>, PesDbError>>()?;
        Ok(PesSnapshotReceipt {
            snapshot_id,
            frame_hashes,
        })
    }

    /// Reload and validate one immutable network snapshot.
    pub fn read_snapshot(
        &self,
        snapshot_id: u64,
        descriptor_space: &DescriptorSpace,
    ) -> Result<StoredPesNetwork, PesDbError> {
        let trajectory = self
            .corpus
            .traj_meta(snapshot_id)?
            .ok_or_else(|| PesDbError::InvalidRecord("snapshot trajectory is missing".into()))?;
        if trajectory.n_frames == 0 {
            return Err(PesDbError::InvalidRecord(
                "snapshot trajectory is empty".into(),
            ));
        }

        let mut minima = Vec::<Option<MinimumRecord>>::new();
        let mut saddles = Vec::<Option<SaddleConnection>>::new();
        let mut provenance = None;
        let mut frame_hashes = Vec::with_capacity(trajectory.n_frames as usize);
        for frame_idx in 0..trajectory.n_frames {
            let key = FrameKey {
                traj_id: snapshot_id,
                frame_idx,
            };
            let text = self.corpus.get_frame_text(key)?;
            let expected_hash = self.corpus.frame_hash(key)?;
            if hash_frame_bytes(text.as_bytes()) != expected_hash {
                return Err(PesDbError::InvalidRecord(format!(
                    "frame {frame_idx} content hash does not match its index"
                )));
            }
            frame_hashes.push(expected_hash.to_hex());
            let frame = self.corpus.get_frame(key)?;
            let envelope = decode_envelope(&frame, snapshot_id)?;
            envelope.provenance.units.validate()?;
            match &provenance {
                None => provenance = Some(envelope.provenance.clone()),
                Some(existing) if *existing == envelope.provenance => {}
                Some(_) => {
                    return Err(PesDbError::InvalidRecord(
                        "frames disagree on snapshot provenance".into(),
                    ));
                }
            }
            let coordinates = decode_coordinates(&frame, &envelope)?;
            let context = envelope.context.to_context()?;
            validate_context(&context, coordinates.len())?;
            validate_physical_frame(&frame, &coordinates, &context, &envelope.provenance.units)?;
            if descriptor_space.geometry() != context.geometry() {
                return Err(PesDbError::InvalidRecord(
                    "descriptor payload geometry disagrees with the requested space".into(),
                ));
            }
            let descriptor = descriptor_space.describe(coordinates.view(), context.species())?;
            envelope.descriptor.validate(&descriptor)?;

            match envelope.record {
                RecordMetadata::Minimum {
                    id,
                    energy_bits,
                    max_gradient_bits,
                } => {
                    let energy = decode_finite_scalar(energy_bits, "minimum energy")?;
                    let max_gradient =
                        decode_finite_scalar(max_gradient_bits, "minimum maximum gradient")?;
                    insert_record(
                        &mut minima,
                        id,
                        MinimumRecord {
                            id,
                            energy,
                            coordinates,
                            context,
                            max_gradient,
                            descriptor,
                        },
                        "minimum",
                    )?
                }
                RecordMetadata::Saddle {
                    id,
                    origin,
                    endpoints,
                    energy_bits,
                    curvature_bits,
                    lowest_mode_bits,
                    negative_modes,
                    max_gradient_bits,
                    ride_method,
                    irc_at_minimum,
                } => {
                    let energy = decode_finite_scalar(energy_bits, "saddle energy")?;
                    let curvature = decode_finite_scalar(curvature_bits, "saddle curvature")?;
                    let max_gradient =
                        decode_finite_scalar(max_gradient_bits, "saddle maximum gradient")?;
                    let lowest_mode = decode_finite_bits(
                        &lowest_mode_bits,
                        coordinates.len(),
                        "saddle lowest mode",
                    )?;
                    insert_record(
                        &mut saddles,
                        id,
                        SaddleConnection {
                            id,
                            origin,
                            endpoints,
                            saddle_energy: energy,
                            saddle_coordinates: coordinates,
                            context,
                            curvature,
                            lowest_mode,
                            negative_modes,
                            saddle_max_gradient: max_gradient,
                            descriptor,
                            ride_method: ride_method.parse_ride_method()?,
                            irc_at_minimum,
                        },
                        "saddle",
                    )?
                }
            }
        }

        let minima = collect_records(minima, "minimum")?;
        let saddles = collect_records(saddles, "saddle")?;
        let network = PesNetwork::from_records(minima, saddles);
        validate_network(&network)?;
        Ok(StoredPesNetwork {
            network,
            provenance: provenance.expect("a nonempty trajectory sets provenance"),
            frame_hashes,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Envelope {
    schema: String,
    version: u32,
    snapshot_id: u64,
    coordinate_bits: Vec<u64>,
    context: ContextMetadata,
    descriptor: DescriptorMetadata,
    provenance: PesProvenance,
    record: RecordMetadata,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ContextMetadata {
    species: Option<Vec<u32>>,
    masses: Option<Vec<f64>>,
    geometry: Option<GeometryMetadata>,
    identity_domain: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct GeometryMetadata {
    length_scale: f64,
    cell: Option<[f64; 9]>,
    periodic: [bool; 3],
}

impl ContextMetadata {
    fn from_context(context: &StructureContext) -> Self {
        Self {
            species: context.species().map(<[u32]>::to_vec),
            masses: context.masses().map(<[f64]>::to_vec),
            geometry: context.geometry().map(|geometry| GeometryMetadata {
                length_scale: geometry.length_scale(),
                cell: geometry.cell(),
                periodic: geometry.periodic(),
            }),
            identity_domain: context.identity_domain().map(str::to_owned),
        }
    }

    fn to_context(&self) -> Result<StructureContext, PesDbError> {
        let geometry = self
            .geometry
            .as_ref()
            .map(|geometry| {
                DescriptorGeometry::new(geometry.length_scale, geometry.cell, geometry.periodic)
                    .map_err(PesDbError::from)
            })
            .transpose()?;
        Ok(
            StructureContext::new(self.species.clone(), geometry, self.identity_domain.clone())
                .with_masses(self.masses.clone()),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct DescriptorMetadata {
    schema_name: String,
    schema_version: u32,
    value_bits: Vec<u64>,
    blocks: Vec<BlockMetadata>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct BlockMetadata {
    kind: String,
    n_max: usize,
    l_max: usize,
    cutoff_bits: u64,
    offset: usize,
    len: usize,
    raw_norm_bits: u64,
    normalization: String,
}

impl DescriptorMetadata {
    fn from_descriptor(descriptor: &DescriptorVector) -> Self {
        Self {
            schema_name: descriptor.schema_name().into(),
            schema_version: descriptor.schema_version(),
            value_bits: descriptor
                .values()
                .iter()
                .map(|value| value.to_bits())
                .collect(),
            blocks: descriptor
                .blocks()
                .iter()
                .map(|block| BlockMetadata {
                    kind: block_kind_name(block.kind()).into(),
                    n_max: block.n_max(),
                    l_max: block.l_max(),
                    cutoff_bits: block.cutoff().to_bits(),
                    offset: block.offset(),
                    len: block.len(),
                    raw_norm_bits: block.raw_norm().to_bits(),
                    normalization: block.normalization().into(),
                })
                .collect(),
        }
    }

    fn validate(&self, descriptor: &DescriptorVector) -> Result<(), PesDbError> {
        let reproduced = Self::from_descriptor(descriptor);
        if self.schema_name != reproduced.schema_name
            || self.schema_version != reproduced.schema_version
        {
            return Err(PesDbError::InvalidRecord(format!(
                "descriptor schema {:?}/{} reproduces as {:?}/{}",
                self.schema_name,
                self.schema_version,
                reproduced.schema_name,
                reproduced.schema_version
            )));
        }
        if self.value_bits.len() != reproduced.value_bits.len() {
            return Err(PesDbError::InvalidRecord(format!(
                "descriptor value count {} reproduces as {}",
                self.value_bits.len(),
                reproduced.value_bits.len()
            )));
        }
        if let Some((index, (stored, computed))) = self
            .value_bits
            .iter()
            .zip(&reproduced.value_bits)
            .enumerate()
            .find(|(_, (stored, computed))| stored != computed)
        {
            return Err(PesDbError::InvalidRecord(format!(
                "descriptor value {index} has bits {:016x}, reproduced {:016x}",
                stored, computed
            )));
        }
        if self.blocks.len() != reproduced.blocks.len() {
            return Err(PesDbError::InvalidRecord(format!(
                "descriptor block count {} reproduces as {}",
                self.blocks.len(),
                reproduced.blocks.len()
            )));
        }
        if let Some((index, (stored, computed))) = self
            .blocks
            .iter()
            .zip(&reproduced.blocks)
            .enumerate()
            .find(|(_, (stored, computed))| stored != computed)
        {
            return Err(PesDbError::InvalidRecord(format!(
                "descriptor block {index} metadata differs: stored {stored:?}, reproduced {computed:?}"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum RecordMetadata {
    Minimum {
        id: usize,
        energy_bits: u64,
        max_gradient_bits: u64,
    },
    Saddle {
        id: usize,
        origin: usize,
        endpoints: [usize; 2],
        energy_bits: u64,
        curvature_bits: u64,
        lowest_mode_bits: Vec<u64>,
        negative_modes: usize,
        max_gradient_bits: u64,
        ride_method: String,
        irc_at_minimum: [bool; 2],
    },
}

fn frame_for_minimum(
    snapshot_id: u64,
    minimum: &MinimumRecord,
    provenance: &PesProvenance,
) -> Result<ConFrame, PesDbError> {
    frame_for_record(
        snapshot_id,
        &minimum.coordinates,
        &minimum.context,
        &minimum.descriptor,
        minimum.energy,
        RecordMetadata::Minimum {
            id: minimum.id,
            energy_bits: minimum.energy.to_bits(),
            max_gradient_bits: minimum.max_gradient.to_bits(),
        },
        provenance,
    )
}

fn frame_for_saddle(
    snapshot_id: u64,
    saddle: &SaddleConnection,
    provenance: &PesProvenance,
) -> Result<ConFrame, PesDbError> {
    frame_for_record(
        snapshot_id,
        &saddle.saddle_coordinates,
        &saddle.context,
        &saddle.descriptor,
        saddle.saddle_energy,
        RecordMetadata::Saddle {
            id: saddle.id,
            origin: saddle.origin,
            endpoints: saddle.endpoints,
            energy_bits: saddle.saddle_energy.to_bits(),
            curvature_bits: saddle.curvature.to_bits(),
            lowest_mode_bits: saddle
                .lowest_mode
                .iter()
                .map(|value| value.to_bits())
                .collect(),
            negative_modes: saddle.negative_modes,
            max_gradient_bits: saddle.saddle_max_gradient.to_bits(),
            ride_method: saddle.ride_method.name().into(),
            irc_at_minimum: saddle.irc_at_minimum,
        },
        provenance,
    )
}

fn frame_for_record(
    snapshot_id: u64,
    coordinates: &Array1<f64>,
    context: &StructureContext,
    descriptor: &DescriptorVector,
    energy: f64,
    record: RecordMetadata,
    provenance: &PesProvenance,
) -> Result<ConFrame, PesDbError> {
    validate_context(context, coordinates.len())?;
    if !energy.is_finite() {
        return Err(PesDbError::InvalidRecord(
            "stationary energy is nonfinite".into(),
        ));
    }
    let envelope = Envelope {
        schema: STORE_SCHEMA.into(),
        version: STORE_VERSION,
        snapshot_id,
        coordinate_bits: coordinates.iter().map(|value| value.to_bits()).collect(),
        context: ContextMetadata::from_context(context),
        descriptor: DescriptorMetadata::from_descriptor(descriptor),
        provenance: provenance.clone(),
        record,
    };
    let units = &provenance.units;
    let (cell, angles, lattice, periodic) = physical_cell(context, units)?;
    let mut metadata = BTreeMap::new();
    metadata.insert(
        meta::UNITS.into(),
        serde_json::json!({"length": "angstrom", "energy": "eV", "mass": "amu"}),
    );
    metadata.insert(
        meta::PBC.into(),
        serde_json::json!([periodic[0], periodic[1], periodic[2]]),
    );
    if let Some(lattice) = lattice {
        metadata.insert(meta::LATTICE_VECTORS.into(), serde_json::json!(lattice));
    }
    metadata.insert(METADATA_KEY.into(), serde_json::to_value(envelope)?);

    let atoms = coordinates.len() / 3;
    let species = context
        .species()
        .map(<[u32]>::to_vec)
        .unwrap_or_else(|| vec![0; atoms]);
    let masses = context
        .masses()
        .map(<[f64]>::to_vec)
        .unwrap_or_else(|| vec![1.0; atoms]);
    let mut builder = ConFrameBuilder::new(cell, angles);
    builder.prebox_header("Anneal exact PES network");
    builder.metadata(metadata);
    builder.set_energy(energy * units.energy_to_ev);
    for atom in 0..atoms {
        let symbol = atomic_number_to_symbol(u64::from(species[atom]));
        builder.add_atom(
            symbol,
            coordinates[3 * atom] * units.length_to_angstrom,
            coordinates[3 * atom + 1] * units.length_to_angstrom,
            coordinates[3 * atom + 2] * units.length_to_angstrom,
            [false; 3],
            atom as u64,
            masses[atom] * units.mass_to_amu,
        );
    }
    builder
        .build()
        .map_err(|error| PesDbError::Frame(error.to_string()))
}

fn decode_envelope(frame: &ConFrame, snapshot_id: u64) -> Result<Envelope, PesDbError> {
    let value = frame
        .header
        .metadata
        .get(METADATA_KEY)
        .ok_or_else(|| PesDbError::InvalidRecord("frame lacks anneal_pes metadata".into()))?;
    let envelope: Envelope = serde_json::from_value(value.clone())?;
    if envelope.schema != STORE_SCHEMA || envelope.version != STORE_VERSION {
        return Err(PesDbError::InvalidRecord(
            "unsupported PES snapshot schema".into(),
        ));
    }
    if envelope.snapshot_id != snapshot_id {
        return Err(PesDbError::InvalidRecord(
            "frame snapshot id disagrees with its trajectory".into(),
        ));
    }
    Ok(envelope)
}

fn decode_coordinates(frame: &ConFrame, envelope: &Envelope) -> Result<Array1<f64>, PesDbError> {
    if envelope.coordinate_bits.len() != frame.atom_data.len() * 3 {
        return Err(PesDbError::InvalidRecord(
            "native coordinate payload has the wrong dimension".into(),
        ));
    }
    let coordinates = envelope
        .coordinate_bits
        .iter()
        .map(|bits| f64::from_bits(*bits))
        .collect::<Vec<_>>();
    if coordinates.iter().any(|value| !value.is_finite()) {
        return Err(PesDbError::InvalidRecord(
            "native coordinate payload is nonfinite".into(),
        ));
    }
    Ok(Array1::from_vec(coordinates))
}

fn decode_finite_bits(
    bits: &[u64],
    expected_len: usize,
    field: &str,
) -> Result<Array1<f64>, PesDbError> {
    if bits.len() != expected_len {
        return Err(PesDbError::InvalidRecord(format!(
            "{field} has the wrong dimension"
        )));
    }
    let values = bits
        .iter()
        .map(|bits| f64::from_bits(*bits))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PesDbError::InvalidRecord(format!("{field} is nonfinite")));
    }
    Ok(Array1::from_vec(values))
}

fn decode_finite_scalar(bits: u64, field: &str) -> Result<f64, PesDbError> {
    let value = f64::from_bits(bits);
    if !value.is_finite() {
        return Err(PesDbError::InvalidRecord(format!("{field} is nonfinite")));
    }
    Ok(value)
}

fn validate_physical_frame(
    frame: &ConFrame,
    coordinates: &Array1<f64>,
    context: &StructureContext,
    units: &PesUnits,
) -> Result<(), PesDbError> {
    let atoms = coordinates.len() / 3;
    let mut seen = vec![false; atoms];
    for (stored, atom_id) in frame.atom_ids.iter().copied().enumerate() {
        let atom = usize::try_from(atom_id)
            .map_err(|_| PesDbError::InvalidRecord("atom id exceeds usize".into()))?;
        if atom >= atoms || seen[atom] {
            return Err(PesDbError::InvalidRecord(
                "atom ids are not a permutation".into(),
            ));
        }
        seen[atom] = true;
        let physical = frame.positions.as_f64_row(stored);
        for axis in 0..3 {
            let expected = coordinates[3 * atom + axis] * units.length_to_angstrom;
            let scale = expected.abs().max(1.0);
            if (physical[axis] - expected).abs() > 2e-15 * scale {
                return Err(PesDbError::InvalidRecord(
                    "physical coordinates disagree with the exact native payload".into(),
                ));
            }
        }
    }
    let (_, _, expected_lattice, expected_periodic) = physical_cell(context, units)?;
    if frame.header.pbc() != Some(expected_periodic) {
        return Err(PesDbError::InvalidRecord(
            "frame PBC disagrees with the identity context".into(),
        ));
    }
    if frame.header.lattice_vectors() != expected_lattice {
        return Err(PesDbError::InvalidRecord(
            "frame lattice disagrees with the identity context".into(),
        ));
    }
    Ok(())
}

fn validate_context(context: &StructureContext, coordinate_len: usize) -> Result<(), PesDbError> {
    if coordinate_len == 0 || !coordinate_len.is_multiple_of(3) {
        return Err(PesDbError::InvalidRecord(
            "coordinates must be nonempty 3N Cartesian".into(),
        ));
    }
    let atoms = coordinate_len / 3;
    if context
        .species()
        .is_some_and(|species| species.len() != atoms)
    {
        return Err(PesDbError::InvalidRecord(
            "species count does not match coordinates".into(),
        ));
    }
    if let Some(masses) = context.masses()
        && (masses.len() != atoms || masses.iter().any(|mass| !mass.is_finite() || *mass <= 0.0)) {
            return Err(PesDbError::InvalidRecord(
                "masses must be finite, positive, and one per atom".into(),
            ));
        }
    Ok(())
}

fn validate_network(network: &PesNetwork) -> Result<(), PesDbError> {
    for (id, minimum) in network.minima().iter().enumerate() {
        if minimum.id != id {
            return Err(PesDbError::InvalidRecord(
                "minimum ids are not contiguous".into(),
            ));
        }
        validate_context(&minimum.context, minimum.coordinates.len())?;
    }
    for (id, saddle) in network.saddles().iter().enumerate() {
        if saddle.id != id {
            return Err(PesDbError::InvalidRecord(
                "saddle ids are not contiguous".into(),
            ));
        }
        if saddle.origin >= network.minimum_count()
            || saddle
                .endpoints
                .iter()
                .any(|endpoint| *endpoint >= network.minimum_count())
        {
            return Err(PesDbError::InvalidRecord(
                "saddle references a missing minimum".into(),
            ));
        }
        if saddle.negative_modes != 1 {
            return Err(PesDbError::InvalidRecord(
                "stored saddle is not certified index one".into(),
            ));
        }
        if saddle.lowest_mode.len() != saddle.saddle_coordinates.len()
            || saddle.lowest_mode.iter().any(|value| !value.is_finite())
            || saddle.lowest_mode.dot(&saddle.lowest_mode) <= f64::EPSILON
        {
            return Err(PesDbError::InvalidRecord(
                "stored saddle has an invalid lowest mode".into(),
            ));
        }
        validate_context(&saddle.context, saddle.saddle_coordinates.len())?;
    }
    Ok(())
}

fn physical_cell(
    context: &StructureContext,
    units: &PesUnits,
) -> Result<([f64; 3], [f64; 3], Option<[[f64; 3]; 3]>, [bool; 3]), PesDbError> {
    let Some(geometry) = context.geometry() else {
        return Ok(([1.0; 3], [90.0; 3], None, [false; 3]));
    };
    let periodic = geometry.periodic();
    let Some(cell) = geometry.cell() else {
        return Ok(([1.0; 3], [90.0; 3], None, periodic));
    };
    let scale = units.length_to_angstrom;
    let lattice = [
        [cell[0] * scale, cell[1] * scale, cell[2] * scale],
        [cell[3] * scale, cell[4] * scale, cell[5] * scale],
        [cell[6] * scale, cell[7] * scale, cell[8] * scale],
    ];
    let lengths = [norm(lattice[0]), norm(lattice[1]), norm(lattice[2])];
    let angles = [
        angle_degrees(lattice[1], lattice[2])?,
        angle_degrees(lattice[0], lattice[2])?,
        angle_degrees(lattice[0], lattice[1])?,
    ];
    Ok((lengths, angles, Some(lattice), periodic))
}

fn norm(vector: [f64; 3]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn angle_degrees(left: [f64; 3], right: [f64; 3]) -> Result<f64, PesDbError> {
    let denominator = norm(left) * norm(right);
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(PesDbError::InvalidRecord(
            "cell contains a zero-length lattice vector".into(),
        ));
    }
    let cosine = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f64>()
        / denominator;
    Ok(cosine.clamp(-1.0, 1.0).acos().to_degrees())
}

fn insert_record<T>(
    records: &mut Vec<Option<T>>,
    id: usize,
    record: T,
    kind: &str,
) -> Result<(), PesDbError> {
    if records.len() <= id {
        records.resize_with(id + 1, || None);
    }
    if records[id].is_some() {
        return Err(PesDbError::InvalidRecord(format!(
            "duplicate {kind} id {id}"
        )));
    }
    records[id] = Some(record);
    Ok(())
}

fn collect_records<T>(records: Vec<Option<T>>, kind: &str) -> Result<Vec<T>, PesDbError> {
    records
        .into_iter()
        .enumerate()
        .map(|(id, record)| {
            record.ok_or_else(|| PesDbError::InvalidRecord(format!("missing {kind} id {id}")))
        })
        .collect()
}

fn block_kind_name(kind: DescriptorBlockKind) -> &'static str {
    match kind {
        DescriptorBlockKind::SoapMean => "soap_mean",
        DescriptorBlockKind::SoapVariance => "soap_variance",
        DescriptorBlockKind::AceNu3Mean => "ace_nu3_mean",
        DescriptorBlockKind::SoapLeftover => "soap_leftover",
        DescriptorBlockKind::PairRadial => "pair_radial",
        DescriptorBlockKind::ThreeBodyAngular => "three_body_angular",
        DescriptorBlockKind::GraphTopology => "graph_topology",
        DescriptorBlockKind::InvariantSoapMean => "invariant_soap_mean",
        DescriptorBlockKind::InvariantAceNu3Mean => "invariant_ace_nu3_mean",
        DescriptorBlockKind::ChiralMoment => "chiral_moment",
        DescriptorBlockKind::ProviderFeature => "provider_feature",
    }
}

impl RideMethod {
    fn name(self) -> &'static str {
        match self {
            Self::Dimer => "dimer",
            Self::Lanczos => "lanczos",
        }
    }
}

trait ParseRideMethod {
    fn parse_ride_method(&self) -> Result<RideMethod, PesDbError>;
}

impl ParseRideMethod for str {
    fn parse_ride_method(&self) -> Result<RideMethod, PesDbError> {
        match self {
            "dimer" => Ok(RideMethod::Dimer),
            "lanczos" => Ok(RideMethod::Lanczos),
            _ => Err(PesDbError::InvalidRecord(format!(
                "unknown ride method {self:?}"
            ))),
        }
    }
}
