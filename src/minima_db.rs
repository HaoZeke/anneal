//! Durable corpus of quenched minima, one trajectory per system,
//! temperature and seed, on readcon-db.
//!
//! Every search run leaves its validated minima here rather than in a log:
//! the structure, its plain energy to full precision, the system it belongs
//! to, the temperature of the walk that found it and the seed. A trajectory
//! per seed keeps concurrent seeds from contending on one trajectory, and a
//! system-and-temperature query folds the seeds back together. Energies are
//! stored as exact bit patterns beside the CON energy field, so a minimum
//! read back is the one written.

use std::collections::BTreeMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use ndarray::ArrayView1;
use readcon_core::helpers::atomic_number_to_symbol;
use readcon_core::types::{ConFrameBuilder, meta};
use readcon_db::{ConCorpus, FrameKey};
use serde::{Deserialize, Serialize};

const METADATA_KEY: &str = "anneal_minima";
const SCHEMA: &str = "anneal-minima";
const VERSION: u32 = 1;
/// A box large enough that no cluster used here touches it; the frames are
/// nonperiodic and the box is a format requirement, not a cell.
const OPEN_BOX: f64 = 500.0;

/// Which run a set of minima came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MinimaSet {
    /// System name, such as `lj75` or `water6-gfn2`.
    pub system: String,
    /// Metropolis temperature of the walk, in the objective's energy units.
    pub temperature: f64,
    /// Seed of the run.
    pub seed: u64,
}

impl MinimaSet {
    /// The trajectory this set writes to.
    pub fn trajectory_id(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        (SCHEMA, &self.system, self.temperature.to_bits(), self.seed).hash(&mut hasher);
        hasher.finish()
    }
}

/// Native units of the stored values, recorded beside them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MinimaUnits {
    /// Name of the length unit the coordinates carry, such as `sigma`.
    pub length: String,
    /// Name of the energy unit, such as `epsilon`.
    pub energy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Envelope {
    schema: String,
    version: u32,
    set: MinimaSet,
    units: MinimaUnits,
    energy_bits: u64,
    coordinate_bits: Vec<u64>,
    provenance: serde_json::Value,
}

/// One stored minimum.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredMinimum {
    /// The run that found it.
    pub set: MinimaSet,
    /// Exact plain energy.
    pub energy: f64,
    /// Exact coordinates.
    pub coordinates: Vec<f64>,
}

/// Errors of the minima corpus.
#[derive(Debug, thiserror::Error)]
pub enum MinimaDbError {
    /// The underlying corpus failed.
    #[error("minima corpus failed: {0}")]
    Database(#[from] readcon_db::Error),
    /// A record could not be encoded or decoded.
    #[error("minima record is invalid: {0}")]
    InvalidRecord(String),
    /// Metadata serialization failed.
    #[error("minima metadata failed: {0}")]
    Json(#[from] serde_json::Error),
}

/// A readcon-db corpus of minima.
pub struct MinimaCorpus {
    corpus: Arc<ConCorpus>,
    path: PathBuf,
}

/// One open environment per path per process: the store refuses a second
/// environment on the same path, and a driver that loops over seeds opens
/// the corpus once per seed.
fn registry() -> &'static Mutex<BTreeMap<PathBuf, Arc<ConCorpus>>> {
    static REGISTRY: OnceLock<Mutex<BTreeMap<PathBuf, Arc<ConCorpus>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(BTreeMap::new()))
}

impl MinimaCorpus {
    /// Open or create the corpus at `path`; a second open of the same path in
    /// one process shares the first environment.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, MinimaDbError> {
        let path = path.as_ref().to_path_buf();
        let canonical = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
        let mut open = registry().lock().expect("minima corpus registry");
        let corpus = match open.get(&canonical) {
            Some(shared) => Arc::clone(shared),
            None => {
                let shared = Arc::new(ConCorpus::open(&path)?);
                let canonical = std::fs::canonicalize(&path).unwrap_or(canonical);
                open.insert(canonical, Arc::clone(&shared));
                shared
            }
        };
        Ok(Self { corpus, path })
    }

    /// Where the corpus lives.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append the distinct minima of one run.
    ///
    /// Exact repeated coordinates whose energies differ by at most
    /// `energy_tolerance` are stored once within a call. Equal energy alone is
    /// not a structure identity test. Returns how many frames were appended.
    pub fn record(
        &self,
        set: &MinimaSet,
        species: &[u32],
        units: &MinimaUnits,
        entries: &[(f64, ArrayView1<'_, f64>)],
        energy_tolerance: f64,
        provenance: serde_json::Value,
    ) -> Result<usize, MinimaDbError> {
        let mut kept: Vec<(f64, ArrayView1<'_, f64>)> = Vec::new();
        for &(energy, coordinates) in entries {
            if !energy.is_finite() {
                return Err(MinimaDbError::InvalidRecord(
                    "minimum energy is nonfinite".into(),
                ));
            }
            if coordinates.len() % 3 != 0 || coordinates.is_empty() {
                return Err(MinimaDbError::InvalidRecord(format!(
                    "coordinates have length {}",
                    coordinates.len()
                )));
            }
            if kept.iter().any(|(seen_energy, seen_coordinates)| {
                (seen_energy - energy).abs() <= energy_tolerance
                    && seen_coordinates.len() == coordinates.len()
                    && seen_coordinates
                        .iter()
                        .zip(coordinates.iter())
                        .all(|(seen, value)| seen.to_bits() == value.to_bits())
            }) {
                continue;
            }
            kept.push((energy, coordinates));
        }
        if kept.is_empty() {
            return Ok(0);
        }
        kept.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut frames = Vec::with_capacity(kept.len());
        for (energy, coordinates) in &kept {
            let atoms = coordinates.len() / 3;
            let envelope = Envelope {
                schema: SCHEMA.into(),
                version: VERSION,
                set: set.clone(),
                units: units.clone(),
                energy_bits: energy.to_bits(),
                coordinate_bits: coordinates.iter().map(|v| v.to_bits()).collect(),
                provenance: provenance.clone(),
            };
            let mut metadata = BTreeMap::new();
            metadata.insert(
                meta::UNITS.into(),
                serde_json::json!({"length": units.length, "energy": units.energy}),
            );
            metadata.insert(meta::PBC.into(), serde_json::json!([false, false, false]));
            metadata.insert(METADATA_KEY.into(), serde_json::to_value(&envelope)?);
            let mut builder = ConFrameBuilder::new([OPEN_BOX; 3], [90.0; 3]);
            builder.prebox_header(format!(
                "Anneal minimum {} T={} seed={}",
                set.system, set.temperature, set.seed
            ));
            builder.metadata(metadata);
            builder.set_energy(*energy);
            for atom in 0..atoms {
                let z = species.get(atom).copied().unwrap_or(0);
                builder.add_atom(
                    atomic_number_to_symbol(u64::from(z)),
                    coordinates[3 * atom] + OPEN_BOX / 2.0,
                    coordinates[3 * atom + 1] + OPEN_BOX / 2.0,
                    coordinates[3 * atom + 2] + OPEN_BOX / 2.0,
                    [false; 3],
                    atom as u64,
                    1.0,
                );
            }
            frames.push(
                builder
                    .build()
                    .map_err(|error| MinimaDbError::InvalidRecord(error.to_string()))?,
            );
        }
        let source = serde_json::to_string(set)?;
        let appended = self.corpus.extend_trajectory_frames(
            set.trajectory_id(),
            &frames,
            format!("{SCHEMA} {source}"),
        )?;
        let _ = appended;
        Ok(frames.len())
    }

    /// Every stored minimum of one system at one temperature, across seeds,
    /// sorted by energy.
    pub fn minima(
        &self,
        system: &str,
        temperature: f64,
    ) -> Result<Vec<StoredMinimum>, MinimaDbError> {
        let mut found = Vec::new();
        for key in self.corpus.list_frame_keys()? {
            if let Some(minimum) = self.decode(key)?
                && minimum.set.system == system
                && minimum.set.temperature.to_bits() == temperature.to_bits()
            {
                found.push(minimum);
            }
        }
        found.sort_by(|a, b| a.energy.total_cmp(&b.energy));
        Ok(found)
    }

    /// Distinct energies of one system at one temperature, folded across
    /// seeds at `tolerance`, lowest first.
    pub fn distinct_energies(
        &self,
        system: &str,
        temperature: f64,
        tolerance: f64,
    ) -> Result<Vec<f64>, MinimaDbError> {
        let mut distinct: Vec<f64> = Vec::new();
        for minimum in self.minima(system, temperature)? {
            if distinct
                .last()
                .is_none_or(|last| (minimum.energy - last).abs() > tolerance)
            {
                distinct.push(minimum.energy);
            }
        }
        Ok(distinct)
    }

    fn decode(&self, key: FrameKey) -> Result<Option<StoredMinimum>, MinimaDbError> {
        let frame = self.corpus.get_frame(key)?;
        let Some(value) = frame.header.metadata.get(METADATA_KEY) else {
            return Ok(None);
        };
        let envelope: Envelope = serde_json::from_value(value.clone())?;
        if envelope.schema != SCHEMA {
            return Ok(None);
        }
        Ok(Some(StoredMinimum {
            set: envelope.set,
            energy: f64::from_bits(envelope.energy_bits),
            coordinates: envelope
                .coordinate_bits
                .iter()
                .map(|bits| f64::from_bits(*bits))
                .collect(),
        }))
    }
}
